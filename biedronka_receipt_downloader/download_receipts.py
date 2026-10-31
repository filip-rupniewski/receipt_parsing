"""
Pobiera paragony z https://moja.biedronka.pl/panel/paragons i zapisuje każdy
jako osobny plik JSON w ./biedronka_receipts_jsons. Paragony już pobrane są pomijane.

Logowanie: numer telefonu + hasło z credentials.json (plik w .gitignore).
Strona logowania ma captchę Cloudflare Turnstile, która odrzuca przeglądarki
sterowane przez Playwright. Dlatego Chromium startuje jako zwykły proces z trwałym
profilem (.browser_profile); skrypt tylko wpisuje dane, a captchę i przyciski
klikasz sam. Po zalogowaniu skrypt przejmuje przeglądarkę i pobiera paragony.

Użycie:
    uv run download_receipts.py              # pobierz nowe paragony
    uv run download_receipts.py --discover   # zrzut ruchu sieciowego (diagnostyka)
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import re
import sys
import subprocess
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import BrowserContext, Page, Response, sync_playwright

BASE_DIR = Path(__file__).resolve().parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
PROFILE_DIR = BASE_DIR / ".browser_profile"
OUTPUT_DIR = BASE_DIR / "biedronka_receipts_jsons"
DISCOVERY_DIR = BASE_DIR / "discovery"

PANEL_URL = "https://moja.biedronka.pl/panel/paragons"
PANEL_HOST = "moja.biedronka.pl"
LOGIN_HOST = "konto.biedronka.pl"
LOGIN_TIMEOUT_S = 300  # czas na ręczne dokończenie logowania (captcha itp.)


# --------------------------------------------------------------------------- #
# Konfiguracja
# --------------------------------------------------------------------------- #
def load_credentials() -> dict:
    if not CREDENTIALS_FILE.exists():
        sys.exit(
            f"Brak pliku {CREDENTIALS_FILE}.\n"
            f"Skopiuj credentials.example.json -> credentials.json i uzupełnij dane."
        )
    creds = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
    for key in ("phone", "password"):
        if not creds.get(key):
            sys.exit(f"W {CREDENTIALS_FILE.name} brakuje pola '{key}'.")
    return creds


# Logowanie
# --------------------------------------------------------------------------- #
# Cloudflare Turnstile odrzuca przeglądarkę uruchomioną przez Playwright, więc
# Chromium startuje jako zwykły proces (bez flag automatyzacji). Skrypt podłącza
# się do niego przez CDP tylko na chwilę, żeby wpisać dane, i od razu się
# rozłącza - captchę i przycisk "Dalej"/"Zaloguj" klikasz sam. Na stałe łączy się
# dopiero po zalogowaniu, kiedy captcha już nie występuje.
CDP_PORT = 9229
CDP_URL = f"http://127.0.0.1:{CDP_PORT}"


def host_of(url: str) -> str:
    return urlparse(url).hostname or ""


def is_panel(url: str) -> bool:
    return host_of(url) == PANEL_HOST and "/oauth/" not in url


def start_browser() -> subprocess.Popen:
    with sync_playwright() as p:
        exe = p.chromium.executable_path
    PROFILE_DIR.mkdir(exist_ok=True)
    proc = subprocess.Popen(
        [
            exe,
            f"--user-data-dir={PROFILE_DIR}",
            f"--remote-debugging-port={CDP_PORT}",
            "--no-sandbox",  # Ubuntu blokuje sandbox Chromium z Playwright
            "--no-first-run",
            "--no-default-browser-check",
            "--lang=pl-PL",
            "--window-size=1280,950",
            PANEL_URL,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(100):
        if current_tab_url() is not None:
            return proc
        time.sleep(0.2)
    proc.terminate()
    sys.exit("Nie udało się uruchomić Chromium.")


def current_tab() -> dict | None:
    """Karta przeglądarki (url, title) - przez HTTP DevTools, bez podłączania się."""
    try:
        with urllib.request.urlopen(f"{CDP_URL}/json/list", timeout=2) as r:
            tabs = json.load(r)
    except Exception:
        return None
    pages = [t for t in tabs if t.get("type") == "page"]
    return pages[0] if pages else {"url": "", "title": ""}


def current_tab_url() -> str | None:
    tab = current_tab()
    return None if tab is None else tab.get("url", "")


def fill_once(selector: str, value: str, check_remember: bool = False) -> bool:
    """Podłącza się na chwilę, wpisuje wartość w pole i rozłącza się."""
    pw = sync_playwright().start()
    try:
        browser = pw.chromium.connect_over_cdp(CDP_URL)
        page = browser.contexts[0].pages[0]
        field = page.locator(selector).first
        if not field.is_visible():
            return False
        field.fill(value)
        if check_remember:
            remember = page.locator("#rememberMe")
            if remember.count() and not remember.is_checked():
                remember.check(timeout=1000)
        return True
    except Exception:
        return False
    finally:
        pw.stop()  # rozłącza, przeglądarka działa dalej


def login(autofill: bool, creds: dict | None) -> None:
    deadline = time.time() + LOGIN_TIMEOUT_S
    phone_done = password_done = False
    announced = False
    panel_streak = 0
    time.sleep(2)  # daj stronie czas na ewentualne przekierowanie do logowania
    while time.time() < deadline:
        tab = current_tab()
        if tab is None:
            sys.exit("Przeglądarka została zamknięta.")
        url, title = tab.get("url", ""), tab.get("title", "")
        # adres panelu pojawia się też chwilę przed przekierowaniem do logowania,
        # więc wymagamy załadowanej strony (tytuł) i stabilnego stanu
        if is_panel(url) and title and "zaloguj" not in title.lower():
            panel_streak += 1
            if panel_streak >= 2:
                print("Zalogowano." if announced else "Sesja aktywna - logowanie pominięte.")
                return
        else:
            panel_streak = 0
        if host_of(url) == LOGIN_HOST:
            if not announced:
                print("Zaloguj się w oknie przeglądarki "
                      "(skrypt wpisze dane, Ty klikasz captchę i przyciski).")
                announced = True
            if autofill and not phone_done:
                phone_done = fill_once("input[name=phoneNumber]", creds["phone"], True)
                if phone_done:
                    print("  wpisano numer - kliknij weryfikację / 'Dalej'")
            elif autofill and not password_done:
                password_done = fill_once("input[type=password]", creds["password"])
                if password_done:
                    print("  wpisano hasło - kliknij 'Zaloguj'")
        time.sleep(1.5)
    sys.exit("Nie udało się zalogować w wyznaczonym czasie.")


# --------------------------------------------------------------------------- #
# Tryb diagnostyczny: zapis całego ruchu JSON z panelu
# --------------------------------------------------------------------------- #
def discover(context: BrowserContext, page: Page) -> None:
    DISCOVERY_DIR.mkdir(exist_ok=True)
    log: list[dict] = []

    def on_response(resp: Response) -> None:
        if host_of(resp.url) not in (PANEL_HOST,) and "biedronka" not in resp.url:
            return
        ctype = resp.headers.get("content-type", "")
        entry = {
            "url": resp.url,
            "method": resp.request.method,
            "status": resp.status,
            "content_type": ctype,
            "request_post_data": resp.request.post_data,
        }
        entry["content_disposition"] = resp.headers.get("content-disposition")
        if "json" in ctype or (entry["content_disposition"] and "pdf" not in ctype):
            try:
                entry["body"] = resp.json()
            except Exception:
                try:
                    entry["body_text"] = resp.text()[:5000]
                except Exception:
                    pass
        log.append(entry)

    context.on("response", on_response)

    page.goto(PANEL_URL, wait_until="networkidle")
    page.wait_for_timeout(3000)
    (DISCOVERY_DIR / "panel_paragons.html").write_text(page.content(), encoding="utf-8")
    page.screenshot(path=str(DISCOVERY_DIR / "panel_paragons.png"), full_page=True)

    links = page.eval_on_selector_all(
        "a[href]", "els => els.map(e => ({href: e.href, text: e.innerText.trim()}))"
    )
    (DISCOVERY_DIR / "links.json").write_text(
        json.dumps(links, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def save_log() -> None:
        (DISCOVERY_DIR / "network_log.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(
        "\nPrzeglądarka zostaje otwarta na 60 s - kliknij w jeden-dwa paragony "
        "(pobierz JSON i PDF), żeby zarejestrować ich zapytania."
    )
    try:
        # wait_for_timeout (a nie time.sleep), żeby Playwright obsługiwał zdarzenia
        for _ in range(60):
            open_pages = [pg for pg in context.pages if not pg.is_closed()]
            if not open_pages:
                break
            open_pages[0].wait_for_timeout(1000)
            save_log()
        (DISCOVERY_DIR / "last_page.html").write_text(
            open_pages[0].content(), encoding="utf-8")
    except Exception as e:
        print(f"(przerwano: {e.__class__.__name__})")
    finally:
        save_log()
    print(f"Zapisano {len(log)} odpowiedzi do {DISCOVERY_DIR}/")


# --------------------------------------------------------------------------- #
# Pobieranie paragonów
# --------------------------------------------------------------------------- #
def safe_filename(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", str(name)).strip("_")


LIST_URL = "https://moja.biedronka.pl/panel/ajax/paragons"
JSON_URL = "https://moja.biedronka.pl/panel/download/json/{id}"
WINDOW_DAYS = 365  # zakres dat jednego zapytania o listę


def list_receipts(page: Page, since: date) -> list[dict]:
    """Lista paragonów (pola Id, DateTime, TotalValue, Store, ...) od `since` do dziś."""
    receipts: dict[str, dict] = {}
    end = date.today()
    while end >= since:
        begin = max(since, end - timedelta(days=WINDOW_DAYS - 1))
        page_no, n_pages = 0, 1
        while page_no < n_pages:
            resp = page.request.post(
                LIST_URL,
                form={
                    "page": str(page_no),
                    "begin_data": begin.isoformat(),
                    "end_data": end.isoformat(),
                    "order": "desc",
                    "short": "0",  # "1" (domyślne w panelu) pomija część paragonów
                },
                headers={"X-Requested-With": "XMLHttpRequest", "Referer": PANEL_URL},
            )
            if not resp.ok:
                sys.exit(f"Błąd listy paragonów ({begin}..{end}): HTTP {resp.status}")
            data = resp.json()
            for r in data.get("Receipts") or []:
                receipts[str(r["Id"])] = r
            n_pages = int(data.get("NumberOfPages") or 0)
            page_no += 1
            page.wait_for_timeout(200)
        print(f"  {begin} .. {end}: łącznie {len(receipts)} paragonów")
        end = begin - timedelta(days=1)
    return sorted(receipts.values(), key=lambda r: r.get("DateTime") or "")


def fetch_receipt_json(page: Page, receipt_id: str) -> bytes:
    """Surowy plik JSON e-paragonu - dokładnie to, co daje przycisk 'JSON' w panelu."""
    resp = page.request.get(JSON_URL.format(id=receipt_id), headers={"Referer": PANEL_URL})
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status}")
    body = resp.body()
    json.loads(body)  # upewnij się, że to JSON (a nie np. strona logowania)
    return body


def receipt_path(receipt_id: str) -> Path:
    return OUTPUT_DIR / f"paragon_{safe_filename(receipt_id)}.json"


def download_new(page: Page, since: date) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    page.goto(PANEL_URL, wait_until="domcontentloaded")
    print(f"Pobieram listę paragonów od {since}...")
    receipts = list_receipts(page, since)
    new = [r for r in receipts if not receipt_path(str(r["Id"])).exists()]
    print(f"Paragonów w panelu: {len(receipts)}, już pobranych: "
          f"{len(receipts) - len(new)}, nowych: {len(new)}")

    failed = []
    for i, r in enumerate(new, 1):
        rid = str(r["Id"])
        try:
            body = fetch_receipt_json(page, rid)
        except Exception as e:
            print(f"  [{i}/{len(new)}] {rid}: BŁĄD {e}")
            failed.append(rid)
            continue
        out = receipt_path(rid)
        tmp = out.with_suffix(".json.part")
        tmp.write_bytes(body)
        tmp.replace(out)
        print(f"  [{i}/{len(new)}] {out.name}  ({r.get('DateTime', '')[:10]}, "
              f"{r.get('TotalValue')} zł)")
        page.wait_for_timeout(300)  # nie zasypujemy serwera zapytaniami
    if failed:
        print(f"Nie udało się pobrać {len(failed)} paragonów: {', '.join(failed)}")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--discover", action="store_true",
                        help="zapisz ruch sieciowy panelu do ./discovery (diagnostyka)")
    parser.add_argument("--manual-login", action="store_true",
                        help="nie wpisuj danych automatycznie, zaloguj się ręcznie")
    parser.add_argument("--since", default="2019-01-01",
                        help="najstarsza data paragonów do sprawdzenia (RRRR-MM-DD)")
    args = parser.parse_args()

    creds = None if args.manual_login else load_credentials()
    proc = start_browser()
    try:
        login(autofill=not args.manual_login, creds=creds)
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(CDP_URL)
            context = browser.contexts[0]
            page = context.pages[0]
            if args.discover:
                discover(context, page)
            else:
                download_new(page, date.fromisoformat(args.since))
    finally:
        proc.terminate()

if __name__ == "__main__":
    main()
