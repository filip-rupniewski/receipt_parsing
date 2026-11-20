"""
Pobiera listę zakupów z https://allegro.pl/moje-allegro/zakupy/kupione.

Dane logowania NIE są trzymane w plikach projektu. Do wyboru:
  * systemowy pęk kluczy (GNOME Keyring / KWallet, przez bibliotekę `keyring`),
    zapisany raz poleceniem `--save-credentials`; hasło jest zaszyfrowane
    i odblokowywane razem z sesją użytkownika,
  * wpisywanie loginu i hasła w terminalu przy każdym logowaniu (getpass, bez echa),
    gdy w pęku kluczy nic nie ma albo z flagą `--ask`,
  * `--manual-login`: skrypt niczego nie wpisuje, logujesz się sam w oknie przeglądarki.
Hasło jest potrzebne tylko wtedy, gdy sesja w .browser_profile/ wygasła.

Allegro chroni się przed botami (DataDome), więc Chromium startuje jako zwykły
proces z trwałym profilem. Skrypt podłącza się do niego przez CDP tylko na chwilę,
żeby wpisać dane. Przycisk „Zaloguj”, captchę i kod 2FA obsługujesz sam.
Po zalogowaniu skrypt przegląda stronę „Kupione”, zapisuje odpowiedzi JSON
z danymi zamówień i wyciąga z nich listę zakupów.

Użycie:
    uv run download_purchases.py                    # pobierz zakupy
    uv run download_purchases.py --save-credentials # zapisz login i hasło w pęku kluczy
    uv run download_purchases.py --forget-credentials
    uv run download_purchases.py --ask              # zapytaj o dane, nie używaj pęku kluczy
    uv run download_purchases.py --manual-login     # loguj się ręcznie w przeglądarce
    uv run download_purchases.py --days 60          # przeglądaj 60 dni wstecz (domyślnie 30)
    uv run download_purchases.py --full             # przejrzyj całą listę (odśwież statusy)
    uv run download_purchases.py --rebuild          # odtwórz CSV z pobranych plików
    uv run download_purchases.py --discover         # zrzut ruchu sieciowego (diagnostyka)
"""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from playwright.sync_api import BrowserContext, Page, Response, sync_playwright

BASE_DIR = Path(__file__).resolve().parent
PROFILE_DIR = BASE_DIR / ".browser_profile"
# Dane osobowe trzymamy w ./dane: ignoruje je repozytorium z kodem, a ich historia
# jest w prywatnym, lokalnym repo .git-private w katalogu projektu. Patrz commit_data().
DATA_DIR = BASE_DIR / "dane"
RAW_DIR = DATA_DIR / "allegro_raw"          # surowe odpowiedzi JSON ze strony „Kupione”
OUTPUT_DIR = DATA_DIR / "allegro_purchases"  # wynik: zamówienia i pozycje
DISCOVERY_DIR = BASE_DIR / "discovery"

PURCHASES_URL = "https://allegro.pl/moje-allegro/zakupy/kupione"
LOGIN_TIMEOUT_S = 300  # czas na ręczne dokończenie logowania (captcha, 2FA)

KEYRING_SERVICE = "receipt_parsing-allegro"
KEYRING_USER_KEY = "__login__"  # pod tym kluczem trzymamy sam login


# --------------------------------------------------------------------------- #
# Dane logowania
# --------------------------------------------------------------------------- #
def _keyring():
    try:
        import keyring
        from keyring.backends import fail
    except ImportError:
        return None
    if isinstance(keyring.get_keyring(), fail.Keyring):
        return None  # brak działającego pęku kluczy w systemie
    return keyring


def save_credentials() -> None:
    kr = _keyring()
    if kr is None:
        sys.exit("Brak systemowego pęku kluczy (Secret Service). "
                 "Użyj --ask albo --manual-login.")
    login = input("Login Allegro (e-mail lub nazwa użytkownika): ").strip()
    password = getpass.getpass("Hasło Allegro: ")
    if not login or not password:
        sys.exit("Login i hasło nie mogą być puste.")
    kr.set_password(KEYRING_SERVICE, KEYRING_USER_KEY, login)
    kr.set_password(KEYRING_SERVICE, login, password)
    print(f"Zapisano w pęku kluczy (usługa '{KEYRING_SERVICE}').")


def forget_credentials() -> None:
    kr = _keyring()
    if kr is None:
        sys.exit("Brak systemowego pęku kluczy.")
    login = kr.get_password(KEYRING_SERVICE, KEYRING_USER_KEY)
    for key in filter(None, (login, KEYRING_USER_KEY)):
        try:
            kr.delete_password(KEYRING_SERVICE, key)
        except Exception:
            pass
    print("Usunięto dane logowania z pęku kluczy.")


class LazyCredentials:
    """Pobiera dane dopiero wtedy, gdy są potrzebne (sesja wygasła)."""

    def __init__(self, ask: bool):
        self.ask = ask
        self._value: tuple[str, str] | None = None

    def get(self) -> tuple[str, str]:
        if self._value is None:
            self._value = self._load()
        return self._value

    def _load(self) -> tuple[str, str]:
        if not self.ask:
            kr = _keyring()
            if kr is not None:
                login = kr.get_password(KEYRING_SERVICE, KEYRING_USER_KEY)
                password = kr.get_password(KEYRING_SERVICE, login) if login else None
                if login and password:
                    print("  dane logowania: systemowy pęk kluczy")
                    return login, password
        print("  podaj dane logowania (nie zostaną zapisane):")
        login = input("  Login Allegro: ").strip()
        password = getpass.getpass("  Hasło Allegro: ")
        return login, password


# --------------------------------------------------------------------------- #
# Przeglądarka i logowanie
# --------------------------------------------------------------------------- #
CDP_PORT = 9230  # inny niż w biedronka_receipt_downloader (9229)
CDP_URL = f"http://127.0.0.1:{CDP_PORT}"

LOGIN_SELECTORS = ["input#login", "input[name=login]", "input[type=email]",
                   "input[autocomplete=username]"]
PASSWORD_SELECTORS = ["input#password", "input[name=password]", "input[type=password]"]


def is_login_url(url: str) -> bool:
    u = urlparse(url)
    return (u.hostname or "").endswith("allegro.pl") and (
        "/login" in u.path or "/logowanie" in u.path)


def is_purchases_url(url: str) -> bool:
    u = urlparse(url)
    return u.hostname == "allegro.pl" and u.path.startswith("/moje-allegro/zakupy")


def start_browser() -> subprocess.Popen:
    with sync_playwright() as p:
        exe = p.chromium.executable_path
    if not Path(exe).exists():
        sys.exit(f"Brak Chromium dla tej wersji Playwright ({exe}).\n"
                 "Zainstaluj: uv run playwright install chromium")
    PROFILE_DIR.mkdir(exist_ok=True)
    PROFILE_DIR.chmod(0o700)  # ciasteczka sesji tylko dla właściciela
    proc = subprocess.Popen(
        [
            exe,
            f"--user-data-dir={PROFILE_DIR}",
            f"--remote-debugging-port={CDP_PORT}",
            "--remote-debugging-address=127.0.0.1",
            "--no-sandbox",  # Ubuntu blokuje sandbox Chromium z Playwright
            "--no-first-run",
            "--no-default-browser-check",
            "--lang=pl-PL",
            "--window-size=1280,950",
            PURCHASES_URL,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(100):
        if current_tab() is not None:
            return proc
        time.sleep(0.2)
    proc.terminate()
    sys.exit("Nie udało się uruchomić Chromium.")


def current_tab() -> dict | None:
    """Aktywna karta (url, title) przez HTTP DevTools, bez podłączania się."""
    try:
        with urllib.request.urlopen(f"{CDP_URL}/json/list", timeout=2) as r:
            tabs = json.load(r)
    except Exception:
        return None
    pages = [t for t in tabs if t.get("type") == "page"]
    return pages[0] if pages else {"url": "", "title": ""}


def _visible_field(page: Page, selectors: list[str]):
    for sel in selectors:
        field = page.locator(sel).first
        if field.count() and field.is_visible():
            return field
    return None


def fill_credentials(user: str, password: str) -> bool:
    """Podłącza się na chwilę, wpisuje login i hasło, sprawdza, czy wartości
    zostały w polach, i się rozłącza. Strona logowania Allegro potrafi wyczyścić
    pole wpisane przed pełnym załadowaniem formularza, dlatego wartości są
    weryfikowane, a w razie potrzeby wpisywane znak po znaku."""
    pw = sync_playwright().start()
    try:
        browser = pw.chromium.connect_over_cdp(CDP_URL)
        page = browser.contexts[0].pages[0]
        login_field = _visible_field(page, LOGIN_SELECTORS)
        password_field = _visible_field(page, PASSWORD_SELECTORS)
        if login_field is None or password_field is None:
            return False
        page.wait_for_timeout(1000)  # formularz musi się w pełni załadować
        for field, value in ((login_field, user), (password_field, password)):
            for attempt in range(3):
                if field.input_value() == value:
                    break
                if attempt == 0:
                    field.fill(value)
                else:
                    field.fill("")
                    field.press_sequentially(value, delay=40)
                page.wait_for_timeout(500)
        page.wait_for_timeout(500)
        return (login_field.input_value() == user
                and password_field.input_value() == password)
    except Exception:
        return False
    finally:
        pw.stop()  # rozłącza, przeglądarka działa dalej


def login(creds: LazyCredentials | None) -> None:
    deadline = time.time() + LOGIN_TIMEOUT_S
    filled = False
    announced = False
    ok_streak = 0
    time.sleep(2)  # czas na ewentualne przekierowanie do logowania
    while time.time() < deadline:
        tab = current_tab()
        if tab is None:
            sys.exit("Przeglądarka została zamknięta.")
        url, title = tab.get("url", ""), tab.get("title", "")
        if is_purchases_url(url) and title:
            ok_streak += 1
            if ok_streak >= 2:
                print("Zalogowano." if announced else "Sesja aktywna - logowanie pominięte.")
                return
        else:
            ok_streak = 0
        if is_login_url(url):
            if not announced:
                print("Zaloguj się w oknie przeglądarki "
                      "(skrypt wpisze dane, Ty klikasz 'Zaloguj', captchę i 2FA).")
                announced = True
            if creds is not None and not filled:
                user, password = creds.get()
                filled = fill_credentials(user, password)
                if filled:
                    print("  wpisano login i hasło - kliknij 'Zaloguj się'")
        time.sleep(1.5)
    sys.exit("Nie udało się zalogować w wyznaczonym czasie.")


# --------------------------------------------------------------------------- #
# Zbieranie odpowiedzi JSON ze strony „Kupione”
# --------------------------------------------------------------------------- #
# Wewnętrzne API strony nie jest udokumentowane, więc nie wołamy go wprost:
# zapisujemy odpowiedzi JSON, które strona sama pobiera, przewijając ją
# i klikając „pokaż więcej” / „następna strona”.
IGNORED_URL_PARTS = ("datadome", "analytics", "gtm", "consent", "/events", "metrics",
                     "recommend", "advert", "ads.")
MORE_BUTTONS = re.compile(r"pokaż więcej|załaduj więcej|wczytaj więcej|następna", re.I)


def looks_relevant(resp: Response) -> bool:
    host = urlparse(resp.url).hostname or ""
    if not host.endswith("allegro.pl") and not host.endswith("allegrostatic.com"):
        return False
    if any(p in resp.url.lower() for p in IGNORED_URL_PARTS):
        return False
    return "json" in resp.headers.get("content-type", "")


class Collector:
    def __init__(self, keep_all: bool):
        self.keep_all = keep_all
        self.bodies: list[dict] = []  # {"url", "status", "body"}
        self.order_hits = 0

    def on_response(self, resp: Response) -> None:
        if not looks_relevant(resp):
            return
        try:
            body = resp.json()
        except Exception:
            return
        has_orders = bool(find_orders(body))
        if has_orders:
            self.order_hits += 1
        if has_orders or self.keep_all:
            self.bodies.append({"url": resp.url, "status": resp.status,
                                "method": resp.request.method, "body": body})


def embedded_json(page: Page) -> list[dict]:
    """Dane osadzone w HTML (<script type=application/json>) - pierwsza strona
    listy bywa renderowana po stronie serwera, bez osobnego zapytania XHR."""
    out = []
    scripts = page.eval_on_selector_all(
        "script[type='application/json'], script[type='application/ld+json']",
        "els => els.map(e => e.textContent)")
    for text in scripts:
        try:
            body = json.loads(text)
        except Exception:
            continue
        if find_orders(body):
            out.append({"url": page.url + "#embedded", "status": 200, "method": "GET",
                        "body": body})
    return out


def click_more(page: Page) -> bool:
    for role in ("button", "link"):
        btn = page.get_by_role(role, name=MORE_BUTTONS).last
        try:
            if btn.count() and btn.is_visible() and btn.is_enabled():
                btn.scroll_into_view_if_needed(timeout=2000)
                btn.click(timeout=3000)
                return True
        except Exception:
            continue
    return False


HISTORY_COMPLETE = RAW_DIR / ".history_complete"  # znacznik: lista przejrzana do końca


def _parse_date(value) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def order_date(o: dict) -> datetime | None:
    return _parse_date(_first(o, "orderDate", "boughtAt", "purchaseDate", "createdAt",
                              "date", "updatedAt"))


def load_saved_orders() -> list[dict]:
    bodies = []
    for f in sorted(RAW_DIR.glob("responses_*.json")):
        bodies += json.loads(f.read_text(encoding="utf-8"))
    return extract_orders(bodies)


def browse_cutoff(days: int) -> datetime | None:
    """Data, do której przeglądamy listę: `days` dni przed najnowszym zapisanym
    zakupem. None = przeglądaj całą listę (pierwsze uruchomienie albo historia
    niekompletna - wtedy w zapisanych danych mogą być dziury)."""
    if not HISTORY_COMPLETE.exists():
        return None
    dates = [d for d in map(order_date, load_saved_orders()) if d]
    return max(dates) - timedelta(days=days) if dates else None


def browse_purchases(context: BrowserContext, page: Page, collector: Collector,
                     max_rounds: int, cutoff: datetime | None) -> bool:
    """Przewija listę zakupów (od najnowszych). Zwraca True, jeśli doszła do końca
    listy. Gdy podano `cutoff`, kończy po wczytaniu zamówienia starszego niż
    ta data - nowsze zamówienia są przeglądane ponownie, żeby złapać zwroty
    i zmiany statusów."""
    context.on("response", collector.on_response)
    page.goto(PURCHASES_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(4000)
    collector.bodies.extend(embedded_json(page))

    def report(label: str, seen: int) -> bool:
        batch = extract_orders(collector.bodies[seen:])
        dates = [d for d in map(order_date, batch) if d]
        oldest = min(dates).date().isoformat() if dates else "?"
        print(f"  {label}: wczytano {len(batch)} zamówień (najstarsze z {oldest})")
        if cutoff and dates and min(dates) < cutoff:
            print(f"  doszedłem do zakupów sprzed {cutoff.date()} - koniec przeglądania.")
            return True
        return False

    if report("strona 1", 0):
        return False
    idle = 0
    for i in range(max_rounds):
        seen = len(collector.bodies)
        height = page.evaluate("document.body.scrollHeight")
        page.mouse.wheel(0, 20000)
        page.wait_for_timeout(1500)
        clicked = click_more(page)
        page.wait_for_timeout(2500 if clicked else 1000)
        if len(collector.bodies) == seen and not clicked \
                and page.evaluate("document.body.scrollHeight") == height:
            idle += 1
            if idle >= 3:
                return True  # nic więcej się nie wczytuje: koniec listy
            continue
        idle = 0
        if len(collector.bodies) > seen and report(f"runda {i + 1}", seen):
            return False
    return False


# --------------------------------------------------------------------------- #
# Szczegóły zakupu: które produkty zostały zwrócone
# --------------------------------------------------------------------------- #
# Lista zakupów podaje tylko status całego zamówienia (RETURNED,
# PARTIALLY_RETURNED). Które produkty wróciły, widać dopiero na stronie
# szczegółów, w danych osadzonych w HTML:
#   rescissions[]  - zgłoszone odstąpienia od umowy: rescissionOffers (offerId,
#                    quantity, reason), status, timelineStatus,
#   refunds[]      - zwroty pieniędzy: refundedItems.products (offerId,
#                    refundedQuantity, refundValue), header.status.
DETAILS_DIR = RAW_DIR / "details"


def has_return(o: dict) -> bool:
    primary = _first(o, "status.primary") or {}
    return "RETURN" in json.dumps(primary)


def details_path(order_id: str) -> Path:
    return DETAILS_DIR / f"{re.sub(r'[^\w-]+', '_', order_id)}.json"


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def parse_return_details(page: Page) -> dict | None:
    """Zwroty i zwroty pieniędzy z JSON osadzonego w stronie szczegółów.
    Strona zawiera kilka kopii tych samych danych, więc deduplikujemy po id."""
    scripts = page.eval_on_selector_all("script[type='application/json']",
                                        "els => els.map(e => e.textContent)")
    rescissions: dict[str, dict] = {}
    refunds: dict[str, dict] = {}
    found_order = False
    for text in scripts:
        try:
            data = json.loads(text)
        except Exception:
            continue
        for d in _walk(data):
            if "myorders" in d:
                found_order = True
            if "rescissionOffers" in d:
                rid = str(d.get("rescissionId") or len(rescissions))
                rescissions[rid] = {
                    "rescissionId": d.get("rescissionId"),
                    "status": d.get("status"),
                    "requestDate": d.get("requestDate"),
                    "referenceNumber": d.get("referenceNumber"),
                    "timeline": _first(d, "timelineStatus.label"),
                    "timelineHint": _first(d, "timelineStatus.hint"),
                    "offers": [
                        {"offerId": x.get("offerId"), "orderOfferId": x.get("orderOfferId"),
                         "name": _first(x, "product.name"), "quantity": x.get("quantity"),
                         "price": x.get("price"), "reason": _first(x, "reason.label"),
                         "comment": _first(x, "reason.comment")}
                        for x in d["rescissionOffers"]],
                }
            if "refundedItems" in d and "summary" in d:
                rid = str(_first(d, "summary.refundId") or len(refunds))
                refunds[rid] = {
                    "refundId": _first(d, "summary.refundId"),
                    "status": _first(d, "header.status"),
                    "created": _first(d, "summary.refundCreatedDate"),
                    "total": d.get("totalRefundToBuyerValue"),
                    "products": [
                        {"offerId": x.get("offerId"), "name": x.get("name"),
                         "refundedQuantity": x.get("refundedQuantity"),
                         "refundValue": x.get("refundValue")}
                        for x in _first(d, "refundedItems.products") or []],
                }
    if not found_order:
        return None  # strona się nie wczytała (albo Allegro zmieniło format)
    return {"rescissions": list(rescissions.values()), "refunds": list(refunds.values())}


def fetch_return_details(page: Page, orders: list[dict], refresh: set[str]) -> None:
    """Pobiera szczegóły zamówień ze zwrotem: brakujące oraz te z `refresh`
    (zamówienia z przeglądanego okresu - zwrot mógł się w tym czasie zmienić)."""
    DETAILS_DIR.mkdir(parents=True, exist_ok=True)
    todo = [o for o in orders if has_return(o)
            and (order_key(o) in refresh or not details_path(order_key(o)).exists())]
    if not todo:
        return
    print(f"Sprawdzam szczegóły {len(todo)} zamówień ze zwrotami...")
    for i, o in enumerate(todo, 1):
        oid = order_key(o)
        details = None
        try:
            page.goto(f"{PURCHASES_URL}/{oid}", wait_until="domcontentloaded")
            for _ in range(5):
                page.wait_for_timeout(1500)
                details = parse_return_details(page)
                if details is not None:
                    break
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {oid}: BŁĄD {e.__class__.__name__}")
            continue
        if details is None:
            print(f"  [{i}/{len(todo)}] {oid}: nie znaleziono danych zamówienia")
            continue
        details |= {"orderId": oid, "fetched": datetime.now().isoformat(timespec="seconds"),
                    "orderStatus": _first(o, "status.primary.status")}
        details_path(oid).write_text(json.dumps(details, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        n = sum(x.get("quantity") or 0 for r in details["rescissions"] for x in r["offers"])
        print(f"  [{i}/{len(todo)}] {(order_date(o) or datetime.min).date()} "
              f"{details['orderStatus']}: zwróconych sztuk {n}")
        page.wait_for_timeout(1000)  # nie zasypujemy serwera zapytaniami


def load_returns() -> dict[tuple[str, str], dict]:
    """(order_id, offer_id) -> podsumowanie zwrotu tej pozycji."""
    out: dict[tuple[str, str], dict] = {}
    for f in DETAILS_DIR.glob("*.json") if DETAILS_DIR.exists() else []:
        d = json.loads(f.read_text(encoding="utf-8"))
        oid = d["orderId"]
        for r in d.get("rescissions", []):
            for x in r["offers"]:
                e = out.setdefault((oid, str(x["offerId"])), {})
                e["returned_quantity"] = (e.get("returned_quantity") or 0) + (x["quantity"] or 0)
                e["return_reason"] = x.get("reason")
                # np. "Decyzja o zwrocie: Pełen zwrot pieniędzy."
                e["return_status"] = ": ".join(
                    filter(None, (r.get("timeline"), r.get("timelineHint")))) or r.get("status")
                e["return_date"] = r.get("requestDate")
        for r in d.get("refunds", []):
            for x in r["products"]:
                e = out.setdefault((oid, str(x["offerId"])), {})
                amount, _ = _money(x.get("refundValue"))
                prev = float(e.get("refund_amount") or 0)
                e["refund_amount"] = f"{prev + float(amount or 0):.2f}"
                if x.get("refundedQuantity") is not None:  # przy zwrocie kwoty brak
                    e["refunded_quantity"] = (e.get("refunded_quantity") or 0) + \
                        x["refundedQuantity"]
                e["refund_status"] = r.get("status")
    return out


# --------------------------------------------------------------------------- #
# Wyciąganie zamówień z JSON (heurystycznie, bo format API nie jest publiczny)
# --------------------------------------------------------------------------- #
ITEM_LIST_KEYS = ("offers", "lineItems", "items", "products", "positions")
ORDER_HINT_KEYS = {"seller", "orderDate", "boughtAt", "purchaseDate", "createdAt",
                   "orderId", "checkoutFormId", "totalCost", "summary", "payment",
                   "delivery", "status"}


def _is_order(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    has_items = any(isinstance(d.get(k), list) and d[k] for k in ITEM_LIST_KEYS)
    return has_items and len(ORDER_HINT_KEYS & d.keys()) >= 1


def find_orders(node, depth: int = 0) -> list[dict]:
    """Szuka w drzewie JSON list obiektów wyglądających na zamówienia."""
    if depth > 12:
        return []
    found: list[dict] = []
    if isinstance(node, list):
        orders = [x for x in node if _is_order(x)]
        if orders and len(orders) >= len(node) / 2:
            return orders
        for x in node:
            found += find_orders(x, depth + 1)
    elif isinstance(node, dict):
        for v in node.values():
            found += find_orders(v, depth + 1)
    return found


def order_key(o: dict) -> str:
    for k in ("id", "orderId", "checkoutFormId", "purchaseId", "groupId"):
        if o.get(k):
            return str(o[k])
    return hashlib.sha1(json.dumps(o, sort_keys=True).encode()).hexdigest()[:16]


def extract_orders(bodies: list[dict]) -> list[dict]:
    orders: dict[str, dict] = {}
    for b in bodies:
        for o in find_orders(b["body"]):
            orders[order_key(o)] = o
    return list(orders.values())


def _first(d: dict, *paths: str):
    """Pierwsza niepusta wartość spośród ścieżek typu 'seller.login'."""
    for path in paths:
        cur = d
        for part in path.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
            if cur is None:
                break
        if cur not in (None, "", [], {}):
            return cur
    return None


def _money(v):
    """{'amount': '12.34', 'currency': 'PLN'} albo liczba/napis -> (kwota, waluta)."""
    if isinstance(v, dict):
        amount = v.get("amount", v.get("value"))
        return amount, v.get("currency")
    return v, None


def flatten(orders: list[dict]) -> list[dict]:
    rows = []
    for o in orders:
        oid = order_key(o)
        date = _first(o, "orderDate", "boughtAt", "purchaseDate", "createdAt",
                      "date", "updatedAt")
        seller = _first(o, "seller.login", "seller.name", "seller.companyName", "seller")
        total, cur = _money(_first(o, "totalCost", "summary.totalToPay", "totalPrice",
                                   "summary.total", "total", "price"))
        status = _first(o, "status.primary.status", "status.name", "status.label",
                        "status", "state")
        delivery_cost, _ = _money(_first(o, "delivery.cost"))
        delivery_name = _first(o, "delivery.name")
        items = next((o[k] for k in ITEM_LIST_KEYS if isinstance(o.get(k), list)), [])
        for it in items:
            if not isinstance(it, dict):
                continue
            price, icur = _money(_first(it, "unitPrice", "price", "price.amount",
                                        "originalPrice", "cost"))
            rows.append({
                "order_id": oid,
                "date": date,
                "seller": seller if not isinstance(seller, dict) else json.dumps(seller),
                "order_status": status if not isinstance(status, dict) else json.dumps(status),
                "order_total": total,
                "offer_id": _first(it, "offerId", "id", "offer.id"),
                "title": _first(it, "title", "name", "offer.name", "offer.title"),
                "quantity": _first(it, "quantity", "count", "amount"),
                "unit_price": price,
                "currency": icur or cur,
                "delivery_name": delivery_name,
                "delivery_cost": delivery_cost,
                "url": _first(it, "friendlyUrl", "url", "offerUrl", "offer.url"),
            })
    return rows


def save_raw(bodies: list[dict]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (RAW_DIR / f"responses_{stamp}.json").write_text(
        json.dumps(bodies, ensure_ascii=False, indent=2), encoding="utf-8")


KEPT_FIELDS = ["order_id", "date", "seller", "offer_id", "title", "quantity",
               "unit_price", "total", "currency", "url"]


def kept_items(rows: list[dict]) -> list[dict]:
    """Pozycje, które zostały u kupującego: bez anulowanych zamówień i bez
    zwróconych sztuk (zwrot zgłoszony, także w trakcie weryfikacji). Pozycja
    zwrócona częściowo zostaje z pomniejszoną ilością."""
    out = []
    for r in rows:
        if r["order_status"] == "ORDER_CANCELLED":
            continue
        qty = int(r["quantity"] or 0)
        returned = int(r.get("returned_quantity") or r.get("refunded_quantity") or 0)
        if r.get("refund_amount") and not returned:
            returned = qty  # zwrot samej kwoty bez liczby sztuk: produkt odesłany
        kept_qty = qty - returned
        if kept_qty <= 0:
            continue
        unit = float(r["unit_price"] or 0)
        out.append({k: r.get(k) for k in KEPT_FIELDS}
                   | {"quantity": kept_qty, "total": f"{unit * kept_qty:.2f}"})
    return out


NBP_CACHE = RAW_DIR / "nbp_rates.json"


def pln_rate(currency: str, day: datetime, cache: dict) -> float | None:
    """Średni kurs NBP (tabela A) z ostatniego dnia roboczego przed `day`."""
    if not currency or currency == "PLN":
        return 1.0
    key = f"{currency}:{day.date()}"
    if key not in cache:
        start = (day - timedelta(days=10)).date()
        end = (day - timedelta(days=1)).date()
        url = (f"https://api.nbp.pl/api/exchangerates/rates/a/{currency.lower()}/"
               f"{start}/{end}/?format=json")
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                cache[key] = json.load(r)["rates"][-1]["mid"]
        except Exception as e:
            print(f"  brak kursu {currency} z {day.date()} ({e}) - pomijam kwotę")
            return None
    return cache[key]


def write_simple_csv(kept: list[dict]) -> None:
    """zakupy.csv - wersja do czytania: numer zakupu (1 = najstarszy), data,
    produkt, liczba, cena (razem) i cena jednostkowa w PLN, sklep. Najnowsze
    zakupy na górze. Bez anulowanych i zwróconych pozycji."""
    cache = json.loads(NBP_CACHE.read_text()) if NBP_CACHE.exists() else {}
    n_cache = len(cache)
    rows = sorted(kept, key=lambda r: (str(r["date"]), r["order_id"]))
    numbers: dict[str, int] = {}
    out = []
    for r in rows:
        nr = numbers.setdefault(r["order_id"], len(numbers) + 1)
        day = _parse_date(r["date"])
        local = day.astimezone(ZoneInfo("Europe/Warsaw")) if day else None
        rate = pln_rate(r["currency"], day, cache) if day else None
        unit = float(r["unit_price"] or 0) * rate if rate else None
        out.append({
            "nr": nr,
            "data": local.date().isoformat() if local else "",
            "produkt": normalize_name(r["title"]),
            "liczba": r["quantity"],
            "cena": f"{unit * int(r['quantity']):.2f}" if unit is not None else "",
            "cena jednostkowa": f"{unit:.2f}" if unit is not None else "",
            "sklep": "allegro",
        })
    out.reverse()  # najnowszy zakup na górze (numeracja nadal od najstarszego)
    if len(cache) != n_cache:
        NBP_CACHE.write_text(json.dumps(cache, indent=1))
    # Nazwa produktu zawsze w cudzysłowie (bywają w niej przecinki), pozostałe
    # pola (liczby, daty, "allegro") nigdy - stąd zapis ręczny zamiast csv.writer.
    fields = ["nr", "data", "produkt", "liczba", "cena", "cena jednostkowa", "sklep"]
    with open(OUTPUT_DIR / "zakupy.csv", "w", newline="", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        for row in out:
            f.write(",".join(_quote(row[k]) if k == "produkt" else str(row[k])
                             for k in fields) + "\n")


def normalize_name(name: str) -> str:
    """Małe litery i pojedyncze spacje, np. 'ŚWIECA  Zapłonowa ' -> 'świeca zapłonowa'."""
    return " ".join(str(name or "").split()).lower()


def _quote(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def rebuild_outputs() -> list[dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # łączymy z poprzednimi pobraniami (Allegro pokazuje tylko ostatnie zakupy
    # bez dodatkowego przewijania, a starsze zamówienia mogą kiedyś zniknąć)
    orders = load_saved_orders()

    (OUTPUT_DIR / "orders.json").write_text(
        json.dumps(orders, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = flatten(orders)
    returns = load_returns()
    for r in rows:
        r.update(returns.get((r["order_id"], str(r["offer_id"])), {}))
    rows.sort(key=lambda r: str(r["date"] or ""), reverse=True)
    with open(OUTPUT_DIR / "items.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["order_id", "date", "seller", "order_status", "order_total",
                  "offer_id", "title", "quantity", "unit_price", "currency",
                  "delivery_name", "delivery_cost",
                  "returned_quantity", "return_reason", "return_status", "return_date",
                  "refunded_quantity", "refund_amount", "refund_status", "url"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    kept = kept_items(rows)
    with open(OUTPUT_DIR / "items_kept.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KEPT_FIELDS)
        w.writeheader()
        w.writerows(kept)
    write_simple_csv(kept)
    n_ret = sum(1 for r in rows if r.get("returned_quantity") or r.get("refunded_quantity"))
    print(f"Zamówień: {len(orders)}, pozycji: {len(rows)} (zwróconych: {n_ret}), "
          f"zatrzymanych pozycji: {len(kept)} -> {OUTPUT_DIR}/")
    return orders
    if not orders:
        print("Nie rozpoznano żadnych zamówień. Uruchom z --discover i sprawdź "
              f"{DISCOVERY_DIR}/network_log.json (Allegro mogło zmienić format).")


# --------------------------------------------------------------------------- #
# Historia zmian danych: lokalne repozytorium git w ./dane
# --------------------------------------------------------------------------- #
PROJECT_ROOT = BASE_DIR.parent.parent
PRIVATE_GIT = PROJECT_ROOT / ".git-private"  # wspólne prywatne repo danych projektu


def commit_data(message: str) -> None:
    """Commituje zmiany w ./dane do prywatnego, lokalnego repo .git-private
    w katalogu głównym projektu (bez remote - dane nie opuszczają komputera).
    `add -f`, bo .gitignore repozytorium z kodem ignoruje ./dane."""
    if not PRIVATE_GIT.exists():
        return
    git = ["git", f"--git-dir={PRIVATE_GIT}", f"--work-tree={PROJECT_ROOT}"]
    path = str(DATA_DIR.relative_to(PROJECT_ROOT))
    try:
        subprocess.run(git + ["add", "-f", "-A", "--", path], check=True, cwd=PROJECT_ROOT)
        status = subprocess.run(git + ["diff", "--cached", "--name-status", "--", path],
                                check=True, capture_output=True, text=True,
                                cwd=PROJECT_ROOT).stdout.splitlines()
        if not status:
            print("Dane bez zmian - nic do zapisania w historii.")
            return
        details = sum(1 for line in status if "allegro_raw/details/" in line)
        if details:
            message += f", zmienione szczegóły zwrotów: {details}"
        # commit tylko danych Allegro (reszta danych projektu zostaje bez zmian)
        subprocess.run(git + ["commit", "-q", "-m", f"allegro: {message}", "--", path],
                       check=True, cwd=PROJECT_ROOT)
        print(f"Zapisano w historii danych (.git-private): allegro: {message}")
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"Nie udało się zapisać historii danych: {e}")


# --------------------------------------------------------------------------- #
# Tryb diagnostyczny
# --------------------------------------------------------------------------- #
def discover(context: BrowserContext, page: Page) -> None:
    DISCOVERY_DIR.mkdir(exist_ok=True)
    collector = Collector(keep_all=True)
    context.on("response", collector.on_response)
    page.goto(PURCHASES_URL, wait_until="domcontentloaded")
    print("\nPrzeglądarka zostaje otwarta na 90 s - przewiń listę, otwórz "
          "szczegóły zamówienia, przejdź na kolejną stronę.")
    try:
        for _ in range(90):
            open_pages = [pg for pg in context.pages if not pg.is_closed()]
            if not open_pages:
                break
            open_pages[0].wait_for_timeout(1000)
        (DISCOVERY_DIR / "last_page.html").write_text(page.content(), encoding="utf-8")
        page.screenshot(path=str(DISCOVERY_DIR / "last_page.png"), full_page=True)
    except Exception as e:
        print(f"(przerwano: {e.__class__.__name__})")
    log = [{k: v for k, v in b.items()} | {"has_orders": bool(find_orders(b["body"]))}
           for b in collector.bodies]
    (DISCOVERY_DIR / "network_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Zapisano {len(log)} odpowiedzi JSON do {DISCOVERY_DIR}/ "
          f"(z zamówieniami: {sum(e['has_orders'] for e in log)})")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--save-credentials", action="store_true",
                   help="zapisz login i hasło w systemowym pęku kluczy i zakończ")
    g.add_argument("--forget-credentials", action="store_true",
                   help="usuń login i hasło z pęku kluczy i zakończ")
    g.add_argument("--ask", action="store_true",
                   help="zapytaj o login i hasło w terminalu (bez pęku kluczy)")
    g.add_argument("--manual-login", action="store_true",
                   help="nie wpisuj danych, zaloguj się ręcznie w przeglądarce")
    parser.add_argument("--discover", action="store_true",
                        help="zapisz ruch sieciowy strony 'Kupione' do ./discovery")
    g.add_argument("--rebuild", action="store_true",
                   help="tylko odtwórz orders.json / items.csv z ./dane/allegro_raw (bez przeglądarki)")
    parser.add_argument("--days", type=int, default=30,
                        help="przeglądaj zakupy do N dni przed ostatnim zapisanym "
                             "(domyślnie 30), żeby złapać zwroty i zmiany statusów")
    parser.add_argument("--full", action="store_true",
                        help="przejrzyj całą listę i odśwież szczegóły wszystkich zwrotów")
    parser.add_argument("--max-rounds", type=int, default=200,
                        help="maks. liczba przewinięć / kliknięć 'pokaż więcej'")
    args = parser.parse_args()

    if args.save_credentials:
        return save_credentials()
    if args.forget_credentials:
        return forget_credentials()
    if args.rebuild:
        rebuild_outputs()
        return commit_data(f"przebudowa plików wynikowych {time.strftime('%Y-%m-%d %H:%M')}")

    creds = None if args.manual_login else LazyCredentials(ask=args.ask)
    proc = start_browser()
    try:
        login(creds)
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(CDP_URL)
            context = browser.contexts[0]
            page = context.pages[0]
            if args.discover:
                discover(context, page)
            else:
                collector = Collector(keep_all=False)
                known_before = {order_key(o) for o in load_saved_orders()}
                cutoff = None if args.full else browse_cutoff(args.days)
                print("Przeglądam " + (f"zakupy od {cutoff.date()}..." if cutoff
                                       else "całą listę zakupów..."))
                reached_end = browse_purchases(context, page, collector,
                                               args.max_rounds, cutoff)
                save_raw(collector.bodies)
                if reached_end:
                    HISTORY_COMPLETE.touch()
                browsed = {order_key(o) for o in extract_orders(collector.bodies)}
                orders = load_saved_orders()
                fetch_return_details(page, orders, refresh=browsed)
                orders = rebuild_outputs()
                new = len({order_key(o) for o in orders} - known_before)
                commit_data(f"pobranie {time.strftime('%Y-%m-%d %H:%M')}: "
                            f"zamówień {len(orders)}, nowych {new}")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
