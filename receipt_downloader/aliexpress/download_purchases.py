"""
Pobiera listę zakupów z AliExpress (Moje zamówienia).

Dane logowania NIE są trzymane w plikach projektu. Do wyboru:
  * systemowy pęk kluczy (GNOME Keyring / KWallet, przez bibliotekę `keyring`),
    zapisany raz poleceniem `--save-credentials`; hasło jest zaszyfrowane
    i odblokowywane razem z sesją użytkownika,
  * wpisywanie loginu i hasła w terminalu przy każdym logowaniu (getpass, bez echa),
    gdy w pęku kluczy nic nie ma albo z flagą `--ask`,
  * `--manual-login`: skrypt niczego nie wpisuje, logujesz się sam w oknie przeglądarki.
Hasło jest potrzebne tylko wtedy, gdy sesja w .browser_profile/ wygasła.

Chromium startuje jako zwykły proces z trwałym profilem. Skrypt podłącza się
do niego przez CDP tylko na chwilę, żeby wpisać dane (AliExpress pyta najpierw
o e-mail, a po kliknięciu „Kontynuuj” o hasło). Przyciski, suwak captchy i kod
weryfikacyjny obsługujesz sam.

Strony zamówień AliExpress są budowane w przeglądarce z odpowiedzi JSON
wewnętrznego API (mtop.aliexpress.trade.buyer.order.list / .order.detail).
Skrypt przegląda listę zamówień (klikając „View orders”), otwiera szczegóły
każdego zamówienia i zapisuje te odpowiedzi (bez telemetrii, danych adresowych
i tokenów) w dane/aliexpress_raw/. Dopiero z nich powstają pliki wynikowe,
więc poprawka parsera wymaga tylko `--rebuild`, a nie ponownego pobierania.

Użycie:
    uv run download_purchases.py                    # pobierz zakupy
    uv run download_purchases.py --save-credentials # zapisz login i hasło w pęku kluczy
    uv run download_purchases.py --forget-credentials
    uv run download_purchases.py --ask              # zapytaj o dane, nie używaj pęku kluczy
    uv run download_purchases.py --manual-login     # loguj się ręcznie w przeglądarce
    uv run download_purchases.py --days 60          # przeglądaj 60 dni wstecz (domyślnie 30)
    uv run download_purchases.py --full             # przejrzyj całą historię (odśwież statusy)
    uv run download_purchases.py --rebuild          # odtwórz pliki wynikowe z dane/aliexpress_raw
    uv run download_purchases.py --discover         # zrzut stron i JSON do ./discovery
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import BrowserContext, Page, Response, sync_playwright

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
import common  # noqa: E402  (../common.py: logowanie, przeglądarka, zakupy.csv, wykres, NBP)

PROFILE_DIR = BASE_DIR / ".browser_profile"
# Dane osobowe trzymamy w ./dane: ignoruje je repozytorium z kodem, a ich historia
# jest we wspólnym prywatnym repo .git-private w katalogu projektu (common.commit_data).
DATA_DIR = BASE_DIR / "dane"
RAW_DIR = DATA_DIR / "aliexpress_raw"          # odpowiedzi JSON listy zamówień
DETAILS_DIR = RAW_DIR / "details"               # szczegóły zamówień: <nr_zamówienia>.json
OUTPUT_DIR = DATA_DIR / "aliexpress_purchases"  # wynik: zamówienia i pozycje
DISCOVERY_DIR = BASE_DIR / "discovery"
HISTORY_COMPLETE = RAW_DIR / ".history_complete"  # znacznik: cała lista przejrzana
NBP_CACHE = RAW_DIR / "nbp_rates.json"

ORDERS_URL = "https://www.aliexpress.com/p/order/index.html"
LOGIN_TIMEOUT_S = 300  # czas na ręczne dokończenie logowania (suwak, kod weryfikacyjny)
PAGE_DELAY_MS = 1500   # przerwa między stronami - nie zasypujemy serwera zapytaniami
RESPONSE_TIMEOUT_S = 20  # ile czekamy na odpowiedź API po otwarciu / kliknięciu

KEYRING_SERVICE = "receipt_parsing-aliexpress"  # login i hasło: common.LazyCredentials

LIST_API = "mtop.aliexpress.trade.buyer.order.list"
DETAIL_API = "mtop.aliexpress.trade.buyer.order.detail"


def details_url(order_id: str) -> str:
    return f"https://www.aliexpress.com/p/order/detail.html?orderId={order_id}"


# --------------------------------------------------------------------------- #
# Przeglądarka i logowanie
# --------------------------------------------------------------------------- #
CDP_PORT = 9232  # inny niż Amazon (9231), Allegro (9230) i Biedronka (9229)
CDP_URL = f"http://127.0.0.1:{CDP_PORT}"

# Formularz logowania AliExpress bywa w ramce (iframe), więc pola szukamy we
# wszystkich ramkach strony. Gdy żaden selektor e-maila nie pasuje, na stronie
# logowania bierzemy pierwsze widoczne pole tekstowe (EMAIL_FALLBACK).
EMAIL_SELECTORS = ["input#fm-login-id", "input[name=fm-login-id]", "input[name=loginId]",
                   "input[type=email]", "input[autocomplete=username]", "input[name=email]",
                   "input[autocomplete=email]"]
EMAIL_FALLBACK = ["input[type=text]", "input[type=tel]", "input:not([type])"]
PASSWORD_SELECTORS = ["input#fm-login-password", "input[name=fm-login-password]",
                      "input[name=password]", "input[type=password]"]
# suwak captchy (Alibaba „baxia” / „nocaptcha”), pokazywany w ramce albo w okienku
CAPTCHA_SELECTORS = ["#nc_1_n1z", ".btn_slide", ".nc-container", "#nocaptcha",
                     ".baxia-dialog", "#baxia-dialog-content", "#baxia-punish"]
MAX_FILLS = 3  # ile razy najwyżej wpisujemy to samo pole (strona może je odrzucać)


def _host(url: str) -> str:
    return urlparse(url).hostname or ""


def is_aliexpress(url: str) -> bool:
    return _host(url).endswith(("aliexpress.com", "aliexpress.us"))


def is_login_url(url: str) -> bool:
    """login.aliexpress.com, passport.aliexpress.com albo …/login… na www."""
    return is_aliexpress(url) and (_host(url).startswith(("login.", "passport."))
                                   or "login" in urlparse(url).path.lower())


def is_captcha_url(url: str) -> bool:
    return "_____tmd_____" in url or "punish" in url.lower()


def is_orders_url(url: str) -> bool:
    return is_aliexpress(url) and urlparse(url).path.startswith("/p/order/")


def _visible(frame, selectors: list[str]) -> bool:
    for sel in selectors:
        try:
            el = frame.locator(sel).first
            if el.count() and el.is_visible():
                return True
        except Exception:
            continue
    return False


def needs_user(page: Page) -> str | None:
    """'captcha' / 'login', gdy AliExpress czeka na użytkownika (osobna strona,
    ramka albo okienko na bieżącej stronie), inaczej None."""
    if is_captcha_url(page.url):
        return "captcha"
    if is_login_url(page.url):
        return "login"
    for frame in page.frames:
        try:
            if is_captcha_url(frame.url) or _visible(frame, CAPTCHA_SELECTORS):
                return "captcha"
            if _visible(frame, PASSWORD_SELECTORS) or (
                    is_login_url(frame.url) and _visible(frame, EMAIL_SELECTORS)):
                return "login"
        except Exception:
            continue
    return None


_reported_inputs = False


def _report_inputs(frame) -> None:
    """Raz na uruchomienie wypisuje atrybuty widocznych pól (bez wartości), gdy
    nie udało się znaleźć pola e-maila - żeby można było poprawić selektory."""
    global _reported_inputs
    if _reported_inputs:
        return
    _reported_inputs = True
    try:
        inputs = frame.eval_on_selector_all(
            "input", "els => els.filter(e => e.offsetParent).map(e => "
            "['type','name','id','autocomplete','placeholder'].map(a => a + '=' + "
            "(e.getAttribute(a) || '')).join(' '))")
        print("  (nie rozpoznano pola e-maila; widoczne pola: " + "; ".join(inputs) + ")")
    except Exception:
        pass


def _email_field(page: Page, frame):
    field = common.visible_field(frame, EMAIL_SELECTORS)
    if field is None and (is_login_url(frame.url) or is_login_url(page.url)):
        field = common.visible_field(frame, EMAIL_FALLBACK)
        if field is None and _visible(frame, PASSWORD_SELECTORS):
            _report_inputs(frame)
    return field


def fill_login_page(page: Page, user: str, password: str, counts: Counter) -> list[str]:
    """Wpisuje e-mail i/lub hasło w widoczne pola (we wszystkich ramkach), jeśli
    nie ma w nich już właściwej wartości. AliExpress pokazuje hasło dopiero po
    „Kontynuuj”, a po nieudanej captchy potrafi je wyczyścić - wtedy wpisujemy
    je ponownie (najwyżej MAX_FILLS razy). Zwraca listę wpisanych teraz pól."""
    filled = []
    page.wait_for_timeout(1000)  # formularz musi się w pełni załadować
    for frame in page.frames:
        for name, value in (("email", user), ("password", password)):
            if counts[name] >= MAX_FILLS:
                continue
            try:
                field = _email_field(page, frame) if name == "email" else \
                    common.visible_field(frame, PASSWORD_SELECTORS)
                if field is None or field.input_value() == value:
                    continue
                counts[name] += 1
                if common.type_checked(frame, field, value):
                    filled.append(name)
            except Exception:
                continue
    return filled


def fill_credentials(user: str, password: str, counts: Counter) -> list[str]:
    """Podłącza się na chwilę, wypełnia stronę logowania i się rozłącza."""
    pw = sync_playwright().start()
    try:
        browser = pw.chromium.connect_over_cdp(CDP_URL)
        return fill_login_page(browser.contexts[0].pages[0], user, password, counts)
    except Exception:
        return []
    finally:
        pw.stop()  # rozłącza, przeglądarka działa dalej


FILLED_MSG = {"email": "  wpisano e-mail - kliknij 'Kontynuuj'",
              "password": "  wpisano hasło - kliknij 'Zaloguj się'"}
LOGIN_HELP = ("Zaloguj się w oknie przeglądarki (skrypt wpisze e-mail i hasło, Ty klikasz "
              "'Kontynuuj' / 'Zaloguj się', przesuwasz suwak i wpisujesz kod).")


def login(creds: common.LazyCredentials | None) -> None:
    """Czeka, aż w przeglądarce będzie lista zamówień (sesja aktywna albo
    logowanie zakończone). Na stronie logowania wpisuje dane."""
    deadline = time.time() + LOGIN_TIMEOUT_S
    counts: Counter = Counter()
    announced = False
    ok_streak = 0
    time.sleep(3)  # czas na ewentualne przekierowanie do logowania
    while time.time() < deadline:
        tab = common.current_tab(CDP_PORT)
        if tab is None:
            sys.exit("Przeglądarka została zamknięta.")
        url, title = tab.get("url", ""), tab.get("title", "")
        # po zalogowaniu AliExpress wraca na listę zamówień, czasem na stronę główną
        if (is_orders_url(url) or (announced and is_aliexpress(url))) and title \
                and not is_login_url(url) and not is_captcha_url(url):
            ok_streak += 1
            if ok_streak >= 2:
                print("Zalogowano." if announced else "Sesja aktywna.")
                return
        else:
            ok_streak = 0
        if is_login_url(url) or is_captcha_url(url):
            if not announced:
                print(LOGIN_HELP)
                announced = True
            if creds is not None and is_login_url(url):
                user, password = creds.get()
                for f in fill_credentials(user, password, counts):
                    print(FILLED_MSG[f])
        time.sleep(1.5)
    sys.exit("Nie udało się zalogować w wyznaczonym czasie.")


def wait_if_interrupted(page: Page, creds: common.LazyCredentials | None) -> bool:
    """W trakcie przeglądania AliExpress potrafi pokazać suwak captchy albo
    okienko logowania. Wtedy czekamy, aż użytkownik to obsłuży (wpisując dane,
    jeśli je znamy). Zwraca True, jeśli strona była przerwana - wtedy trzeba
    ją otworzyć ponownie."""
    reason = needs_user(page)
    if reason is None:
        return False
    print("  AliExpress prosi o " + ("przesunięcie suwaka (captcha)" if reason == "captcha"
                                    else "zalogowanie") + " - dokończ w oknie przeglądarki.")
    counts: Counter = Counter()
    deadline = time.time() + LOGIN_TIMEOUT_S
    while time.time() < deadline:
        reason = needs_user(page)
        if reason is None:
            page.wait_for_timeout(2000)
            return True
        if reason == "login" and creds is not None:
            user, password = creds.get()
            for f in fill_login_page(page, user, password, counts):
                print(FILLED_MSG[f])
        page.wait_for_timeout(1500)
    sys.exit("Nie udało się zalogować w wyznaczonym czasie.")


def open_page(page: Page, url: str, creds: common.LazyCredentials | None,
              ready: str) -> None:
    """Otwiera stronę i czeka na `ready`. Po captchy / logowaniu wraca na nią."""
    for _ in range(3):
        page.goto(url, wait_until="domcontentloaded")
        try:
            page.wait_for_selector(ready, timeout=15000)
        except Exception:
            pass  # np. pusta zakładka albo okienko captchy zasłania stronę
        if not wait_if_interrupted(page, creds):
            break
    page.wait_for_timeout(PAGE_DELAY_MS)


# --------------------------------------------------------------------------- #
# Odpowiedzi JSON (mtop API)
# --------------------------------------------------------------------------- #
# Strony zamówień są renderowane w przeglądarce z odpowiedzi wewnętrznego API
# „mtop” (acs.aliexpress.com/h5/mtop.<nazwa>/…), czasem w formie JSONP:
# mtopjsonp3({...}). Odpowiedź ma format „ultron”: data.data to słownik
# komponentów {tag, fields}, np. pc_om_list_order (zamówienie na liście),
# detail_product_block (pozycje), detail_order_price_block (rozbicie kwot).
TELEMETRY_RE = re.compile(r"mmstat|arms-retcode|aplus|fourier|\blog\b|[-.]log[-.]|"
                          r"tracking|analytics|beacon|monitor|doubleclick|google|facebook",
                          re.I)
MTOP_API_RE = re.compile(r"/(mtop\.[\w.]+)/")
JSONP_RE = re.compile(r"^\s*[\w$.]+\s*\((.*)\)\s*;?\s*$", re.S)
# API, których nie zapisujemy (rekomendacje, licznik koszyka i zakładek)
SKIP_APIS = ("recommend", "cart.count", "order.count")


def api_name(url: str) -> str:
    m = MTOP_API_RE.search(urlparse(url).path)
    return m.group(1) if m else urlparse(url).path


def response_json(resp: Response):
    """Treść odpowiedzi JSON / JSONP z domen AliExpress (bez telemetrii) albo None."""
    url = resp.url
    if not is_aliexpress(url) or TELEMETRY_RE.search(_host(url)) \
            or TELEMETRY_RE.search(api_name(url)):
        return None
    ctype = resp.headers.get("content-type", "")
    if "json" not in ctype and not ("javascript" in ctype and "mtop" in url):
        return None
    try:
        text = resp.text()
    except Exception:
        return None
    m = JSONP_RE.match(text)
    try:
        return json.loads(m.group(1) if m and not text.lstrip().startswith(("{", "[")) else text)
    except ValueError:
        return None


class Collector:
    """Zbiera odpowiedzi mtop API (api, body) w kolejności nadejścia."""

    def __init__(self):
        self.items: list[tuple[str, dict]] = []

    def on_response(self, resp: Response) -> None:
        api = api_name(resp.url)
        if any(s in api for s in SKIP_APIS):
            return
        body = response_json(resp)
        if isinstance(body, dict):
            self.items.append((api, body))

    def wait_quiet(self, page: Page, start: int, quiet_ms: int = 2500) -> None:
        """Czeka, aż przez `quiet_ms` nie przyjdzie nic nowego (maks. RESPONSE_TIMEOUT_S)."""
        deadline = time.time() + RESPONSE_TIMEOUT_S
        n, last = len(self.items), time.time()
        while time.time() < deadline:
            page.wait_for_timeout(500)
            if len(self.items) != n:
                n, last = len(self.items), time.time()
            elif len(self.items) > start and time.time() - last >= quiet_ms / 1000:
                return

    def wait_for(self, page: Page, api: str, start: int) -> dict | None:
        """Pierwsza udana odpowiedź `api` od pozycji `start` (czeka do
        RESPONSE_TIMEOUT_S, obsługując w tym czasie captchę)."""
        deadline = time.time() + RESPONSE_TIMEOUT_S
        while time.time() < deadline:
            for name, body in self.items[start:]:
                if name == api and "SUCCESS" in str(body.get("ret")):
                    return body
            page.wait_for_timeout(500)
        return None


def components(body: dict) -> list[dict]:
    """Komponenty odpowiedzi „ultron” jako [{tag, fields}]."""
    data = (body.get("data") or {}).get("data") or {}
    return [c for c in data.values() if isinstance(c, dict) and c.get("tag")]


# Pola bez znaczenia dla zakupów albo zmieniające się przy każdym wczytaniu
# (śledzenie, ikony, obrazki, podpisy linków), a także dane adresowe. Z linków
# zostaje adres bez zapytania - poza identyfikatorami (…Id=).
NOISE_KEYS = {"utParams", "exposeInfo", "orderLineExposeInfoList", "sellerConnectUrl",
              "connectUrl", "shopIcon", "itemImgUrl", "titleHeaderImg", "credentialIcon",
              "credentialUrl", "tagCombineColor", "features", "showSelectBox",
              "hiddenOrderId", "i18n", "bgImg", "serviceIcon", "icon", "addressVO",
              "contentText", "trackInfo", "traceId", "track", "itemPicUrl", "dmIcon",
              "localPriceTag", "mdsTranslations", "contactSellerUrl", "paymentSecurityUrl",
              # dane karty i przelewu zwrotu (początek numeru karty, numer ARN)
              "cardNo", "arnNo", "refundChannelInfo"}
URL_RE = re.compile(r"^(https?:)?//")


def scrub(node):
    if isinstance(node, dict):
        return {k: scrub(v) for k, v in node.items() if k not in NOISE_KEYS}
    if isinstance(node, list):
        return [scrub(v) for v in node]
    if isinstance(node, str) and URL_RE.match(node):
        u = urlparse(node if node.startswith("http") else "https:" + node)
        ids = [f"{k}={v[0]}" for k, v in sorted(parse_qs(u.query).items())
               if re.search(r"id$", k, re.I)]
        return f"https://{u.netloc}{u.path}" + ("?" + "&".join(ids) if ids else "")
    return node


# --------------------------------------------------------------------------- #
# Przeglądanie listy zamówień
# --------------------------------------------------------------------------- #
LIST_READY = ".order-item, [class*=order-item], [class*=orderItem], .order-empty"
DETAILS_READY = "[class*=order-detail], [class*=orderDetail], [class*=order-price]"
MORE_RE = re.compile(r"view orders|więcej zamówień|wyświetl więcej|pokaż więcej|"
                     r"more orders|load more|view more", re.I)


def list_page(body: dict) -> dict:
    """Strona listy: zakładka, numer strony, czy są kolejne, zamówienia."""
    comps = components(body)
    def first(tag: str) -> dict:
        return next((c.get("fields") or {} for c in comps if c["tag"] == tag), {})
    page, action, header = (first("pc_om_list_body"), first("pc_om_list_header_action"),
                            first("pc_om_list_header"))
    return {
        "tab": action.get("statusTab"), "timeOption": action.get("timeOption"),
        "pageIndex": page.get("pageIndex"), "hasMore": bool(page.get("hasMore")),
        "tailTab": (header.get("tailStatusTab") or {}).get("title"),
        "orders": [scrub(c.get("fields") or {}) for c in comps
                   if c["tag"] == "pc_om_list_order"],
    }


def click_more(page: Page) -> bool:
    """Klika „View orders” pod listą. Zwraca False, gdy go nie ma."""
    page.mouse.wheel(0, 20000)
    page.wait_for_timeout(1000)
    for btn in (page.locator(".order-more button, .order-more").first,
                page.get_by_role("button", name=MORE_RE).last):
        try:
            if btn.count() and btn.is_visible() and btn.is_enabled():
                btn.scroll_into_view_if_needed(timeout=2000)
                btn.click(timeout=3000)
                return True
        except Exception:
            continue
    return False


def click_text(page: Page, text: str) -> bool:
    try:
        el = page.get_by_text(text, exact=True).first
        if el.count():
            el.click(timeout=5000)
            return True
    except Exception:
        pass
    return False


def browse_list(page: Page, collector: Collector, creds: common.LazyCredentials | None,
                cutoff: str | None, saved: list[dict], tab_text: str | None = None) -> bool:
    """Przegląda listę zamówień od najnowszych, klikając „View orders”. Każdą
    stronę dopisuje do `saved`. `tab_text` = nazwa zakładki do kliknięcia (np.
    „Deleted orders”), None = „View all”. Zwraca True, jeśli doszła do końca
    listy. Gdy podano `cutoff`, kończy po wczytaniu zamówienia starszego niż ta
    data - nowsze zamówienia są przeglądane ponownie, żeby złapać zwroty."""
    def reopen() -> int:
        start = len(collector.items)
        open_page(page, ORDERS_URL, creds, LIST_READY)
        if tab_text:
            start = len(collector.items)
            if not click_text(page, tab_text):
                print(f"  nie znalazłem zakładki '{tab_text}'")
        return start
    start, failures, pages_seen = reopen(), 0, set()
    while True:
        body = collector.wait_for(page, LIST_API, start)
        if body is None:
            failures += 1
            if failures > 2:
                print("  lista przestała się wczytywać - przerywam przeglądanie.")
                return False
            if wait_if_interrupted(page, creds) or not pages_seen:
                start = reopen()  # po captchy lista zaczyna się od nowa
            else:
                start = len(collector.items)
                click_more(page)
            continue
        failures = 0
        rec = list_page(body)
        rec["fetched"] = datetime.now().isoformat(timespec="seconds")
        key = (rec["tab"], rec["pageIndex"])
        if key not in pages_seen:  # po ponownym otwarciu strony się powtarzają
            pages_seen.add(key)
            saved.append(rec)
            dates = [d for d in (parse_date_text(o.get("orderDateText", ""))
                                 for o in rec["orders"]) if d]
            print(f"  {rec['tab']} strona {rec['pageIndex']}: zamówień {len(rec['orders'])}"
                  + (f" (najstarsze z {min(dates)})" if dates else ""))
            if cutoff and dates and min(dates) < cutoff:
                print(f"  doszedłem do zakupów sprzed {cutoff} - koniec przeglądania.")
                return False
        if not rec["hasMore"]:
            return True
        page.wait_for_timeout(PAGE_DELAY_MS)
        start = len(collector.items)
        if not click_more(page):
            wait_if_interrupted(page, creds)
            start = len(collector.items)
            if not click_more(page):
                print("  nie znalazłem przycisku 'View orders' - przerywam przeglądanie.")
                return False


def browse_orders(page: Page, collector: Collector, creds: common.LazyCredentials | None,
                  cutoff: str | None, saved: list[dict]) -> bool:
    """„View all”, a przy pełnym przeglądaniu także „Deleted orders” (usunięte
    z listy zamówienia to nadal zakupy). Zwraca True, jeśli przejrzała wszystko."""
    complete = browse_list(page, collector, creds, cutoff, saved)
    tail = next((r["tailTab"] for r in saved if r.get("tailTab")), None)
    if complete and not cutoff and tail:
        print(f"Przeglądam zakładkę '{tail}'...")
        complete = browse_list(page, collector, creds, None, saved, tab_text=tail)
    return complete and not cutoff


def save_list_pages(pages: list[dict]) -> None:
    if not pages:
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (RAW_DIR / f"responses_{stamp}.json").write_text(
        json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")


def load_saved_orders() -> dict[str, dict]:
    """Zamówienia ze wszystkich zapisanych stron listy; nowsze pobranie wygrywa.
    Zamówienie z „Deleted orders” nie przykrywa wersji z „View all”."""
    orders: dict[str, dict] = {}
    for f in sorted(RAW_DIR.glob("responses_*.json")):  # od najstarszego pobrania
        for rec in json.loads(f.read_text(encoding="utf-8")):
            for o in rec["orders"]:
                oid = str(o.get("orderId"))
                deleted = rec.get("tab") == "recycle"
                if deleted and oid in orders and not orders[oid]["deleted"]:
                    continue
                orders[oid] = o | {"deleted": deleted}
    return orders


def browse_cutoff(days: int) -> str | None:
    """Data ('RRRR-MM-DD'), do której przeglądamy listę: `days` dni przed
    najnowszym zapisanym zakupem. None = cała lista (pierwsze uruchomienie
    albo historia niekompletna)."""
    if not HISTORY_COMPLETE.exists():
        return None
    dates = [d for d in (parse_date_text(o.get("orderDateText", ""))
                         for o in load_saved_orders().values()) if d]
    if not dates:
        return None
    return (datetime.fromisoformat(max(dates)) - timedelta(days=days)).date().isoformat()


# --------------------------------------------------------------------------- #
# Szczegóły zamówień (rozbicie kwot, sprzedawca, zwroty)
# --------------------------------------------------------------------------- #
# Lista podaje sumę zamówienia i ceny pozycji, ale nie dostawę, kupony, monety
# ani zwroty pieniędzy. Te są na stronie szczegółów (order.detail). Zapisujemy
# jej komponenty do details/<nr_zamówienia>.json, razem z odpowiedziami innych
# API wczytanymi przez tę stronę (np. o sporach). Bez znaczników czasu, adresu
# i tokenów, więc plik zmienia się tylko wtedy, gdy zmieni się treść.
# Komponenty bez danych o zakupie albo zmieniające się z czasem: kontenery,
# „AliExpress Commitment”, kupony na kolejne zakupy (z kodem kuponu), monety
# do odebrania, pasek „Fast delivery”.
SKIP_DETAIL_TAGS = ("detail_root_container", "detail_tail_container", "detail_order_services",
                    "detail_om_order_coupon", "detail_omd_resource", "detail_service_progress_bar")
# API stron sporu, które są tylko formularzem zgłoszenia (bez danych o zwrocie)
SKIP_REFUND_APIS = ("initiatereverserender", "querynrinterceptrule")
# Przyciski pozycji prowadzące do stron sporu / zwrotu pieniędzy, np.
#   END_ISSUE  „See request”        -> /p/refund-dispute/detail.html?tradeOrderLineId=…
#   REFUND_INFO „Refund information” -> /p/reverse-pages/detail.html?detailId=…
# Te strony też są budowane z JSON; zapisujemy ich odpowiedzi w „refund_pages”.
REFUND_PAGE_RE = re.compile(r"refund|dispute|reverse|return", re.I)


def refund_links(detail: dict) -> list[tuple[str, str]]:
    """(typ przycisku, adres) stron sporu / zwrotu z pozycji zamówienia."""
    out = []
    for c in components(detail):
        for line in (c.get("fields") or {}).get("productVOList") or []:
            for b in line.get("buttonVOS") or []:
                href = str(b.get("href") or "")
                if href and REFUND_PAGE_RE.search(urlparse(href).path) \
                        and (b.get("type"), href) not in out:
                    out.append((b.get("type"), href))
    return out


def details_path(order_id: str) -> Path:
    return DETAILS_DIR / f"{re.sub(r'[^\w-]+', '_', order_id)}.json"


def details_record(order_id: str, responses: list[tuple[str, dict]]) -> dict:
    detail = next(b for api, b in responses if api == DETAIL_API)
    comps: dict[str, dict] = {}
    for c in components(detail):
        if c["tag"].startswith(SKIP_DETAIL_TAGS) or not c.get("fields"):
            continue
        key, n = c["tag"], 2
        while key in comps:  # ten sam komponent kilka razy
            key, n = f"{c['tag']}#{n}", n + 1
        comps[key] = scrub(c["fields"])
    other = {api: scrub(b.get("data")) for api, b in responses
             if api not in (DETAIL_API, LIST_API) and b.get("data")}
    return {"orderId": order_id, "components": comps, "apis": other, "refund_pages": []}


def fetch_refund_pages(page: Page, collector: Collector, detail: dict) -> list[dict]:
    """Otwiera strony sporu / zwrotu z przycisków pozycji i zapisuje odpowiedzi
    API, które te strony wczytały (format jeszcze nieznany - parser na ich
    podstawie powstanie później, wystarczy wtedy --rebuild)."""
    out = []
    for kind, href in refund_links(detail):
        start = len(collector.items)
        try:
            page.goto(href, wait_until="domcontentloaded")
        except Exception as e:
            print(f"    {kind}: BŁĄD {e.__class__.__name__}")
            continue
        collector.wait_quiet(page, start)
        apis = {api: scrub(b.get("data", b)) for api, b in collector.items[start:]
                if api not in (DETAIL_API, LIST_API) and not api.endswith(SKIP_REFUND_APIS)}
        out.append({"button": kind, "url": scrub(href), "apis": apis})
        page.wait_for_timeout(PAGE_DELAY_MS)
    return out


def fetch_details(page: Page, collector: Collector, creds: common.LazyCredentials | None,
                  orders: dict[str, dict], refresh: set[str]) -> None:
    """Pobiera szczegóły zamówień: brakujące oraz te z `refresh` (zamówienia
    z przeglądanego okresu - zwrot mógł się w tym czasie zmienić)."""
    DETAILS_DIR.mkdir(parents=True, exist_ok=True)
    todo = [oid for oid in orders if oid in refresh or not details_path(oid).exists()]
    if not todo:
        return
    todo.sort(key=lambda oid: parse_date_text(orders[oid].get("orderDateText", "")) or "",
              reverse=True)
    print(f"Sprawdzam szczegóły {len(todo)} zamówień...")
    for i, oid in enumerate(todo, 1):
        rec = None
        for _ in range(2):
            start = len(collector.items)
            try:
                page.goto(details_url(oid), wait_until="domcontentloaded")
            except Exception as e:
                print(f"  [{i}/{len(todo)}] {oid}: BŁĄD {e.__class__.__name__}")
                break
            detail = collector.wait_for(page, DETAIL_API, start)
            if detail is not None:
                page.wait_for_timeout(PAGE_DELAY_MS)  # dodatkowe API (np. spory)
                rec = details_record(oid, collector.items[start:])
                rec["refund_pages"] = fetch_refund_pages(page, collector, detail)
                break
            wait_if_interrupted(page, creds)
        if rec is None:
            print(f"  [{i}/{len(todo)}] {oid}: brak danych zamówienia - pomijam")
            continue
        path = details_path(oid)
        text = json.dumps(rec, ensure_ascii=False, indent=1)
        old = path.read_text(encoding="utf-8") if path.exists() else None
        if old != text:  # bez zmian treści -> plik bez zmian (czysta historia)
            path.write_text(text, encoding="utf-8")
        d = parse_details(rec)
        print(f"  [{i}/{len(todo)}] {d['date'] or '?'} {oid}: {d['status_text'].strip()}, "
              f"pozycji {len(d['items'])}"
              + (f", strony zwrotu / sporu: {len(rec['refund_pages'])}"
                 if rec["refund_pages"] else "")
              + (" (zmiana)" if old and old != text else ""))


def load_details() -> dict[str, dict]:
    out = {}
    for f in sorted(DETAILS_DIR.glob("*.json")) if DETAILS_DIR.exists() else []:
        rec = json.loads(f.read_text(encoding="utf-8"))
        out[str(rec["orderId"])] = parse_details(rec)
    return out


# --------------------------------------------------------------------------- #
# Parsowanie (bez przeglądarki)
# --------------------------------------------------------------------------- #
# Daty: lista i szczegóły podają tylko dzień („Nov 25, 2025”), w strefie czasowej
# przeglądarki (API dostaje parametr timeZone), czyli w czasie polskim.
MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
          "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
          "sty": 1, "lut": 2, "kwi": 4, "maj": 5, "cze": 6, "lip": 7, "sie": 8,
          "wrz": 9, "paź": 10, "paz": 10, "lis": 11, "gru": 12}
DATE_RES = [
    re.compile(r"\b([^\W\d_]{3,})\.?\s+(\d{1,2}),?\s+(\d{4})\b"),   # Nov 25, 2025
    re.compile(r"\b(\d{1,2})\.?\s+([^\W\d_]{3,})\.?\s+(\d{4})\b"),  # 25 lis 2025
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),                     # 2025-11-25
]


def parse_date_text(text: str) -> str | None:
    """Data z tekstu -> 'RRRR-MM-DD'."""
    for n, rx in enumerate(DATE_RES):
        for m in rx.finditer(text or ""):
            a, b, c = m.groups()
            if n == 0:
                y, month, day = int(c), MONTHS.get(a.lower()[:3]), int(b)
            elif n == 1:
                y, month, day = int(c), MONTHS.get(b.lower()[:3]), int(a)
            else:
                y, month, day = int(a), int(b), int(c)
            if month and 1 <= day <= 31:
                return f"{y:04d}-{month:02d}-{day:02d}"
    return None


def parse_amount(text) -> float | None:
    """Kwota z napisu w formacie strony: '€4.12', '-CHF1.40', 'zł1,234.56',
    '1 234,56 zł', "CHF1'234.50". 'Free shipping' -> 0. Brak liczby -> None."""
    text = str(text or "")
    m = re.search(r"\d[\d\s.,'’ ]*", text)
    if not m:
        return 0.0 if re.search(r"free|darmow|bezpłat", text, re.I) else None
    num = re.sub(r"[\s'’ ]", "", m.group()).rstrip(".,")
    dec = re.search(r"[.,](\d{1,2})$", num)
    whole = re.sub(r"[.,]", "", num[:dec.start()] if dec else num)
    value = float(whole + ("." + dec.group(1) if dec else ""))
    return -value if "-" in text[:m.start()] else value


def price_of(fields: dict) -> float | None:
    """Cena z 'formatPriceInfo' ('€3.50|3|50': część całkowita | grosze),
    a gdy jej nie ma - z tekstu ceny."""
    parts = str(fields.get("formatPriceInfo") or "").split("|")
    if len(parts) == 3 and parts[2].isdigit():
        whole = re.sub(r"\D", "", parts[1])
        if whole:
            return float(f"{whole}.{parts[2]}")
    for k in ("itemPriceText", "totalPriceText", "value"):
        if fields.get(k):
            return parse_amount(fields[k])
    return None


# Status zamówienia: „Expired” = nieopłacone w terminie, „Closed” = zamknięte
# przed realizacją albo po pełnym zwrocie pieniędzy. Oba nie są zakupem.
CANCELLED_RE = re.compile(r"expired|cancel|closed|unpaid|awaiting payment|"
                          r"wygas|anulow|zamknię|nieopłac|oczekuje na płatność", re.I)
REFUND_RE = re.compile(r"refund|zwrot|return|dispute|spór|reklamac", re.I)
# Wiersze rozbicia kwoty („priceDetails”): Subtotal, Shipping, AliExpress Coupons,
# Store discount, Coins, Tax… Kwoty ujemne to rabaty.
ROW_KEYS = {
    "subtotal": re.compile(r"^(subtotal|suma częściowa|wartość produktów|produkty|items)",
                           re.I),
    "shipping": re.compile(r"shipping|delivery|dostaw|wysył|przesył", re.I),
    "tax": re.compile(r"\btax|vat|podat|\bcło\b|duty|customs|import|handling", re.I),
    "total": re.compile(r"^(total|razem|suma|do zapłaty)$", re.I),
    "refund": REFUND_RE,
}


def parse_price_rows(rows: list[dict]) -> dict:
    """{tytuł: kwota} i wyliczone pola: delivery_cost, tax, discounts (ujemne),
    other (pozostałe dodatnie), refund_total, subtotal."""
    summary: dict[str, float] = {}
    hidden: set[str] = set()
    for r in rows or []:
        amount = parse_amount(r.get("value"))
        if r.get("title") and amount is not None:
            title = str(r["title"]).strip().rstrip(":")
            summary[title] = round(summary.get(title, 0) + amount, 2)
            if r.get("hide"):
                hidden.add(title)
    out = {"summary": summary, "delivery_cost": 0.0, "tax": 0.0, "discounts": 0.0,
           "other": 0.0, "refund_total": 0.0, "subtotal": None, "hidden_charges": {}}
    for title, v in summary.items():
        kind = next((k for k, rx in ROW_KEYS.items() if rx.search(title)), None)
        if title in hidden and v > 0 and kind in ("shipping", "tax"):
            # ukryte („hide”) dodatnie wiersze bywają niedoliczone do sumy,
            # np. „Shipping 0.02” - decyduje kontrola w merge_orders
            out["hidden_charges"][kind] = round(out["hidden_charges"].get(kind, 0) + v, 2)
        if kind == "subtotal":
            out["subtotal"] = v
        elif kind == "refund":
            out["refund_total"] += abs(v)
        elif kind == "total":
            continue
        elif v < 0:
            out["discounts"] += v
        elif kind == "shipping":
            out["delivery_cost"] += v
        elif kind == "tax":
            out["tax"] += v
        else:
            out["other"] += v
    return {k: round(v, 2) if isinstance(v, float) else v for k, v in out.items()}


# Zwroty pieniędzy: strony z przycisków pozycji wczytują
#   ….reverse.refundrecord.querydetail  (data: zwrot pieniędzy - kwota, liczba sztuk,
#                                        refundStatus FINISHED, refundSource),
#   ….reverse.reverseorderlinerenderforbuyer (module: spór - reachSolution.refundMoney,
#                                        reverseDetailStatusText „Request complete”).
# Oba mają tradeOrderLineId (pozycja) i reverseOrderLineId (zgłoszenie); ten sam
# zwrot bywa w obu, więc łączymy je po reverseOrderLineId (zapis zwrotu wygrywa).
# Sam przycisk „Refund information” nie znaczy, że był zwrot: w starszych
# zamówieniach prowadzi do formularza zgłoszenia, a spór może skończyć się kwotą 0.
def _amount(money) -> tuple[float | None, str | None]:
    if not isinstance(money, dict):
        return None, None
    unit = money.get("unit")
    if unit is None and money.get("cent") is not None:
        unit = int(money["cent"]) / 100
    return (float(unit) if unit is not None else None), money.get("currency")


def _join(*texts) -> str:
    return "; ".join(" ".join(str(t).split()) for t in texts if t and str(t).strip())


def parse_refunds(rec: dict) -> dict[str, list[dict]]:
    """tradeOrderLineId -> lista zwrotów {amount, currency, quantity, done, info}."""
    by_key: dict[str, dict] = {}
    for page in rec.get("refund_pages") or []:
        for api, v in (page.get("apis") or {}).items():
            if not isinstance(v, dict):
                continue
            if api.endswith("refundrecord.querydetail") and isinstance(v.get("data"), dict):
                d = v["data"]
                amount, cur = _amount(d.get("refundAmount"))
                key = str(d.get("reverseOrderLineId") or d.get("id"))
                by_key[key] = {
                    "line": str(d.get("tradeOrderLineId")), "amount": amount, "currency": cur,
                    "quantity": d.get("quantity"), "source": "record",
                    "done": d.get("refundStatus") == "FINISHED"
                    or d.get("reverseStatus") == "REFUND_SUCCESS",
                    "info": _join(d.get("reason"), d.get("refundSource"), d.get("refundStatus")),
                }
            elif api.endswith("reverseorderlinerenderforbuyer") and \
                    isinstance(v.get("module"), dict):
                m = v["module"]
                key = str(m.get("reverseOrderLineId"))
                if by_key.get(key, {}).get("source") == "record":
                    continue
                solution = m.get("reachSolution") or {}
                amount, cur = _amount(solution.get("refundMoney"))
                if amount is None:
                    amount, cur = _amount(((m.get("reverseFinishInfo") or {}).get(
                        "refundInfo") or {}).get("refundDetails", {}).get("cashMoney"))
                status = str(m.get("reverseDetailStatusText") or "")
                by_key[key] = {
                    "line": str(m.get("tradeOrderLineId")), "amount": amount, "currency": cur,
                    "quantity": (m.get("item") or {}).get("itemCount"), "source": "dispute",
                    "done": bool(m.get("reverseFinishInfo")) or "complete" in status.lower(),
                    "info": _join((m.get("applyReason") or {}).get("reasonName"),
                                  solution.get("solutionTypeText"), status),
                }
    out: dict[str, list[dict]] = {}
    for r in by_key.values():
        out.setdefault(r["line"], []).append(r)
    return out


def _comp(rec: dict, tag: str) -> dict:
    return rec["components"].get(tag) or {}


def parse_details(rec: dict) -> dict:
    status = _comp(rec, "detail_order_status_block")
    products = _comp(rec, "detail_product_block")
    prices = _comp(rec, "detail_order_price_block")
    info = _comp(rec, "detail_simple_order_info_component")
    refunds = parse_refunds(rec)
    items = []
    for line in products.get("productVOList") or []:
        items.append(_item(line) | {"refunds": refunds.pop(str(line.get("orderLineId")), [])})
    total = prices.get("totalPrice") or {}
    known = {"detail_order_status_block", "detail_product_block",
             "detail_order_price_block", "detail_simple_order_info_component",
             "detail_aync_block", "detail_expose_block"}
    return {
        "date": parse_date_text(info.get("orderCreatTime", "")),
        "paid": parse_date_text(info.get("payTime", "")),
        "status_text": status.get("title") or "",
        "reminder": status.get("reminder") or "",
        "seller": (products.get("sellerVO") or {}).get("sellerName") or "",
        "items": items,
        "total": price_of(total),
        "currency": total.get("currencyCode"),
        "vat_included": (prices.get("totalPriceHint") or {}).get("hintPriceValue"),
        "payment_method": info.get("paymentMethod"),
        # zwroty, których nie da się przypisać do pozycji zamówienia
        "unassigned_refunds": [r for rs in refunds.values() for r in rs],
        # komponenty i API, których nie znamy - mogą opisywać spór / zwrot
        "unknown_parts": sorted([k for k in rec["components"] if k.split("#")[0] not in known]
                                + list(rec.get("apis") or {})),
    } | parse_price_rows(prices.get("priceDetails"))


def _item(line: dict) -> dict:
    sku = "; ".join(f"{a.get('name')}: {a.get('text')}" for a in line.get("skuAttrs") or []
                    if a.get("name") != "Ships From")
    url = str(line.get("itemDetailUrl") or "")
    return {
        "product_id": str(line.get("productId") or ""), "sku_id": str(line.get("skuId") or ""),
        "order_line_id": str(line.get("orderLineId") or ""),
        "title": " ".join(str(line.get("itemTitle") or "").split()), "sku": sku,
        "quantity": int(line.get("quantity") or 1),
        "unit_price": price_of(line), "currency": line.get("currencyCode"),
        "url": ("https:" + url if url.startswith("//") else url).split("?")[0],
    }


# --------------------------------------------------------------------------- #
# Pliki wynikowe
# --------------------------------------------------------------------------- #
def apply_refunds(it: dict, cancelled: bool) -> None:
    """Ustala stan pozycji z jej zwrotów (it["refunds"]):
      * zakończony zwrot = wartość pozycji (albo k × cena sztuki) -> zwrócone
        całe / k sztuk (także „zwrot bez odsyłania” - pieniądze wróciły),
      * mniejsza kwota -> częściowy zwrot pieniędzy: pozycja zostaje, a w
        items_kept.csv ma cenę pomniejszoną o zwrot,
      * spór niezakończony -> tylko informacja w return_info."""
    done = [r for r in it["refunds"] if r["done"] and r["amount"]]
    amount = round(sum(r["amount"] for r in done), 2)
    other_cur = any(r["currency"] and it["currency"] and r["currency"] != it["currency"]
                    for r in done)
    it["refund_amount"] = amount or None
    it["refund_currency"] = done[0]["currency"] if done else None
    it["return_info"] = " | ".join(
        f"{r['amount'] if r['amount'] is not None else '?'} {r['currency'] or ''}: {r['info']}"
        + ("" if r["done"] else " (w toku)") for r in it["refunds"])
    it["state"], it["returned_quantity"] = "", 0
    if cancelled:
        it["state"] = "cancelled"
    elif amount and it["unit_price"] and not other_cur:
        unit, qty = it["unit_price"], it["quantity"]
        k = round(amount / unit)
        if amount >= unit * qty - 0.015:
            it["state"], it["returned_quantity"] = "returned", qty
        elif 1 <= k < qty and abs(amount - k * unit) < 0.015:
            it["state"], it["returned_quantity"] = "returned", k
        else:
            it["state"] = "partially_refunded"
    elif amount:
        it["state"] = "partially_refunded"  # inna waluta zwrotu - zostaje w raporcie
    elif any(not r["done"] for r in it["refunds"]):
        it["state"] = "dispute"


def merge_orders() -> list[dict]:
    """Zamówienia z listy uzupełnione o szczegóły (rozbicie kwot, zwroty)."""
    listed = load_saved_orders()
    details = load_details()
    groups = Counter(o.get("paymentOutId") for o in listed.values() if o.get("paymentOutId"))
    orders = []
    for oid, o in listed.items():
        d = details.get(oid) or {}
        lines = d["items"] if d.get("items") else \
            [_item(line) | {"refunds": []} for line in o.get("orderLines") or []]
        status_text = (d.get("status_text") or o.get("statusText") or "").strip()
        # „Expired” / „Canceled” / „Closed”: nie ma zakupu, nawet jeśli pieniądze
        # pobrano i potem zwrócono
        cancelled = bool(CANCELLED_RE.search(status_text))
        items = [dict(it) for it in lines]
        for it in items:
            apply_refunds(it, cancelled)
        states = {it["state"] for it in items}
        status = ("CANCELLED" if states == {"cancelled"} else
                  "RETURNED" if states <= {"returned", "cancelled"} and "returned" in states
                  else "PARTIALLY_RETURNED" if states & {"returned", "partially_refunded"}
                  else "DISPUTE" if "dispute" in states
                  else "PARTIALLY_CANCELLED" if "cancelled" in states else "OK")
        items_sum = round(sum((it["unit_price"] or 0) * it["quantity"] for it in items), 2)
        total = d.get("total")
        if total is None:
            total = price_of({"formatPriceInfo": o.get("formatPriceInfo"),
                              "totalPriceText": o.get("totalPriceText")})
        charges = {k: d.get(k) for k in ("delivery_cost", "tax", "discounts", "other")}
        # ukryte wiersze (np. „Shipping 0.02”), których AliExpress nie doliczył do sumy
        hidden = d.get("hidden_charges") or {}
        if total is not None and hidden and charges["delivery_cost"] is not None:
            charged = items_sum + sum(charges.values())
            if abs(charged - sum(hidden.values()) - total) < 0.015:
                for kind, v in hidden.items():
                    key = "delivery_cost" if kind == "shipping" else kind
                    charges[key] = round(charges[key] - v, 2)
        refund = round(sum(it["refund_amount"] or 0 for it in items)
                       + (d.get("refund_total") or 0), 2)
        orders.append({
            "orderId": oid,
            "date": parse_date_text(o.get("orderDateText", "")) or d.get("date"),
            "status": status, "status_text": status_text,
            "deleted": o.get("deleted", False),
            "seller": o.get("storeName") or d.get("seller") or "",
            "total": total,
            "total_from": "details" if d.get("total") is not None else
                          "list" if total is not None else None,
            "currency": o.get("currencyCode") or d.get("currency")
            or next((it["currency"] for it in items if it["currency"]), None)
            or o.get("intentionCurrency"),
            "items_sum": items_sum,
            "subtotal": d.get("subtotal"),
            "delivery_cost": charges["delivery_cost"],
            "tax": charges["tax"],
            "discounts": charges["discounts"],
            "other_charges": charges["other"],
            "refund_total": refund or None,
            "vat_included": d.get("vat_included"),
            "summary": d.get("summary"),
            "payment_id": o.get("paymentOutId"),
            "orders_in_payment": groups.get(o.get("paymentOutId"), 1),
            "paid": d.get("paid"),
            "payment_method": d.get("payment_method"),
            "reminder": d.get("reminder"),
            "unassigned_refunds": d.get("unassigned_refunds") or [],
            "unknown_parts": d.get("unknown_parts"),
            "details_fetched": bool(d),
            "items": items,
        })
    orders.sort(key=lambda o: (o["date"] or "", o["orderId"]), reverse=True)
    return orders


ITEM_FIELDS = ["order_id", "date", "seller", "order_status", "status_text", "order_total",
               "product_id", "title", "sku", "quantity", "unit_price", "currency",
               "delivery_cost", "item_status", "returned_quantity", "refund_amount",
               "refund_currency", "order_refund_total", "return_info", "payment_id", "url"]
KEPT_FIELDS = ["order_id", "date", "seller", "product_id", "title", "sku", "quantity",
               "unit_price", "total", "currency", "url"]


def flatten(orders: list[dict]) -> list[dict]:
    rows = []
    for o in orders:
        for it in o["items"]:
            rows.append({
                "order_id": o["orderId"], "date": o["date"], "seller": o["seller"],
                "order_status": o["status"], "status_text": o["status_text"],
                "order_total": o["total"], "product_id": it["product_id"],
                "title": it["title"], "sku": it["sku"], "quantity": it["quantity"],
                "unit_price": it["unit_price"], "currency": it["currency"] or o["currency"],
                "delivery_cost": o["delivery_cost"], "item_status": it["state"],
                "returned_quantity": it["returned_quantity"] or "",
                "refund_amount": it["refund_amount"] or "",
                "refund_currency": it["refund_currency"] or "",
                "order_refund_total": o["refund_total"] or "",
                "return_info": it["return_info"], "payment_id": o["payment_id"],
                "url": it["url"],
            })
    return rows


def kept_items(rows: list[dict]) -> list[dict]:
    """Pozycje, które zostały u kupującego: bez anulowanych / nieopłaconych
    zamówień i bez zwróconych sztuk. Przy częściowym zwrocie pieniędzy (np.
    rekompensata za brakującą część) pozycja zostaje z ceną pomniejszoną
    o zwrot. Pozycje bez ceny są pomijane."""
    out = []
    for r in rows:
        if r["item_status"] == "cancelled" or r["unit_price"] is None:
            continue
        qty = int(r["quantity"] or 0) - int(r["returned_quantity"] or 0)
        if qty <= 0:
            continue
        total = float(r["unit_price"]) * qty
        if r["item_status"] == "partially_refunded" and r["refund_currency"] in ("", r["currency"]):
            total = max(float(r["unit_price"]) * int(r["quantity"]) - float(r["refund_amount"]), 0)
        out.append({k: r.get(k) for k in KEPT_FIELDS}
                   | {"quantity": qty, "unit_price": round(total / qty, 2),
                      "total": f"{total:.2f}"})
    return out


def check_consistency(orders: list[dict]) -> None:
    """Suma pozycji + dostawa + podatek/cło + rabaty (kupony, monety) + inne
    opłaty = kwota zapłacona za zamówienie; zwroty pieniędzy nieprzypisane do
    pozycji."""
    bad, ok = [], 0
    for o in orders:
        if o["total"] is None or o["total_from"] != "details" or o["status"] == "CANCELLED":
            continue
        o["expected_total"] = round(o["items_sum"] + o["delivery_cost"] + o["tax"]
                                    + o["discounts"] + o["other_charges"], 2)
        if abs(o["expected_total"] - o["total"]) > 0.015:
            bad.append(o)
        else:
            ok += 1
    print(f"  spójność: suma pozycji + dostawa + podatek + rabaty = zapłacona kwota "
          f"w {ok} z {ok + len(bad)} zamówień z rozbiciem kwot")
    if bad:
        print(f"  uwaga: niezgodnych zamówień: {len(bad)}, np. "
              + ", ".join(f"{o['orderId']} ({o['expected_total']:.2f} vs {o['total']:.2f} "
                          f"{o['currency']}, wiersze: {o['summary']})" for o in bad[:3]))
    no_breakdown = [o for o in orders if o["status"] != "CANCELLED"
                    and o["total_from"] != "details"]
    if no_breakdown:
        print(f"  bez rozbicia kwot (AliExpress go nie podaje, najczęściej stare "
              f"zamówienia): {len(no_breakdown)}, np. "
              + ", ".join(f"{o['orderId']} ({o['date']})" for o in no_breakdown[:3]))
    unassigned = [o for o in orders if o["unassigned_refunds"]]
    if unassigned:
        print(f"  uwaga: zwroty pieniędzy nieprzypisane do pozycji: {len(unassigned)}, np. "
              + ", ".join(f"{o['orderId']} ({[r['amount'] for r in o['unassigned_refunds']]})"
                          for o in unassigned[:3]))
    refunded = [(o, it) for o in orders for it in o["items"] if it["refunds"]]
    print(f"  zwroty / spory: pozycji {len(refunded)} - zwróconych "
          f"{sum(it['state'] == 'returned' for _, it in refunded)}, częściowy zwrot "
          f"pieniędzy {sum(it['state'] == 'partially_refunded' for _, it in refunded)}, "
          f"spór w toku {sum(it['state'] == 'dispute' for _, it in refunded)}, "
          f"bez zwrotu {sum(it['state'] == '' for _, it in refunded)}, "
          f"w anulowanych {sum(it['state'] == 'cancelled' for _, it in refunded)}")
    unknown = Counter(p for o in orders for p in o["unknown_parts"] or [])
    if unknown:
        print("  nieznane części szczegółów (mogą opisywać spór / zwrot): "
              + ", ".join(f"{k} ×{n}" for k, n in unknown.most_common()))
    statuses = Counter(o["status_text"] for o in orders)
    print("  statusy AliExpress: " + ", ".join(f"{s or '(brak)'} ×{n}"
                                               for s, n in statuses.most_common()))


CHART_RECENT_YEARS = 5  # drugi wykres: tylko ostatnie lata (pełna historia jest długa)


def last_years(rows: list[dict], years: int) -> list[dict]:
    """Wiersze zakupy.csv z ostatnich `years` lat (12 × years miesięcy), licząc
    od miesiąca najnowszego zakupu, a nie od dzisiaj - dzięki temu --rebuild
    daje ten sam wykres niezależnie od dnia uruchomienia."""
    months = [r["data"][:7] for r in rows if r["data"]]
    if not months:
        return []
    y, m = map(int, max(months).split("-"))
    y, m = (y - years, m + 1) if m < 12 else (y - years + 1, 1)
    start = f"{y:04d}-{m:02d}"
    return [r for r in rows if r["data"] and r["data"][:7] >= start]


def rebuild_outputs() -> list[dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    orders = merge_orders()
    check_consistency(orders)
    (OUTPUT_DIR / "orders.json").write_text(
        json.dumps(orders, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = flatten(orders)
    with open(OUTPUT_DIR / "items.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ITEM_FIELDS)
        w.writeheader()
        w.writerows(rows)
    kept = kept_items(rows)
    with open(OUTPUT_DIR / "items_kept.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KEPT_FIELDS)
        w.writeheader()
        w.writerows(kept)
    simple = common.write_simple_csv(kept, OUTPUT_DIR / "zakupy.csv", NBP_CACHE, "aliexpress")
    common.write_monthly_chart(simple, OUTPUT_DIR / "wydatki_miesiecznie.png", "AliExpress")
    recent = last_years(simple, CHART_RECENT_YEARS)
    common.write_monthly_chart(
        recent, OUTPUT_DIR / f"wydatki_miesiecznie_{CHART_RECENT_YEARS}_lat.png", "AliExpress",
        title=f"Wydatki na AliExpress w ostatnich {CHART_RECENT_YEARS} latach")
    n_ret = sum(1 for r in rows if r["item_status"] == "returned")
    print(f"Zamówień: {len(orders)}, pozycji: {len(rows)} (zwróconych: {n_ret}), "
          f"zatrzymanych pozycji: {len(kept)} -> {OUTPUT_DIR}/")
    if not orders:
        print("Nie rozpoznano żadnych zamówień. Uruchom z --discover i sprawdź "
              f"{DISCOVERY_DIR}/ (AliExpress mógł zmienić format danych).")
    return orders


def commit_data(message: str) -> None:
    common.commit_data(DATA_DIR, "aliexpress", message,
                       {"aliexpress_raw/details/": "zmienione szczegóły zamówień"})


# --------------------------------------------------------------------------- #
# Tryb diagnostyczny
# --------------------------------------------------------------------------- #
# Ze strony zostaje tylko jej treść, bez skryptów, stylów, obrazków (z obrazków
# zostaje `alt`), rekomendacji i komentarzy. Generowane klasy CSS
# („order--title--3xHk8aZ”) tracą losową końcówkę, a linki - parametry śledzenia.
SIMPLIFY_JS = r"""
(roots) => {
  let root = null;
  for (const sel of roots) { root = document.querySelector(sel); if (root) break; }
  root = (root || document.body).cloneNode(true);
  root.querySelectorAll(
    'script,style,noscript,svg,iframe,link,meta,template,canvas,video,input[type=hidden],' +
    'header,footer,#header,#footer,.site-footer,[class*=recommend],[class*=Recommend],' +
    '[class*=baxia],[id*=baxia]'
  ).forEach(e => e.remove());
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_COMMENT);
  const comments = [];
  while (walker.nextNode()) comments.push(walker.currentNode);
  comments.forEach(c => c.remove());
  const keep = new Set(['class', 'href', 'alt', 'title', 'aria-label', 'role']);
  const idParam = /^(order|dispute|issue|reverse|refund|return|trade)\w*id$/i;
  for (const el of [root, ...root.querySelectorAll('*')]) {
    for (const a of [...el.attributes]) {
      if (!keep.has(a.name)) { el.removeAttribute(a.name); continue; }
      if (a.name === 'class') {
        const cls = a.value.split(/\s+/).filter(Boolean)
          .map(c => c.replace(/^(.+--[^-]+)--[A-Za-z0-9_]{4,10}$/, '$1'));
        el.setAttribute('class', cls.join(' '));
      }
      if (a.name === 'href') {
        try {
          const u = new URL(a.value, location.href);
          const q = [...u.searchParams].filter(([k]) => idParam.test(k))
            .map(([k, v]) => k + '=' + v).join('&');
          el.setAttribute('href', u.host + u.pathname + (q ? '?' + q : ''));
        } catch (e) { el.removeAttribute('href'); }
      }
    }
  }
  return root.outerHTML;
}
"""
PAGE_ROOTS = ["#root", "#app", "body"]


def simplified_html(page: Page) -> str:
    html = page.evaluate(SIMPLIFY_JS, PAGE_ROOTS)
    return re.sub(r">\s+<", "><", re.sub(r"\s+", " ", html)).strip()


TAB_SELECTORS = [".comet-tabs-nav-item", "[role=tab]"]
INTERESTING_RE = re.compile(r"zwrot|refund|spór|spor|dispute|reklam|anul|cancel|closed|"
                            r"zamknię|return|nieudan|failed|expired|deleted|usunię", re.I)
ORDER_KEYS = {"orderId", "orderID", "order_id", "tradeOrderId", "orderNo"}

# Zamówienia widoczne na liście: dla każdego linku ze „orderId=” szukamy
# najszerszego przodka, który zawiera tylko to jedno zamówienie (karta zamówienia).
ORDERS_ON_PAGE_JS = r"""
() => {
  const idOf = a => { try { return new URL(a.href, location.href).searchParams.get('orderId'); }
                      catch (e) { return null; } };
  const out = {};
  for (const a of document.querySelectorAll('a[href*="orderId="]')) {
    const id = idOf(a);
    if (!id || out[id]) continue;
    let card = a;
    while (card.parentElement && card.parentElement !== document.body) {
      const ids = new Set([...card.parentElement.querySelectorAll('a[href*="orderId="]')]
                          .map(idOf).filter(Boolean));
      if (ids.size > 1) break;
      card = card.parentElement;
    }
    const st = card.querySelector('[class*=status], [class*=Status]');
    out[id] = {orderId: id, status: st ? st.innerText.trim() : '',
               products: card.querySelectorAll('img').length,
               text: card.innerText.replace(/\s+/g, ' ').slice(0, 300)};
  }
  return Object.values(out);
}
"""


def _slug(text: str) -> str:
    return re.sub(r"\W+", "_", text.lower()).strip("_")[:30] or "x"


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def json_statuses(bodies: list[dict]) -> Counter:
    """Pola statusu („…status…” = wartość) w obiektach JSON z numerem zamówienia,
    liczone raz na zamówienie."""
    seen: dict[tuple[str, str], str] = {}
    for b in bodies:
        for d in _walk(b["body"]):
            oid = next((str(d[k]) for k in ORDER_KEYS if d.get(k)), None)
            if not oid:
                continue
            for k, v in d.items():
                if "status" in k.lower() and isinstance(v, (str, int)) and str(v):
                    seen[(oid, k)] = f"{k} = {v}"
    return Counter(seen.values())


class Snapshots:
    """Zapisuje do ./discovery pełny HTML, uproszczony HTML i zrzut ekranu strony."""

    def __init__(self):
        self.urls: dict[str, str] = {}  # nazwa pliku -> adres strony

    def save(self, page: Page, name: str) -> None:
        name = f"{len(self.urls) + 1:02d}_{name}"
        stem = DISCOVERY_DIR / name
        try:
            stem.with_suffix(".html").write_text(page.content(), encoding="utf-8")
            stem.with_suffix(".simple.html").write_text(simplified_html(page),
                                                        encoding="utf-8")
            page.screenshot(path=str(stem.with_suffix(".png")), full_page=True)
        except Exception as e:
            print(f"  (nie zapisano {name}: {e.__class__.__name__})")
            return
        self.urls[name] = page.url
        print(f"  zapisano {name}: {page.url[:100]}")


def tab_names(page: Page) -> list[str]:
    """Zakładki listy zamówień oraz „Deleted orders” (link w prawym górnym rogu)."""
    names = []
    for sel in TAB_SELECTORS:
        names = [" ".join(t.split()) for t in page.locator(sel).all_inner_texts()]
        if names:
            break
    try:
        tail = page.get_by_text(re.compile(r"deleted orders|usunięte zamówienia", re.I)).first
        if tail.count():
            names.append(" ".join(tail.inner_text().split()))
    except Exception:
        pass
    return names


def discover_auto(page: Page, creds: common.LazyCredentials | None,
                  snaps: Snapshots, bodies: list[dict]) -> None:
    """Sam przegląda listę zamówień: każdą zakładkę (kilka stron w „View all”,
    dwie w pozostałych), potem szczegóły zamówień ze zwrotem / sporem /
    anulowaniem (maks. 15) i 4 zwykłych z największą liczbą produktów, a na
    koniec strony sporów i zwrotów, do których prowadzą linki ze szczegółów."""
    open_page(page, ORDERS_URL, creds, LIST_READY)
    seen: dict[str, dict] = {}   # orderId -> zamówienie z listy
    names = tab_names(page) or ["lista"]
    print(f"Zakładki listy zamówień: {', '.join(names)}")
    for t, name in enumerate(names):
        if t:
            open_page(page, ORDERS_URL, creds, LIST_READY)
            if not click_text(page, name):
                print(f"  nie udało się kliknąć zakładki '{name}'")
                continue
            page.wait_for_timeout(3000)
            wait_if_interrupted(page, creds)
        max_pages = 5 if t == 0 else 2
        for p in range(1, max_pages + 1):
            found = page.evaluate(ORDERS_ON_PAGE_JS)
            for o in found:
                seen.setdefault(o["orderId"], o).setdefault("tabs", []).append(name)
            snaps.save(page, f"list_{t}_{_slug(name)}_p{p}")
            print(f"  zakładka '{name}', strona {p}: zamówień na stronie {len(found)}")
            if p == max_pages or not click_more(page):
                break
            page.wait_for_timeout(3000)
            wait_if_interrupted(page, creds)

    print(f"\nZamówień na zapisanych stronach: {len(seen)}. Statusy (z karty zamówienia):")
    for s, n in Counter(o["status"] for o in seen.values()).most_common():
        print(f"  {n:4d}  {s or '(brak)'}")
    print("Pola statusu w odpowiedziach JSON (raz na zamówienie):")
    for s, n in json_statuses(bodies).most_common(40):
        print(f"  {n:4d}  {s}")

    odd = [o for o in seen.values() if INTERESTING_RE.search(o["status"])
           or any(INTERESTING_RE.search(t) for t in o.get("tabs", []))]
    normal = sorted((o for o in seen.values() if o not in odd), key=lambda o: -o["products"])
    picked = odd[:15] + normal[:4]
    print(f"\nZapisuję szczegóły {len(picked)} zamówień "
          f"(ze zwrotem / sporem / anulowaniem / usuniętych: {len(odd[:15])})...")
    extra_links: dict[str, str] = {}
    for o in picked:
        try:
            open_page(page, details_url(o["orderId"]), creds, DETAILS_READY)
            snaps.save(page, f"details_{o['orderId']}")
            for href in page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)"):
                if is_aliexpress(href) and re.search(r"dispute|refund|reverse|return|issue",
                                                     href, re.I):
                    extra_links.setdefault(scrub(href), href)
        except Exception as e:
            print(f"  {o['orderId']}: BŁĄD {e.__class__.__name__}")
    if extra_links:
        print(f"\nZapisuję strony sporów / zwrotów z linków w szczegółach "
              f"({min(len(extra_links), 8)} z {len(extra_links)})...")
    for href in list(extra_links.values())[:8]:
        try:
            open_page(page, href, creds, "body")
            page.wait_for_timeout(2000)
            snaps.save(page, f"dispute_{_slug(urlparse(href).path)}")
        except Exception as e:
            print(f"  {href[:80]}: BŁĄD {e.__class__.__name__}")
    (DISCOVERY_DIR / "orders_seen.json").write_text(
        json.dumps(list(seen.values()), ensure_ascii=False, indent=1), encoding="utf-8")


def discover(context: BrowserContext, page: Page, creds: common.LazyCredentials | None) -> None:
    """Zapisuje strony zamówień i odpowiedzi JSON (bez telemetrii) do ./discovery:
    najpierw automatycznie (discover_auto), potem każdą stronę, którą sam
    otworzysz, dopóki nie naciśniesz Enter (maks. 15 min)."""
    DISCOVERY_DIR.mkdir(exist_ok=True)
    DISCOVERY_DIR.chmod(0o700)
    bodies: list[dict] = []

    def on_response(resp: Response) -> None:
        body = response_json(resp)
        if body is not None:
            bodies.append({"api": api_name(resp.url), "url": resp.url,
                           "status": resp.status, "body": body})
    context.on("response", on_response)
    snaps = Snapshots()

    def save_log() -> None:
        (DISCOVERY_DIR / "network_log.json").write_text(
            json.dumps(bodies, ensure_ascii=False, indent=1), encoding="utf-8")
        (DISCOVERY_DIR / "urls.json").write_text(
            json.dumps(snaps.urls, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        discover_auto(page, creds, snaps, bodies)
    finally:
        save_log()  # zapisz to, co już zebrano, nawet po błędzie

    finished = threading.Event()
    threading.Thread(target=lambda: (input(), finished.set()), daemon=True).start()
    print("\nGotowe. Możesz jeszcze otworzyć w przeglądarce inne strony (np. spór albo\n"
          "zamówienie ze zwrotem, którego skrypt nie znalazł) - każda zostanie zapisana.\n"
          "Naciśnij Enter w terminalu, żeby zakończyć (maks. 15 min).")
    known = set(snaps.urls.values())
    deadline = time.time() + 900
    while time.time() < deadline and not finished.is_set():
        for pg in [p for p in context.pages if not p.is_closed()]:
            try:
                pg.wait_for_load_state("load", timeout=3000)
                url = pg.url
                if url in known or not is_aliexpress(url) or needs_user(pg):
                    continue
                pg.wait_for_timeout(2500)
                known.add(url)
                snaps.save(pg, f"manual_{_slug(urlparse(url).path)}")
            except Exception as e:
                print(f"  (nie zapisano strony: {e.__class__.__name__})")
        time.sleep(1)
    save_log()
    apis = Counter(b["api"] for b in bodies)
    print(f"\nZapisano {len(snaps.urls)} stron i {len(bodies)} odpowiedzi JSON "
          f"do {DISCOVERY_DIR}/. Odpowiedzi według API:")
    for name, n in apis.most_common():
        print(f"  {n:4d}  {name}")


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
                        help="zapisz strony zamówień i odpowiedzi JSON do ./discovery")
    g.add_argument("--rebuild", action="store_true",
                   help="tylko odtwórz pliki wynikowe z ./dane/aliexpress_raw (bez przeglądarki)")
    parser.add_argument("--days", type=int, default=30,
                        help="przeglądaj zakupy do N dni przed ostatnim zapisanym "
                             "(domyślnie 30), żeby złapać zwroty i anulowania")
    parser.add_argument("--full", action="store_true",
                        help="przejrzyj całą listę i odśwież szczegóły wszystkich zamówień")
    args = parser.parse_args()

    if args.save_credentials:
        return common.save_credentials(KEYRING_SERVICE, "AliExpress",
                                       "e-mail lub numer telefonu")
    if args.forget_credentials:
        return common.forget_credentials(KEYRING_SERVICE)
    if args.rebuild:
        rebuild_outputs()
        return commit_data(f"przebudowa plików wynikowych {time.strftime('%Y-%m-%d %H:%M')}")

    creds = None if args.manual_login else \
        common.LazyCredentials(KEYRING_SERVICE, "AliExpress", ask=args.ask)
    proc = common.start_browser(PROFILE_DIR, CDP_PORT, ORDERS_URL)
    try:
        login(creds)
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(CDP_URL)
            context = browser.contexts[0]
            page = context.pages[0]
            wait_if_interrupted(page, creds)  # np. okienko logowania na liście
            if args.discover:
                return discover(context, page, creds)
            collector = Collector()
            context.on("response", collector.on_response)
            known_before = set(load_saved_orders())
            cutoff = None if args.full else browse_cutoff(args.days)
            print("Przeglądam " + (f"zakupy od {cutoff}..." if cutoff
                                   else "całą listę zamówień..."))
            pages: list[dict] = []
            try:
                reached_end = browse_orders(page, collector, creds, cutoff, pages)
            finally:
                save_list_pages(pages)  # zapisz to, co już pobrano, nawet po błędzie
            if reached_end:
                HISTORY_COMPLETE.touch()
            browsed = {str(o.get("orderId")) for pg in pages for o in pg["orders"]}
            orders = load_saved_orders()
            fetch_details(page, collector, creds, orders,
                          refresh=set(orders) if args.full else browsed)
            orders_out = rebuild_outputs()
            new = len({o["orderId"] for o in orders_out} - known_before)
            commit_data(f"pobranie {time.strftime('%Y-%m-%d %H:%M')}: "
                        f"zamówień {len(orders_out)}, nowych {new}")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
