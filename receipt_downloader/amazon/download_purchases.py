"""
Pobiera listę zakupów z Amazona (Twoje zamówienia, domyślnie amazon.pl).

Dane logowania NIE są trzymane w plikach projektu. Do wyboru:
  * systemowy pęk kluczy (GNOME Keyring / KWallet, przez bibliotekę `keyring`),
    zapisany raz poleceniem `--save-credentials`; hasło jest zaszyfrowane
    i odblokowywane razem z sesją użytkownika,
  * wpisywanie loginu i hasła w terminalu przy każdym logowaniu (getpass, bez echa),
    gdy w pęku kluczy nic nie ma albo z flagą `--ask`,
  * `--manual-login`: skrypt niczego nie wpisuje, logujesz się sam w oknie przeglądarki.
Hasło jest potrzebne tylko wtedy, gdy sesja w .browser_profile/ wygasła.

Chromium startuje jako zwykły proces z trwałym profilem. Skrypt podłącza się
do niego przez CDP tylko na chwilę, żeby wpisać dane (Amazon pyta najpierw
o e-mail, a dopiero na następnej stronie o hasło). Przyciski „Dalej” / „Zaloguj
się”, captchę i kod OTP/2FA obsługujesz sam.

Amazon nie ma API z historią zamówień, a strony zamówień są renderowane po stronie
serwera (HTML, nie JSON). Skrypt przegląda listę zamówień rok po roku, zapisuje
uproszczony HTML każdej strony listy i strony szczegółów każdego zamówienia
(dane/amazon_raw/), a dopiero potem wyciąga z niego zakupy. Dzięki temu poprawka
parsera wymaga tylko `--rebuild`, a nie ponownego pobierania.

Użycie:
    uv run download_purchases.py                    # pobierz zakupy
    uv run download_purchases.py --save-credentials # zapisz login i hasło w pęku kluczy
    uv run download_purchases.py --forget-credentials
    uv run download_purchases.py --ask              # zapytaj o dane, nie używaj pęku kluczy
    uv run download_purchases.py --manual-login     # loguj się ręcznie w przeglądarce
    uv run download_purchases.py --days 60          # przeglądaj 60 dni wstecz (domyślnie 30)
    uv run download_purchases.py --full             # przejrzyj wszystkie lata (odśwież statusy)
    uv run download_purchases.py --rebuild          # odtwórz pliki wynikowe z dane/amazon_raw
    uv run download_purchases.py --discover         # zrzut stron do ./discovery (diagnostyka)
    uv run download_purchases.py --domain amazon.de # inna domena Amazona
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
from urllib.parse import urlparse

from playwright.sync_api import BrowserContext, Page, sync_playwright

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
import common  # noqa: E402  (../common.py: logowanie, przeglądarka, zakupy.csv, wykres, NBP)

PROFILE_DIR = BASE_DIR / ".browser_profile"
# Dane osobowe trzymamy w ./dane: ignoruje je repozytorium z kodem, a ich historia
# jest we wspólnym prywatnym repo .git-private w katalogu projektu (common.commit_data).
DATA_DIR = BASE_DIR / "dane"
RAW_DIR = DATA_DIR / "amazon_raw"          # uproszczony HTML stron listy i szczegółów
DETAILS_DIR = RAW_DIR / "details"           # szczegóły zamówień: <nr_zamówienia>.json
OUTPUT_DIR = DATA_DIR / "amazon_purchases"  # wynik: zamówienia i pozycje
DISCOVERY_DIR = BASE_DIR / "discovery"
HISTORY_COMPLETE = RAW_DIR / ".history_complete"  # znacznik: wszystkie lata przejrzane
NBP_CACHE = RAW_DIR / "nbp_rates.json"

DOMAIN = "amazon.pl"  # zmieniane przez --domain (patrz set_domain)
LOGIN_TIMEOUT_S = 300  # czas na ręczne dokończenie logowania (captcha, OTP)
PAGE_DELAY_MS = 1500   # przerwa między stronami - nie zasypujemy serwera zapytaniami

KEYRING_SERVICE = "receipt_parsing-amazon"  # login i hasło: common.LazyCredentials


def base_url() -> str:
    return f"https://www.{DOMAIN}"


def orders_url(time_filter: str | None = None, start: int = 0) -> str:
    url = f"{base_url()}/your-orders/orders"
    if time_filter:
        url += f"?timeFilter={time_filter}&startIndex={start}"
    return url


def details_url(order_id: str) -> str:
    if order_id.startswith("D"):  # zamówienia cyfrowe (Kindle, aplikacje…)
        return f"{base_url()}/gp/digital/your-account/order-summary.html?orderID={order_id}"
    return f"{base_url()}/your-orders/order-details?orderID={order_id}"


def set_domain(domain: str) -> None:
    global DOMAIN
    DOMAIN = domain.removeprefix("https://").removeprefix("www.").strip("/")


# --------------------------------------------------------------------------- #
# Przeglądarka i logowanie
# --------------------------------------------------------------------------- #
CDP_PORT = 9231  # inny niż w allegro (9230) i biedronka_receipt_downloader (9229)
CDP_URL = f"http://127.0.0.1:{CDP_PORT}"

EMAIL_SELECTORS = ["input#ap_email", "input#ap_email_login", "input[name=email]",
                   "input[type=email]", "input[autocomplete=username]"]
PASSWORD_SELECTORS = ["input#ap_password", "input[name=password]", "input[type=password]"]
REMEMBER_ME = "input[name=rememberMe]"  # „Nie wylogowuj mnie”


def is_login_url(url: str) -> bool:
    """Strony logowania Amazona: /ap/signin, /ap/mfa (OTP), /ap/cvf (weryfikacja),
    /ax/claim (nowy formularz e-maila)."""
    u = urlparse(url)
    return (u.hostname or "").endswith(DOMAIN) and u.path.startswith(("/ap/", "/ax/"))


def is_captcha_url(url: str) -> bool:
    return "validateCaptcha" in url or "/errors/" in urlparse(url).path


def is_orders_url(url: str) -> bool:
    u = urlparse(url)
    return (u.hostname or "").endswith(DOMAIN) and u.path.startswith(
        ("/your-orders", "/gp/css/order-history", "/gp/your-account/order-history"))


def fill_login_page(page: Page, user: str, password: str, done: set[str]) -> list[str]:
    """Wypełnia to, co jest na bieżącej stronie logowania: e-mail i/lub hasło
    (Amazon pokazuje je zwykle na osobnych stronach). `done` - co już wpisano
    na poprzednich stronach. Zwraca listę wpisanych teraz pól."""
    filled = []
    page.wait_for_timeout(1000)  # formularz musi się w pełni załadować
    email_field = common.visible_field(page, EMAIL_SELECTORS)
    if email_field is not None and "email" not in done:
        if common.type_checked(page, email_field, user):
            filled.append("email")
    password_field = common.visible_field(page, PASSWORD_SELECTORS)
    if password_field is not None and "password" not in done:
        if common.type_checked(page, password_field, password):
            filled.append("password")
            try:  # dłuższa sesja = rzadsze logowanie
                box = page.locator(REMEMBER_ME).first
                if box.count() and box.is_visible() and not box.is_checked():
                    box.check(timeout=2000)
            except Exception:
                pass
    return filled


def fill_credentials(user: str, password: str, done: set[str]) -> list[str]:
    """Podłącza się na chwilę, wypełnia stronę logowania i się rozłącza."""
    pw = sync_playwright().start()
    try:
        browser = pw.chromium.connect_over_cdp(CDP_URL)
        return fill_login_page(browser.contexts[0].pages[0], user, password, done)
    except Exception:
        return []
    finally:
        pw.stop()  # rozłącza, przeglądarka działa dalej


FILLED_MSG = {"email": "  wpisano e-mail - kliknij 'Dalej'",
              "password": "  wpisano hasło - kliknij 'Zaloguj się'"}


def login(creds: common.LazyCredentials | None) -> None:
    deadline = time.time() + LOGIN_TIMEOUT_S
    done: set[str] = set()
    last_url = ""
    announced = False
    ok_streak = 0
    time.sleep(2)  # czas na ewentualne przekierowanie do logowania
    while time.time() < deadline:
        tab = common.current_tab(CDP_PORT)
        if tab is None:
            sys.exit("Przeglądarka została zamknięta.")
        url, title = tab.get("url", ""), tab.get("title", "")
        if is_orders_url(url) and title:
            ok_streak += 1
            if ok_streak >= 2:
                print("Zalogowano." if announced else "Sesja aktywna - logowanie pominięte.")
                return
        else:
            ok_streak = 0
        if is_login_url(url) or is_captcha_url(url):
            if not announced:
                print("Zaloguj się w oknie przeglądarki (skrypt wpisze e-mail i hasło, "
                      "Ty klikasz 'Dalej' / 'Zaloguj się', captchę i kod OTP).")
                announced = True
            if url != last_url and "password" in done:
                done.discard("password")  # np. ponowne pytanie o hasło po captchy
            last_url = url
            if creds is not None and len(done) < 2:
                user, password = creds.get()
                for f in fill_credentials(user, password, done):
                    done.add(f)
                    print(FILLED_MSG[f])
        time.sleep(1.5)
    sys.exit("Nie udało się zalogować w wyznaczonym czasie.")


def wait_if_interrupted(page: Page, creds: common.LazyCredentials | None) -> None:
    """W trakcie przeglądania Amazon potrafi poprosić o captchę albo ponownie
    o hasło (np. przy starszych zamówieniach). Wtedy czekamy, aż użytkownik to
    obsłuży, i wypełniamy hasło, jeśli je znamy."""
    if not (is_login_url(page.url) or is_captcha_url(page.url)):
        return
    print("  Amazon prosi o zalogowanie / captchę - dokończ w oknie przeglądarki.")
    done: set[str] = set()
    deadline = time.time() + LOGIN_TIMEOUT_S
    while time.time() < deadline:
        if not (is_login_url(page.url) or is_captcha_url(page.url)):
            page.wait_for_timeout(2000)
            return
        if creds is not None and is_login_url(page.url) and len(done) < 2:
            user, password = creds.get()
            for f in fill_login_page(page, user, password, done):
                done.add(f)
                print(FILLED_MSG[f])
        page.wait_for_timeout(1500)
    sys.exit("Nie udało się zalogować w wyznaczonym czasie.")


# --------------------------------------------------------------------------- #
# Zapis stron jako uproszczony HTML
# --------------------------------------------------------------------------- #
# Ze strony zostaje tylko główny kontener (bez nagłówka, stopki, rekomendacji),
# bez skryptów, stylów i obrazków (z obrazków zostaje `alt`, czyli często nazwa
# produktu) i tylko z atrybutami potrzebnymi do parsowania. Z linków znika
# zapytanie (tokeny sesji, śledzenie), zostaje tylko orderID. Dzięki temu pliki
# są małe i zmieniają się tylko wtedy, gdy zmieni się treść.
SIMPLIFY_JS = r"""
(roots) => {
  let root = null;
  for (const sel of roots) { root = document.querySelector(sel); if (root) break; }
  root = (root || document.body).cloneNode(true);
  root.querySelectorAll(
    'script,style,noscript,svg,iframe,link,meta,template,input[type=hidden],' +
    '#navbar,#nav-main,header,footer,#navFooter,#rhf,' +
    '[data-component=recommendations],[data-component=personalization],' +
    '[data-component=debugBanner],[data-component=aapiDebug],.p13n-sc-desktop-section'
  ).forEach(e => e.remove());
  // komentarze HTML (wstrzykiwane widżety) - zmieniają się przy każdym wczytaniu
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_COMMENT);
  const comments = [];
  while (walker.nextNode()) comments.push(walker.currentNode);
  comments.forEach(c => c.remove());
  const keep = new Set(['class', 'id', 'href', 'alt', 'data-component', 'aria-label',
                        'value', 'name', 'selected', 'dir']);
  for (const el of [root, ...root.querySelectorAll('*')]) {
    for (const a of [...el.attributes]) {
      if (!keep.has(a.name)) { el.removeAttribute(a.name); continue; }
      // id z liczbami (a-autoid-8, anonCarousel1), klasy śledzące (pf_rd_r-…)
      // i generowane klasy stylów (css-1jxf684, r-150rngu)
      // są losowe albo zależne od kolejności wczytania - bez nich plik jest stabilny
      if (a.name === 'id' && /\d/.test(a.value)) { el.removeAttribute('id'); continue; }
      if (a.name === 'class') {
        const cls = a.value.split(/\s+/).filter(c => c && !/^(pd_rd|pf_rd|content-id|c-f|css-|r-)/.test(c));
        el.setAttribute('class', cls.join(' '));
      }
      if (a.name === 'href') {
        try {
          const u = new URL(a.value, location.href);
          const id = u.searchParams.get('orderID') || u.searchParams.get('orderId');
          el.setAttribute('href', u.pathname.replace(/\/ref=[^/]*$/, '') +
                                  (id ? '?orderID=' + id : ''));
        } catch (e) { el.removeAttribute('href'); }
      }
    }
  }
  return root.outerHTML;
}
"""
LIST_ROOTS = ["#yourOrdersContent", "#ordersContainer", ".your-orders-content-container",
              "#yourOrders", "#a-page"]
DETAILS_ROOTS = ["#orderDetails", "[data-component=orderDetails]", "#a-page"]


def simplified_html(page: Page, roots: list[str]) -> str:
    html = page.evaluate(SIMPLIFY_JS, roots)
    return re.sub(r">\s+<", "><", re.sub(r"\s+", " ", html)).strip()


# --------------------------------------------------------------------------- #
# Parsowanie (bez przeglądarki: BeautifulSoup na zapisanym HTML)
# --------------------------------------------------------------------------- #
ORDER_ID_RE = re.compile(r"\b(\d{3}-\d{7}-\d{7}|D\d{2}-\d{7}-\d{7})\b")
ASIN_RE = re.compile(r"/(?:dp|gp/product|gp/aw/d)/([A-Z0-9]{10})")

# Miesiące: polski (dopełniacz i mianownik), angielski, niemiecki - po 3 literach.
MONTHS = {"sty": 1, "lut": 2, "mar": 3, "kwi": 4, "maj": 5, "cze": 6, "lip": 7,
          "sie": 8, "wrz": 9, "paź": 10, "paz": 10, "lis": 11, "gru": 12,
          "jan": 1, "feb": 2, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
          "sep": 9, "oct": 10, "nov": 11, "dec": 12,
          "mär": 3, "mai": 5, "okt": 10, "dez": 12}
DATE_RES = [
    re.compile(r"\b(\d{1,2})\.?\s+([^\W\d_]{3,})\.?\s+(\d{4})\b"),   # 20 listopada 2025
    re.compile(r"\b([^\W\d_]{3,})\.?\s+(\d{1,2}),?\s+(\d{4})\b"),   # November 20, 2025
    re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b"),               # 20.11.2025
]


def parse_date_text(text: str) -> str | None:
    """Pierwsza data z rokiem w tekście -> 'RRRR-MM-DD'."""
    best = None
    for n, rx in enumerate(DATE_RES):
        for m in rx.finditer(text):
            a, b, y = m.groups()
            if n == 0:
                day, month = int(a), MONTHS.get(b.lower()[:3])
            elif n == 1:
                day, month = int(b), MONTHS.get(a.lower()[:3])
            else:
                day, month = int(a), int(b)
            if month and 1 <= month <= 12 and 1 <= day <= 31 and 1995 <= int(y) <= 2100:
                if best is None or m.start() < best[0]:
                    best = (m.start(), f"{int(y):04d}-{month:02d}-{day:02d}")
                break
    return best[1] if best else None


CURRENCIES = {"zł": "PLN", "PLN": "PLN", "€": "EUR", "EUR": "EUR", "$": "USD",
              "USD": "USD", "£": "GBP", "GBP": "GBP", "SEK": "SEK", "kr": "SEK"}
_CUR = r"(zł|PLN|€|EUR|\$|USD|£|GBP|SEK|kr)"
MONEY_RE = re.compile(
    rf"(?:{_CUR}\s?)?(-?\d{{1,3}}(?:[ .,  ]\d{{3}})*[.,]\d{{2}}|-?\d+[.,]\d{{2}})"
    rf"(?:\s?{_CUR})?")


def parse_money(text: str) -> tuple[float, str] | None:
    """Pierwsza kwota z walutą w tekście, np. '1 234,56 zł' -> (1234.56, 'PLN')."""
    for m in MONEY_RE.finditer(text or ""):
        cur = m.group(1) or m.group(3)
        if not cur:
            continue
        num = m.group(2)
        num = re.sub(r"[ .,  ](?=\d{3}(?:\D|$))", "", num[:-3]) + "." + num[-2:]
        return float(num), CURRENCIES[cur]
    return None


def _soup(html: str):
    from bs4 import BeautifulSoup
    return BeautifulSoup(html, "html.parser")


def _text(el) -> str:
    return " ".join(el.get_text(" ").split()) if el is not None else ""


def _lines(el) -> list[str]:
    return [" ".join(s.split()) for s in el.get_text("\n").split("\n") if s.strip()]


def _outermost(elements: list) -> list:
    """Bez elementów zagnieżdżonych w innych z tej samej listy."""
    ids = {id(e) for e in elements}
    return [e for e in elements if not any(id(p) in ids for p in e.parents)]


def product_links(el) -> dict[str, dict]:
    """ASIN -> {title, url} z linków do produktów (link z obrazka bywa bez tekstu,
    wtedy nazwa z `alt` obrazka)."""
    out: dict[str, dict] = {}
    for a in el.select("a[href]"):
        m = ASIN_RE.search(a["href"])
        if not m:
            continue
        title = _text(a) or next((i.get("alt", "") for i in a.select("img[alt]")), "")
        e = out.setdefault(m.group(1), {"title": "", "url": f"{base_url()}/dp/{m.group(1)}"})
        if len(title) > len(e["title"]):
            e["title"] = title
    return out


# Statusy przesyłek / pozycji. „Zwrot możliwy do …” / „Return window closed” to
# tylko informacja o terminie, nie zwrot - dlatego wzorce są konkretne.
RETURNED_RE = re.compile(
    r"zwrot (zakończ|rozpocz|przyj|w toku|otrzym|zrealiz)|zwrócon|zwrot pieniędzy (zosta|zrealiz|wysła)"
    r"|return (complete|started|received|in transit|requested)|\breturned\b"
    r"|refund(ed| issued| complete)|rückgabe (abgeschlossen|gestartet|eingeleitet)"
    r"|erstattet|zurückgesendet|rücksendung (abgeschlossen|eingegangen)", re.I)
CANCELLED_RE = re.compile(r"anulowan|\bcancel+ed\b|storniert", re.I)


def classify(text: str) -> str:
    if CANCELLED_RE.search(text):
        return "cancelled"
    if RETURNED_RE.search(text):
        return "returned"
    return ""


def parse_list_page(html: str) -> list[dict]:
    """Zamówienia z jednej strony listy: numer, data, suma, statusy przesyłek
    i produkty (bez cen - te są dopiero w szczegółach)."""
    soup = _soup(html)
    cards = _outermost(soup.select(".order-card, .js-order-card")) or \
        _outermost([d for d in soup.select(".order") if d.select_one(".order-info, .order-header")])
    orders = []
    for card in cards:
        header = card.select_one(".order-header, .order-info") or card
        m = ORDER_ID_RE.search(_text(header)) or ORDER_ID_RE.search(_text(card))
        if not m:
            continue
        money = parse_money(_text(header))
        statuses = [_text(s) for s in card.select(
            ".delivery-box__primary-text, .yohtmlc-shipment-status-primaryText, "
            ".shipment-top-row .a-size-medium, .js-shipment-info-container .a-row:first-child")]
        detail_link = next((a["href"] for a in card.select("a[href]")
                            if "orderID=" in a["href"]), None)
        orders.append({
            "orderId": m.group(1),
            "date": parse_date_text(_text(header)),
            "total": money[0] if money else None,
            "currency": money[1] if money else None,
            "statuses": [s for s in dict.fromkeys(statuses) if s],
            "cancelled": bool(CANCELLED_RE.search(" ".join(statuses))),
            "items": [{"asin": a} | v for a, v in product_links(card).items()],
            "detailsPath": detail_link,
        })
    return orders


def parse_time_filters(html: str) -> list[str]:
    """Wartości filtra czasu z listy zamówień, np. ['year-2026', 'year-2025', 'archived']."""
    soup = _soup(html)
    values = [o.get("value", "") for o in soup.select("select option")]
    years = sorted({v for v in values if re.fullmatch(r"year-\d{4}", v)}, reverse=True)
    return years + [v for v in values if v == "archived"]


def _is_shipment_box(el) -> bool:
    classes = el.get("class") or []
    return "a-box" in classes or "shipment" in classes or \
        el.get("data-component") in ("shipments", "shipment")


def _item_container(link, asin: str):
    """Najszerszy przodek linku do produktu, który zawiera tylko ten jeden produkt
    (ale nie wychodzi poza pudełko przesyłki)."""
    node = link
    while node.parent is not None and node.parent.name != "[document]" \
            and not _is_shipment_box(node.parent):
        asins = {m.group(1) for a in node.parent.select("a[href]")
                 if (m := ASIN_RE.search(a["href"]))}
        if asins - {asin}:
            break
        node = node.parent
    return node


def _shipment_box(el):
    """Najbliższe pudełko przesyłki (.a-box) zawierające pozycję."""
    return next((p for p in el.parents if p.name and _is_shipment_box(p)), None)


def _shipment_status(box) -> str:
    if box is None:
        return ""
    el = box.select_one("[data-component=shipmentStatus], .od-status-message, "
                        ".yohtmlc-shipment-status-primaryText, .delivery-box__primary-text, "
                        ".shipment-top-row .a-size-medium")
    return _text(el) if el is not None else next(iter(_lines(box)), "")


def _quantity(container) -> int:
    for el in container.select("[class*=qty], [class*=quantity], [data-component=quantity]"):
        m = re.search(r"\d+", _text(el))
        if m:
            return int(m.group())
    m = re.search(r"(?:ilość|quantity|menge|qty)\s*:?\s*(\d+)|\b(\d+)\s*(?:of|z|x|×)\s*:",
                  _text(container), re.I)
    return int(m.group(1) or m.group(2)) if m else 1


# Wiersze „Podsumowanie zamówienia” (etykieta -> kwota), np. na amazon.pl:
#   Produkty: 88,73 zł | Pakowanie i wysyłka: 10,56 zł | Suma bez VAT: 99,29 zł |
#   VAT: 22,84 zł | Suma: 122,13 zł | Darmowa dostawa: -12,99 zł | Łączna kwota: 109,14 zł
# Gdy jest wiersz „Suma bez VAT”, produkty i wysyłka są podane netto, a ceny pozycji
# i rabaty brutto.
SUMMARY_KEYS = {
    "shipping": re.compile(r"^(pakowanie i wysyłka|wysyłka|dostawa|koszt wysyłki|"
                           r"shipping|postage|versand)", re.I),
    "net": re.compile(r"^(suma bez vat|total before tax|summe ohne mwst)", re.I),
    "gross": re.compile(r"^(suma|total|summe)$", re.I),
    "total": re.compile(r"^(łączna kwota|suma końcowa|grand total|order total|"
                        r"gesamtsumme|gesamtbetrag)", re.I),
    "refund": re.compile(r"(zwrot|refund|erstattung)", re.I),
}
DELIVERY_DISCOUNT_RE = re.compile(r"dostaw|wysył|shipping|postage|versand", re.I)


def parse_summary(soup) -> dict:
    """Wiersze podsumowania zamówienia -> {etykieta: kwota} oraz wyliczone pola:
    delivery_cost (zapłacona dostawa brutto, po rabatach na dostawę), discounts
    (pozostałe rabaty, ujemne), total, refund_total."""
    rows: dict[str, float] = {}
    for row in soup.select(".od-line-item-row"):
        label = _text(row.select_one(".od-line-item-row-label")).rstrip(":").strip()
        money = parse_money(_text(row.select_one(".od-line-item-row-content")))
        if label and money:
            rows[label] = money[0]
    if not rows:  # stary układ: etykieta i kwota w kolejnych wierszach tekstu
        lines = _lines(soup.select_one("[data-component=chargeSummary], #od-subtotals") or soup)
        for i, line in enumerate(lines[:-1]):
            if line.endswith(":") and (money := parse_money(lines[i + 1])):
                rows[line.rstrip(":")] = money[0]

    def find(key: str) -> list[tuple[str, float]]:
        return [(k, v) for k, v in rows.items() if SUMMARY_KEYS[key].search(k)]
    shipping = sum(v for _, v in find("shipping"))
    net, gross = find("net"), find("gross")
    if shipping and net and gross and net[0][1]:
        shipping = round(shipping * gross[0][1] / net[0][1], 2)  # netto -> brutto
    negative = [(k, v) for k, v in rows.items() if v < 0]
    delivery_discount = sum(v for k, v in negative if DELIVERY_DISCOUNT_RE.search(k))
    total = find("total")
    refund = [v for k, v in rows.items() if SUMMARY_KEYS["refund"].search(k) and v > 0]
    return {
        "summary": rows,
        "delivery_cost": round(shipping + delivery_discount, 2) if rows else None,
        "discounts": round(sum(v for k, v in negative
                               if not DELIVERY_DISCOUNT_RE.search(k)), 2),
        "total": total[0][1] if total else None,
        "refund_total": sum(refund) if refund else None,
    }


def _component(el, name: str) -> str:
    return _text(el.select_one(f"[data-component={name}]"))


def parse_details(html: str) -> dict:
    """Szczegóły zamówienia: pozycje z ceną, ilością i sprzedawcą, status przesyłki
    każdej pozycji (zwrot / anulowanie), koszt dostawy, rabaty, suma, suma zwrotu."""
    soup = _soup(html)
    items = []
    # amazon.pl (2026): pozycja = [data-component=purchasedItemsRightGrid] z polami
    # itemTitle, unitPrice, quantity, orderedMerchant, itemReturnEligibility;
    # przesyłka = .a-box w [data-component=shipments] ze statusem shipmentStatus.
    for box in soup.select("[data-component=purchasedItemsRightGrid]"):
        title_el = box.select_one("[data-component=itemTitle]")
        link = title_el.select_one("a[href]") if title_el else None
        m = ASIN_RE.search(link["href"]) if link else None
        ship = _shipment_box(box)
        ship_status = _component(ship, "shipmentStatus") if ship else ""
        money = parse_money(_component(box, "unitPrice"))
        qty = re.search(r"\d+", _component(box, "quantity"))
        items.append({
            "asin": m.group(1) if m else "", "title": _text(title_el),
            "url": f"{base_url()}/dp/{m.group(1)}" if m else "",
            "seller": re.sub(r"^\w+:\s*", "", _component(box, "orderedMerchant")),
            "quantity": int(qty.group()) if qty else 1,
            "unit_price": money[0] if money else None,
            "currency": money[1] if money else None,
            "shipment_status": ship_status,
            "return_info": _component(box, "itemReturnEligibility"),
            "state": classify(ship_status) or classify(_text(box)),  # "", returned, cancelled
        })
    if not items:  # inny układ strony: każdy link do produktu poza rekomendacjami
        for asin, info in product_links(soup).items():
            link = next(a for a in soup.select("a[href]") if asin in a["href"])
            box = _item_container(link, asin)
            ship_status = _shipment_status(_shipment_box(box))
            money = parse_money(_text(box))
            items.append({
                "asin": asin, "title": info["title"], "url": info["url"], "seller": "",
                "quantity": _quantity(box),
                "unit_price": money[0] if money else None,
                "currency": money[1] if money else None,
                "shipment_status": ship_status, "return_info": "",
                "state": classify(_text(box)) or classify(ship_status),
            })
    cancelled_text = " ".join(_text(e) for e in soup.select(
        "[data-component=cancelled], [data-component=orderStatus], .od-status-message"))
    return {"date": parse_date_text(_component(soup, "orderDate") or _text(soup)),
            "items": items, "cancelled": bool(CANCELLED_RE.search(cancelled_text))} \
        | parse_summary(soup)


# --------------------------------------------------------------------------- #
# Przeglądanie listy zamówień
# --------------------------------------------------------------------------- #
def load_list_pages() -> list[dict]:
    pages = []
    for f in sorted(RAW_DIR.glob("pages_*.json")):  # od najstarszego pobrania
        pages += json.loads(f.read_text(encoding="utf-8"))
    return pages


def load_saved_orders() -> dict[str, dict]:
    """Zamówienia ze wszystkich zapisanych stron listy; nowsze pobranie wygrywa."""
    orders: dict[str, dict] = {}
    for p in load_list_pages():
        for o in parse_list_page(p["html"]):
            orders[o["orderId"]] = o
    return orders


def browse_cutoff(days: int) -> str | None:
    """Data ('RRRR-MM-DD'), do której przeglądamy listę: `days` dni przed
    najnowszym zapisanym zakupem. None = wszystkie lata (pierwsze uruchomienie
    albo historia niekompletna)."""
    if not HISTORY_COMPLETE.exists():
        return None
    dates = [o["date"] for o in load_saved_orders().values() if o["date"]]
    if not dates:
        return None
    return (datetime.fromisoformat(max(dates)) - timedelta(days=days)).date().isoformat()


def open_page(page: Page, url: str, creds: common.LazyCredentials | None, ready: str) -> None:
    page.goto(url, wait_until="domcontentloaded")
    if is_login_url(page.url) or is_captcha_url(page.url):
        wait_if_interrupted(page, creds)
        page.goto(url, wait_until="domcontentloaded")  # po logowaniu wróć na stronę
    try:
        page.wait_for_selector(ready, timeout=10000)
    except Exception:
        pass  # np. pusty rok - strona bez zamówień
    page.wait_for_timeout(PAGE_DELAY_MS)


CARD_SELECTOR = ".order-card, .js-order-card, .order, .your-orders-content-container"


def browse_orders(page: Page, creds: common.LazyCredentials | None, cutoff: str | None,
                  saved: list[dict]) -> bool:
    """Przegląda listę zamówień rok po roku, od najnowszych. Każdą stronę zapisuje
    do `saved`. Zwraca True, jeśli przejrzała wszystkie lata (i archiwum). Gdy
    podano `cutoff`, kończy po wczytaniu zamówienia starszego niż ta data -
    nowsze zamówienia są przeglądane ponownie, żeby złapać zwroty i anulowania."""
    open_page(page, orders_url(), creds, CARD_SELECTOR)
    filters = parse_time_filters(simplified_html(page, LIST_ROOTS))
    guessed = not filters
    if guessed:
        print("  nie znalazłem filtra lat na liście zamówień - sprawdzam lata wstecz")
        filters = [f"year-{y}" for y in range(datetime.now().year, 1994, -1)]
    if cutoff:
        filters = [f for f in filters
                   if f.startswith("year-") and f[5:] >= cutoff[:4]]
    empty_years = 0
    for flt in filters:
        start = 0
        while True:
            url = orders_url(flt, start)
            open_page(page, url, creds, CARD_SELECTOR)
            html = simplified_html(page, LIST_ROOTS)
            orders = parse_list_page(html)
            saved.append({"url": url, "filter": flt, "startIndex": start,
                          "fetched": datetime.now().isoformat(timespec="seconds"),
                          "html": html})
            dates = [o["date"] for o in orders if o["date"]]
            if orders or start:
                print(f"  {flt} od {start + 1}: zamówień {len(orders)}"
                      + (f" (najstarsze z {min(dates)})" if dates else ""))
            if cutoff and dates and min(dates) < cutoff:
                print(f"  doszedłem do zakupów sprzed {cutoff} - koniec przeglądania.")
                return False
            if len(orders) < 10:  # Amazon pokazuje po 10 zamówień na stronę
                break
            start += 10
        empty_years = empty_years + 1 if start == 0 and not orders else 0
        if guessed and empty_years >= 5:
            break  # przy zgadywaniu lat: 5 pustych lat z rzędu = koniec historii
    return not cutoff


def save_list_pages(pages: list[dict]) -> None:
    if not pages:
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (RAW_DIR / f"pages_{stamp}.json").write_text(
        json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Szczegóły zamówień (ceny, ilości, zwroty)
# --------------------------------------------------------------------------- #
# Lista zamówień podaje tylko sumę zamówienia i status przesyłek. Ceny pozycji,
# ilości i to, które produkty wróciły (także gdy zwrot dotyczy kilku produktów),
# widać dopiero na stronie szczegółów. Zapisujemy jej uproszczony HTML do
# details/<nr_zamówienia>.json - plik zmienia się tylko, gdy zmieni się treść.
def details_path(order_id: str) -> Path:
    return DETAILS_DIR / f"{re.sub(r'[^\w-]+', '_', order_id)}.json"


def fetch_details(page: Page, creds: common.LazyCredentials | None, orders: dict[str, dict],
                  refresh: set[str]) -> None:
    """Pobiera szczegóły zamówień: brakujące oraz te z `refresh` (zamówienia
    z przeglądanego okresu - zwrot mógł się w tym czasie zmienić)."""
    DETAILS_DIR.mkdir(parents=True, exist_ok=True)
    todo = [o for o in orders.values()
            if o["orderId"] in refresh or not details_path(o["orderId"]).exists()]
    if not todo:
        return
    print(f"Sprawdzam szczegóły {len(todo)} zamówień...")
    for i, o in enumerate(sorted(todo, key=lambda o: o["date"] or "", reverse=True), 1):
        oid = o["orderId"]
        url = (base_url() + o["detailsPath"]) if o.get("detailsPath") else details_url(oid)
        try:
            open_page(page, url, creds, "[data-component=purchasedItems], .yohtmlc-item, "
                                        "#orderDetails, .a-box")
            html = simplified_html(page, DETAILS_ROOTS)
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {oid}: BŁĄD {e.__class__.__name__}")
            continue
        if oid not in html:
            print(f"  [{i}/{len(todo)}] {oid}: strona bez danych zamówienia - pomijam")
            continue
        path = details_path(oid)
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if old.get("html") != html:  # bez zmian treści -> plik bez zmian (czysta historia)
            path.write_text(json.dumps({
                "orderId": oid, "url": url,
                "fetched": datetime.now().isoformat(timespec="seconds"), "html": html,
            }, ensure_ascii=False, indent=1), encoding="utf-8")
        d = parse_details(html)
        n_ret = sum(it["quantity"] for it in d["items"] if it["state"] == "returned")
        print(f"  [{i}/{len(todo)}] {o['date']} {oid}: pozycji {len(d['items'])}"
              + (f", zwróconych sztuk {n_ret}" if n_ret else "")
              + (" (zmiana)" if old and old.get("html") != html else ""))


def load_details() -> dict[str, dict]:
    out = {}
    for f in sorted(DETAILS_DIR.glob("*.json")) if DETAILS_DIR.exists() else []:
        d = json.loads(f.read_text(encoding="utf-8"))
        out[d["orderId"]] = parse_details(d["html"]) | {"fetched": d.get("fetched")}
    return out


# --------------------------------------------------------------------------- #
# Pliki wynikowe
# --------------------------------------------------------------------------- #
def merge_orders() -> list[dict]:
    """Zamówienia z listy uzupełnione o szczegóły (ceny, ilości, zwroty)."""
    listed = load_saved_orders()
    details = load_details()
    orders = []
    for oid, o in listed.items():
        d = details.get(oid)
        items = d["items"] if d and d["items"] else \
            [it | {"seller": "", "quantity": 1, "unit_price": None, "currency": None,
                   "shipment_status": "", "return_info": "", "state": ""}
             for it in o["items"]]
        # anulowane całe zamówienie: wg szczegółów albo (bez szczegółów) gdy
        # wszystkie przesyłki na liście mają status anulowania
        whole_cancelled = d["cancelled"] if d and d["items"] else (
            bool(o["statuses"]) and all(CANCELLED_RE.search(x) for x in o["statuses"]))
        if whole_cancelled:
            for it in items:
                it["state"] = it["state"] or "cancelled"
        states = {it["state"] for it in items}
        status = ("CANCELLED" if states == {"cancelled"} else
                  "RETURNED" if states <= {"returned", "cancelled"} and "returned" in states
                  else "PARTIALLY_RETURNED" if "returned" in states
                  else "PARTIALLY_CANCELLED" if "cancelled" in states else "OK")
        items_sum = sum((it["unit_price"] or 0) * it["quantity"] for it in items)
        orders.append({
            "orderId": oid, "domain": DOMAIN, "date": o["date"] or (d or {}).get("date"),
            "status": status, "statuses": o["statuses"],
            "total": (d or {}).get("total") or o["total"],
            "currency": o["currency"] or next((it["currency"] for it in items
                                               if it["currency"]), None),
            "delivery_cost": (d or {}).get("delivery_cost"),
            "discounts": (d or {}).get("discounts"),
            "refund_total": (d or {}).get("refund_total"),
            "summary": (d or {}).get("summary"),
            "items_sum": round(items_sum, 2),
            "details_fetched": (d or {}).get("fetched"),
            "items": items,
        })
    orders.sort(key=lambda o: (o["date"] or "", o["orderId"]), reverse=True)
    return orders


ITEM_FIELDS = ["order_id", "date", "seller", "order_status", "order_total", "asin",
               "title", "quantity", "unit_price", "currency", "delivery_cost",
               "item_status", "shipment_status", "returned_quantity",
               "order_refund_total", "return_info", "url"]
KEPT_FIELDS = ["order_id", "date", "seller", "asin", "title", "quantity", "unit_price", "total",
               "currency", "url"]


def flatten(orders: list[dict]) -> list[dict]:
    rows = []
    for o in orders:
        for it in o["items"]:
            rows.append({
                "order_id": o["orderId"], "date": o["date"], "seller": it["seller"],
                "order_status": o["status"],
                "order_total": o["total"], "asin": it["asin"], "title": it["title"],
                "quantity": it["quantity"], "unit_price": it["unit_price"],
                "currency": it["currency"] or o["currency"],
                "delivery_cost": o["delivery_cost"],
                "item_status": it["state"], "shipment_status": it["shipment_status"],
                "returned_quantity": it["quantity"] if it["state"] == "returned" else "",
                "order_refund_total": o["refund_total"],
                "return_info": it["return_info"], "url": it["url"],
            })
    return rows


def kept_items(rows: list[dict]) -> list[dict]:
    """Pozycje, które zostały u kupującego: bez anulowanych i zwróconych
    (także zwrotów w toku). Pozycje bez ceny (brak szczegółów) są pomijane."""
    out = []
    for r in rows:
        if r["item_status"] in ("cancelled", "returned") or r["unit_price"] is None:
            continue
        qty = int(r["quantity"] or 0) - int(r["returned_quantity"] or 0)
        if qty <= 0:
            continue
        out.append({k: r.get(k) for k in KEPT_FIELDS}
                   | {"quantity": qty, "total": f"{float(r['unit_price']) * qty:.2f}"})
    return out


def check_consistency(orders: list[dict]) -> None:
    """Suma pozycji + dostawa + rabaty vs. łączna kwota zamówienia oraz zwroty
    pieniędzy bez zwróconej pozycji."""
    bad = []
    for o in orders:
        if o["total"] is None or not o["details_fetched"] or o["status"] == "CANCELLED":
            continue
        o["expected_total"] = round(o["items_sum"] + (o["delivery_cost"] or 0)
                                    + (o["discounts"] or 0), 2)
        if abs(o["expected_total"] - o["total"]) > 0.02:
            bad.append(o)
    ok = sum(1 for o in orders if "expected_total" in o) - len(bad)
    print(f"  spójność: suma pozycji + dostawa + rabaty = łączna kwota w {ok} "
          f"z {ok + len(bad)} zamówień ze szczegółami")
    no_price = sum(1 for o in orders for it in o["items"]
                   if it["unit_price"] is None and it["state"] != "cancelled")
    refund_unassigned = [o for o in orders if o["refund_total"]
                         and o["status"] not in ("RETURNED", "PARTIALLY_RETURNED")]
    if bad:
        print(f"  uwaga: zamówień, w których suma pozycji + dostawa + rabaty ≠ łączna "
              f"kwota: {len(bad)}, np. "
              + ", ".join(f"{o['orderId']} ({o['expected_total']:.2f} vs {o['total']:.2f})"
                          for o in bad[:3]))
    if no_price:
        print(f"  uwaga: pozycji bez ceny (brak szczegółów zamówienia): {no_price}")
    if refund_unassigned:
        print(f"  uwaga: zamówień ze zwrotem pieniędzy, ale bez zwróconej pozycji: "
              f"{len(refund_unassigned)}, np. "
              + ", ".join(o["orderId"] for o in refund_unassigned[:3]))


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
    simple = common.write_simple_csv(kept, OUTPUT_DIR / "zakupy.csv", NBP_CACHE, "amazon")
    common.write_monthly_chart(simple, OUTPUT_DIR / "wydatki_miesiecznie.png", "Amazonie")
    n_ret = sum(1 for r in rows if r["item_status"] == "returned")
    print(f"Zamówień: {len(orders)}, pozycji: {len(rows)} (zwróconych: {n_ret}), "
          f"zatrzymanych pozycji: {len(kept)} -> {OUTPUT_DIR}/")
    if not orders:
        print("Nie rozpoznano żadnych zamówień. Uruchom z --discover i sprawdź "
              f"{DISCOVERY_DIR}/ (Amazon mógł zmienić format strony).")
    return orders


def commit_data(message: str) -> None:
    common.commit_data(DATA_DIR, "amazon", message,
                       {"amazon_raw/details/": "zmienione szczegóły zamówień"})


# --------------------------------------------------------------------------- #
# Tryb diagnostyczny
# --------------------------------------------------------------------------- #
def _snapshot(pg: Page, name: str) -> None:
    """Pełny HTML, uproszczony HTML i zrzut ekranu bieżącej strony do ./discovery."""
    stem = DISCOVERY_DIR / name
    stem.with_suffix(".html").write_text(pg.content(), encoding="utf-8")
    roots = LIST_ROOTS if is_orders_url(pg.url) else DETAILS_ROOTS
    stem.with_suffix(".simple.html").write_text(simplified_html(pg, roots), encoding="utf-8")
    pg.screenshot(path=str(stem.with_suffix(".png")), full_page=True)
    print(f"  zapisano {stem.name}: {pg.url[:100]}")


INTERESTING_RE = re.compile(r"zwr|anul|refund|cancel|return|pieniądz|zastąp|wymian", re.I)


def discover_auto(page: Page, creds: common.LazyCredentials | None) -> list[str]:
    """Sam zapisuje pierwszą stronę listy z każdego roku (i drugą stronę z pierwszego
    roku, który ją ma) oraz szczegóły: wszystkich zamówień, których status wygląda
    na zwrot / anulowanie (maks. 20), i 3 zwykłych z największą liczbą produktów.
    Zwraca adresy zapisanych stron."""
    urls = []
    open_page(page, orders_url(), creds, CARD_SELECTOR)
    filters = parse_time_filters(simplified_html(page, LIST_ROOTS))
    print(f"Filtr lat: {', '.join(filters) or 'brak'}")
    orders: list[dict] = []
    second_page_done = False
    for flt in filters:
        for start in (0, 10):
            if start and second_page_done:
                break
            open_page(page, orders_url(flt, start), creds, CARD_SELECTOR)
            found = parse_list_page(simplified_html(page, LIST_ROOTS))
            if start == 0 and not found:
                break
            _snapshot(page, f"list_{flt}_{start}")
            urls.append(page.url)
            orders += found
            if start:
                second_page_done = True
            if len(found) < 10:
                break
    statuses = Counter(s for o in orders for s in o["statuses"])
    print(f"\nZamówień na zapisanych stronach: {len(orders)}. Statusy przesyłek:")
    for s, n in statuses.most_common():
        print(f"  {n:4d}  {s}")
    odd = [o for o in orders if any(INTERESTING_RE.search(s) for s in o["statuses"])]
    normal = sorted((o for o in orders if o not in odd), key=lambda o: -len(o["items"]))
    picked = odd[:20] + normal[:3]
    print(f"\nZapisuję szczegóły {len(picked)} zamówień "
          f"(ze zwrotem / anulowaniem: {len(odd[:20])})...")
    for o in picked:
        url = (base_url() + o["detailsPath"]) if o.get("detailsPath") \
            else details_url(o["orderId"])
        try:
            open_page(page, url, creds, "#orderDetails, .a-box")
            _snapshot(page, f"details_{o['orderId']}")
            urls.append(page.url)
        except Exception as e:
            print(f"  {o['orderId']}: BŁĄD {e.__class__.__name__}")
    return urls


def discover(context: BrowserContext, page: Page, creds: common.LazyCredentials | None) -> None:
    """Zapisuje strony zamówień do ./discovery: najpierw automatycznie (discover_auto),
    potem każdą stronę, którą sam otworzysz, dopóki nie naciśniesz Enter. Zapisuje
    też odpowiedzi JSON (poza telemetrią)."""
    DISCOVERY_DIR.mkdir(exist_ok=True)
    DISCOVERY_DIR.chmod(0o700)
    responses: list[dict] = []

    def on_response(resp) -> None:
        host = urlparse(resp.url).hostname or ""
        if host.endswith(DOMAIN) and not host.startswith("unagi.") and \
                "json" in resp.headers.get("content-type", ""):
            try:
                responses.append({"url": resp.url, "status": resp.status, "body": resp.json()})
            except Exception:
                pass
    context.on("response", on_response)
    saved = {u: n for n, u in enumerate(discover_auto(page, creds), 1)}
    finished = threading.Event()
    threading.Thread(target=lambda: (input(), finished.set()), daemon=True).start()
    print("\nGotowe. Możesz jeszcze otworzyć w przeglądarce inne strony (np. fakturę albo\n"
          "zamówienie ze zwrotem, którego nie było na liście) - każda zostanie zapisana.\n"
          "Naciśnij Enter w terminalu, żeby zakończyć (maks. 15 min).")
    deadline = time.time() + 900
    while time.time() < deadline and not finished.is_set():
        for pg in [p for p in context.pages if not p.is_closed()]:
            try:
                pg.wait_for_load_state("load", timeout=3000)
                url = pg.url
                if url in saved or not (urlparse(url).hostname or "").endswith(DOMAIN):
                    continue
                pg.wait_for_timeout(1500)
                n = saved[url] = len(saved) + 1
                name = re.sub(r"[^\w]+", "_", urlparse(url).path).strip("_")[:50] or "root"
                _snapshot(pg, f"manual_{n:02d}_{name}")
            except Exception as e:
                print(f"  (nie zapisano strony: {e.__class__.__name__})")
        time.sleep(1)
    (DISCOVERY_DIR / "network_log.json").write_text(
        json.dumps(responses, ensure_ascii=False, indent=1), encoding="utf-8")
    (DISCOVERY_DIR / "urls.json").write_text(json.dumps(saved, indent=1), encoding="utf-8")
    print(f"Zapisano {len(saved)} stron i {len(responses)} odpowiedzi JSON do {DISCOVERY_DIR}/")


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
                        help="zapisz odwiedzane strony zamówień do ./discovery")
    g.add_argument("--rebuild", action="store_true",
                   help="tylko odtwórz pliki wynikowe z ./dane/amazon_raw (bez przeglądarki)")
    parser.add_argument("--days", type=int, default=30,
                        help="przeglądaj zakupy do N dni przed ostatnim zapisanym "
                             "(domyślnie 30), żeby złapać zwroty i anulowania")
    parser.add_argument("--full", action="store_true",
                        help="przejrzyj wszystkie lata i odśwież szczegóły wszystkich zamówień")
    parser.add_argument("--domain", default=DOMAIN,
                        help=f"domena Amazona (domyślnie {DOMAIN})")
    args = parser.parse_args()
    set_domain(args.domain)

    if args.save_credentials:
        return common.save_credentials(KEYRING_SERVICE, "Amazon", "e-mail lub numer telefonu")
    if args.forget_credentials:
        return common.forget_credentials(KEYRING_SERVICE)
    if args.rebuild:
        rebuild_outputs()
        return commit_data(f"przebudowa plików wynikowych {time.strftime('%Y-%m-%d %H:%M')}")

    creds = None if args.manual_login else \
        common.LazyCredentials(KEYRING_SERVICE, "Amazon", ask=args.ask)
    proc = common.start_browser(PROFILE_DIR, CDP_PORT, orders_url())
    try:
        login(creds)
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(CDP_URL)
            context = browser.contexts[0]
            page = context.pages[0]
            if args.discover:
                return discover(context, page, creds)
            known_before = set(load_saved_orders())
            cutoff = None if args.full else browse_cutoff(args.days)
            print("Przeglądam " + (f"zakupy od {cutoff}..." if cutoff
                                   else "wszystkie lata zamówień..."))
            pages: list[dict] = []
            try:
                reached_end = browse_orders(page, creds, cutoff, pages)
            finally:
                save_list_pages(pages)  # zapisz to, co już pobrano, nawet po błędzie
            if reached_end:
                HISTORY_COMPLETE.touch()
            browsed = {o["orderId"] for pg in pages for o in parse_list_page(pg["html"])}
            orders = load_saved_orders()
            fetch_details(page, creds, orders, refresh=browsed)
            orders_out = rebuild_outputs()
            new = len({o["orderId"] for o in orders_out} - known_before)
            commit_data(f"pobranie {time.strftime('%Y-%m-%d %H:%M')}: "
                        f"zamówień {len(orders_out)}, nowych {new}")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
