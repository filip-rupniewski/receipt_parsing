"""
Kod wspólny dla skryptów pobierających zakupy (allegro/, amazon/, aliexpress/):
  * dane logowania: systemowy pęk kluczy albo terminal,
  * Chromium jako zwykły proces z trwałym profilem (CDP) i wpisywanie danych w pola,
  * kursy walut NBP (z cache),
  * zakupy.csv - uproszczona lista zakupów do czytania,
  * wydatki_miesiecznie.png - wykres wydatków w kolejnych miesiącach,
  * commit danych do prywatnego, lokalnego repo .git-private.

Skrypty importują ten moduł po dopisaniu katalogu receipt_downloader/ do sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import common
"""

from __future__ import annotations

import getpass
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

PL_TZ = ZoneInfo("Europe/Warsaw")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIVATE_GIT = PROJECT_ROOT / ".git-private"  # wspólne prywatne repo danych projektu


def parse_date(value) -> datetime | None:
    """Data ISO ('2026-11-16T23:10:38.421Z' albo '2026-11-16'). Data bez strefy
    czasowej jest traktowana jako czas polski."""
    try:
        d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=PL_TZ)


def normalize_name(name: str) -> str:
    """Małe litery i pojedyncze spacje, np. 'ŚWIECA  Zapłonowa ' -> 'świeca zapłonowa'."""
    return " ".join(str(name or "").split()).lower()


def quote(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


# --------------------------------------------------------------------------- #
# Dane logowania
# --------------------------------------------------------------------------- #
# Dane logowania NIE są trzymane w plikach projektu: albo systemowy pęk kluczy
# (GNOME Keyring / KWallet, przez bibliotekę `keyring`), albo terminal (getpass).
KEYRING_USER_KEY = "__login__"  # pod tym kluczem trzymamy sam login


def _keyring():
    try:
        import keyring
        from keyring.backends import fail
    except ImportError:
        return None
    if isinstance(keyring.get_keyring(), fail.Keyring):
        return None  # brak działającego pęku kluczy w systemie
    return keyring


def save_credentials(service: str, site: str, login_hint: str) -> None:
    """Pyta o login i hasło i zapisuje je w pęku kluczy pod usługą `service`
    (np. 'receipt_parsing-amazon'). `site` i `login_hint` są tylko w pytaniu:
    „Login Amazon (e-mail lub numer telefonu): ”."""
    kr = _keyring()
    if kr is None:
        sys.exit("Brak systemowego pęku kluczy (Secret Service). "
                 "Użyj --ask albo --manual-login.")
    login = input(f"Login {site} ({login_hint}): ").strip()
    password = getpass.getpass(f"Hasło {site}: ")
    if not login or not password:
        sys.exit("Login i hasło nie mogą być puste.")
    kr.set_password(service, KEYRING_USER_KEY, login)
    kr.set_password(service, login, password)
    print(f"Zapisano w pęku kluczy (usługa '{service}').")


def forget_credentials(service: str) -> None:
    kr = _keyring()
    if kr is None:
        sys.exit("Brak systemowego pęku kluczy.")
    login = kr.get_password(service, KEYRING_USER_KEY)
    for key in filter(None, (login, KEYRING_USER_KEY)):
        try:
            kr.delete_password(service, key)
        except Exception:
            pass
    print("Usunięto dane logowania z pęku kluczy.")


class LazyCredentials:
    """Pobiera dane dopiero wtedy, gdy są potrzebne (sesja wygasła): z pęku
    kluczy, a gdy go nie ma, jest pusty albo `ask` - z terminala."""

    def __init__(self, service: str, site: str, ask: bool):
        self.service, self.site, self.ask = service, site, ask
        self._value: tuple[str, str] | None = None

    def get(self) -> tuple[str, str]:
        if self._value is None:
            self._value = self._load()
        return self._value

    def _load(self) -> tuple[str, str]:
        if not self.ask:
            kr = _keyring()
            if kr is not None:
                login = kr.get_password(self.service, KEYRING_USER_KEY)
                password = kr.get_password(self.service, login) if login else None
                if login and password:
                    print("  dane logowania: systemowy pęk kluczy")
                    return login, password
        print("  podaj dane logowania (nie zostaną zapisane):")
        login = input(f"  Login {self.site}: ").strip()
        password = getpass.getpass(f"  Hasło {self.site}: ")
        return login, password


# --------------------------------------------------------------------------- #
# Przeglądarka
# --------------------------------------------------------------------------- #
# Sklepy chronią się przed botami, więc Chromium startuje jako zwykły proces
# z trwałym profilem (ciasteczka sesji), a skrypt podłącza się do niego przez CDP
# tylko wtedy, gdy musi. Każdy skrypt ma własny port CDP.
def start_browser(profile_dir: Path, port: int, start_url: str) -> subprocess.Popen:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        exe = p.chromium.executable_path
    if not Path(exe).exists():
        sys.exit(f"Brak Chromium dla tej wersji Playwright ({exe}).\n"
                 "Zainstaluj: uv run playwright install chromium")
    profile_dir.mkdir(exist_ok=True)
    profile_dir.chmod(0o700)  # ciasteczka sesji tylko dla właściciela
    proc = subprocess.Popen(
        [
            exe,
            f"--user-data-dir={profile_dir}",
            f"--remote-debugging-port={port}",
            "--remote-debugging-address=127.0.0.1",
            "--no-sandbox",  # Ubuntu blokuje sandbox Chromium z Playwright
            "--no-first-run",
            "--no-default-browser-check",
            "--lang=pl-PL",
            "--window-size=1280,950",
            start_url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(100):
        if current_tab(port) is not None:
            return proc
        time.sleep(0.2)
    proc.terminate()
    sys.exit("Nie udało się uruchomić Chromium.")


def current_tab(port: int) -> dict | None:
    """Aktywna karta (url, title) przez HTTP DevTools, bez podłączania się.
    None = przeglądarka nie odpowiada (zamknięta)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2) as r:
            tabs = json.load(r)
    except Exception:
        return None
    pages = [t for t in tabs if t.get("type") == "page"]
    return pages[0] if pages else {"url": "", "title": ""}


def visible_field(page, selectors: list[str]):
    """Pierwsze widoczne i edytowalne pole spośród selektorów (strona albo ramka)."""
    for sel in selectors:
        field = page.locator(sel).first
        try:
            if field.count() and field.is_visible() and field.is_editable():
                return field
        except Exception:
            continue
    return None


def type_checked(page, field, value: str) -> bool:
    """Wpisuje wartość i sprawdza, czy została w polu. Formularz potrafi wyczyścić
    pole wpisane przed pełnym załadowaniem, więc w razie potrzeby wpisujemy
    jeszcze raz, znak po znaku."""
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
    return field.input_value() == value


# --------------------------------------------------------------------------- #
# Kursy NBP
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# zakupy.csv
# --------------------------------------------------------------------------- #
SIMPLE_FIELDS = ["nr", "data", "produkt", "liczba", "cena", "cena jednostkowa", "sklep"]


def write_simple_csv(kept: list[dict], out_path: Path, nbp_cache: Path,
                     shop: str) -> list[dict]:
    """zakupy.csv - wersja do czytania: numer zakupu (1 = najstarszy), data,
    produkt, liczba, cena (razem) i cena jednostkowa w PLN, sklep. Najnowsze
    zakupy na górze. `kept` to pozycje z polami order_id, date, title, quantity,
    unit_price, currency (bez anulowanych i zwróconych)."""
    cache = json.loads(nbp_cache.read_text()) if nbp_cache.exists() else {}
    n_cache = len(cache)
    rows = sorted(kept, key=lambda r: (str(r["date"]), r["order_id"]))
    numbers: dict[str, int] = {}
    out = []
    for r in rows:
        nr = numbers.setdefault(r["order_id"], len(numbers) + 1)
        day = parse_date(r["date"])
        local = day.astimezone(PL_TZ) if day else None
        rate = pln_rate(r["currency"], day, cache) if day else None
        unit = float(r["unit_price"] or 0) * rate if rate else None
        out.append({
            "nr": nr,
            "data": local.date().isoformat() if local else "",
            "produkt": normalize_name(r["title"]),
            "liczba": r["quantity"],
            "cena": f"{unit * int(r['quantity']):.2f}" if unit is not None else "",
            "cena jednostkowa": f"{unit:.2f}" if unit is not None else "",
            "sklep": shop,
        })
    out.reverse()  # najnowszy zakup na górze (numeracja nadal od najstarszego)
    if len(cache) != n_cache:
        nbp_cache.parent.mkdir(parents=True, exist_ok=True)
        nbp_cache.write_text(json.dumps(cache, indent=1))
    # Nazwa produktu zawsze w cudzysłowie (bywają w niej przecinki), pozostałe
    # pola (liczby, daty, nazwa sklepu) nigdy - stąd zapis ręczny zamiast csv.writer.
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        f.write(",".join(SIMPLE_FIELDS) + "\n")
        for row in out:
            f.write(",".join(quote(row[k]) if k == "produkt" else str(row[k])
                             for k in SIMPLE_FIELDS) + "\n")
    return out


# --------------------------------------------------------------------------- #
# Wykres wydatków w kolejnych miesiącach
# --------------------------------------------------------------------------- #
MONTHS_PL = ["sty", "lut", "mar", "kwi", "maj", "cze",
             "lip", "sie", "wrz", "paź", "lis", "gru"]

TOP_N = 3            # ile najdroższych pozycji wyróżnić w kolumnie
TOP_MIN_SHARE = 0.10  # ...tylko w kolumnach >= 10% najwyższej
LIST_MIN_PRICE = 100  # w legendzie także wszystkie pozycje droższe niż tyle zł


def write_monthly_chart(rows: list[dict], out_path: Path, shop_name: str,
                        title: str | None = None) -> None:
    """Wykres kolumnowy wydatków w kolejnych miesiącach z wierszy zakupy.csv
    (z uwzględnieniem zwrotów, bez anulowanych zamówień i dostawy). W kolumnach
    >= 10% najwyższej trzy najdroższe pozycje (cena = liczba × cena jedn.) są
    wyróżnione odcieniami niebieskiego, a ich nazwy i ceny są w legendzie pod
    wykresem. `shop_name` trafia do tytułu: „Wydatki na <shop_name> w kolejnych
    miesiącach”, chyba że podano własny `title`."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("  (brak matplotlib - pomijam wykres; uv sync)")
        return
    by_month: dict[str, list[tuple[float, str]]] = {}
    for r in rows:
        if r["data"] and r["cena"]:
            by_month.setdefault(r["data"][:7], []).append((float(r["cena"]), r["produkt"]))
    if not by_month:
        return
    # wszystkie miesiące od pierwszego do ostatniego, także bez zakupów (0 zł)
    y, m = map(int, min(by_month).split("-"))
    months = []
    while f"{y}-{m:02d}" <= max(by_month):
        months.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    values = [sum(p for p, _ in by_month.get(k, [])) for k in months]
    total, mean, peak = sum(values), sum(values) / len(values), max(values)

    # (miesiąc -> lista [(cena, nazwa)] wyróżnionych, od najdroższej)
    top: dict[int, list[tuple[float, str]]] = {}
    for i, k in enumerate(months):
        if values[i] and values[i] >= TOP_MIN_SHARE * peak:
            top[i] = sorted(by_month[k], key=lambda t: -t[0])[:TOP_N]

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    rank_colors = ["#184f95", "#2a78d6", "#86b6ef"]  # 1., 2., 3. najdroższa (rampa niebieska)
    rest_color = "#c9c8c2"                           # pozostałe zakupy
    def zl(v: float) -> str:
        return f"{v:,.0f} zł".replace(",", " ")  # spacja tysięcy, bez łamania
    def month_name(k: str) -> str:
        return f"{MONTHS_PL[int(k[5:]) - 1]} {k[:4]}"

    # Legenda z nazwami: w każdym miesiącu wyróżnione pozycje (kolor rangi)
    # oraz wszystkie pozostałe droższe niż LIST_MIN_PRICE (szary kwadrat).
    listed: dict[int, list[tuple[float, str, str]]] = {}
    for i, k in enumerate(months):
        ranked = sorted(by_month.get(k, []), key=lambda t: -t[0])
        n_top = len(top.get(i, []))
        entries = [(p, name, rank_colors[r] if r < n_top else rest_color)
                   for r, (p, name) in enumerate(ranked)
                   if r < n_top or p > LIST_MIN_PRICE]
        if entries:
            listed[i] = entries

    # Układ w calach od góry: tytuł, wykres, klucz kolorów, legenda z nazwami
    # (blok na każdy miesiąc z listy, najnowsze pierwsze, w dwóch kolumnach).
    blocks = sorted(listed, reverse=True)
    sizes = [len(listed[i]) + 1 for i in blocks]
    split = min(range(len(blocks) + 1),  # podział na 2 kolumny o najbliższej wysokości
                key=lambda k: max(sum(sizes[:k]), sum(sizes[k:])))
    key_lines = max(sum(sizes[:split]), sum(sizes[split:]), 0)
    W, LINE = 11.0, 0.2
    AX_TOP, AX_H = 0.95, 3.9
    KEY_ROW = AX_TOP + AX_H + 0.62
    LIST_TOP = KEY_ROW + 0.62
    H = LIST_TOP + key_lines * LINE + 0.2
    def Y(inch: float) -> float:  # cale od góry -> współrzędna figury
        return 1 - inch / H
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(W, H), dpi=150)
    fig.patch.set_facecolor(surface)
    ax = fig.add_axes((0.075, Y(AX_TOP + AX_H), 0.85, AX_H / H))
    ax.set_facecolor(surface)

    width = 0.62
    for i, k in enumerate(months):
        if not values[i]:
            continue
        bottom = 0.0
        segments = [(p, rank_colors[r]) for r, (p, _) in enumerate(top.get(i, []))]
        rest = values[i] - sum(p for p, _ in segments)
        if rest > 0.005:
            segments.append((rest, rest_color))
        for p, color in segments:  # najdroższa przy osi, pozostałe na górze
            ax.add_patch(Rectangle((i - width / 2, bottom), width, p, facecolor=color,
                                   edgecolor=surface, linewidth=1.2, zorder=2))
            bottom += p

    fig.text(0.012, Y(0.18), title or f"Wydatki na {shop_name} w kolejnych miesiącach",
             color=ink, fontsize=13, ha="left", va="top")
    fig.text(0.012, Y(0.47), f"razem {zl(total)} · średnio {zl(mean)} / miesiąc · "
             "z uwzględnieniem zwrotów, bez anulowanych zamówień i dostawy",
             color=ink2, fontsize=9.5, ha="left", va="top")
    # podpisy tylko przy największym i ostatnim miesiącu (reszta: oś Y)
    hi = max(range(len(values)), key=values.__getitem__)
    for i in {hi, len(values) - 1}:
        if values[i] > 0:
            ax.annotate(zl(values[i]), (i, values[i]), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        color=ink, fontsize=9)
    labels = [f"{MONTHS_PL[int(k[5:]) - 1]}\n{k[:4]}" if k[5:] == "01" or i == 0
              else MONTHS_PL[int(k[5:]) - 1] for i, k in enumerate(months)]
    step = 1 if len(months) <= 18 else 2
    ticks = [i for i in range(len(months)) if i % step == 0 or months[i][5:] == "01"]
    if len(months) > 48:  # wiele lat: tylko styczeń każdego roku, podpisany rokiem
        ticks = [i for i, k in enumerate(months) if k[5:] == "01"]
        labels = [k[:4] for k in months]
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], color=ink2, fontsize=8.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: zl(v)))
    ax.tick_params(axis="y", colors=ink2, labelsize=8.5, length=0,
                   labelleft=True, labelright=True)  # kwoty po obu stronach
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(ink2)
    ax.set_xlim(-0.6, len(months) - 0.1)  # miejsce na podpis ostatniej kolumny
    ax.set_ylim(0, peak * 1.12)

    # klucz kolorów
    items = [(c, f"{n}. najdroższa pozycja") for n, c in enumerate(rank_colors, 1)]
    items.append((rest_color, "pozostałe"))
    for (color, text), kx in zip(items, (0.012, 0.19, 0.368, 0.546)):
        fig.patches.append(Rectangle((kx, Y(KEY_ROW + 0.135)), 0.011, 0.12 / H,
                                     facecolor=color, transform=fig.transFigure, figure=fig))
        fig.text(kx + 0.017, Y(KEY_ROW), text, color=ink2, fontsize=8.5, va="top")
    fig.text(0.012, Y(KEY_ROW + 0.24), f"kolorem: {TOP_N} najdroższe pozycje w miesiącach "
             f"≥ {TOP_MIN_SHARE:.0%} najwyższego · lista niżej: te pozycje oraz "
             f"wszystkie pozostałe droższe niż {LIST_MIN_PRICE} zł",
             color=ink2, fontsize=8.5, va="top")

    # nazwy i ceny wyróżnionych pozycji
    for col, chunk in enumerate((blocks[:split], blocks[split:])):
        x0, yy = (0.012, 0.507)[col], LIST_TOP
        for i in chunk:
            fig.text(x0, Y(yy), f"{month_name(months[i])} · razem {zl(values[i])}",
                     color=ink, fontsize=8.5, va="top", weight="bold")
            yy += LINE
            for p, name, color in listed[i]:
                fig.patches.append(Rectangle((x0, Y(yy + 0.125)), 0.009, 0.1 / H,
                                             facecolor=color,
                                             transform=fig.transFigure, figure=fig))
                short = name if len(name) <= 52 else name[:51].rstrip() + "…"
                fig.text(x0 + 0.014, Y(yy), short, color=ink, fontsize=8.2, va="top")
                fig.text(x0 + 0.475, Y(yy), zl(p), color=ink2, fontsize=8.2,
                         va="top", ha="right")
                yy += LINE

    # bez metadanych, żeby ten sam wykres dawał ten sam plik (czysta historia w git)
    fig.savefig(out_path, facecolor=surface, metadata={"Software": None})
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Historia zmian danych: prywatne repo .git-private w katalogu głównym projektu
# --------------------------------------------------------------------------- #
def commit_data(data_dir: Path, prefix: str, message: str,
                counted: dict[str, str] | None = None) -> None:
    """Commituje zmiany w `data_dir` do prywatnego, lokalnego repo .git-private
    w katalogu głównym projektu (bez remote - dane nie opuszczają komputera).
    `add -f`, bo .gitignore repozytorium z kodem ignoruje katalogi z danymi.
    `counted` = {fragment ścieżki: opis}: do opisu commitu dopisywana jest liczba
    zmienionych plików, których ścieżka zawiera ten fragment."""
    if not PRIVATE_GIT.exists():
        return
    git = ["git", f"--git-dir={PRIVATE_GIT}", f"--work-tree={PROJECT_ROOT}"]
    path = str(data_dir.relative_to(PROJECT_ROOT))
    try:
        subprocess.run(git + ["add", "-f", "-A", "--", path], check=True, cwd=PROJECT_ROOT)
        status = subprocess.run(git + ["diff", "--cached", "--name-status", "--", path],
                                check=True, capture_output=True, text=True,
                                cwd=PROJECT_ROOT).stdout.splitlines()
        if not status:
            print("Dane bez zmian - nic do zapisania w historii.")
            return
        for part, label in (counted or {}).items():
            n = sum(1 for line in status if part in line)
            if n:
                message += f", {label}: {n}"
        # commit tylko danych tego sklepu (reszta danych projektu zostaje bez zmian)
        subprocess.run(git + ["commit", "-q", "-m", f"{prefix}: {message}", "--", path],
                       check=True, cwd=PROJECT_ROOT)
        print(f"Zapisano w historii danych (.git-private): {prefix}: {message}")
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"Nie udało się zapisać historii danych: {e}")
