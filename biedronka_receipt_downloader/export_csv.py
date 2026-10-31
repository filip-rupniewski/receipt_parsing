"""
Eksportuje skonwertowane paragony Biedronki (biedronka_receipts_parsed/all_receipts.json)
do CSV w formacie zbliżonym do reszty projektu (separator ';', przecinek dziesiętny,
data dd.mm.rrrr):

    biedronka_receipts_parsed/biedronka_receipts.csv

Kolumny główne:
    data,
    nazwa              - czytelna nazwa z biedronka_product_names.csv (lub nazwa z kasy),
    procent/sztuk/kg   - ilość w jednostce z kolumny "jednostka":
                         produkty ważone -> kg z kasy,
                         wielkość z biedronka_package_sizes.csv (jeśli wpisana),
                         produkty z wagą/objętością w nazwie (np. "Jogurt 400g") ->
                         sztuki * rozmiar (kg lub l),
                         pozostałe -> liczba sztuk,
    cena               - zapłacona kwota za pozycję (po rabacie),
    za kg/l/sztukę     - cena / procent/sztuk/kg,
    sklep,
    W tym rabat        - kwota rabatu zawarta w cenie (zł),
    rabat w %          - rabat / cena przed rabatem * 100.
Dalej kolumny z informacjami z paragonu (godzina, adres sklepu, kasa, ...).

Użycie:
    uv run export_csv.py
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PARSED_DIR = BASE_DIR / "biedronka_receipts_parsed"
INPUT_FILE = PARSED_DIR / "all_receipts.json"
OUTPUT_FILE = PARSED_DIR / "biedronka_receipts.csv"
# ręcznie uzupełniane wielkości opakowań (w repo, nie w .gitignore)
SIZES_FILE = BASE_DIR / "biedronka_package_sizes.csv"
SIZES_HEADER = ["nazwa", "wielkość opakowania", "jednostka", "przykładowa data", "przykładowa cena"]
# ręcznie uzupełniane czytelne nazwy produktów (w repo)
NAMES_FILE = BASE_DIR / "biedronka_product_names.csv"
NAMES_HEADER = ["nazwa z kasy", "nazwa czytelna", "kategoria"]

HEADER = [
    "data", "nazwa", "procent/sztuk/kg", "cena", "za kg/l/sztukę", "sklep", "W tym rabat",
    "rabat w %",
    # informacje dodatkowe
    "nazwa z kasy", "kategoria", "jednostka", "ilość z kasy", "cena jednostkowa z kasy", "cena przed rabatem",
    "stawka VAT", "godzina", "nr sklepu", "adres", "kod pocztowy", "miasto",
    "nr kasy", "kasjer", "nr paragonu", "id paragonu", "pozycja na paragonie",
    "suma paragonu", "rabat na paragonie", "płatność", "sprzedawca", "NIP",
]

# rozmiar opakowania w nazwie: "400g", "1kg", "1l", "500 ml", "0,5L", "4X100g"
SIZE_RE = re.compile(
    r"(?:(\d+)\s*x\s*)?(\d+(?:[.,]\d+)?)\s*(kg|g|ml|l)(?![a-ząćęłńóśźż])", re.IGNORECASE
)
UNIT_FACTORS = {"kg": ("kg", 1.0), "g": ("kg", 0.001), "l": ("l", 1.0), "ml": ("l", 0.001),
                "szt": ("szt", 1.0)}


def size_from_name(name: str) -> tuple[float, str] | None:
    """Ostatni rozmiar w nazwie, przeliczony na kg lub l. 'MLEKO 3,2 1l' -> (1.0, 'l')."""
    matches = list(SIZE_RE.finditer(name))
    if not matches:
        return None
    count, value, unit = matches[-1].groups()
    base_unit, factor = UNIT_FACTORS[unit.lower()]
    size = float(value.replace(",", ".")) * factor * int(count or 1)
    return (size, base_unit) if size > 0 else None


def load_manual_sizes() -> dict[str, tuple[float, str] | None]:
    """Wszystkie nazwy z pliku wielkości; wartość None, gdy wielkość nie jest wpisana."""
    sizes: dict[str, tuple[float, str] | None] = {}
    if not SIZES_FILE.exists():
        return sizes
    with SIZES_FILE.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            name = (row.get("nazwa") or "").strip()
            if not name:
                continue
            value = (row.get("wielkość opakowania") or "").strip().replace(",", ".")
            unit = (row.get("jednostka") or "").strip().lower()
            sizes[name] = None
            if not value:
                continue
            if unit not in UNIT_FACTORS:
                print(f"  {SIZES_FILE.name}: '{name}' - nieznana jednostka '{unit}' "
                      f"(dozwolone: {', '.join(UNIT_FACTORS)})")
                continue
            try:
                size = float(value)
            except ValueError:
                print(f"  {SIZES_FILE.name}: '{name}' - niepoprawna wielkość '{value}'")
                continue
            base_unit, factor = UNIT_FACTORS[unit]
            sizes[name] = (size * factor, base_unit)
    return sizes


def amount_and_unit(item: dict, manual: dict) -> tuple[float, str]:
    if item["is_weighted"]:
        return item["quantity"], "kg"
    size = manual.get(item["name"]) or size_from_name(item["name"])
    if size:
        return item["quantity"] * size[0], size[1]
    return item["quantity"], "szt"


def read_names_file() -> dict[str, tuple[str, str]]:
    """{nazwa z kasy: (nazwa czytelna, kategoria)} - obie mogą być puste."""
    names: dict[str, tuple[str, str]] = {}
    if NAMES_FILE.exists():
        with NAMES_FILE.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader, None)
            for row in reader:
                if row and row[0].strip():
                    row = [c.strip() for c in row] + ["", ""]
                    names[row[0]] = (row[1], row[2])
    return names


def update_names_file(receipts: list[dict], names: dict[str, tuple[str, str]]) -> int:
    """Dopisuje nowe nazwy z kasy (z pustą nazwą i kategorią); wpisane zostają bez zmian."""
    names = dict(names)
    for r in receipts:
        for item in r["items"]:
            names.setdefault(item["name"], ("", ""))
    tmp = NAMES_FILE.with_suffix(".csv.part")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(NAMES_HEADER)
        for raw in sorted(names, key=str.lower):
            writer.writerow([raw, *names[raw]])
    tmp.replace(NAMES_FILE)
    return sum(1 for name, cat in names.values() if not name or not cat)


def update_sizes_file(receipts: list[dict], manual: dict) -> int:
    """Dopisuje do pliku wielkości produkty, dla których nie znaleziono wielkości.
    Istniejące wiersze (także wypełnione ręcznie) zostają bez zmian."""
    rows: dict[str, list[str]] = {}
    if SIZES_FILE.exists():
        with SIZES_FILE.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader, None)
            for row in reader:
                if row and row[0].strip():
                    rows[row[0].strip()] = row
    for r in receipts:
        for item in r["items"]:
            name = item["name"]
            if (name in rows or item["is_weighted"] or size_from_name(name)):
                continue
            date = datetime.fromisoformat(r["datetime"]).strftime("%d.%m.%Y")
            unit_price = item["price_before_discount"] / item["quantity"]
            rows[name] = [name, "", "", date, num(unit_price)]
    tmp = SIZES_FILE.with_suffix(".csv.part")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(SIZES_HEADER)
        for name in sorted(rows, key=str.lower):
            writer.writerow(rows[name])
    tmp.replace(SIZES_FILE)
    return sum(1 for name in rows if not rows[name][1].strip())


def num(value: float | None, decimals: int = 2) -> str:
    return "" if value is None else f"{value:.{decimals}f}".replace(".", ",")


def payments_text(receipt: dict) -> str:
    parts = [
        f"{p['method']} {num(p['amount'])}"
        for p in receipt["payments"] if not p["is_change"]
    ]
    return ", ".join(parts)


def rows(receipts: list[dict], manual: dict, names: dict[str, tuple[str, str]]):
    for r in receipts:
        when = datetime.fromisoformat(r["datetime"])
        store = r["store"]
        for item in r["items"]:
            amount, unit = amount_and_unit(item, manual)
            before = item["price_before_discount"]
            discount_pct = item["discount"] / before * 100 if item["discount"] and before else None
            per_unit = item["price"] / amount if amount else None
            yield [
                when.strftime("%d.%m.%Y"),
                names.get(item["name"], ("", ""))[0] or item["name"],
                num(amount, 3),
                num(item["price"]),
                num(per_unit),
                r["shop"],
                num(item["discount"]) if item["discount"] else "",
                f"{num(discount_pct, 1)}%" if discount_pct else "",
                item["name"],
                names.get(item["name"], ("", ""))[1],
                unit,
                num(item["quantity"], 3),
                num(item["unit_price"]),
                num(item["price_before_discount"]),
                f"{item['vat_id']} ({num(item['vat_rate'], 0)}%)"
                if isinstance(item["vat_rate"], (int, float)) else item["vat_id"],
                when.strftime("%H:%M:%S"),
                store["number"],
                store["street"],
                store["postal_code"],
                store["city"],
                r["register"],
                r["cashier"],
                r["receipt_number"],
                r["receipt_id"],
                item["line"],
                num(r["total"]),
                num(r["total_discount"]),
                payments_text(r),
                r["seller"]["name"],
                r["seller"]["nip"],
            ]


def main() -> None:
    receipts = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    manual = load_manual_sizes()
    names = read_names_file()
    tmp = OUTPUT_FILE.with_suffix(".csv.part")
    n = 0
    # utf-8-sig, żeby Excel/LibreOffice poprawnie pokazały polskie znaki
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(HEADER)
        for row in rows(receipts, manual, names):
            writer.writerow(row)
            n += 1
    tmp.replace(OUTPUT_FILE)
    print(f"Zapisano {n} pozycji -> {OUTPUT_FILE.relative_to(BASE_DIR)}")
    untranslated = update_names_file(receipts, names)
    if untranslated:
        print(f"Produkty bez czytelnej nazwy lub kategorii: {untranslated} "
              f"- uzupełnij {NAMES_FILE.name}")
    missing = update_sizes_file(receipts, manual)
    if missing:
        print(f"Produkty bez wielkości opakowania: {missing} - uzupełnij {SIZES_FILE.name}")


if __name__ == "__main__":
    main()
