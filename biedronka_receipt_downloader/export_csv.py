"""
Eksportuje skonwertowane paragony Biedronki (biedronka_receipts_parsed/all_receipts.json)
do CSV w formacie zbliżonym do reszty projektu (separator ';', przecinek dziesiętny,
data dd.mm.rrrr):

    biedronka_receipts_parsed/biedronka_receipts.csv

Kolumny główne:
    data, nazwa,
    procent/sztuk/kg   - ilość w jednostce z kolumny "jednostka":
                         produkty ważone -> kg z kasy,
                         produkty z wagą/objętością w nazwie (np. "Jogurt 400g") ->
                         sztuki * rozmiar (kg lub l),
                         pozostałe -> liczba sztuk,
    cena               - zapłacona kwota za pozycję (po rabacie),
    za kg/l/sztukę     - cena / procent/sztuk/kg,
    sklep,
    W tym rabat        - kwota rabatu zawarta w cenie (zł).
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

HEADER = [
    "data", "nazwa", "procent/sztuk/kg", "cena", "za kg/l/sztukę", "sklep", "W tym rabat",
    # informacje dodatkowe
    "jednostka", "ilość z kasy", "cena jednostkowa z kasy", "cena przed rabatem",
    "stawka VAT", "godzina", "nr sklepu", "adres", "kod pocztowy", "miasto",
    "nr kasy", "kasjer", "nr paragonu", "id paragonu", "pozycja na paragonie",
    "suma paragonu", "rabat na paragonie", "płatność", "sprzedawca", "NIP",
]

# rozmiar opakowania w nazwie: "400g", "1kg", "1l", "500 ml", "0,5L", "4X100g"
SIZE_RE = re.compile(
    r"(?:(\d+)\s*x\s*)?(\d+(?:[.,]\d+)?)\s*(kg|g|ml|l)(?![a-ząćęłńóśźż])", re.IGNORECASE
)
UNIT_FACTORS = {"kg": ("kg", 1.0), "g": ("kg", 0.001), "l": ("l", 1.0), "ml": ("l", 0.001)}


def size_from_name(name: str) -> tuple[float, str] | None:
    """Ostatni rozmiar w nazwie, przeliczony na kg lub l. 'MLEKO 3,2 1l' -> (1.0, 'l')."""
    matches = list(SIZE_RE.finditer(name))
    if not matches:
        return None
    count, value, unit = matches[-1].groups()
    base_unit, factor = UNIT_FACTORS[unit.lower()]
    size = float(value.replace(",", ".")) * factor * int(count or 1)
    return (size, base_unit) if size > 0 else None


def amount_and_unit(item: dict) -> tuple[float, str]:
    if item["is_weighted"]:
        return item["quantity"], "kg"
    size = size_from_name(item["name"])
    if size:
        return item["quantity"] * size[0], size[1]
    return item["quantity"], "szt"


def num(value: float | None, decimals: int = 2) -> str:
    return "" if value is None else f"{value:.{decimals}f}".replace(".", ",")


def payments_text(receipt: dict) -> str:
    parts = [
        f"{p['method']} {num(p['amount'])}"
        for p in receipt["payments"] if not p["is_change"]
    ]
    return ", ".join(parts)


def rows(receipts: list[dict]):
    for r in receipts:
        when = datetime.fromisoformat(r["datetime"])
        store = r["store"]
        for item in r["items"]:
            amount, unit = amount_and_unit(item)
            per_unit = item["price"] / amount if amount else None
            yield [
                when.strftime("%d.%m.%Y"),
                item["name"],
                num(amount, 3),
                num(item["price"]),
                num(per_unit),
                r["shop"],
                num(item["discount"]) if item["discount"] else "",
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
    tmp = OUTPUT_FILE.with_suffix(".csv.part")
    n = 0
    # utf-8-sig, żeby Excel/LibreOffice poprawnie pokazały polskie znaki
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(HEADER)
        for row in rows(receipts):
            writer.writerow(row)
            n += 1
    tmp.replace(OUTPUT_FILE)
    print(f"Zapisano {n} pozycji -> {OUTPUT_FILE.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
