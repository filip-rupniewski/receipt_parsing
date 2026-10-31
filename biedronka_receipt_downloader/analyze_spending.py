"""
Statystyki zakupów w Biedronce: ile wydajesz tygodniowo, na co, kiedy i gdzie.

Czyta biedronka_receipts_parsed/all_receipts.json (paragony) oraz
biedronka_receipts_parsed/biedronka_receipts.csv (pozycje z czytelnymi nazwami
i kategoriami) i zapisuje raport w Markdown:

    analiza_wydatkow.md

Średnia tygodniowa = suma wydatków / liczba dni od pierwszego do ostatniego
paragonu * 7.

Użycie:
    uv run analyze_spending.py
    uv run analyze_spending.py --since 2026-10-01 --until 2026-10-31
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PARSED_DIR = BASE_DIR / "biedronka_receipts_parsed"
RECEIPTS_FILE = PARSED_DIR / "all_receipts.json"
ITEMS_FILE = PARSED_DIR / "biedronka_receipts.csv"
REPORT_FILE = BASE_DIR / "analiza_wydatkow.md"

WEEKDAYS = ["poniedziałek", "wtorek", "środa", "czwartek", "piątek", "sobota", "niedziela"]
TIME_OF_DAY = [  # (od godziny, do godziny wyłącznie, nazwa)
    (5, 12, "rano (5–12)"),
    (12, 16, "w południe (12–16)"),
    (16, 22, "wieczorem (16–22)"),
    (22, 29, "w nocy (22–5)"),
]
TOP_PRODUCTS = 15


def zl(value: float) -> str:
    return f"{value:,.2f} zł".replace(",", " ").replace(".", ",")


def pct(part: float, whole: float) -> str:
    return f"{part / whole * 100:.0f}%" if whole else "–"


def num(s: str) -> float:
    return float(s.replace(",", ".")) if s else 0.0


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def time_of_day(hour: int) -> str:
    for start, end, name in TIME_OF_DAY:
        if start <= hour < end or start <= hour + 24 < end:
            return name
    return "?"


def load(since: date | None, until: date | None) -> tuple[list[dict], list[dict]]:
    receipts = json.loads(RECEIPTS_FILE.read_text(encoding="utf-8"))
    for r in receipts:
        r["dt"] = datetime.fromisoformat(r["datetime"])
    receipts = [
        r for r in receipts
        if (not since or r["dt"].date() >= since) and (not until or r["dt"].date() <= until)
    ]
    ids = {r["receipt_id"] for r in receipts}
    with ITEMS_FILE.open(encoding="utf-8-sig", newline="") as f:
        items = [row for row in csv.DictReader(f, delimiter=";") if row["id paragonu"] in ids]
    return receipts, items


def report(receipts: list[dict], items: list[dict]) -> str:
    receipts.sort(key=lambda r: r["dt"])
    first, last = receipts[0]["dt"].date(), receipts[-1]["dt"].date()
    days = (last - first).days + 1
    total = sum(r["total"] for r in receipts)
    discount = sum(r["total_discount"] for r in receipts)
    weekly = total / days * 7

    out = [
        "# Analiza wydatków w Biedronce",
        "",
        f"Okres: {first:%d.%m.%Y} – {last:%d.%m.%Y} ({days} dni), "
        f"{len(receipts)} paragonów. Wygenerowano {datetime.now():%d.%m.%Y %H:%M} "
        f"skryptem `analyze_spending.py`.",
        "",
        "Średnia tygodniowa = suma wydatków / liczba dni od pierwszego do ostatniego "
        "paragonu × 7. Przy krótkim okresie traktuj ją jako przybliżenie.",
        "",
        "## Ile wydajesz",
        "",
        table(["", ""], [
            ["Suma", f"**{zl(total)}**"],
            ["Rabaty", f"{zl(discount)}: przed rabatami byłoby {zl(total + discount)}, "
                       f"czyli rabaty obniżyły ceny o {pct(discount, total + discount)}"],
            ["Średnio na tydzień", f"**{zl(weekly)}**"],
            ["Średnio na miesiąc (30 dni)", zl(total / days * 30)],
            ["Średni paragon", zl(total / len(receipts))],
            ["Wizyty w tygodniu (średnio)", f"{len(receipts) / days * 7:.1f}".replace(".", ",")],
        ]),
        "",
        "### Tydzień po tygodniu",
        "",
    ]

    weeks: dict[date, list[dict]] = defaultdict(list)
    for r in receipts:
        monday = r["dt"].date() - timedelta(days=r["dt"].weekday())
        weeks[monday].append(r)
    out.append(table(["Tydzień", "Wydatki", "Wizyty"], [
        [f"{m:%d.%m}–{m + timedelta(days=6):%d.%m.%Y}", zl(sum(r["total"] for r in rs)), len(rs)]
        for m, rs in sorted(weeks.items())
    ]))
    out.append("")
    out.append("Pierwszy i ostatni tydzień mogą być niepełne.")

    # --- na co ---
    by_cat: dict[str, float] = defaultdict(float)
    for it in items:
        by_cat[it["kategoria"] or "bez kategorii"] += num(it["cena"])
    items_total = sum(by_cat.values())
    out += [
        "",
        "## Na co",
        "",
        "Kategorie pochodzą z kolumny `kategoria` w `biedronka_product_names.csv`.",
        "",
        table(["Kategoria", "Kwota", "Udział"], [
            [c, zl(v), pct(v, items_total)]
            for c, v in sorted(by_cat.items(), key=lambda kv: -kv[1])
        ]),
    ]

    prod: dict[str, list] = defaultdict(lambda: [0, 0.0, 0.0])
    for it in items:
        p = prod[it["nazwa"]]
        p[0] += 1
        p[1] += num(it["cena"])
        p[2] += num(it["W tym rabat"])
    out += [
        "",
        f"### Produkty: największe wydatki (top {TOP_PRODUCTS})",
        "",
        table(["Produkt", "Ile razy", "Wydatki", "W tym rabaty"], [
            [n, c, zl(s), zl(d) if d else ""]
            for n, (c, s, d) in sorted(prod.items(), key=lambda kv: -kv[1][1])[:TOP_PRODUCTS]
        ]),
        "",
        f"### Produkty: najczęściej kupowane (top {TOP_PRODUCTS})",
        "",
        table(["Produkt", "Ile razy", "Wydatki"], [
            [n, c, zl(s)]
            for n, (c, s, _) in sorted(prod.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))
            [:TOP_PRODUCTS]
        ]),
    ]

    # --- kiedy ---
    wd = Counter(r["dt"].weekday() for r in receipts)
    wd_sum: dict[int, float] = defaultdict(float)
    for r in receipts:
        wd_sum[r["dt"].weekday()] += r["total"]
    tod = Counter(time_of_day(r["dt"].hour) for r in receipts)
    hours = Counter(r["dt"].hour for r in receipts)
    out += [
        "",
        "## Kiedy",
        "",
        "### Dni tygodnia",
        "",
        table(["Dzień", "Wizyty", "Wydatki"], [
            [WEEKDAYS[d], wd.get(d, 0), zl(wd_sum.get(d, 0.0))] for d in range(7)
        ]),
        "",
        "### Pora dnia",
        "",
        table(["Pora", "Wizyty"], [
            [name, tod.get(name, 0)] for _, _, name in TIME_OF_DAY if tod.get(name)
        ]),
        "",
        "Godziny wizyt: " + ", ".join(
            f"{h}:00 ({n}×)" if n > 1 else f"{h}:00" for h, n in sorted(hours.items())
        ) + ".",
        "",
        "### Wszystkie wizyty",
        "",
        table(["Data", "Dzień", "Godzina", "Kwota", "Sklep"], [
            [f"{r['dt']:%d.%m.%Y}", WEEKDAYS[r["dt"].weekday()], f"{r['dt']:%H:%M}",
             zl(r["total"]), r["store"]["number"]]
            for r in receipts
        ]),
    ]

    # --- gdzie ---
    stores: dict[str, list] = defaultdict(lambda: [0, 0.0, ""])
    for r in receipts:
        s = stores[r["store"]["number"]]
        s[0] += 1
        s[1] += r["total"]
        s[2] = f"{r['store']['street']}, {r['store']['city']}"
    out += [
        "",
        "## Gdzie",
        "",
        table(["Sklep", "Adres", "Wizyty", "Wydatki"], [
            [n, addr, c, zl(v)] for n, (c, v, addr) in sorted(stores.items(), key=lambda kv: -kv[1][0])
        ]),
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", type=date.fromisoformat, help="od daty (RRRR-MM-DD)")
    parser.add_argument("--until", type=date.fromisoformat, help="do daty (RRRR-MM-DD)")
    args = parser.parse_args(argv)

    receipts, items = load(args.since, args.until)
    if not receipts:
        print("Brak paragonów w wybranym okresie.")
        return
    REPORT_FILE.write_text(report(receipts, items), encoding="utf-8")
    print(f"Zapisano analizę wydatków -> {REPORT_FILE.name}")


if __name__ == "__main__":
    main()
