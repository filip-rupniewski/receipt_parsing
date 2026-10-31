"""
Konwertuje e-paragony z Biedronki (biedronka_receipts_jsons/paragon_<Id>.json)
na prosty JSON do dalszej obróbki (biedronka_receipts_parsed/).

Źródłem danych jest pole "data" e-paragonu: podpisany (JWS) dokument JPK_KASA_PARAGON
w formacie Ministerstwa Finansów. Kwoty są tam w groszach - tu zamieniane na złote.
Podpis nie jest weryfikowany, dokument jest tylko dekodowany.

Wynik:
    biedronka_receipts_parsed/paragon_<Id>.json   - jeden paragon
    biedronka_receipts_parsed/all_receipts.json   - wszystkie paragony (lista, wg daty)
    biedronka_receipts_parsed/all_items.json      - wszystkie pozycje (płaska lista)
    biedronka_receipts_parsed/biedronka_receipts.csv - CSV (patrz export_csv.py)

Użycie:
    uv run convert_receipts.py
"""

from __future__ import annotations

import base64
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "biedronka_receipts_jsons"
OUTPUT_DIR = BASE_DIR / "biedronka_receipts_parsed"
LOCAL_TZ = ZoneInfo("Europe/Warsaw")

# kody form płatności z JPK_KASA; dla pozostałych kodów używana jest nazwa
# z paragonu (np. "BLIK"), a gdy jej brak - "forma_<kod>"
PAYMENT_FORMS = {
    "1": "gotówka",
    "4": "zwrot opakowań",
}


def pln(grosze: int | None) -> float | None:
    return None if grosze is None else round(grosze / 100, 2)


def to_float(value: str | int | float) -> float:
    return float(str(value).replace(",", "."))


def b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def decode_jpk(raw: dict) -> dict:
    """Wyciąga dokument JPK z pola 'data' (JWS: nagłówek.payload.podpis)."""
    payload = raw["data"].split(".")[1]
    return json.loads(b64url_decode(payload))["dokument"]


def clean_name(name: str, vat_id: str) -> str:
    """'MlekoŚw3,2 1l            C' -> 'MlekoŚw3,2 1l' (usuwa dopełnienie i literę VAT)."""
    name = name.rstrip()
    if vat_id and re.search(rf"\s{re.escape(vat_id)}$", name):
        name = name[: -len(vat_id)]
    return name.strip()


def parse_idz(idz: str) -> dict:
    """'c=...|g={...}|s=5213|p=1|t=1235' -> {'c': ..., 's': '5213', ...}"""
    return dict(part.split("=", 1) for part in idz.split("|") if "=" in part)


def local_time(iso_utc: str) -> datetime:
    return datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).astimezone(LOCAL_TZ)


def convert(raw: dict, receipt_id: str) -> dict:
    doc = decode_jpk(raw)
    par = doc["paragon"]
    seller = doc.get("podmiot1", {})
    addr = seller.get("adresPod", {})
    idz = parse_idz(raw.get("IDZ", ""))
    when = local_time(par["zakSprzed"])

    vat_rates = {}
    for st in par.get("stPTU", []):
        w = st.get("wart")
        vat_rates[st["id"]] = to_float(w) / 100 if isinstance(w, (int, float)) else w  # "ZW"

    items = []
    for i, poz in enumerate(par.get("pozycja", []), 1):
        t = poz["towar"]
        vat_id = t.get("idStPTU")
        qty = to_float(t["ilosc"])
        discount = -(t.get("rabat") or {}).get("wart", 0)
        name = clean_name(t["nazwa"], vat_id)
        items.append({
            "line": i,
            "name": name,
            "quantity": qty,
            "is_weighted": not qty.is_integer() or bool(re.search(r"\bkg\b", name, re.I)),
            "unit_price": pln(t["cena"]),
            "price_before_discount": pln(t["brutto"] + discount),
            "discount": pln(discount),
            "price": pln(t["brutto"]),
            "vat_id": vat_id,
            "vat_rate": vat_rates.get(vat_id),
        })

    payments = []
    for p in par.get("platnosc", []):
        code = str(p.get("forma"))
        payments.append({
            "method": PAYMENT_FORMS.get(code) or p.get("nazwa") or f"forma_{code}",
            "method_code": code,
            "amount": pln(p.get("wart")),
            "is_change": bool(p.get("reszta")),  # reszta wydana klientowi
        })

    podsum = par.get("podsum", {})
    return {
        "receipt_id": receipt_id,
        "shop": "biedronka",
        "datetime": when.isoformat(),
        "date": when.date().isoformat(),
        "store": {
            "number": idz.get("s"),
            "street": addr.get("ulica"),
            "postal_code": addr.get("kodPoczt"),
            "city": addr.get("miejsc"),
        },
        "seller": {"name": seller.get("nazwaPod"), "nip": seller.get("NIP")},
        "register": par.get("nrKasy"),
        "cashier": par.get("kasjer"),
        "receipt_number": par.get("nrParag"),
        "currency": podsum.get("waluta", "PLN"),
        "total": pln(podsum.get("sumaBrutto")),
        "total_discount": pln(-podsum.get("sumaOpust", 0)),
        "total_vat": pln(podsum.get("sumaPod")),
        "vat_summary": [
            {
                "vat_id": s["idStPTU"],
                "vat_rate": vat_rates.get(s["idStPTU"]),
                "gross": pln(s["brutto"]),
                "vat": pln(s["vat"]),
            }
            for s in podsum.get("sumaNetto", [])
        ],
        "payments": payments,
        "items": items,
    }


def check(rec: dict) -> list[str]:
    """Kontrola spójności - suma pozycji i rabatów vs podsumowanie paragonu."""
    problems = []
    items_sum = round(sum(i["price"] for i in rec["items"]), 2)
    if items_sum != rec["total"]:
        problems.append(f"suma pozycji {items_sum} != total {rec['total']}")
    disc_sum = round(sum(i["discount"] for i in rec["items"]), 2)
    if disc_sum != rec["total_discount"]:
        problems.append(f"suma rabatów {disc_sum} != total_discount {rec['total_discount']}")
    return problems


def write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    receipts = []
    for src in sorted(INPUT_DIR.glob("paragon_*.json")):
        receipt_id = src.stem.removeprefix("paragon_")
        try:
            rec = convert(json.loads(src.read_text(encoding="utf-8")), receipt_id)
        except Exception as e:
            print(f"  {src.name}: BŁĄD {e.__class__.__name__}: {e}")
            continue
        for problem in check(rec):
            print(f"  {src.name}: UWAGA {problem}")
        write_json(OUTPUT_DIR / src.name, rec)
        receipts.append(rec)

    receipts.sort(key=lambda r: r["datetime"])
    write_json(OUTPUT_DIR / "all_receipts.json", receipts)
    all_items = [
        {"receipt_id": r["receipt_id"], "date": r["date"], "datetime": r["datetime"],
         "shop": r["shop"], "store_number": r["store"]["number"], **item}
        for r in receipts for item in r["items"]
    ]
    write_json(OUTPUT_DIR / "all_items.json", all_items)
    print(f"Skonwertowano {len(receipts)} paragonów ({len(all_items)} pozycji) "
          f"-> {OUTPUT_DIR.name}/")

    import export_csv
    export_csv.main()

    import analyze_spending
    analyze_spending.main([])


if __name__ == "__main__":
    main()
