# biedronka_receipt_downloader

Pobiera e-paragony z [Moja Biedronka](https://moja.biedronka.pl/panel/paragons),
zamienia je na czytelny JSON i CSV i liczy statystyki wydatków.

```
moja.biedronka.pl ──download_receipts.py──▶ biedronka_receipts_jsons/   (oryginalne e-paragony)
                    convert_receipts.py ──▶ biedronka_receipts_parsed/  (JSON: paragony, pozycje)
                    export_csv.py       ──▶ biedronka_receipts_parsed/biedronka_receipts.csv
                    analyze_spending.py ──▶ analiza_wydatkow.md
```

Wystarczy uruchomić `download_receipts.py`: po pobraniu nowych paragonów pozostałe kroki
uruchamiają się automatycznie. Każdy krok można też uruchomić osobno.

## Pierwsza konfiguracja

```
cd biedronka_receipt_downloader
cp credentials.example.json credentials.json   # wpisz numer telefonu i hasło
uv sync
uv run playwright install chromium
```

Ten folder ma własne środowisko uv (tylko Playwright), niezależne od głównego projektu.

## Użycie

```
uv run download_receipts.py                    # pobierz nowe paragony (od 2019-01-01) i przetwórz
uv run download_receipts.py --since 2026-10-01 # szybciej: krótszy zakres dat
uv run download_receipts.py --manual-login     # bez automatycznego wpisywania danych logowania

uv run convert_receipts.py                     # tylko konwersja + CSV + analiza
uv run export_csv.py                           # tylko CSV (np. po zmianie plików nazw/wielkości)
uv run analyze_spending.py                     # tylko analiza wydatków
uv run analyze_spending.py --since 2026-10-01 --until 2026-10-31
```

### Logowanie

Logowanie chroni captcha Cloudflare Turnstile, która odrzuca przeglądarki sterowane przez
Playwright. Dlatego Chromium startuje jako zwykły proces, a skrypt podłącza się do niego
tylko na chwilę, żeby wpisać numer telefonu i hasło. Weryfikację Cloudflare i przyciski
„Dalej” / „Zaloguj” klikasz sam. Po zalogowaniu skrypt przejmuje przeglądarkę i pobiera
paragony. Dzięki „Zapamiętaj mnie” sesja zostaje w `.browser_profile/`, więc zwykle
logowanie jest pomijane.

## Skrypty

### `download_receipts.py`

Pobiera listę paragonów rocznymi zakresami dat i zapisuje każdy nowy jako
`biedronka_receipts_jsons/paragon_<Id>.json`. To ten sam plik, co przycisk „JSON”
w panelu. Paragony już pobrane są pomijane.

API panelu (odczytane z ruchu sieciowego):

- `POST /panel/ajax/paragons` (form: `page`, `begin_data`, `end_data`, `order`, `short=0`)
  → `{"Receipts": [...], "NumberOfPages": n}`. Panel wysyła `short=1`, które pomija część
  paragonów.
- `GET /panel/download/json/{Id}` → plik e-paragonu.

Tryb `--discover` zapisuje ruch sieciowy panelu do `discovery/` (przydatne, gdy
Biedronka zmieni stronę).

### `convert_receipts.py`

Dekoduje z pola `data` e-paragonu podpisany dokument JPK_KASA_PARAGON (format Ministerstwa
Finansów; podpis nie jest weryfikowany). Kwoty zamienia z groszy na złote. Zapisuje do
`biedronka_receipts_parsed/`:

- `paragon_<Id>.json`: jeden paragon (data, sklep, sumy, VAT, płatności, pozycje),
- `all_receipts.json`: wszystkie paragony,
- `all_items.json`: płaska lista wszystkich pozycji z datą i numerem paragonu.

Pozycja ma pola `name`, `quantity`, `is_weighted`, `unit_price`, `price_before_discount`,
`discount`, `price` (po rabacie), `vat_id`, `vat_rate`. Konwerter sprawdza, czy suma pozycji
i rabatów zgadza się z podsumowaniem paragonu, i ostrzega, jeśli nie.

### `export_csv.py`

Tworzy `biedronka_receipts_parsed/biedronka_receipts.csv` w formacie reszty projektu
(separator `;`, przecinek dziesiętny, data `dd.mm.rrrr`, UTF-8 z BOM dla Excela/LibreOffice).

| Kolumna | Zawartość |
|---|---|
| data | `02.10.2026` |
| nazwa | nazwa czytelna z `biedronka_product_names.csv` (lub nazwa z kasy) |
| procent/sztuk/kg | ilość: kg z kasy (produkty ważone), sztuki × wielkość opakowania (kg/l), albo sztuki |
| cena | zapłacona kwota za pozycję, po rabacie |
| za kg/l/sztukę | cena / procent/sztuk/kg |
| sklep | `biedronka` |
| W tym rabat | rabat w zł |
| rabat w % | rabat / cena przed rabatem |

Dalej kolumny dodatkowe: nazwa z kasy, kategoria, jednostka, ilość i cena jednostkowa
z kasy, cena przed rabatem, stawka VAT, godzina, nr sklepu, adres, kod pocztowy, miasto,
nr kasy, kasjer, nr paragonu, id paragonu, pozycja na paragonie, suma i rabat paragonu,
płatność, sprzedawca, NIP.

Wielkość opakowania jest brana z nazwy produktu (`400g`, `1l`, `4X100g`) albo
z `biedronka_package_sizes.csv`.

### `analyze_spending.py`

Zapisuje raport `analiza_wydatkow.md`:

- suma, rabaty, średnio na tydzień i miesiąc, średni paragon, wizyty w tygodniu,
- wydatki tydzień po tygodniu,
- wydatki według kategorii, najdroższe i najczęściej kupowane produkty,
- dni tygodnia, pora dnia i godziny zakupów, lista wszystkich wizyt,
- sklepy.

Średnia tygodniowa = suma wydatków / liczba dni od pierwszego do ostatniego paragonu × 7.

## Pliki do ręcznego uzupełniania

Oba pliki są w repo. Nowe produkty dopisują się do nich automatycznie z pustymi polami,
a wpisane wartości zostają. Po zmianach uruchom `uv run export_csv.py`
(i ewentualnie `uv run analyze_spending.py`).

### `biedronka_product_names.csv`

`nazwa z kasy;nazwa czytelna;kategoria`, np.
`HerbRemseyOwoceLeśn;herbata Owoce Leśne Remsey;spiżarnia`.

Kategorie są dowolne. Obecnie używane: nabiał, słodycze, spiżarnia, owoce i warzywa,
ryby, pieczywo, wędliny.

### `biedronka_package_sizes.csv`

Produkty, dla których nie udało się odczytać wielkości z nazwy. Wpisz wielkość jednej
sztuki i jednostkę: `kg`, `g`, `l`, `ml` albo `szt` (np. `30;szt` dla herbaty
w saszetkach). Można tu też dopisać dowolny produkt, żeby nadpisać wielkość odczytaną
z nazwy. Plik używa nazw z kasy.

## Pozostałe pliki

- `analiza_zakupow_biedronka.md`: jednorazowa analiza produktów pod kątem zdrowia
  i ceny do jakości (październik 2026), napisana ręcznie.
- `credentials.example.json`: wzór pliku z danymi logowania.

W `.gitignore` (dane osobowe i lokalne): `credentials.json`, `.browser_profile/`,
`biedronka_receipts_jsons/`, `biedronka_receipts_parsed/`, `discovery/`,
`analiza_wydatkow.md`, `paragon_*.json`.
