# biedronka_receipt_downloader

Pobiera e-paragony z https://moja.biedronka.pl/panel/paragons jako JSON
(ten sam plik, co przycisk „JSON” w panelu) do `biedronka_receipts_jsons/`.
Paragony już pobrane są pomijane.

Po pobraniu nowych paragonów automatycznie uruchamia się `convert_receipts.py`
(można go też uruchomić osobno: `uv run convert_receipts.py`). Konwerter dekoduje
z pola `data` podpisany dokument JPK_KASA_PARAGON i zapisuje do
`biedronka_receipts_parsed/`:

- `paragon_<Id>.json`: jeden paragon (data, sklep, sumy, VAT, płatności, pozycje),
- `all_receipts.json`: wszystkie paragony,
- `all_items.json`: płaska lista wszystkich pozycji z datą i numerem paragonu,
- `biedronka_receipts.csv`: CSV (`;`, przecinek dziesiętny) z kolumnami
  `data; nazwa; procent/sztuk/kg; cena; za kg/l/sztukę; sklep; W tym rabat; rabat w %` oraz
  danymi z paragonu (jednostka, godzina, adres sklepu, kasa, kasjer, nr paragonu,
  płatność, ...). Wielkość opakowania (kg/l) jest brana z nazwy produktu
  (np. `400g`, `1l`, `4X100g`); produkty ważone mają kg z kasy, pozostałe sztuki.
  Osobno: `uv run export_csv.py`.

Kwoty są w złotych. Pozycja ma pola `name`, `quantity`, `is_weighted`, `unit_price`,
`price_before_discount`, `discount`, `price` (po rabacie), `vat_id`, `vat_rate`.

## Konfiguracja

```
cp credentials.example.json credentials.json   # wpisz numer telefonu i hasło
uv sync
uv run playwright install chromium
```

`credentials.json`, `.browser_profile/`, `biedronka_receipts_jsons/`,
`biedronka_receipts_parsed/` i `discovery/`
są w `.gitignore`.

## Użycie

```
uv run download_receipts.py                    # nowe paragony od 2019-01-01
uv run download_receipts.py --since 2026-01-01 # krótszy zakres
uv run download_receipts.py --manual-login     # bez automatycznego wpisywania danych
```

Logowanie chroni captcha Cloudflare Turnstile, która odrzuca przeglądarki sterowane
przez Playwright. Dlatego Chromium startuje jako zwykły proces: skrypt tylko wpisuje
numer i hasło, a weryfikację i przyciski „Dalej”/„Zaloguj” klikasz sam. Dzięki
„Zapamiętaj mnie” sesja zostaje w `.browser_profile/`, więc zwykle logowanie jest
pomijane.

API panelu (odczytane z ruchu sieciowego):
- `POST /panel/ajax/paragons` (form: `page`, `begin_data`, `end_data`, `order`, `short=0`; `short=1` pomija część paragonów)
  → `{"Receipts": [...], "NumberOfPages": n}`
- `GET /panel/download/json/{Id}` → plik e-paragonu

## Wielkości opakowań

`biedronka_package_sizes.csv` zawiera produkty, dla których nie udało się odczytać
wielkości z nazwy. Wpisz wielkość jednej sztuki i jednostkę (`kg`, `g`, `l`, `ml` albo
`szt`, np. `30;szt` dla herbaty w saszetkach), a potem uruchom `uv run export_csv.py`.
Nowe produkty są dopisywane automatycznie, a wpisane wartości zostają.
Można tu też dopisać dowolny produkt, żeby nadpisać wielkość odczytaną z nazwy.

## Czytelne nazwy produktów

`biedronka_product_names.csv` (`nazwa z kasy;nazwa czytelna`) tłumaczy skrócone nazwy
z kasy, np. `HerbRemseyOwoceLeśn` → `herbata Owoce Leśne Remsey`. W CSV kolumna
`nazwa` ma nazwę czytelną (lub nazwę z kasy, jeśli czytelna jest pusta), a oryginał
jest w kolumnie `nazwa z kasy`. Nowe produkty dopisują się automatycznie z pustą
nazwą czytelną. Plik wielkości opakowań nadal używa nazw z kasy.
