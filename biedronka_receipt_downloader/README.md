# biedronka_receipt_downloader

Pobiera e-paragony z https://moja.biedronka.pl/panel/paragons jako JSON
(ten sam plik, co przycisk „JSON” w panelu) do `biedronka_receipts_jsons/`.
Paragony już pobrane są pomijane.

## Konfiguracja

```
cp credentials.example.json credentials.json   # wpisz numer telefonu i hasło
uv sync
uv run playwright install chromium
```

`credentials.json`, `.browser_profile/`, `biedronka_receipts_jsons/` i `discovery/`
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
- `POST /panel/ajax/paragons` (form: `page`, `begin_data`, `end_data`, `order`, `short=1`)
  → `{"Receipts": [...], "NumberOfPages": n}`
- `GET /panel/download/json/{Id}` → plik e-paragonu
