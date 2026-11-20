# allegro: pobieranie listy zakupów

Pobiera zakupy z [Moje Allegro → Kupione](https://allegro.pl/moje-allegro/zakupy/kupione).

```
allegro.pl ──download_purchases.py──▶ allegro_raw/responses_<data>.json   (surowe odpowiedzi JSON)
                                  └──▶ allegro_purchases/orders.json       (zamówienia)
                                  └──▶ allegro_purchases/items.csv         (jedna pozycja = jeden wiersz)
```

## Pierwsza konfiguracja

```
cd receipt_downloader/allegro
uv sync
uv run playwright install chromium
uv run download_purchases.py --save-credentials   # opcjonalnie, patrz niżej
```

## Dane logowania

Login i hasło nie są zapisywane w żadnym pliku projektu. Do wyboru:

| Sposób | Polecenie |
|---|---|
| Systemowy pęk kluczy (GNOME Keyring / KWallet, zaszyfrowany, odblokowywany z sesją) | `--save-credentials` raz, potem zwykłe `uv run download_purchases.py` |
| Wpisywanie w terminalu przy każdym logowaniu (bez echa, nic nie jest zapisywane) | `--ask` (albo domyślnie, gdy pęk kluczy jest pusty) |
| Logowanie ręczne w oknie przeglądarki | `--manual-login` |
| Usunięcie danych z pęku kluczy | `--forget-credentials` |

O dane skrypt pyta tylko wtedy, gdy sesja w `.browser_profile/` wygasła. Ten katalog
zawiera ciasteczka sesji, więc ma uprawnienia 700 i jest w `.gitignore`.

## Logowanie

Allegro blokuje boty (DataDome), więc Chromium startuje jako zwykły proces. Skrypt
podłącza się do niego tylko na chwilę, żeby wpisać login i hasło. „Zaloguj się”, captchę
i kod 2FA obsługujesz sam. Masz na to 5 minut.

Paski „Chrome do testowania…” i „nieobsługiwana flaga --no-sandbox” są niegroźne.
Skrypt używa Chromium dostarczanego z Playwright, a Ubuntu (AppArmor) nie pozwala
mu działać z sandboxem.

## Wynik

`allegro_purchases/items.csv`: jedna pozycja na wiersz, kolumny `order_id`, `date`,
`seller`, `order_status` (DELIVERED, RETURNED, PARTIALLY_RETURNED, ORDER_CANCELLED…),
`order_total`, `offer_id`, `title`, `quantity`, `unit_price`, `currency`,
`delivery_name`, `delivery_cost`, `url`. Pełne dane zamówień są w `orders.json`.

`allegro_purchases/items_kept.csv` zawiera tylko to, co zostało u Ciebie. Pomija
anulowane zamówienia i zwrócone sztuki, także zwroty jeszcze w trakcie weryfikacji.
Pozycja zwrócona częściowo zostaje z pomniejszoną ilością. Kolumny: `order_id`, `date`,
`seller`, `offer_id`, `title`, `quantity`, `unit_price`, `total` (= ilość × cena),
`currency`, `url`. Koszt dostawy nie jest wliczony.

`allegro_purchases/zakupy.csv` to uproszczona wersja `items_kept.csv` do czytania.
Ma kolumny `nr`, `data`, `produkt`, `liczba`, `cena` (= liczba × cena jednostkowa),
`cena jednostkowa` i `sklep` (zawsze `allegro`). Kwoty są w PLN, a najnowszy zakup jest
na górze. `nr` to numer zakupu: 1 to najstarszy, a numery
idą kolejno bez luk. Pozycje z jednego zamówienia mają ten sam numer. Numery są liczone
tylko z zachowanych zakupów, więc mogą się przesunąć, jeśli później zwrócisz całe
zamówienie. Data jest w czasie polskim. Kwoty w innych walutach są przeliczane na PLN
po średnim kursie NBP z dnia roboczego przed zakupem. Kursy są zapisywane w
`allegro_raw/nbp_rates.json`.

`uv run download_purchases.py --rebuild` odtwarza pliki wynikowe z `allegro_raw/` bez
logowania, np. po zmianie kolumn.

## Jak działa pobieranie

Wewnętrzne API strony „Kupione” nie jest publiczne. Dlatego skrypt nie woła go
bezpośrednio, tylko przewija stronę i klika „pokaż więcej” / „następna”. W tym czasie
zapisuje odpowiedzi JSON, które wyglądają na listę zamówień (obiekty z listą
`offers`/`lineItems`/`items` i polami takimi jak `seller` czy `orderDate`). Wyniki są
łączone ze wszystkimi wcześniejszymi pobraniami z `allegro_raw/`.

Pierwsze uruchomienie przegląda całą listę. Znacznik `allegro_raw/.history_complete`
zapisuje, że historia jest kompletna. Kolejne uruchomienia przeglądają listę tylko do
30 dni przed najnowszym zapisanym zakupem (`--days N` zmienia ten okres). Dzięki temu
łapią nowe zakupy oraz zwroty i zmiany statusów w ostatnich zamówieniach. `--full`
przegląda całą listę i odświeża szczegóły wszystkich zwrotów.

## Zwroty

Lista zakupów podaje tylko status całego zamówienia (`RETURNED`, `PARTIALLY_RETURNED`).
Dlatego dla zamówień ze zwrotem skrypt otwiera stronę „Szczegóły zakupu” i odczytuje
z osadzonego w niej JSON-a:

- zgłoszone zwroty (`rescissions`): produkt, ilość, powód, etap,
- zwroty pieniędzy (`refunds`): produkt, zwrócona ilość i kwota, status.

Wynik trafia do `allegro_raw/details/<id_zamówienia>.json`. Szczegóły są pobierane dla
nowych zamówień ze zwrotem i dla tych z przeglądanego okresu, bo zwrot mógł się w tym
czasie zmienić. W `items.csv` są kolumny `returned_quantity`, `return_reason`,
`return_status` (np. „Decyzja o zwrocie: Pełen zwrot pieniędzy.”), `return_date`,
`refunded_quantity`, `refund_amount` i `refund_status`. Gdy Allegro podaje tylko kwotę
zwrotu, `refunded_quantity` jest puste. Liczy się `refund_amount`.
