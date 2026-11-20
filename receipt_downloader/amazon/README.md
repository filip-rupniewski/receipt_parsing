# amazon: pobieranie listy zakupów

Pobiera zakupy z [Twoje zamówienia](https://www.amazon.pl/your-orders/orders) (domyślnie
amazon.pl, inna domena: `--domain amazon.de`).

```
amazon.pl ──download_purchases.py──▶ dane/amazon_raw/pages_<data>.json   (uproszczony HTML stron listy)
                                 └──▶ dane/amazon_raw/details/*.json     (szczegóły zamówień: ceny, zwroty)
                                 └──▶ dane/amazon_purchases/             (orders.json, items.csv,
                                                                         items_kept.csv, zakupy.csv,
                                                                         wydatki_miesiecznie.png)
```

## Pierwsza konfiguracja

```
cd receipt_downloader/amazon
uv sync
uv run playwright install chromium
uv run download_purchases.py --save-credentials   # opcjonalnie, patrz niżej
```

## Dane osobowe i ich historia

Wszystkie pobrane dane są w `dane/`. Repozytorium z kodem ignoruje ten katalog
(`.gitignore`), a historia danych jest zapisywana we wspólnym, prywatnym repozytorium
projektu `.git-private` w katalogu głównym. To repozytorium nie ma remote, więc nic nie
trafia na GitHub (opis w głównym README). Po każdym pobraniu i po `--rebuild`, jeśli
coś się zmieniło, skrypt sam robi tam commit, np. „amazon: pobranie 2026-11-20 12:00:
zamówień 85, nowych 1, zmienione szczegóły zamówień: 3”. Historię oglądasz tak
(z katalogu głównego projektu):

```
git --git-dir=.git-private log --stat -- receipt_downloader/amazon/dane
git --git-dir=.git-private diff HEAD~1 -- receipt_downloader/amazon/dane/amazon_purchases/zakupy.csv
```

Jeśli `.git-private` nie istnieje, skrypt pomija commit. Katalog `dane/` jest też
na liście `DANE` w `zapisz_dane_prywatne.sh`.

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

Chromium startuje jako zwykły proces z trwałym profilem (port CDP 9231, inny niż
Allegro 9230 i Biedronka 9229). Skrypt podłącza się do niego tylko na chwilę, żeby
wpisać dane, i sprawdza, czy wpisane wartości zostały w polach. Amazon pyta najpierw
o e-mail, a dopiero na następnej stronie o hasło, więc: skrypt wpisuje e-mail, Ty
klikasz „Dalej”, skrypt wpisuje hasło i zaznacza „Nie wylogowuj mnie”, Ty klikasz
„Zaloguj się”. Captchę i kod OTP/2FA obsługujesz sam. Masz na to 5 minut.

Amazon czasem prosi o hasło albo captchę także w trakcie przeglądania (np. przy
starszych zamówieniach). Skrypt wtedy czeka, wpisuje hasło i wraca do przerwanej strony.

Paski „Chrome do testowania…” i „nieobsługiwana flaga --no-sandbox” są niegroźne.
Skrypt używa Chromium dostarczanego z Playwright, a Ubuntu (AppArmor) nie pozwala
mu działać z sandboxem.

## Wynik

`dane/amazon_purchases/items.csv`: jedna pozycja na wiersz, kolumny `order_id`, `date`,
`seller`, `order_status` (OK, RETURNED, PARTIALLY_RETURNED, CANCELLED,
PARTIALLY_CANCELLED), `order_total` (łączna kwota zamówienia), `asin`, `title`,
`quantity`, `unit_price` (brutto), `currency`, `delivery_cost` (zapłacona dostawa po
rabacie „Darmowa dostawa”, brutto), `item_status` (`returned`, `cancelled` albo puste),
`shipment_status` (tekst statusu przesyłki z Amazona, np. „Odebrano w dniu: 18
listopada”), `returned_quantity`, `order_refund_total` (suma zwrotów pieniędzy
w zamówieniu), `return_info` (np. „Termin zwrotu upłynął w dniu 10 stycznia 2026”)
i `url`. Pełne dane zamówień, łącznie z wierszami „Podsumowania zamówienia”, są
w `orders.json`.

`dane/amazon_purchases/items_kept.csv` zawiera tylko to, co zostało u Ciebie. Pomija
anulowane pozycje i zwrócone sztuki, także zwroty jeszcze w toku. Kolumny: `order_id`,
`date`, `asin`, `title`, `quantity`, `unit_price`, `total` (= ilość × cena), `currency`,
`url`, `seller` (sprzedawca). Koszt dostawy nie jest wliczony.

`dane/amazon_purchases/zakupy.csv` to uproszczona wersja `items_kept.csv` do czytania,
w tym samym formacie co w Allegro (wspólny kod w `../common.py`). Nazwa produktu jest
zapisana małymi literami, z pojedynczymi spacjami i zawsze w cudzysłowie, bo bywają
w niej przecinki. Ma kolumny `nr`, `data`, `produkt`, `liczba`, `cena` (= liczba × cena
jednostkowa), `cena jednostkowa` i `sklep` (zawsze `amazon`). Kwoty są w PLN, a najnowszy
zakup jest na górze. `nr` to numer zakupu: 1 to najstarszy, a numery idą kolejno bez luk.
Pozycje z jednego zamówienia mają ten sam numer. Numery są liczone tylko z zachowanych
zakupów, więc mogą się przesunąć, jeśli później zwrócisz całe zamówienie. Amazon podaje
tylko dzień zamówienia (bez godziny), traktowany jako dzień w czasie polskim. Kwoty
w innych walutach są przeliczane na PLN po średnim kursie NBP z dnia roboczego przed
zakupem. Kursy są zapisywane w `dane/amazon_raw/nbp_rates.json`.

`dane/amazon_purchases/wydatki_miesiecznie.png` to wykres kolumnowy wydatków
w kolejnych miesiącach, taki sam jak w Allegro (wspólny kod w `../common.py`), policzony
z `zakupy.csv` (z uwzględnieniem zwrotów, bez anulowanych zamówień i dostawy). Miesiące
bez zakupów mają 0 zł. W kolumnach, które mają co najmniej 10% wysokości najwyższej,
trzy najdroższe pozycje są zaznaczone odcieniami niebieskiego. Pod wykresem jest lista
pogrupowana po miesiącach: wyróżnione pozycje oraz wszystkie pozostałe droższe niż
100 zł. Progi ustawiają `TOP_N`, `TOP_MIN_SHARE` i `LIST_MIN_PRICE` w `../common.py`.

Na koniec `--rebuild` i każde pobranie sprawdzają spójność danych: czy suma pozycji
+ zapłacona dostawa + pozostałe rabaty = „Łączna kwota” z podsumowania zamówienia.
Gdy podsumowanie podaje kwoty netto (wiersz „Suma bez VAT”), wysyłka jest przeliczana
na brutto. Skrypt wypisuje też pozycje bez ceny i zwroty pieniędzy bez zwróconej
pozycji.

`uv run download_purchases.py --rebuild` odtwarza pliki wynikowe z `dane/amazon_raw/` bez
logowania, np. po zmianie kolumn albo poprawce parsera.

## Jak działa pobieranie

Amazon nie ma API z historią zamówień, a strony zamówień są zwykłym HTML-em
renderowanym po stronie serwera. Skrypt otwiera listę zamówień osobno dla każdego roku
z filtra (`timeFilter=year-2025`, a przy pełnym przeglądaniu także `archived`) i przechodzi
po kolejnych stronach (po 10 zamówień). Z każdej strony zapisuje uproszczony HTML: tylko
główny kontener, bez skryptów, stylów, obrazków (zostaje ich `alt`), rekomendacji,
komentarzy i losowych identyfikatorów, z linkami bez parametrów śledzenia. Zamówienia są wyciągane z tego HTML-a dopiero potem
(BeautifulSoup), więc poprawka parsera nie wymaga ponownego pobierania. Wyniki są
łączone ze wszystkimi wcześniejszymi pobraniami z `dane/amazon_raw/`.

Pierwsze uruchomienie przegląda wszystkie lata. Znacznik `dane/amazon_raw/.history_complete`
zapisuje, że historia jest kompletna. Kolejne uruchomienia przeglądają listę tylko do
30 dni przed najnowszym zapisanym zakupem (`--days N` zmienia ten okres). Dzięki temu
łapią nowe zakupy oraz zwroty i anulowania w ostatnich zamówieniach. `--full`
przegląda wszystkie lata i odświeża szczegóły wszystkich zamówień.

## Szczegóły zamówień i zwroty

Lista zamówień podaje tylko sumę zamówienia i status przesyłek. Ceny pozycji, ilości
i to, które produkty wróciły, są dopiero na stronie „Szczegóły zamówienia”. Dlatego
skrypt otwiera ją dla każdego zamówienia i zapisuje jej uproszczony HTML w
`dane/amazon_raw/details/<nr_zamówienia>.json`. Szczegóły są pobierane dla nowych
zamówień i dla tych z przeglądanego okresu (zwrot mógł się w tym czasie zmienić).
Plik zmienia się tylko wtedy, gdy zmieni się treść strony.

Na stronie szczegółów produkty są pogrupowane w przesyłki, a każda przesyłka ma
status („Odebrano w dniu…”, „Zwrot zakończony”, „Zwrócono pieniądze”, „Anulowano”).
Pola pozycji są oznaczone atrybutami `data-component` (`itemTitle`, `unitPrice`,
`quantity`, `orderedMerchant`, `itemReturnEligibility`, `shipmentStatus`). Zwrotów
i anulowań nie dało się jeszcze sprawdzić na prawdziwej stronie (do listopada 2026
nie było takich zamówień), więc ich rozpoznawanie może wymagać poprawki.
Pozycja jest zwrócona, gdy jej przesyłka albo ona sama ma status zwrotu. Dzięki temu
zwrot kilku produktów z jednego zamówienia przypisuje się do właściwych pozycji, a
reszta zamówienia zostaje. Informacje „Możliwość wygasa w dniu …” i „Termin zwrotu
upłynął …” to tylko termin zwrotu i nie są liczone jako zwrot.

## Diagnostyka

`uv run download_purchases.py --discover` otwiera przeglądarkę i zapisuje do
`discovery/` pierwszą stronę listy zamówień z każdego roku, szczegóły zamówień ze
statusem zwrotu lub anulowania (i 3 zwykłych) oraz każdą stronę, którą sam otworzysz
(pełny HTML, uproszczony HTML, zrzut ekranu), aż naciśniesz Enter. Wypisuje też
wszystkie spotkane statusy przesyłek. Przydaje się, gdy Amazon
zmieni wygląd stron. Katalog zawiera dane osobowe, więc jest w `.gitignore`.

## Oficjalny eksport: „Pobierz swoje dane”

Amazon ma też oficjalny eksport danych (Konto → „Pobierz swoje dane” / *Request your
data*, kategoria „Twoje zamówienia”). Po potwierdzeniu e-mailem przysyła po kilku
godzinach lub dniach archiwum ZIP z plikami CSV, m.in. `Retail.OrderHistory.1.csv`
(pozycje zamówień) i plikami o zwrotach. To dobre źródło do jednorazowego sprawdzenia
całej historii, ale nie nadaje się do regularnego, automatycznego pobierania: każdy
eksport trzeba zamówić ręcznie, potwierdzić i czekać. Dlatego ten skrypt czyta strony
zamówień.
