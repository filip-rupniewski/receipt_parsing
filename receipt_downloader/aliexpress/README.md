# aliexpress: pobieranie listy zakupów

Pobiera zakupy z [Moje zamówienia](https://www.aliexpress.com/p/order/index.html) na AliExpress.

```
aliexpress.com ──download_purchases.py──▶ dane/aliexpress_raw/responses_<data>.json  (odpowiedzi JSON listy zamówień)
                                      └──▶ dane/aliexpress_raw/details/*.json        (szczegóły: kwoty, sklep, zwroty)
                                      └──▶ dane/aliexpress_purchases/                (orders.json, items.csv,
                                                                                     items_kept.csv, zakupy.csv,
                                                                                     wydatki_miesiecznie.png,
                                                                                     wydatki_miesiecznie_5_lat.png)
```

## Pierwsza konfiguracja

```
cd receipt_downloader/aliexpress
uv sync
uv run playwright install chromium
uv run download_purchases.py --save-credentials   # opcjonalnie, patrz niżej
```

## Dane osobowe i ich historia

Wszystkie pobrane dane są w `dane/`. Repozytorium z kodem ignoruje ten katalog
(`.gitignore`), a historia danych jest zapisywana we wspólnym, prywatnym repozytorium
projektu `.git-private` w katalogu głównym. To repozytorium nie ma remote, więc nic nie
trafia na GitHub (opis w głównym README). Po każdym pobraniu i po `--rebuild`, jeśli
coś się zmieniło, skrypt sam robi tam commit, np. „aliexpress: pobranie 2026-11-20 13:20:
zamówień 335, nowych 1, zmienione szczegóły zamówień: 3”. Historię oglądasz tak
(z katalogu głównego projektu):

```
git --git-dir=.git-private log --stat -- receipt_downloader/aliexpress/dane
git --git-dir=.git-private diff HEAD~1 -- receipt_downloader/aliexpress/dane/aliexpress_purchases/zakupy.csv
```

Jeśli `.git-private` nie istnieje, skrypt pomija commit. Katalog `dane/` jest też
na liście `DANE` w `zapisz_dane_prywatne.sh`.

Odpowiedzi AliExpress zawierają też dane, których skrypt nie potrzebuje. Przed
zapisem usuwa z nich adres dostawy, imię i nazwisko, telefon, dane karty i przelewu
zwrotu (początek numeru karty, numer ARN), kody kuponów na kolejne zakupy, ikony,
obrazki i parametry śledzenia w linkach (zostają tylko identyfikatory, np. `orderId`).

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

Chromium startuje jako zwykły proces z trwałym profilem (port CDP 9232, inny niż
Amazon 9231, Allegro 9230 i Biedronka 9229). Skrypt podłącza się do niego tylko na
chwilę, żeby wpisać dane, i sprawdza, czy wpisane wartości zostały w polach. Formularz
logowania AliExpress bywa w ramce (iframe), więc pól szukamy we wszystkich ramkach.
Logowanie bywa dwuetapowe: skrypt wpisuje e-mail, Ty klikasz „Kontynuuj”, skrypt
wpisuje hasło, Ty klikasz „Zaloguj się”. Suwak captchy i kod weryfikacyjny obsługujesz
sam. Masz na to 5 minut.

Jeśli skrypt nie rozpozna pola e-maila, wypisze atrybuty widocznych pól (bez wartości).
Wtedy wystarczy dopisać właściwy selektor do `EMAIL_SELECTORS` w skrypcie.

AliExpress czasem pokazuje suwak captchy albo okienko logowania także w trakcie
przeglądania. Skrypt wtedy czeka, wpisuje dane, jeśli je zna, i wraca do przerwanej
strony.

Paski „Chrome do testowania…” i „nieobsługiwana flaga --no-sandbox” są niegroźne.
Skrypt używa Chromium dostarczanego z Playwright, a Ubuntu (AppArmor) nie pozwala
mu działać z sandboxem.

## Wynik

`dane/aliexpress_purchases/items.csv`: jedna pozycja na wiersz, kolumny `order_id`,
`date`, `seller` (sklep), `order_status` (OK, RETURNED, PARTIALLY_RETURNED, DISPUTE,
CANCELLED), `status_text` (status z AliExpress: Completed, Awaiting delivery, Expired,
Canceled…), `order_total` (kwota zapłacona za zamówienie), `product_id`, `title`, `sku`
(wariant, np. „Color: Black”), `quantity`, `unit_price`, `currency`, `delivery_cost`,
`item_status` (`returned`, `partially_refunded`, `dispute`, `cancelled` albo puste),
`returned_quantity`, `refund_amount`, `refund_currency`, `order_refund_total`,
`return_info` (kwota, powód i stan zwrotu albo sporu), `payment_id` (płatność; kilka
zamówień opłaconych razem ma ten sam numer) i `url`. Pełne dane zamówień, łącznie
z wierszami rozbicia kwoty („Subtotal”, „Shipping”, „AliExpress Coupons”, „Estimated
import charges”…), są w `orders.json`.

`dane/aliexpress_purchases/items_kept.csv` zawiera tylko to, co zostało u Ciebie.
Pomija anulowane i nieopłacone zamówienia oraz zwrócone sztuki. Przy częściowym zwrocie
pieniędzy (np. rekompensata za brakującą część) pozycja zostaje, ale z ceną
pomniejszoną o zwrot. Kolumny: `order_id`, `date`, `seller`, `product_id`, `title`,
`sku`, `quantity`, `unit_price`, `total` (= ilość × cena), `currency`, `url`. Koszt
dostawy, cło i kupony AliExpress dla całego zamówienia nie są wliczone. Ceny są cenami
pozycji, tak jak w Allegro i Amazonie.

`dane/aliexpress_purchases/zakupy.csv` to uproszczona wersja `items_kept.csv` do
czytania, w tym samym formacie co w Allegro i Amazonie (wspólny kod w `../common.py`).
Nazwa produktu jest zapisana małymi literami, z pojedynczymi spacjami i zawsze
w cudzysłowie. Ma kolumny `nr`, `data`, `produkt`, `liczba`, `cena` (= liczba × cena
jednostkowa), `cena jednostkowa` i `sklep` (zawsze `aliexpress`). Kwoty są w PLN,
a najnowszy zakup jest na górze. `nr` to numer zakupu: 1 to najstarszy, a numery idą
kolejno bez luk. Pozycje z jednego zamówienia mają ten sam numer. Numery są liczone
tylko z zachowanych zakupów, więc mogą się przesunąć, jeśli później zwrócisz całe
zamówienie. AliExpress podaje ceny w walucie, w której płaciłeś (PLN, CHF, EUR, USD).
Kwoty są przeliczane na PLN po średnim kursie NBP z dnia roboczego przed zakupem.
Kursy są zapisywane w `dane/aliexpress_raw/nbp_rates.json`.

Daty: AliExpress podaje tylko dzień zamówienia, bez godziny. Strona wysyła do API
strefę czasową przeglądarki (`timeZone`), więc dzień jest już liczony w czasie polskim.
Dla zamówień z czasu letniego przesunięcie może wynosić najwyżej godzinę.

`dane/aliexpress_purchases/wydatki_miesiecznie.png` to wykres kolumnowy wydatków
w kolejnych miesiącach, taki sam jak w Allegro i Amazonie (wspólny kod
w `../common.py`), policzony z `zakupy.csv`. Przy historii dłuższej niż 4 lata oś
pokazuje tylko kolejne lata.

`dane/aliexpress_purchases/wydatki_miesiecznie_5_lat.png` to ten sam wykres, ale tylko
z ostatnich 5 lat (60 miesięcy). Pełna historia sięga 2015 roku, więc na pierwszym
wykresie ostatnie zakupy są słabo widoczne. Okres jest liczony od miesiąca najnowszego
zakupu, a nie od dzisiejszej daty, więc `--rebuild` daje ten sam plik niezależnie od
dnia uruchomienia. Liczbę lat ustawia `CHART_RECENT_YEARS` w skrypcie.

`--rebuild` i każde pobranie sprawdzają spójność danych: czy suma pozycji + dostawa
+ podatek/cło + rabaty (kupony, monety) = kwota zapłacona za zamówienie. Ukryty
wiersz dostawy, którego AliExpress nie doliczył do sumy (np. „Shipping 0.02”), nie
jest liczony jako koszt. Najstarsze zamówienia (2015–2019) nie mają rozbicia kwot,
więc ich nie da się sprawdzić. Skrypt wypisuje ich liczbę, zwroty pieniędzy
nieprzypisane do pozycji, podsumowanie zwrotów i sporów, nieznane części strony
szczegółów i wszystkie statusy zamówień.

`uv run download_purchases.py --rebuild` odtwarza pliki wynikowe z `dane/aliexpress_raw/`
bez logowania, np. po zmianie kolumn albo poprawce parsera.

## Jak działa pobieranie

Strony zamówień AliExpress są budowane w przeglądarce z odpowiedzi JSON wewnętrznego
API (`acs.aliexpress.com/h5/mtop.aliexpress.trade.buyer.order.list` i `….order.detail`).
To format „ultron”: `data.data` to słownik komponentów `{tag, fields}`, np.
`pc_om_list_order` (zamówienie na liście), `detail_product_block` (pozycje i sklep),
`detail_order_price_block` (rozbicie kwoty). Skrypt nie woła tego API sam (wymaga
podpisanych zapytań). Otwiera listę zamówień, klika „View orders” (po 10 zamówień)
i zapisuje odpowiedzi, które strona sama pobiera. Zamówienia z zakładki „Deleted
orders” też są zakupami, więc przy pełnym przeglądaniu skrypt przechodzi również
przez nią. Wyniki są łączone ze wszystkimi wcześniejszymi pobraniami
z `dane/aliexpress_raw/`.

Pierwsze uruchomienie przegląda całą listę (w listopadzie 2026: 36 stron, zamówienia
od 2015 roku). Znacznik `dane/aliexpress_raw/.history_complete` zapisuje, że historia
jest kompletna. Kolejne uruchomienia przeglądają listę tylko do 30 dni przed
najnowszym zapisanym zakupem (`--days N` zmienia ten okres). Dzięki temu łapią nowe
zakupy oraz zwroty i spory w ostatnich zamówieniach. `--full` przegląda całą listę
i odświeża szczegóły wszystkich zamówień.

## Szczegóły zamówień i zwroty

Lista podaje ceny pozycji i sumę zamówienia, ale nie dostawę, kupony, monety, cło
ani zwroty. Dlatego skrypt otwiera szczegóły każdego zamówienia i zapisuje ich
komponenty w `dane/aliexpress_raw/details/<nr_zamówienia>.json`. Szczegóły są
pobierane dla nowych zamówień i dla tych z przeglądanego okresu. Plik nie zawiera
czasu pobrania ani tokenów, więc zmienia się tylko wtedy, gdy zmieni się treść.

Zwrot pieniędzy nie jest widoczny w statusie zamówienia (zostaje „Completed”) ani
w rozbiciu kwoty. Pozycja ze zwrotem lub sporem ma za to przyciski „See request”
(`END_ISSUE`) i „Refund information” (`REFUND_INFO`), prowadzące do stron
`/p/refund-dispute/…` i `/p/reverse-pages/…`. Skrypt otwiera te strony i zapisuje
w `refund_pages` odpowiedzi ich API:

- `mtop.aliexpress.buyer.reverse.refundrecord.querydetail`: zwrot pieniędzy, czyli
  kwota, liczba sztuk, `refundStatus` (`FINISHED`) i źródło („Refunds with no return
  needed”, „Order closed”),
- `mtop.aliexpress.buyer.reverse.reverseorderlinerenderforbuyer`: spór, czyli powód,
  rozwiązanie („Refund” / „No refund”), kwota i stan („Request complete”).

Oba API podają `tradeOrderLineId`, więc zwrot przypisuje się do właściwej pozycji
także w zamówieniu z kilkoma produktami. Ten sam zwrot bywa w obu API, więc są
łączone po `reverseOrderLineId`. Sam przycisk „Refund information” nie oznacza zwrotu:
w starszych zamówieniach prowadzi tylko do formularza zgłoszenia, a spór może
skończyć się kwotą 0. Zasady:

- zakończony zwrot równy wartości pozycji oznacza, że pozycja jest zwrócona (także przy „zwrocie
  bez odsyłania”, bo pieniądze wróciły); równy k × cena sztuki oznacza k zwróconych sztuk,
- mniejsza kwota oznacza częściowy zwrot pieniędzy (`partially_refunded`): pozycja zostaje
  w `items_kept.csv` z ceną pomniejszoną o zwrot,
- niezakończony spór oznacza `dispute`: pozycja zostaje, informacja jest w `return_info`,
- zamówienie „Expired” (nieopłacone w terminie), „Canceled” albo „Closed” nie jest
  zakupem, nawet jeśli pobrane pieniądze zostały potem zwrócone.

Każde zamówienie AliExpress ma jeden sklep. Koszyk z produktami z kilku sklepów
staje się kilkoma zamówieniami opłaconymi jedną płatnością (wspólny `payment_id`
i `orders_in_payment` w `orders.json`).

## Diagnostyka

`uv run download_purchases.py --discover` otwiera przeglądarkę i zapisuje do
`discovery/` strony listy zamówień (kilka stron „View all” i każdą zakładkę, także
„Deleted orders”), szczegóły zamówień nietypowych (zwrot, spór, anulowanie, usunięte)
i 4 zwykłych, strony sporów z linków oraz każdą stronę, którą sam otworzysz (pełny
HTML, uproszczony HTML, zrzut ekranu), aż naciśniesz Enter. Zapisuje też odpowiedzi
JSON bez telemetrii (`network_log.json`), wypisuje statusy zamówień i liczbę
odpowiedzi każdego API. Przydaje się, gdy AliExpress zmieni format danych. Katalog
zawiera dane osobowe (także adres), więc jest w `.gitignore`.

## Oficjalny eksport danych (RODO)

AliExpress ma eksport danych osobowych w Privacy Center (privacy.aliexpress.com →
„Get a copy of your personal data” → „Create File”). Według polityki prywatności
obejmuje on historię zakupów, ale AliExpress nie opisuje formatu, pól (ceny, kupony,
monety, zwroty) ani zakresu czasu. Jedyna znana relacja to skarga noyb z lipca 2025:
plik dało się otworzyć tylko raz i był uszkodzony. Każdy eksport trzeba zamówić
ręcznie i na niego czekać, więc nie nadaje się do regularnego pobierania. Dlatego
ten skrypt czyta strony zamówień. Eksport można zamówić raz, żeby porównać go
z wynikiem skryptu.
