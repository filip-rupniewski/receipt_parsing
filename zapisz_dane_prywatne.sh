#!/usr/bin/env bash
# Zapisuje (commituje) dane osobowe projektu w prywatnym, lokalnym repo .git-private
# (bez remote - nic nie wychodzi poza ten komputer). Kod jest w zwykłym .git.
#
#   ./zapisz_dane_prywatne.sh ["opis zmian"]
#
# Podgląd historii:  git --git-dir=.git-private log --stat
set -euo pipefail
cd "$(dirname "$0")"
G=(git --git-dir=.git-private --work-tree=.)

# katalogi i pliki z danymi osobowymi (zdjęcia paragonów, pobrane zakupy, wyniki)
DANE=(
  "paragony"
  "paragony stare"
  "archives"
  "denner" "lidl" "lidl_wyrazne" "migros"
  "output_data"
  "biedronka_receipt_downloader/biedronka_receipts_jsons"
  "biedronka_receipt_downloader/biedronka_receipts_parsed"
  "biedronka_receipt_downloader/analiza_wydatkow.md"
  "receipt_downloader/allegro/dane"
)
# nigdy: hasła, sesje przeglądarek, pliki tymczasowe
POMIN=(
  ':(exclude,glob)**/credentials.json'
  ':(exclude,glob)**/.browser_profile/**'
  ':(exclude,glob)**/__pycache__/**'
  ':(exclude,glob)**/*.pyc'
  ':(exclude,glob)**/*.part'
)

ISTNIEJACE=()
for p in "${DANE[@]}"; do [ -e "$p" ] && ISTNIEJACE+=("$p"); done
"${G[@]}" add -f -A -- "${ISTNIEJACE[@]}" "${POMIN[@]}"
if "${G[@]}" diff --cached --quiet; then
  echo "Dane bez zmian."
  exit 0
fi
"${G[@]}" diff --cached --shortstat
"${G[@]}" commit -q -m "${1:-dane $(date '+%Y-%m-%d %H:%M')}"
"${G[@]}" log --oneline -1
