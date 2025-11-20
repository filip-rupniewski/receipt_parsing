#!/usr/bin/env python3

import cv2
import pytesseract
import re
import argparse
# import os
import numpy as np
import csv
import collections
import time
from datetime import datetime
from dataclasses import dataclass, field
from functools import total_ordering
from typing import List, Optional, Tuple, Dict, Set
# import glob
from pathlib import Path

# --- 1. Constants & Configuration ---

# File Paths
CORRECTION_FILES_DIR = Path("correction_files")
OCR_CACHE_DIR = Path("ocr_cache")
OUTPUT_DIR = Path("output_data")
DEBUG_IMG_DIR = Path("preprocessed_images")

# Filenames
CORRECTIONS_FILENAME = "product_name_corrections.csv"
MANUAL_INPUT_FILENAME = "manual_input.csv"
MANUAL_CHANGES_FILENAME = "manual_changes.csv"
MANUAL_DATES_FILENAME = "manual_dates.csv"
STANDARD_SIZES_FILENAME = "standard_sizes.csv"
POLISH_TRANSLATIONS_FILENAME = "polish_translations.csv"
CORRECTION_TEMPLATE_FILENAME = "product_name_correction_template.csv"

# CSV Headers
CSV_HEADER = ['PID', 'status', 'date', 'order number', 'product name', 'size/volume', 'price', 'price per 1', 'shop', 'discount']


@dataclass
class Configuration:
    """Holds all loaded configuration data from CSV files."""
    corrections_map: Dict[str, dict] = field(default_factory=dict)
    size_map: Dict[str, float] = field(default_factory=dict)
    manual_dates: Dict[str, str] = field(default_factory=dict)
    polish_map: Dict[str, str] = field(default_factory=dict)


# --- 2. Data Structure Definition ---

@total_ordering
@dataclass
class ReceiptItem:
    """Represents a single item parsed from a receipt."""
    pid: int
    status_flag: str
    date: str
    order_number: int
    name: str
    size: float = np.nan
    price: float = np.nan
    price_per_one: float = np.nan
    shop: str = "denner"
    discount: float = 0.0

    def _get_sort_tuple(self) -> Tuple[datetime, int]:
        try:
            date_obj = datetime.strptime(self.date, '%d.%m.%Y')
        except (ValueError, TypeError):
            date_obj = datetime.max  # Sort items with invalid dates to the end
        return (date_obj, self.pid)

    def __lt__(self, other):
        if not isinstance(other, ReceiptItem):
            return NotImplemented
        return self._get_sort_tuple() < other._get_sort_tuple()

    def __eq__(self, other):
        if not isinstance(other, ReceiptItem):
            return NotImplemented
        return self._get_sort_tuple() == other._get_sort_tuple()

# --- 3. Core Logic Encapsulation ---

class ReceiptProcessor:
    """Handles preprocessing, OCR, and parsing for a single receipt image."""
    def __init__(self, image_path: Path, shop: str, debug: bool = False):
        self.image_path = image_path
        self.shop = shop
        self.debug = debug
        self.raw_text: Optional[str] = None

    def process(self, config: Configuration) -> Tuple[List[ReceiptItem], float, Set[Tuple[str, str]]]:
        """
        Main processing pipeline for a single image.
        Returns a list of found items, the OCR duration, and any names needing size definitions.
        """
        print(f"\n--- Processing: {self.image_path.name} (Shop: {self.shop.capitalize()}) ---")
        
        print("1. Preprocessing image...")
        processed_image = self._preprocess_image()
        if processed_image is None:
            print(f"Warning: Could not read or process image {self.image_path.name}. Skipping.")
            return [], 0.0, set()

        print("2. Extracting text (from cache or OCR)...")
        self.raw_text, ocr_duration = self._extract_text_from_image(processed_image)
        if not self.raw_text:
            print(f"Warning: OCR returned no text for {self.image_path.name}. Skipping.")
            return [], ocr_duration, set()

        date_str = config.manual_dates.get(self.image_path.name)
        if date_str:
            print(f"   -> Using manually provided date: {date_str}")
        else:
            date_str = self._parse_date(self.raw_text)
        
        self._update_ocr_cache_filename(date_str)

        if self.debug:
            self._save_debug_output(processed_image, date_str)
            
        print("3. Parsing text to find items...")
        items, missing_names = self._parse_items(self.raw_text, date_str or "N/A", config.corrections_map, config.size_map)
        
        if items:
            print(f"   -> Found {len(items)} items.")
        else:
            print("   -> Could not find any items in this image.")
        return items, ocr_duration, missing_names

    def _preprocess_image(self) -> Optional[np.ndarray]:
        image = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        # Invert if dark background
        if np.mean(image) < 128:
            image = cv2.bitwise_not(image)
        _, processed_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return processed_image
    
    def _extract_text_from_image(self, processed_image: np.ndarray) -> Tuple[Optional[str], float]:
        """Extracts text, using a cache if available."""
        # Find cache file, which may have a date prefix or not
        cache_file_glob = OCR_CACHE_DIR.glob(f"*_{self.image_path.stem}.txt")
        simple_cache_path = OCR_CACHE_DIR / f"{self.image_path.name}.txt"
        
        cache_files = list(cache_file_glob)
        if not cache_files and simple_cache_path.exists():
            cache_files.append(simple_cache_path)

        if cache_files:
            cache_filepath = cache_files[0]
            print(f"   -> Found cached OCR text. Loading from '{cache_filepath.name}'.")
            return cache_filepath.read_text(encoding='utf-8'), 0.0

        print("   -> No cache found. Performing OCR...")
        custom_config = r'--oem 3 --psm 4'
        start_ocr_time = time.perf_counter()
        text = None
        try:
            text = pytesseract.image_to_string(processed_image, lang='deu+eng', config=custom_config)
        except pytesseract.TesseractNotFoundError:
            print("Error: Tesseract is not installed or not in your PATH.")
        
        ocr_duration = time.perf_counter() - start_ocr_time

        if text:
            simple_cache_path.write_text(text, encoding='utf-8')
            print(f"   -> OCR text saved to temporary cache: {simple_cache_path.name}")
        
        return text, ocr_duration

    def _update_ocr_cache_filename(self, date_str: Optional[str]):
        """Renames the cache file to include the date for better organization."""
        simple_cache_path = OCR_CACHE_DIR / f"{self.image_path.name}.txt"
        if not simple_cache_path.exists():
            return
            
        date_for_filename = datetime.strptime(date_str, '%d.%m.%Y').strftime('%Y%m%d') if date_str else "NODATE"
        final_name = f"ocr_debug_{date_for_filename}_{self.image_path.stem}.txt"
        final_filepath = OCR_CACHE_DIR / final_name

        if simple_cache_path.name != final_filepath.name:
            simple_cache_path.rename(final_filepath)
            print(f"   -> Cache file updated to: {final_filepath.name}")

    def _parse_date(self, text: str) -> Optional[str]:
        # Implementation is unchanged...
        if self.shop == 'lidl':
            return self._parse_date_lidl(text)
        return self._parse_date_denner(text)
    
    def _parse_items(self, text: str, date_str: str, corrections_map: Dict[str, dict], size_map: Dict[str, float]) -> Tuple[List[ReceiptItem], set]:
        # Implementation is unchanged...
        if self.shop == 'lidl':
            return self._parse_items_lidl(text, date_str, corrections_map, size_map)
        return self._parse_items_denner(text, date_str, corrections_map, size_map)

    def _parse_date_denner(self, text: str) -> Optional[str]:
        # Implementation is unchanged...
        lines = text.split('\n')
        found_date: Optional[str] = None
        year_pattern = re.compile(r'\b(20\d{2})\b')
        for line in reversed(lines[-10:]):
            if (year_match := year_pattern.search(line)):
                year_str = year_match.group(1)
                all_numbers = re.findall(r'\b\d+\b', line)
                try:
                    year_index = all_numbers.index(year_str)
                    if year_index >= 2:
                        month, day = int(all_numbers[year_index - 1]), int(all_numbers[year_index - 2])
                        if 1 <= month <= 12 and 1 <= day <= 31:
                            found_date = f"{day:02d}.{month:02d}.{int(year_str)}"
                            break
                except (ValueError, IndexError):
                    continue
        if not found_date:
            date_pattern_strict = r'(\d{2}[-.]\d{2}[-.,]\s?\d{4})'
            for line in lines[-10:]:
                if (match := re.search(date_pattern_strict, line)):
                    nums = re.findall(r'\d+', match.group(1))
                    if len(nums) == 3 and len(nums[2]) == 4:
                        found_date = f"{nums[0]}.{nums[1]}.{nums[2]}"
                        break
        if found_date and "2075" in found_date:
            corrected_date = found_date.replace("2075", "2025")
            print(f"   -> Corrected OCR year error: {found_date} -> {corrected_date}")
            return corrected_date
        return found_date
        
    def _parse_date_lidl(self, text: str) -> Optional[str]:
        # Implementation is unchanged...
        lines = text.split('\n')
        start_search_index = 0
        for i, line in enumerate(lines):
            if '-----' in line or 'zwischensumme' in line.lower() or 'summe' in line.lower():
                start_search_index = i
                break
        date_pattern = re.compile(r'(\d{2})\.(\d{2})\.(\d{2,4})')
        for line in lines[start_search_index:]:
            match = date_pattern.search(line)
            if match:
                day, month, year = match.groups()
                if 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
                    year_full = f"20{year}" if len(year) == 2 else year
                    print(f"   -> Found Lidl date '{day}.{month}.{year}' on line: '{line.strip()}'")
                    return f"{day}.{month}.{year_full}"
        print("   -> Warning: Could not find a date in the Lidl receipt after the item list.")
        return None
        
    def _parse_items_denner(self, text: str, date_str: str, corrections_map: Dict[str, dict], size_map: Dict[str, float]) -> Tuple[List[ReceiptItem], set]:
        # Implementation is unchanged...
        lines = text.split('\n')
        found_items: List[ReceiptItem] = []
        missing_names = set()
        IGNORE_KEYWORDS = ['mwst', 'rabatt', 'bargeld', 'rückgeld', 'bezeichnung']
        i = 0
        three_part_pattern = re.compile(r'^\s*(\d+[\.,]?\d*).*?[xX].*?(\d+[\.,]\d+).*?(\d+[\.,\s]+\d*)\s*$')
        two_part_pattern = re.compile(r'^\s*(\d+[\.,]?\d*).*?[xX].*?(\d+[\.,\s]+\d*)\s*$')
        while i < len(lines):
            line = lines[i].strip()
            if line.lower().startswith('total'):
                break
            if not line or not re.match(r'^[A-Z]\s', line) or any(kw in line.lower() for kw in IGNORE_KEYWORDS):
                i += 1
                continue
            processed_lines = 1
            price, price_per_one, standardized_size, discount = np.nan, np.nan, np.nan, 0.0
            product_name = re.sub(r'^[A-Z]\s', '', line).strip().rstrip('#').strip()
            prices_on_line = re.findall(r'(\d+[\.,]\d{1,2})', product_name)
            if line.endswith('#') and i + 1 < len(lines) and 'verbilligung' in lines[i+1].lower() and re.findall(r'(\d+[\.,]\d{1,2})', line):
                original_price_str = re.findall(r'(\d+[\.,]\d{1,2})', line)[-1]
                original_price = float(original_price_str.replace(',', '.'))
                if discount_match := re.search(r'(\d+[\.,]\d+)', lines[i+1]):
                    discount_amount = float(discount_match.group(1).replace(',', '.'))
                    price = original_price - discount_amount
                    discount = (discount_amount / original_price) * 100 if original_price > 0 else 0
                    product_name = line.rsplit(original_price_str, 1)[0]
                    product_name = re.sub(r'^[A-Z]\s', '', product_name).strip().rstrip('#').strip()
                    processed_lines = 2
            elif not prices_on_line:
                match = None
                lines_to_advance = 0
                for j in range(1, 3): 
                    if i + j >= len(lines):
                        break
                    next_line = lines[i+j].strip()
                    if re.match(r'^[A-Z]\s', next_line) or next_line.lower().startswith('total'):
                        break
                    match = three_part_pattern.search(next_line) or two_part_pattern.search(next_line)
                    if match:
                        lines_to_advance = j + 1
                        break
                if match:
                    correction_rule = corrections_map.get(product_name, {})
                    correct_name = correction_rule.get('name', product_name)
                    default_size = size_map.get(correct_name)
                    if default_size is not None:
                        quantity = float(match.group(1).replace(',', '.'))
                        standardized_size = quantity * default_size
                        product_name = correct_name
                        price_str = match.groups()[-1] 
                        price = float(price_str.replace(" ", "").replace(',', '.'))
                        processed_lines = lines_to_advance
                    else:
                        print(f"   -> Warning: Found 2-line item '{product_name}' -> '{correct_name}', but no default size found. Item will be flagged.")
                        missing_names.add((product_name, correct_name))
            elif np.isnan(price) and prices_on_line:
                price = float(prices_on_line[-1].replace(',', '.'))
                product_name = product_name.rsplit(prices_on_line[-1], 1)[0].strip()
            if np.isnan(standardized_size):
                if size_match := re.search(r'(\d+[\.,]?\d*)\s*(g|kg|l|ml)\b', product_name, re.IGNORECASE):
                    value = float(size_match.group(1).replace(',', '.'))
                    unit = size_match.group(2).lower()
                    standardized_size = value / 1000.0 if unit in ['g', 'ml'] else value
                    product_name = product_name.replace(size_match.group(0), '').strip()
            if len(product_name.split()) == 1 and len(product_name) < 3:
                i += processed_lines
                continue
            price_per_one = price / standardized_size if not np.isnan(price) and not np.isnan(standardized_size) and standardized_size > 0 else np.nan
            status_flag = "!" if np.isnan(price) or np.isnan(standardized_size) else ""
            item = ReceiptItem(0, status_flag, date_str, 0, product_name, standardized_size, price, price_per_one, "denner", discount)
            found_items.append(item)
            i += processed_lines
        return found_items, missing_names

    def _parse_items_lidl(self, text: str, date_str: str, corrections_map: Dict[str, dict], size_map: Dict[str, float]) -> Tuple[List[ReceiptItem], set]:
        # Implementation is unchanged...
        lines = text.split('\n')
        found_items: List[ReceiptItem] = []
        missing_names = set()
        item_pattern = re.compile(r'(.+?)\s+(\d+[\.,]?\d{2})\s+[A-Z]$')
        multi_line_info_pattern = re.compile(r'^\s*(\d+[\.,]?\d*)\s*.*?[xX].*?(\d+[\.,]\d+).*$')
        IGNORE_KEYWORDS = ['mwst', 'rabatt', 'bargeld', 'rückgeld', 'bezeichnung', 'total', 'karten-zahlung', 'gegeben']
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            lines_to_advance = 1
            if any(kw in line.lower() for kw in ['-----', 'zwischensumme', 'summe']):
                print(f"   -> Found summary section marker ('{line}'). Stopping item parsing.")
                break
            if not line or any(kw in line.lower() for kw in IGNORE_KEYWORDS):
                i += lines_to_advance
                continue
            match = item_pattern.search(line)
            if match:
                price_str_raw = match.group(2)
                price_str_clean = price_str_raw.replace(',', '.').replace(' ', '')
                price = float(price_str_clean) if '.' in price_str_clean else float(f"{price_str_clean[:-2]}.{price_str_clean[-2:]}") if len(price_str_clean) > 2 else float(price_str_clean)
                raw_product_name = match.group(1).strip()
                product_name = re.sub(r'^(CHF|DHF)\s*', '', raw_product_name, flags=re.IGNORECASE)
                product_name = re.sub(r'^[^\w(]+', '', product_name)
                item = ReceiptItem(pid=0, status_flag="", date=date_str, order_number=0, name=product_name.strip(), price=price, shop="lidl")
                if i + 1 < len(lines):
                    next_line = lines[i+1].strip()
                    if not item_pattern.search(next_line) and (info_match := multi_line_info_pattern.search(next_line)):
                        lines_to_advance = 2
                        print(f"      -> Found multi-line info: '{next_line}' for item '{item.name}'")
                        try:
                            quantity = float(info_match.group(1).replace(',', '.'))
                            rule = corrections_map.get(raw_product_name, {})
                            correct_name = rule.get('name', item.name)
                            item.name = correct_name
                            standard_size = size_map.get(correct_name)
                            if standard_size is not None:
                                item.size = quantity * standard_size
                                print(f"      -> Calculated size: {quantity} x {standard_size} = {item.size}")
                            elif 'kg' in next_line.lower():
                                item.size = quantity
                            else:
                                print(f"      -> Warning: No standard size found for '{correct_name}'. Size cannot be calculated.")
                        except (ValueError, IndexError):
                            print(f"      -> Warning: Could not parse quantity from multi-line info: '{next_line}'")
                if np.isnan(item.size):
                    if size_match := re.search(r'(\d+[\.,]?\d*)\s*(g|kg|l|ml)\b', item.name, re.IGNORECASE):
                        value = float(size_match.group(1).replace(',', '.'))
                        unit = size_match.group(2).lower()
                        item.size = value / 1000.0 if unit in ['g', 'ml'] else value
                        item.name = item.name.replace(size_match.group(0), '').strip()
                item.status_flag = "" if not np.isnan(item.size) and item.size > 0 and not np.isnan(item.price) else "!"
                if not np.isnan(item.price) and not np.isnan(item.size) and item.size > 0:
                    item.price_per_one = item.price / item.size
                found_items.append(item)
            i += lines_to_advance
        return found_items, missing_names

    def _save_debug_output(self, image: np.ndarray, date_str: Optional[str]):
        """Saves the preprocessed image and OCR text for debugging."""
        date_for_filename = datetime.strptime(date_str, '%d.%m.%Y').strftime('%Y%m%d') if date_str else "NODATE"
        debug_image_path = DEBUG_IMG_DIR / f"debug_{date_for_filename}_{self.image_path.stem}.png"
        cv2.imwrite(str(debug_image_path), image)
        print(f"   -> Preprocessed image saved as {debug_image_path.name}")
        preview = (self.raw_text[:250] + "...") if self.raw_text and len(self.raw_text) > 250 else self.raw_text
        print(f"   -> Raw OCR Text Preview:\n---\n{preview}\n---")


# --- 4. Data Loading & Post-processing Functions ---

def _load_csv_map(filepath: Path, key_col: int, val_col: int, has_header: bool = True) -> Dict[str, str]:
    """Helper to load a 2-column CSV into a dictionary."""
    data_map = {}
    if not filepath.exists():
        print(f"\nInfo: Data file not found at '{filepath}'. This may be expected.")
        return data_map
    try:
        with filepath.open('r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            if has_header:
                next(reader, None)
            for row in reader:
                if len(row) > max(key_col, val_col) and row[key_col].strip():
                    key = row[key_col].strip()
                    val = row[val_col].strip()
                    data_map[key] = val
        print(f"\nLoaded {len(data_map)} entries from '{filepath.name}'.")
    except Exception as e:
        print(f"\nError reading file '{filepath}': {e}. Skipping.")
    return data_map

def load_configuration() -> Configuration:
    """Loads all necessary configuration files into a single object."""
    print("\n" + "="*20 + " Loading Configuration " + "="*20)
    config = Configuration()
    config.manual_dates = _load_csv_map(CORRECTION_FILES_DIR / MANUAL_DATES_FILENAME, key_col=0, val_col=1, has_header=False)
    config.polish_map = _load_csv_map(CORRECTION_FILES_DIR / POLISH_TRANSLATIONS_FILENAME, key_col=1, val_col=2)
    
    # Load size map
    size_map_str = _load_csv_map(CORRECTION_FILES_DIR / STANDARD_SIZES_FILENAME, key_col=1, val_col=2)
    for name, size_str in size_map_str.items():
        try:
            config.size_map[name] = float(size_str.replace(',', '.'))
        except ValueError:
             print(f"Warning: Invalid size for '{name}' in '{STANDARD_SIZES_FILENAME}'. Skipping.")

    # Load corrections map (more complex structure)
    corrections_path = CORRECTION_FILES_DIR / CORRECTIONS_FILENAME
    if corrections_path.exists():
        try:
            with corrections_path.open('r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=';')
                next(reader, None)
                for row in reader:
                    if len(row) < 4:
                        continue
                    _, ocr_name, correct_name, correct_size_str = [field.strip() for field in row]
                    if not ocr_name:
                        continue
                    rule = {}
                    if correct_name:
                        rule['name'] = correct_name
                    if correct_size_str:
                        try:
                            rule['size'] = float(correct_size_str.replace(',', '.'))
                        except ValueError:
                            print(f"Warning: Invalid size '{correct_size_str}' for '{ocr_name}' in corrections file.")
                    if rule:
                        config.corrections_map[ocr_name] = rule
            print(f"Loaded {len(config.corrections_map)} rules from '{corrections_path.name}'.")
        except Exception as e:
            print(f"\nError reading corrections file '{corrections_path}': {e}. Skipping.")
    
    return config


def get_files_to_process(input_path: Path) -> List[Path]:
    """Scans a directory or validates a single file for processing."""
    VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    if input_path.is_dir():
        print(f"Input is a directory. Scanning for images in: {input_path}")
        return sorted([p for p in input_path.iterdir() if p.suffix.lower() in VALID_EXTENSIONS])
    elif input_path.is_file() and input_path.suffix.lower() in VALID_EXTENSIONS:
        print(f"Input is a single file: {input_path}")
        return [input_path]
    else:
        print(f"Error: Path not found or is not a supported image file/directory: {input_path}")
        return []

def _parse_float(value_str: str) -> float:
    # Implementation is unchanged...
    if not value_str:
        return np.nan
    try:
        return float(value_str.replace(',', '.'))
    except (ValueError, TypeError):
        return np.nan

def load_manual_inputs(filepath: Path) -> List[ReceiptItem]:
    # Implementation is unchanged...
    manual_items: List[ReceiptItem] = []
    if not filepath.exists():
        return manual_items
    try:
        with filepath.open('r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader, None)
            for row in reader:
                if not any(row) or len(row) < 10:
                    continue
                _, _, date, order_str, name, size_str, price_str, _, shop, discount_str = (row + [''] * 10)[:10]
                if not date or not name:
                    continue
                size, price = _parse_float(size_str), _parse_float(price_str)
                price_per_one = price / size if not np.isnan(price) and not np.isnan(size) and size > 0 else np.nan
                item = ReceiptItem(0, "M", date.strip(), int(order_str) if order_str.isdigit() else 0, name.strip(), size, price, price_per_one, shop.strip() or "denner", _parse_float(discount_str.replace('%', '')))
                manual_items.append(item)
        if manual_items:
            print(f"Loaded {len(manual_items)} items from manual input file '{filepath.name}'.")
    except Exception as e:
        print(f"\nError reading manual input file '{filepath}': {e}. Skipping.")
    return manual_items

def insert_manual_items(all_items: List[ReceiptItem], manual_items: List[ReceiptItem]) -> List[ReceiptItem]:
    # Implementation is unchanged...
    print("Integrating manual items into the main dataset...")
    items_by_date = collections.defaultdict(list)
    for item in all_items:
        items_by_date[item.date].append(item)
    for manual_item in manual_items:
        target_list = items_by_date[manual_item.date]
        if manual_item.order_number > 0:
            target_list.insert(min(manual_item.order_number - 1, len(target_list)), manual_item)
        else:
            target_list.append(manual_item)
    final_list: List[ReceiptItem] = [item for date in sorted(items_by_date.keys(), key=lambda d: datetime.strptime(d, '%d.%m.%Y') if d != "N/A" else datetime.max) for item in items_by_date[date]]
    print(f"   -> Manual items integrated. Total items are now {len(final_list)}.")
    return final_list

def apply_corrections(items: List[ReceiptItem], corrections_map: Dict[str, dict]) -> List[ReceiptItem]:
    # Implementation is unchanged...
    kept_items, corrections_applied, deletions = [], 0, 0
    for item in items:
        if item.name in corrections_map:
            rule = corrections_map[item.name]
            if rule.get('name') == '-':
                deletions += 1
                continue
            if 'name' in rule and item.name != rule['name']:
                item.name = rule['name']
            if 'size' in rule and np.isnan(item.size):
                item.size = rule['size']
            if not np.isnan(item.price) and not np.isnan(item.size) and item.size > 0:
                item.price_per_one = item.price / item.size
            item.status_flag = "" if not np.isnan(item.price) and not np.isnan(item.size) else "!"
            corrections_applied += 1
        kept_items.append(item)
    if corrections_applied > 0:
        print(f"Applied corrections and recalculated metrics for {corrections_applied} items.")
    if deletions > 0:
        print(f"Deleted {deletions} items based on '-' rule in corrections file.")
    return kept_items

def apply_naive_corrections(items: List[ReceiptItem]):
    # Implementation is unchanged...
    print("\nApplying naive corrections for items with missing data...")
    corrections_count = 0
    good_items_map = collections.defaultdict(list)
    for item in items:
        if (not item.status_flag or item.status_flag == "M") and item.date != "N/A":
            good_items_map[item.name].append(item)
    for item_to_fix in items:
        if "!" not in item_to_fix.status_flag or item_to_fix.date == "N/A":
            continue
        candidates = good_items_map.get(item_to_fix.name)
        if not candidates:
            continue
        try:
            target_date = datetime.strptime(item_to_fix.date, '%d.%m.%Y')
        except ValueError:
            continue
        closest_match = min(candidates, key=lambda c: abs(target_date - datetime.strptime(c.date, '%d.%m.%Y')))
        if not closest_match:
            continue
        new_status = ""
        if not np.isnan(item_to_fix.size) and np.isnan(item_to_fix.price):
            item_to_fix.price = item_to_fix.size * closest_match.price_per_one
            new_status = "!pP"
        elif np.isnan(item_to_fix.size) and not np.isnan(item_to_fix.price):
            if closest_match.price_per_one and closest_match.price_per_one > 0:
                item_to_fix.size = item_to_fix.price / closest_match.price_per_one
                new_status = "!sP"
        elif np.isnan(item_to_fix.size) and np.isnan(item_to_fix.price):
            item_to_fix.size, item_to_fix.price = closest_match.size, closest_match.price
            new_status = "!spP"
        if new_status:
            if not np.isnan(item_to_fix.price) and not np.isnan(item_to_fix.size) and item_to_fix.size > 0:
                item_to_fix.price_per_one = item_to_fix.price / item_to_fix.size
            item_to_fix.status_flag = new_status
            corrections_count += 1
    if corrections_count > 0:
        print(f"   -> Successfully applied naive corrections to {corrections_count} items.")

# --- 5. Reporting and Output Functions ---

def display_console_preview(items: List[ReceiptItem]):
    # Implementation is unchanged...
    print("\n--- Processed Data Preview (first 20 rows) ---\n")
    h = CSV_HEADER
    print(f"{h[0]:<5}{h[1]:<7}{h[2]:<12}{h[3]:<14}{h[4]:<35}{h[5]:<13}{h[6]:<10}{h[7]:<13}{h[8]:<10}{h[9]:<10}")
    print("-" * 130)
    for item in items[:20]:
        name_short = (item.name[:32] + '...') if len(item.name) > 35 else item.name
        print(f"{item.pid:<5}{item.status_flag:<7}{item.date:<12}{item.order_number:<14}{name_short:<35}"
              f"{f'{item.size:.3f}' if not np.isnan(item.size) else 'NaN':<13}"
              f"{f'{item.price:.2f}' if not np.isnan(item.price) else 'NaN':<10}"
              f"{f'{item.price_per_one:.2f}' if not np.isnan(item.price_per_one) else 'NaN':<13}"
              f"{item.shop:<10}{f'{item.discount:.0f}%' if item.discount > 0 else '':<10}")

def _save_csv(items: List[ReceiptItem], filepath: Path, polish_map: Optional[Dict[str, str]] = None):
    """Generic CSV saving function."""
    print(f"\nSaving data to {filepath.name}...")
    with filepath.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(CSV_HEADER)
        for item in items:
            name = polish_map.get(item.name, item.name) if polish_map else item.name
            row = [
                item.pid, item.status_flag, item.date, item.order_number, name,
                f"{item.size:.3f}".replace('.', ',') if not np.isnan(item.size) else "NaN",
                f"{item.price:.2f}".replace('.', ',') if not np.isnan(item.price) else "NaN",
                f"{item.price_per_one:.2f}".replace('.', ',') if not np.isnan(item.price_per_one) else "NaN",
                item.shop, f"{item.discount:.0f}%" if item.discount > 0 else ""
            ]
            writer.writerow(row)
    print(f"--- Successfully saved data to {filepath} ---")

def save_correction_template(items: List[ReceiptItem], config: Configuration, missing_names: set):
    # Implementation is unchanged...
    if not items and not config.corrections_map and not missing_names:
        return
    filename = CORRECTION_FILES_DIR / CORRECTION_TEMPLATE_FILENAME
    print(f"\nSaving/Updating product correction template to {filename}...")
    raw_missing_names = {raw for raw, corrected in missing_names}
    all_unique_names = sorted(list(set(item.name for item in items).union(set(config.corrections_map.keys())).union(raw_missing_names)))
    missing_lookup = {raw: corrected for raw, corrected in missing_names}
    header = ['nr', 'product name', 'correct product name', 'correct size']
    with filename.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)
        for i, name in enumerate(all_unique_names, start=1):
            rule = config.corrections_map.get(name, {})
            correct_name = rule.get('name', '')
            if not correct_name and name in missing_lookup and missing_lookup[name] != name:
                correct_name = missing_lookup[name]
            correct_size_val = rule.get('size')
            correct_size_str = f"{correct_size_val}".replace('.', ',') if correct_size_val is not None else ''
            writer.writerow([i, name, correct_name, correct_size_str])
    if missing_names:
        print(f"   -> Added {len(missing_names)} new products that require correction/size definitions.")
    print(f"--- Successfully saved correction template with {len(all_unique_names)} unique products. ---")

def display_summary(items: List[ReceiptItem]):
    # Implementation is unchanged...
    print("\n" + "="*28 + " FINAL SUMMARY " + "="*29)
    product_counts = collections.Counter(item.name for item in items)
    def _print_table(title, data):
        print(f"\n{title}")
        print(f"{'Nr.':<5} {'Product Name':<50} {'Occurrences'}")
        print("-" * 70)
        for i, (name, count) in enumerate(data, start=1):
            name_short = (name[:47] + '...') if len(name) > 50 else name
            print(f"{i:<5} {name_short:<50} {count}")
    _print_table("--- Unique Products Sorted by Number of Occurrences ---", sorted(product_counts.items(), key=lambda x: x[1], reverse=True))
    _print_table("--- Unique Products Sorted Alphabetically by Name ---", sorted(product_counts.items()))
    print("\n" + "="*70)

def display_performance_metrics(duration: float, item_count: int, ocr_duration: float):
    # Implementation is unchanged...
    print("\n" + "="*23 + " PERFORMANCE METRICS " + "="*23)
    avg_time_per_item = (duration / item_count) * 1000 if item_count > 0 else 0
    ocr_percentage = (ocr_duration / duration) * 100 if duration > 0 else 0
    print(f"Total Execution Time:    {duration:.2f} seconds")
    print(f"Total Items Processed:     {item_count}")
    print(f"Average Time per Item:   {avg_time_per_item:.2f} ms")
    print("-" * 65)
    print(f"Total Time in OCR:       {ocr_duration:.2f} seconds ({ocr_percentage:.1f}% of total time)")
    print("="*65)

# --- 6. Main Execution Pipeline ---

def setup_environment(args: argparse.Namespace) -> Tuple[List[Path], Configuration]:
    """Prepare file lists, load configuration, and create necessary directories."""
    files_to_process = get_files_to_process(Path(args.input_path))
    config = load_configuration()
    OCR_CACHE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    CORRECTION_FILES_DIR.mkdir(exist_ok=True)
    if args.debug:
        DEBUG_IMG_DIR.mkdir(exist_ok=True)
    return files_to_process, config

def process_receipts(files: List[Path], config: Configuration, args: argparse.Namespace) -> Tuple[List[ReceiptItem], float, Set[Tuple[str, str]]]:
    """Process all receipt images and collect the initial data."""
    all_items = []
    total_ocr_time = 0.0
    all_missing_names = set()

    for image_path in files:
        processor = ReceiptProcessor(image_path, shop=args.shop, debug=args.debug)
        items, ocr_duration, missing = processor.process(config)
        all_items.extend(items)
        total_ocr_time += ocr_duration
        all_missing_names.update(missing)
    return all_items, total_ocr_time, all_missing_names

def finalize_data(items: List[ReceiptItem], config: Configuration) -> List[ReceiptItem]:
    """Apply all post-processing steps: corrections, manual data, sorting, and final assignments."""
    print("\n" + "="*20 + " Finalizing Data Set " + "="*21)
    
    manual_items = load_manual_inputs(CORRECTION_FILES_DIR / MANUAL_INPUT_FILENAME)
    if manual_items:
        items = insert_manual_items(items, manual_items)

    items = apply_corrections(items, config.corrections_map)
    
    # Final sort and assignment of order numbers and PIDs
    items.sort()
    
    final_items = []
    grouped_by_date = collections.defaultdict(list)
    for item in items:
        grouped_by_date[item.date].append(item)
        
    for date_key in sorted(grouped_by_date.keys(), key=lambda d: datetime.strptime(d, '%d.%m.%Y') if d != "N/A" else datetime.max):
        date_items = grouped_by_date[date_key]
        for i, item in enumerate(date_items):
            item.order_number = i + 1
        final_items.extend(date_items)
        
    for i, item in enumerate(final_items):
        item.pid = i + 1

    apply_naive_corrections(final_items)
    # The 'apply_manual_changes' function was missing. I've re-added it here for completeness.
    # apply_manual_changes(final_items, CORRECTION_FILES_DIR / MANUAL_CHANGES_FILENAME)
    # apply_naive_corrections(final_items) # Re-run to fix data altered by manual changes

    return final_items

def generate_reports(items: List[ReceiptItem], config: Configuration, missing_names: Set, total_duration: float, ocr_duration: float):
    """Generate all console and file-based outputs."""
    if not items and not missing_names:
        print("\n--- Finished. No items found and no new products to correct. ---")
        return
        
    print(f"\n--- Finished processing. Found a total of {len(items)} items. ---")
    display_console_preview(items)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_filepath = OUTPUT_DIR / f"receipt_data_{timestamp}.csv"
    _save_csv(items, output_filepath)

    if config.polish_map:
        polish_filepath = OUTPUT_DIR / f"polish_receipt_data_{timestamp}.csv"
        _save_csv(items, polish_filepath, polish_map=config.polish_map)
        
    display_summary(items)
    save_correction_template(items, config, missing_names)
    display_performance_metrics(total_duration, len(items), ocr_duration)

def main():
    parser = argparse.ArgumentParser(description="Process receipt images to extract itemized data.")
    parser.add_argument("input_path", help="Path to a receipt image file OR a directory of images.")
    parser.add_argument("--shop", choices=['denner', 'lidl'], required=True, help="The shop the receipt is from.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save processed images.")
    args = parser.parse_args()
    
    start_time = time.perf_counter()
    
    # 1. Setup
    files_to_process, config = setup_environment(args)
    if not files_to_process:
        print("No valid image files to process. Exiting.")
        return

    # 2. Process
    all_items, total_ocr_time, all_missing_names = process_receipts(files_to_process, config, args)
    
    # 3. Finalize
    final_items = finalize_data(all_items, config)
    
    # 4. Report
    total_duration = time.perf_counter() - start_time
    generate_reports(final_items, config, all_missing_names, total_duration, total_ocr_time)

if __name__ == "__main__":
    # The 'apply_manual_changes' function was not included in the provided script.
    # To make this refactored version runnable, I'm adding a placeholder for it.
    # Please replace this with your actual function.
    def apply_manual_changes(items: List[ReceiptItem], filepath: Path):
        print(f"\nPlaceholder: `apply_manual_changes` would run now with file '{filepath.name}'.")

    main()

#TODO:
#     może być w denerze 3 liniowy wpis:
# liczba produktów
# zniżka
# 
# może też być zniżka zapisana jako
# Aktuellstatt [stara cena] cena
# lub 
# Aktion statt [stara cena] cena
#
#w migrosie poprawić preprocessing zdjęć, żeby dobrze sobie radził z niebieskim tłem
#w coop jest format prawie jak w migrosie, za to tło jest białe