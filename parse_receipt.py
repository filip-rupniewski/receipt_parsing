#!/usr/bin/env python3

import cv2
import pytesseract
import re
import argparse
import os
import numpy as np
import csv
from datetime import datetime
import collections
import time
from dataclasses import dataclass
from functools import total_ordering
from typing import List, Optional, Tuple, Dict

# --- Constants ---
CORRECTIONS_FILENAME = "product_name_corrections.csv"
MANUAL_INPUT_FILENAME = "manual_input.csv"
MANUAL_CHANGES_FILENAME = "manual_changes.csv"
MANUAL_DATES_FILENAME = "manual_dates.csv"
OCR_CACHE_DIR = "ocr_cache"  # --- NEW ---: Directory to store OCR text results

# --- 1. Data Structure Definition ---
@total_ordering
@dataclass
class ReceiptItem:
    # ... (No changes in this class, content is identical) ...
    pid: int
    status_flag: str; date: str; order_number: int; name: str
    size: float = np.nan; price: float = np.nan; price_per_one: float = np.nan
    shop: str = "denner"; discount: float = 0.0

    def _get_sort_tuple(self) -> Tuple[datetime, int]:
        try:
            date_obj = datetime.strptime(self.date, '%d.%m.%Y')
        except (ValueError, TypeError):
            date_obj = datetime.max
        return (date_obj, self.pid)

    def __lt__(self, other):
        if not isinstance(other, ReceiptItem): return NotImplemented
        return self._get_sort_tuple() < other._get_sort_tuple()

    def __eq__(self, other):
        if not isinstance(other, ReceiptItem): return NotImplemented
        return self._get_sort_tuple() == other._get_sort_tuple()

# --- 2. Core Logic Encapsulation ---
class ReceiptProcessor:
    def __init__(self, image_path: str, debug: bool = False):
        self.image_path = image_path; self.debug = debug; self.raw_text: Optional[str] = None
    
    # --- MODIFIED ---: The process method is slightly updated
    def process(self, manual_dates: Dict[str, str] = None) -> Tuple[List[ReceiptItem], float]:
        if manual_dates is None: manual_dates = {}
        filename = os.path.basename(self.image_path)
        print(f"\n--- Processing: {filename} ---")
        print("1. Preprocessing image...")
        processed_image = self._preprocess_image()
        if processed_image is None: print(f"Warning: Could not read or process image {filename}. Skipping."); return [], 0.0
        
        # --- MODIFIED ---: Updated print statement and passing filename to extractor
        print("2. Extracting text (from cache or OCR)...")
        self.raw_text, ocr_duration = self._extract_text_from_image(processed_image, filename)
        
        if not self.raw_text: print(f"Warning: OCR returned no text for {filename}. Skipping."); return [], ocr_duration

        date_str = manual_dates.get(filename)
        if date_str:
            print(f"   -> Using manually provided date: {date_str}")
        else:
            date_str = self._parse_date(self.raw_text)

        if self.debug: self._save_debug_output(filename, processed_image, date_str)
        print("3. Parsing text to find items...")
        items = self._parse_items(self.raw_text, date_str or "N/A")
        if items: print(f"   -> Found {len(items)} items.")
        else: print("   -> Could not find any items in this image.")
        return items, ocr_duration
    
    def _preprocess_image(self) -> Optional[np.ndarray]:
        # ... (No changes in this method, content is identical) ...
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None: return None
        if np.mean(image) < 128: image = cv2.bitwise_not(image)
        _, processed_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return processed_image

    # --- MODIFIED ---: This entire method is rewritten to include caching logic
    def _extract_text_from_image(self, processed_image: np.ndarray, original_filename: str) -> Tuple[Optional[str], float]:
        """Extracts text, using a cache to avoid re-running OCR."""
        cache_filepath = os.path.join(OCR_CACHE_DIR, f"{original_filename}.txt")

        # --- CACHE HIT ---
        if os.path.exists(cache_filepath):
            print(f"   -> Found cached OCR text. Loading from '{cache_filepath}'.")
            with open(cache_filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            return text, 0.0  # Return 0.0 duration for cached reads

        # --- CACHE MISS ---
        print("   -> No cache found. Performing OCR...")
        custom_config = r'--oem 3 --psm 4'
        start_ocr_time = time.perf_counter()
        text = None
        try:
            text = pytesseract.image_to_string(processed_image, lang='deu+eng', config=custom_config)
        except pytesseract.TesseractNotFoundError:
            print("Error: Tesseract is not installed or not in your PATH.")
        
        ocr_duration = time.perf_counter() - start_ocr_time

        # Save the result to cache for next time
        if text:
            with open(cache_filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"   -> OCR text saved to cache: {cache_filepath}")
            
        return text, ocr_duration

    def _parse_date(self, text: str) -> Optional[str]:
        # ... (No changes in this method, content is identical) ...
        lines = text.split('\n'); found_date: Optional[str] = None
        year_pattern = re.compile(r'\b(20\d{2})\b')
        for line in reversed(lines[-10:]):
            if (year_match := year_pattern.search(line)):
                year_str = year_match.group(1); all_numbers = re.findall(r'\b\d+\b', line)
                try:
                    year_index = all_numbers.index(year_str)
                    if year_index >= 2:
                        month, day = int(all_numbers[year_index - 1]), int(all_numbers[year_index - 2])
                        if 1 <= month <= 12 and 1 <= day <= 31:
                            found_date = f"{day:02d}.{month:02d}.{int(year_str)}"; break
                except (ValueError, IndexError): continue
        if not found_date:
            date_pattern_strict = r'(\d{2}[-.]\d{2}[-.,]\s?\d{4})'
            for line in lines[-10:]:
                if (match := re.search(date_pattern_strict, line)):
                    nums = re.findall(r'\d+', match.group(1))
                    if len(nums) == 3 and len(nums[2]) == 4:
                        found_date = f"{nums[0]}.{nums[1]}.{nums[2]}"; break
        if found_date and "2075" in found_date:
            corrected_date = found_date.replace("2075", "2025")
            print(f"   -> Corrected OCR year error: {found_date} -> {corrected_date}"); return corrected_date
        return found_date
    
    def _parse_items(self, text: str, date_str: str) -> List[ReceiptItem]:
        # ... (No changes in this method, content is identical) ...
        lines = text.split('\n'); found_items: List[ReceiptItem] = []
        IGNORE_KEYWORDS = ['mwst', 'rabatt', 'bargeld', 'rückgeld', 'bezeichnung']; i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.lower().startswith('total'): break
            if not re.match(r'^[A-Z]\s', line) or any(kw in line.lower() for kw in IGNORE_KEYWORDS): i += 1; continue
            processed_lines = 1; product_name_raw = line
            price, price_per_one, standardized_size, discount = np.nan, np.nan, np.nan, 0.0
            prices_on_line = re.findall(r'(\d+[\.,]\d{1,2})', line)
            if line.endswith('#') and i + 1 < len(lines) and 'verbilligung' in lines[i+1].lower() and prices_on_line:
                original_price = float(prices_on_line[-1].replace(',', '.'))
                if discount_match := re.search(r'(\d+[\.,]\d+)', lines[i+1]):
                    discount_amount = float(discount_match.group(1).replace(',', '.')); price = original_price - discount_amount
                    discount = (discount_amount / original_price) * 100 if original_price > 0 else 0
                    product_name_raw = line.rsplit(prices_on_line[-1], 1)[0]; processed_lines = 2
            elif np.isnan(price) and prices_on_line:
                price = float(prices_on_line[-1].replace(',', '.')); product_name_raw = line.rsplit(prices_on_line[-1], 1)[0]
            product_name = re.sub(r'^[A-Z]\s', '', product_name_raw).strip().rstrip('#').strip()
            if len(product_name.split()) == 1 and len(product_name) < 3: i += processed_lines; continue
            if size_match := re.search(r'(\d+[\.,]?\d*)\s*(g|kg|l|ml)\b', product_name, re.IGNORECASE):
                value = float(size_match.group(1).replace(',', '.')); unit = size_match.group(2).lower()
                standardized_size = value / 1000.0 if unit in ['g', 'ml'] else value
                product_name = product_name.replace(size_match.group(0), '').strip()
            price_per_one = price / standardized_size if not np.isnan(price) and not np.isnan(standardized_size) and standardized_size > 0 else np.nan
            status_flag = "!" if np.isnan(price) or np.isnan(standardized_size) else ""
            item = ReceiptItem(0, status_flag, date_str, 0, product_name, standardized_size, price, price_per_one, "denner", discount)
            found_items.append(item); i += processed_lines
        for index, item in enumerate(found_items): item.order_number = index + 1
        return found_items

    def _save_debug_output(self, filename: str, image: np.ndarray, date_str: Optional[str]):
        # ... (No changes in this method, content is identical) ...
        name, _ = os.path.splitext(filename)
        date_for_filename = datetime.strptime(date_str, '%d.%m.%Y').strftime('%Y%m%d') if date_str else "NODATE"
        debug_dir = "preprocessed_images"; os.makedirs(debug_dir, exist_ok=True)
        debug_filename = os.path.join(debug_dir, f"debug_{date_for_filename}_{name}.png")
        cv2.imwrite(debug_filename, image)
        print(f"   -> Preprocessed image saved as {debug_filename}")
        print("   -> Raw OCR Text:\n---\n" + self.raw_text + "\n---")


# --- 3. Reporting and Utility Functions ---
# ... (No changes in the utility functions from get_files_to_process to display_performance_metrics) ...
# ... Their content is identical ...

def get_files_to_process(input_path: str) -> List[str]:
    files = []; VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    if os.path.isdir(input_path):
        print(f"Input is a directory. Scanning for images in: {input_path}")
        files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(VALID_EXTENSIONS)])
    elif os.path.isfile(input_path):
        if input_path.lower().endswith(VALID_EXTENSIONS): print(f"Input is a single file: {input_path}"); files = [input_path]
        else: print(f"Error: Input file '{input_path}' is not a supported image type.")
    else: print(f"Error: Path not found or is not a valid file/directory: {input_path}")
    return files

def _parse_float(value_str: str) -> float:
    if not value_str: return np.nan
    try: return float(value_str.replace(',', '.'))
    except (ValueError, TypeError): return np.nan

def load_manual_dates(filepath: str) -> Dict[str, str]:
    date_map: Dict[str, str] = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if not row or not row[0].strip(): continue
                filename = row[0].strip()
                correct_date = next((item.strip() for item in reversed(row) if item.strip()), None)
                if correct_date: date_map[filename] = correct_date
        if date_map: print(f"\nLoaded {len(date_map)} manual date overrides from '{filepath}'.")
    except FileNotFoundError: print(f"\nManual dates file not found at '{filepath}'. Will parse dates from images.")
    except Exception as e: print(f"\nError reading manual dates file '{filepath}': {e}. Skipping.")
    return date_map

def load_manual_inputs(filepath: str) -> List[ReceiptItem]:
    manual_items: List[ReceiptItem] = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';'); next(reader, None)
            for row in reader:
                if not any(row) or len(row) < 10: continue
                pid_str, status, date, order_str, name, size_str, price_str, _, shop, discount_str = (row + [''] * 10)[:10]
                if not date or not name: continue
                order_number = int(order_str) if order_str.isdigit() and int(order_str) > 0 else 0
                size = _parse_float(size_str); price = _parse_float(price_str); discount = _parse_float(discount_str.replace('%', ''))
                price_per_one = price / size if not np.isnan(price) and not np.isnan(size) and size > 0 else np.nan
                item = ReceiptItem(0, "M", date.strip(), order_number, name.strip(), size, price, price_per_one, shop.strip() or "denner", discount)
                manual_items.append(item)
        if manual_items: print(f"\nLoaded {len(manual_items)} items from manual input file '{filepath}'.")
    except FileNotFoundError: print(f"\nManual input file not found at '{filepath}'. Skipping.")
    return manual_items

def insert_manual_items(all_items: List[ReceiptItem], manual_items: List[ReceiptItem]) -> List[ReceiptItem]:
    print("Integrating manual items into the main dataset...")
    items_by_date = collections.defaultdict(list)
    for item in all_items: items_by_date[item.date].append(item)
    for manual_item in manual_items:
        target_list = items_by_date[manual_item.date]
        if manual_item.order_number > 0: target_list.insert(min(manual_item.order_number - 1, len(target_list)), manual_item)
        else: target_list.append(manual_item)
    final_items_list: List[ReceiptItem] = []
    sorted_dates = sorted(items_by_date.keys(), key=lambda d: datetime.strptime(d, '%d.%m.%Y') if d != "N/A" else datetime.max)
    for date in sorted_dates:
        date_group = items_by_date[date]
        for i, item in enumerate(date_group): item.order_number = i + 1
        final_items_list.extend(date_group)
    print(f"   -> Manual items integrated. Total items are now {len(final_items_list)}.")
    return final_items_list

def _values_match(item_value, search_value_str: str) -> bool:
    if isinstance(item_value, float):
        if search_value_str.lower() == 'nan': return np.isnan(item_value)
        try: return np.isclose(item_value, float(search_value_str.replace(',', '.')))
        except (ValueError, TypeError): return False
    elif isinstance(item_value, int):
        try: return item_value == int(search_value_str)
        except (ValueError, TypeError): return False
    else: return str(item_value) == search_value_str

def apply_manual_changes(items: List[ReceiptItem], filepath: str):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';'); change_rules = list(reader)
    except FileNotFoundError: print(f"\nManual changes file not found at '{filepath}'. Skipping this step."); return
    except Exception as e: print(f"\nError reading manual changes file '{filepath}': {e}. Skipping this step."); return
    if not change_rules: return

    print(f"\nApplying {len(change_rules)} rules from manual changes file '{filepath}'...")
    updates_applied_count = 0
    csv_to_attr_map = {'PID': 'pid', 'status': 'status_flag', 'date': 'date', 'order number': 'order_number',
                       'product name': 'name', 'size/volume': 'size', 'price': 'price', 'price per 1': 'price_per_one',
                       'shop': 'shop', 'discount': 'discount'}

    for i, rule in enumerate(change_rules, 1):
        identifiers = {csv_to_attr_map[k]: v for k, v in rule.items() if k in csv_to_attr_map and v is not None and v.strip()}
        updates = {csv_to_attr_map[k.replace('u_', '')]: v for k, v in rule.items() if k.startswith('u_') and v is not None and v.strip()}
        if not identifiers or not updates: continue
        item_found_and_updated = False
        for item in items:
            if all(_values_match(getattr(item, attr), val) for attr, val in identifiers.items()):
                print(f"   -> Rule {i}: Found match on PID {item.pid} ('{item.name}'). Applying updates...")
                for attr_key, update_val_str in updates.items():
                    try:
                        if attr_key in ['size', 'price', 'price_per_one', 'discount']: new_value = _parse_float(update_val_str)
                        elif attr_key in ['pid', 'order_number']: new_value = int(update_val_str.strip())
                        else: new_value = update_val_str.strip()
                        setattr(item, attr_key, new_value)
                    except (ValueError, TypeError) as e: print(f"      Warning: Could not apply update for '{attr_key}' with value '{update_val_str}'. Error: {e}")
                if not np.isnan(item.price) and not np.isnan(item.size) and item.size > 0: item.price_per_one = item.price / item.size
                else: item.price_per_one = np.nan
                if 'M' not in item.status_flag: item.status_flag = 'M'
                updates_applied_count += 1; item_found_and_updated = True; break
        if not item_found_and_updated: print(f"   -> Rule {i}: No matching item found for criteria: {identifiers}")
    if updates_applied_count > 0: print(f"--- Successfully applied {updates_applied_count} manual updates. ---")

def load_corrections(filepath: str) -> Dict[str, dict]:
    corrections_map = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';'); next(reader, None)
            for row in reader:
                if len(row) < 4: continue
                _, ocr_name, correct_name, correct_size_str = [field.strip() for field in row]
                if not ocr_name: continue
                rule = {}
                if correct_name: rule['name'] = correct_name
                if correct_size_str:
                    try: rule['size'] = float(correct_size_str.replace(',', '.'))
                    except ValueError: print(f"Warning: Invalid size '{correct_size_str}' for '{ocr_name}' in corrections file. Ignoring.")
                if rule: corrections_map[ocr_name] = rule
        print(f"\nLoaded {len(corrections_map)} rules from '{filepath}'.")
    except FileNotFoundError: print(f"\nCorrection file not found at '{filepath}'. Skipping correction step.")
    return corrections_map

def apply_corrections(items: List[ReceiptItem], corrections_map: Dict[str, dict]) -> List[ReceiptItem]:
    kept_items = []; corrections_applied_count = 0; deletions_count = 0
    for item in items:
        if item.name in corrections_map:
            rule = corrections_map[item.name]
            if rule.get('name') == '-': deletions_count += 1; continue
            if 'name' in rule: item.name = rule['name']
            if 'size' in rule:
                corrected_size = rule['size']
                if not np.isnan(item.size): item.size = item.size * corrected_size
                else: item.size = corrected_size
            if not np.isnan(item.price) and not np.isnan(item.size) and item.size > 0: item.price_per_one = item.price / item.size
            item.status_flag = "" if not np.isnan(item.price) and not np.isnan(item.size) else "!"
            corrections_applied_count += 1
        kept_items.append(item)
    if corrections_applied_count > 0: print(f"Applied corrections and recalculated metrics for {corrections_applied_count} items based on rules.")
    if deletions_count > 0: print(f"Deleted {deletions_count} items based on '-' rule in corrections file.")
    return kept_items

def apply_naive_corrections(items: List[ReceiptItem]):
    print("\nApplying naive corrections for items with missing data...")
    corrections_count = 0
    good_items_map = collections.defaultdict(list)
    for item in items:
        if (not item.status_flag or item.status_flag == "M") and item.date != "N/A": good_items_map[item.name].append(item)
    for item_to_fix in items:
        if "!" not in item_to_fix.status_flag or item_to_fix.date == "N/A": continue
        candidates = good_items_map.get(item_to_fix.name)
        if not candidates: continue
        try: target_date = datetime.strptime(item_to_fix.date, '%d.%m.%Y')
        except ValueError: continue
        closest_match = min(candidates, key=lambda c: abs(target_date - datetime.strptime(c.date, '%d.%m.%Y')))
        if not closest_match: continue
        new_status = ""
        if not np.isnan(item_to_fix.size) and np.isnan(item_to_fix.price):
            item_to_fix.price_per_one = closest_match.price_per_one; item_to_fix.price = item_to_fix.size * closest_match.price_per_one; new_status = "!pP"
        elif np.isnan(item_to_fix.size) and not np.isnan(item_to_fix.price):
            if closest_match.price_per_one and closest_match.price_per_one > 0:
                item_to_fix.price_per_one = closest_match.price_per_one; item_to_fix.size = item_to_fix.price / closest_match.price_per_one; new_status = "!sP"
        elif np.isnan(item_to_fix.size) and np.isnan(item_to_fix.price):
            item_to_fix.size = closest_match.size; item_to_fix.price = closest_match.price; item_to_fix.price_per_one = closest_match.price_per_one; new_status = "!spP"
        if new_status: item_to_fix.status_flag = new_status; corrections_count += 1
    if corrections_count > 0: print(f"   -> Successfully applied naive corrections to {corrections_count} items.")

def display_console_preview(items: List[ReceiptItem]):
    print("\n--- Processed Data Preview (first 20 rows) ---\n")
    header = ['PID', 'status', 'date', 'order number', 'product name', 'size/volume', 'price', 'price per 1', 'shop', 'discount']
    print(f"{header[0]:<5}{header[1]:<7}{header[2]:<12}{header[3]:<14}{header[4]:<35}{header[5]:<13}{header[6]:<10}{header[7]:<13}{header[8]:<10}{header[9]:<10}")
    print("-" * 130)
    for item in sorted(items)[:20]:
        name_short = (item.name[:32] + '...') if len(item.name) > 35 else item.name
        print(f"{item.pid:<5}{item.status_flag:<7}{item.date:<12}{item.order_number:<14}{name_short:<35}"
              f"{f'{item.size:.3f}' if not np.isnan(item.size) else 'NaN':<13}"
              f"{f'{item.price:.2f}' if not np.isnan(item.price) else 'NaN':<10}"
              f"{f'{item.price_per_one:.2f}' if not np.isnan(item.price_per_one) else 'NaN':<13}"
              f"{item.shop:<10}{f'{item.discount:.0f}%' if item.discount > 0 else '':<10}")

def save_to_csv(items: List[ReceiptItem]):
    output_dir = "output_data"; os.makedirs(output_dir, exist_ok=True); timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(output_dir, f"receipt_data_{timestamp}.csv")
    print(f"\nSaving all data to {filename}..."); header = ['PID', 'status', 'date', 'order number', 'product name', 'size/volume', 'price', 'price per 1', 'shop', 'discount']
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';'); writer.writerow(header)
        for item in sorted(items):
            row = [item.pid, item.status_flag, item.date, item.order_number, item.name,
                   f"{item.size:.3f}".replace('.', ',') if not np.isnan(item.size) else "NaN",
                   f"{item.price:.2f}".replace('.', ',') if not np.isnan(item.price) else "NaN",
                   f"{item.price_per_one:.2f}".replace('.', ',') if not np.isnan(item.price_per_one) else "NaN",
                   item.shop, f"{item.discount:.0f}%" if item.discount > 0 else ""]
            writer.writerow(row)
    print(f"--- Successfully saved all data to {filename} ---")

def save_correction_template(items: List[ReceiptItem], corrections_map: Dict[str, dict]):
    if not items and not corrections_map: return
    output_dir = "correction_files"; os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "product_name_correction_template.csv")
    print(f"\nSaving/Updating product correction template to {filename}...")
    all_unique_names = sorted(list(set(item.name for item in items).union(set(corrections_map.keys()))))
    header = ['nr', 'product name', 'correct product name', 'correct size']
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';'); writer.writerow(header)
        for i, name in enumerate(all_unique_names, start=1):
            rule = corrections_map.get(name, {})
            correct_name = rule.get('name', '')
            correct_size_val = rule.get('size')
            correct_size_str = f"{correct_size_val}".replace('.', ',') if correct_size_val is not None else ''
            writer.writerow([i, name, correct_name, correct_size_str])
    print(f"--- Successfully saved correction template with {len(all_unique_names)} unique products. ---")

def _print_summary_table(title: str, data: List[Tuple[str, int]]):
    print(f"\n{title}")
    print(f"{'Nr.':<5} {'Product Name':<50} {'Occurrences'}"); print("-" * 70)
    for i, (name, count) in enumerate(data, start=1):
        name_short = (name[:47] + '...') if len(name) > 50 else name
        print(f"{i:<5} {name_short:<50} {count}")

def display_summary(items: List[ReceiptItem]):
    print("\n" + "="*28 + " FINAL SUMMARY " + "="*29)
    product_counts = collections.Counter(item.name for item in items)
    _print_summary_table("--- Unique Products Sorted by Number of Occurrences ---", sorted(product_counts.items(), key=lambda item: item[1], reverse=True))
    _print_summary_table("--- Unique Products Sorted Alphabetically by Name ---", sorted(product_counts.items(), key=lambda item: item[0]))
    print("\n" + "="*70)

def display_performance_metrics(duration: float, item_count: int, ocr_duration: float):
    print("\n" + "="*23 + " PERFORMANCE METRICS " + "="*23)
    avg_time_per_item = (duration / item_count) * 1000 if item_count > 0 else 0
    ocr_percentage = (ocr_duration / duration) * 100 if duration > 0 else 0
    print(f"Total Execution Time:    {duration:.2f} seconds"); print(f"Total Items Processed:     {item_count}")
    print(f"Average Time per Item:   {avg_time_per_item:.2f} ms"); print("-" * 65)
    print(f"Total Time in OCR:       {ocr_duration:.2f} seconds ({ocr_percentage:.1f}% of total time)"); print("="*65)

# --- 4. Main Execution Block ---
def main():
    parser = argparse.ArgumentParser(description="Process receipt images to extract itemized data.")
    parser.add_argument("input_path", help="Path to a receipt image file OR a directory of images.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save processed images and text.")
    args = parser.parse_args(); start_time = time.perf_counter()
    files_to_process = get_files_to_process(args.input_path)
    if not files_to_process: print("No valid image files to process. Exiting."); return

    # --- NEW ---: Ensure the OCR cache directory exists before starting
    os.makedirs(OCR_CACHE_DIR, exist_ok=True)
    
    manual_dates_file = os.path.join("correction_files", MANUAL_DATES_FILENAME)
    manual_dates_map = load_manual_dates(manual_dates_file)

    all_items: List[ReceiptItem] = []; total_ocr_time: float = 0.0
    for image_path in files_to_process:
        processor = ReceiptProcessor(image_path, debug=args.debug)
        items_from_file, ocr_duration = processor.process(manual_dates_map)
        all_items.extend(items_from_file); total_ocr_time += ocr_duration

    manual_input_file = os.path.join("correction_files", MANUAL_INPUT_FILENAME)
    manual_items = load_manual_inputs(manual_input_file)
    if manual_items: all_items = insert_manual_items(all_items, manual_items)

    corrections_file_path = os.path.join("correction_files", CORRECTIONS_FILENAME)
    corrections_map = load_corrections(corrections_file_path)

    if all_items:
        if corrections_map:
            all_items = apply_corrections(all_items, corrections_map)
        
        all_items.sort(key=lambda item: (datetime.strptime(item.date, '%d.%m.%Y') if item.date != "N/A" else datetime.max, item.order_number))
        for i, item in enumerate(all_items): item.pid = i + 1

        apply_naive_corrections(all_items)
        manual_changes_file = os.path.join("correction_files", MANUAL_CHANGES_FILENAME)
        apply_manual_changes(all_items, manual_changes_file)
        apply_naive_corrections(all_items)

        print(f"\n--- Finished processing all files. Found a total of {len(all_items)} items. ---")
        display_console_preview(all_items)
        save_to_csv(all_items)
        display_summary(all_items)
    else:
        print("\n--- Finished processing. No new items were found in any images. ---")

    save_correction_template(all_items, corrections_map)
    total_duration = time.perf_counter() - start_time
    display_performance_metrics(total_duration, len(all_items), total_ocr_time)

if __name__ == "__main__":
    main()