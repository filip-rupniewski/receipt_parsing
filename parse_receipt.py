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

# --- 1. Data Structure Definition ---
@total_ordering
@dataclass
class ReceiptItem:
    pid: int
    status_flag: str; date: str; order_number: int; name: str
    size: float = np.nan; price: float = np.nan; price_per_one: float = np.nan
    shop: str = "denner"; discount: float = 0.0

    def _get_sort_tuple(self) -> Tuple[datetime, int]:
        """Creates the tuple used for comparison and sorting by date, then PID."""
        try:
            date_obj = datetime.strptime(self.date, '%d.%m.%Y')
        except (ValueError, TypeError):
            date_obj = datetime.max  # Sort items with invalid dates last
        # Use self.pid as the secondary sort key for a stable, unique order.
        return (date_obj, self.pid)

    def __lt__(self, other):
        # The '<' operator will use this method
        if not isinstance(other, ReceiptItem):
            return NotImplemented
        return self._get_sort_tuple() < other._get_sort_tuple()

    def __eq__(self, other):
        # The '==' operator will use this method
        if not isinstance(other, ReceiptItem):
            return NotImplemented
        return self._get_sort_tuple() == other._get_sort_tuple()

# --- 2. Core Logic Encapsulation ---
class ReceiptProcessor:
    def __init__(self, image_path: str, debug: bool = False):
        self.image_path = image_path; self.debug = debug; self.raw_text: Optional[str] = None
    def process(self) -> Tuple[List[ReceiptItem], float]:
        filename = os.path.basename(self.image_path)
        print(f"\n--- Processing: {filename} ---")
        print("1. Preprocessing image...")
        processed_image = self._preprocess_image()
        if processed_image is None: print(f"Warning: Could not read or process image {filename}. Skipping."); return [], 0.0
        print("2. Performing OCR to extract text...")
        self.raw_text, ocr_duration = self._extract_text_from_image(processed_image)
        if not self.raw_text: print(f"Warning: OCR returned no text for {filename}. Skipping."); return [], ocr_duration
        date_str = self._parse_date(self.raw_text)
        if self.debug: self._save_debug_output(filename, processed_image, date_str)
        print("3. Parsing text to find items...")
        items = self._parse_items(self.raw_text, date_str or "N/A")
        if items: print(f"   -> Found {len(items)} items.")
        else: print("   -> Could not find any items in this image.")
        return items, ocr_duration
    def _preprocess_image(self) -> Optional[np.ndarray]:
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None: return None
        if np.mean(image) < 128: image = cv2.bitwise_not(image)
        _, processed_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return processed_image
    def _extract_text_from_image(self, processed_image: np.ndarray) -> Tuple[Optional[str], float]:
        custom_config = r'--oem 3 --psm 4'; start_ocr_time = time.perf_counter()
        try: text = pytesseract.image_to_string(processed_image, lang='deu+eng', config=custom_config)
        except pytesseract.TesseractNotFoundError: print("Error: Tesseract is not installed or not in your PATH."); text = None
        return text, time.perf_counter() - start_ocr_time
    def _parse_date(self, text: str) -> Optional[str]:
        lines = text.split('\n')
        year_pattern = re.compile(r'\b(20\d{2})\b')
        for line in reversed(lines[-10:]):
            if (year_match := year_pattern.search(line)):
                year_str = year_match.group(1)
                all_numbers = re.findall(r'\b\d+\b', line)
                try:
                    year_index = all_numbers.index(year_str)
                    if year_index >= 2:
                        month, day = int(all_numbers[year_index - 1]), int(all_numbers[year_index - 2])
                        if 1 <= month <= 12 and 1 <= day <= 31: return f"{day:02d}.{month:02d}.{int(year_str)}"
                except (ValueError, IndexError): continue
        date_pattern_strict = r'(\d{2}[-.]\d{2}[-.,]\s?\d{4})'
        for line in lines[-10:]:
            if (match := re.search(date_pattern_strict, line)):
                nums = re.findall(r'\d+', match.group(1))
                if len(nums) == 3 and len(nums[2]) == 4: return f"{nums[0]}.{nums[1]}.{nums[2]}"
        return None
    def _parse_items(self, text: str, date_str: str) -> List[ReceiptItem]:
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
            # Add a placeholder '0' for PID and order_number.
            item = ReceiptItem(0, status_flag, date_str, 0, product_name, standardized_size, price, price_per_one, "denner", discount)
            found_items.append(item); i += processed_lines
        for index, item in enumerate(found_items): item.order_number = index + 1
        return found_items
    def _save_debug_output(self, filename: str, image: np.ndarray, date_str: Optional[str]):
        name, _ = os.path.splitext(filename)
        date_for_filename = datetime.strptime(date_str, '%d.%m.%Y').strftime('%Y%m%d') if date_str else "NODATE"
        debug_dir = "preprocessed_images"; os.makedirs(debug_dir, exist_ok=True)
        debug_filename = os.path.join(debug_dir, f"{name}_debug_{date_for_filename}.png")
        cv2.imwrite(debug_filename, image)
        print(f"   -> Preprocessed image saved as {debug_filename}")
        print("   -> Raw OCR Text:\n---\n" + self.raw_text + "\n---")

# --- 3. Reporting and Utility Functions ---
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
    except FileNotFoundError:
        print(f"\nCorrection file not found at '{filepath}'. Skipping correction step.")
    return corrections_map

def apply_corrections(items: List[ReceiptItem], corrections_map: Dict[str, dict]):
    """Applies loaded corrections and recalculates derived fields for each item."""
    corrections_applied_count = 0
    for item in items:
        if item.name in corrections_map:
            rule = corrections_map[item.name]
            
            if 'name' in rule:
                item.name = rule['name']
            
            if 'size' in rule:
                corrected_size = rule['size']
                if not np.isnan(item.size):
                    item.size = item.size * corrected_size
                else:
                    item.size = corrected_size
            
            if not np.isnan(item.price) and not np.isnan(item.size) and item.size > 0:
                item.price_per_one = item.price / item.size
            
            if not np.isnan(item.price) and not np.isnan(item.size):
                item.status_flag = ""
            else:
                item.status_flag = "!"
                
            corrections_applied_count += 1
            
    if corrections_applied_count > 0:
        print(f"Applied corrections and recalculated metrics for {corrections_applied_count} items based on rules.")

def apply_naive_corrections(items: List[ReceiptItem]):
    """
    Attempts to fix items with missing data (status '!') by using data
    from valid historical entries of the same product.
    """
    print("\nApplying naive corrections for items with missing data...")
    corrections_count = 0
    
    good_items_map = collections.defaultdict(list)
    for item in items:
        if not item.status_flag and item.date != "N/A":
            good_items_map[item.name].append(item)

    for item_to_fix in items:
        if item_to_fix.status_flag != "!" or item_to_fix.date == "N/A":
            continue

        candidates = good_items_map.get(item_to_fix.name)
        if not candidates:
            continue

        try:
            target_date = datetime.strptime(item_to_fix.date, '%d.%m.%Y')
        except ValueError:
            continue

        closest_match = None
        min_time_diff = None

        for candidate in candidates:
            try:
                candidate_date = datetime.strptime(candidate.date, '%d.%m.%Y')
                time_diff = abs(target_date - candidate_date)
                if min_time_diff is None or time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_match = candidate
            except ValueError:
                continue

        if not closest_match:
            continue

        new_status = ""
        if not np.isnan(item_to_fix.size) and np.isnan(item_to_fix.price):
            item_to_fix.price_per_one = closest_match.price_per_one
            item_to_fix.price = item_to_fix.size * closest_match.price_per_one
            new_status = "!pP"

        elif np.isnan(item_to_fix.size) and not np.isnan(item_to_fix.price):
            if closest_match.price_per_one and closest_match.price_per_one > 0:
                item_to_fix.price_per_one = closest_match.price_per_one
                item_to_fix.size = item_to_fix.price / closest_match.price_per_one
                new_status = "!sP"

        elif np.isnan(item_to_fix.size) and np.isnan(item_to_fix.price):
            item_to_fix.size = closest_match.size
            item_to_fix.price = closest_match.price
            item_to_fix.price_per_one = closest_match.price_per_one
            new_status = "!spP"

        if new_status:
            item_to_fix.status_flag = new_status
            corrections_count += 1
            
    if corrections_count > 0:
        print(f"   -> Successfully applied naive corrections to {corrections_count} items.")

def display_console_preview(items: List[ReceiptItem]):
    print("\n--- Processed Data Preview (first 20 rows) ---\n")
    header = ['PID', 'status', 'date', 'order number', 'product name', 'size/volume', 'price', 'price per 1', 'shop', 'discount']
    print(f"{header[0]:<5}{header[1]:<7}{header[2]:<12}{header[3]:<14}{header[4]:<35}{header[5]:<13}{header[6]:<10}{header[7]:<13}{header[8]:<10}{header[9]:<10}")
    print("-" * 130)
    # Sort for preview consistency, though the main list is already sorted for saving
    preview_items = sorted(items)
    for item in preview_items[:20]:
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
    
    # Python's built-in sorted() will now use the rich comparison methods in ReceiptItem
    sorted_items = sorted(items)
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';'); writer.writerow(header)
        for item in sorted_items:
            row = [item.pid, item.status_flag, item.date, item.order_number, item.name,
                   f"{item.size:.3f}".replace('.', ',') if not np.isnan(item.size) else "NaN",
                   f"{item.price:.2f}" if not np.isnan(item.price) else "NaN",
                   f"{item.price_per_one:.2f}" if not np.isnan(item.price_per_one) else "NaN",
                   item.shop, f"{item.discount:.0f}%" if item.discount > 0 else ""]
            writer.writerow(row)
    print(f"--- Successfully saved all data to {filename} ---")

def save_correction_template(items: List[ReceiptItem], corrections_map: Dict[str, dict]):
    """
    Saves a template for correcting product names, pre-filling known corrections.
    """
    if not items and not corrections_map:
        return

    output_dir = "correction_files"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "product_name_correction_template.csv")
    print(f"\nSaving/Updating product correction template to {filename}...")

    names_from_current_scan = set(item.name for item in items)
    names_from_existing_corrections = set(corrections_map.keys())
    all_unique_names = sorted(list(names_from_current_scan.union(names_from_existing_corrections)))

    header = ['nr', 'product name', 'correct product name', 'correct size']
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)
        for i, name in enumerate(all_unique_names, start=1):
            if name in corrections_map:
                rule = corrections_map[name]
                correct_name = rule.get('name', '')
                correct_size_val = rule.get('size')
                correct_size_str = ''
                if correct_size_val is not None:
                    correct_size_str = f"{correct_size_val}".replace('.', ',')
                writer.writerow([i, name, correct_name, correct_size_str])
            else:
                writer.writerow([i, name, '', ''])
            
    print(f"--- Successfully saved correction template with {len(all_unique_names)} unique products. ---")

def _print_summary_table(title: str, data: List[Tuple[str, int]]):
    """A private helper function to print a formatted summary table."""
    print(f"\n{title}")
    print(f"{'Nr.':<5} {'Product Name':<50} {'Occurrences'}")
    print("-" * 70)
    for i, (name, count) in enumerate(data, start=1):
        name_short = (name[:47] + '...') if len(name) > 50 else name
        print(f"{i:<5} {name_short:<50} {count}")

def display_summary(items: List[ReceiptItem]):
    """Counts and prints product occurrences, sorted in two ways."""
    print("\n" + "="*28 + " FINAL SUMMARY " + "="*29)
    product_counts = collections.Counter(item.name for item in items)
    sorted_by_occurrence = sorted(product_counts.items(), key=lambda item: item[1], reverse=True)
    sorted_by_name = sorted(product_counts.items(), key=lambda item: item[0])
    _print_summary_table(
        "--- Unique Products Sorted by Number of Occurrences ---",
        sorted_by_occurrence
    )
    _print_summary_table(
        "--- Unique Products Sorted Alphabetically by Name ---",
        sorted_by_name
    )
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
    args = parser.parse_args()
    start_time = time.perf_counter()
    files_to_process = get_files_to_process(args.input_path)
    if not files_to_process: print("No valid image files to process. Exiting."); return
    
    all_items: List[ReceiptItem] = []; total_ocr_time: float = 0.0
    for image_path in files_to_process:
        processor = ReceiptProcessor(image_path, debug=args.debug)
        items_from_file, ocr_duration = processor.process()
        all_items.extend(items_from_file); total_ocr_time += ocr_duration
    
    corrections_file_path = os.path.join("correction_files", CORRECTIONS_FILENAME)
    corrections_map = load_corrections(corrections_file_path)

    if all_items:
        # --- ASSIGN UNIQUE PID ---
        # 1. Define a temporary key for initial sorting based on original parse order.
        def get_initial_sort_key(item: ReceiptItem) -> Tuple[datetime, int]:
            try:
                date_obj = datetime.strptime(item.date, '%d.%m.%Y')
            except (ValueError, TypeError):
                date_obj = datetime.max
            return (date_obj, item.order_number)

        # 2. Sort all items chronologically to establish a stable order.
        all_items.sort(key=get_initial_sort_key)

        # 3. Assign the final, sequential PID to each item.
        for i, item in enumerate(all_items):
            item.pid = i + 1
        # --- END PID ASSIGNMENT ---

        if corrections_map:
            apply_corrections(all_items, corrections_map)

        apply_naive_corrections(all_items)
        
        print(f"\n--- Finished processing all files. Found a total of {len(all_items)} items. ---")
        
        display_console_preview(all_items)
        save_to_csv(all_items)
        display_summary(all_items)
    else:
        print("\n--- Finished processing. No new items were found in any images. ---")
    
    save_correction_template(all_items, corrections_map)
    
    end_time = time.perf_counter()
    total_duration = end_time - start_time
    
    display_performance_metrics(total_duration, len(all_items), total_ocr_time)

if __name__ == "__main__":
    main()

# TODO
# dodać do paragonów:
# ręczne wpisy - plik
# ręczne poprawki - plik
# ręczna biblioteka z price per 1 i datą