# reporting.py

import csv
import collections
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set

import numpy as np

from data_models import ReceiptItem
from config import CSV_HEADER, CORRECTION_FILES_DIR, CORRECTION_TEMPLATE_FILENAME, Configuration

def display_console_preview(items: List[ReceiptItem]):
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

def save_csv(items: List[ReceiptItem], filepath: Path, polish_map: Optional[Dict[str, str]] = None):
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
    print("\n" + "="*23 + " PERFORMANCE METRICS " + "="*23)
    avg_time_per_item = (duration / item_count) * 1000 if item_count > 0 else 0
    ocr_percentage = (ocr_duration / duration) * 100 if duration > 0 else 0
    print(f"Total Execution Time:    {duration:.2f} seconds")
    print(f"Total Items Processed:     {item_count}")
    print(f"Average Time per Item:   {avg_time_per_item:.2f} ms")
    print("-" * 65)
    print(f"Total Time in OCR:       {ocr_duration:.2f} seconds ({ocr_percentage:.1f}% of total time)")
    print("="*65)