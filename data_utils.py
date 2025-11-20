# data_utils.py

import csv
import collections
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from data_models import ReceiptItem

def _parse_float(value_str: str) -> float:
    if not value_str:
        return np.nan
    try:
        return float(value_str.replace(',', '.'))
    except (ValueError, TypeError):
        return np.nan

def load_manual_inputs(filepath: Path) -> List[ReceiptItem]:
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