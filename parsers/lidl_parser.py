# parsers/lidl_parser.py

import re
import numpy as np
from typing import List, Optional, Tuple, Dict, Set

from data_models import ReceiptItem
from .base_parser import BaseParser

class LidlParser(BaseParser):
    def parse_date(self, text: str) -> Optional[str]:
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

    def parse_items(self, text: str, date_str: str, corrections_map: Dict[str, dict], size_map: Dict[str, float]) -> Tuple[List[ReceiptItem], set]:
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