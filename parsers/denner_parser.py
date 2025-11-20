# parsers/denner_parser.py

import re
import numpy as np
from typing import List, Optional, Tuple, Dict, Set

from data_models import ReceiptItem
from .base_parser import BaseParser

class DennerParser(BaseParser):
    def parse_date(self, text: str) -> Optional[str]:
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

    def parse_items(self, text: str, date_str: str, corrections_map: Dict[str, dict], size_map: Dict[str, float]) -> Tuple[List[ReceiptItem], set]:
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