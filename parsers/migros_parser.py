# parsers/migros_parser.py

import re
import numpy as np
from typing import List, Optional, Tuple, Dict

# Assuming data_models.py and base_parser.py exist in the same structure
from data_models import ReceiptItem
from .base_parser import BaseParser

class MigrosParser(BaseParser):
    """
    Parses the OCR text from a Migros receipt to extract the date and a list of purchased items.
    """

    def parse_date(self, text: str) -> Optional[str]:
        """
        Extracts the purchase date from the receipt text, allowing for various separators.
        Migros receipts typically have the date at the very bottom.
        Looks for patterns like '15.11.2025', '15 11 2025', '15,11,2025', etc.
        """
        # Search in the last 5 lines, which is a very common location for the date.
        for line in reversed(text.split('\n')[-5:]):
            # Regex to find a date pattern DD<sep>MM<sep>YYYY.
            # Separators can be a dot, comma, colon, semicolon, or space.
            # It captures day, month, and year in separate groups for validation and reformatting.
            match = re.search(r'(\d{2})\s*[\.,:;\s]\s*(\d{2})\s*[\.,:;\s]\s*(20\d{2})', line)
            
            if match:
                day, month, year = match.groups()
                
                # Perform a basic validation to ensure the matched numbers are a plausible date.
                # This helps avoid accidentally matching other numeric sequences.
                if 1 <= int(day) <= 31 and 1 <= int(month) <= 12 and int(year) > 2000:
                    # Standardize the output format to DD.MM.YYYY for consistency.
                    return f"{day}.{month}.{year}"
                    
        # If the flexible pattern fails, try the original strict one as a fallback.
        for line in reversed(text.split('\n')[-5:]):
            match = re.search(r'(\d{2}\.\d{2}\.\d{4})', line)
            if match:
                return match.group(1)

        return None # Return None if no valid date pattern is found.

    def parse_items(self, text: str, date_str: str, corrections_map: Dict[str, dict], size_map: Dict[str, float]) -> Tuple[List[ReceiptItem], set]:
        """
        Parses the main body of the receipt to extract individual items.
        - Uses flexible 'any' logic to find start and end of the item block.
        - Handles both integer and float quantities (e.g., for weighed items).
        """
        lines = text.split('\n')
        found_items: List[ReceiptItem] = []
        missing_names = set()

        # --- 1. & 2. Find Start and End of Item Block ---
        start_index, end_index = -1, -1
        
        # Keywords to identify the header row (start marker)
        header_keywords = ['artikelbezeichnung', 'menge', 'nenge', 'preis', 'gespart', 'total']
        # Keywords to identify the summary/footer row (end marker)
        summary_keywords = ['total', 'total chf']

        # 1. Find the first line that contains ANY of the header keywords.
        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in header_keywords):
                start_index = i + 1  # The items start on the *next* line.
                break
        
        # If a start line was found, search for the end line *after* it.
        if start_index != -1:
            for i, line in enumerate(lines[start_index:], start=start_index):
                # We check for 'total' but also make sure it's not the header again.
                # A good summary line usually starts with 'total'.
                if any(line.lower().strip().startswith(kw) for kw in summary_keywords):
                    end_index = i # This is the line that ends the item list.
                    break

        # If we couldn't define a clear item block, exit gracefully.
        if start_index == -1 or end_index == -1:
            print("Warning: Could not reliably find the start and/or end of the item list.")
            return [], set()

        item_lines = lines[start_index:end_index]
        
        # --- End of Start/End Block ---

        price_pattern = re.compile(r'\b\d+[\.,]\d{2}\b')

        for line in item_lines:
            line = line.strip()
            prices_found = price_pattern.findall(line)

            if not prices_found:
                continue

            price_str = prices_found[-1]
            price = float(price_str.replace(',', '.'))
            
            price_per_one_str = prices_found[-2] if len(prices_found) > 1 else price_str
            price_per_one = float(price_per_one_str.replace(',', '.'))
            
            discount = 0.0
            if price_per_one > price:
                discount_amount = price_per_one - price
                discount = (discount_amount / price_per_one) * 100 if price_per_one > 0 else 0

            product_part = line.split(prices_found[0])[0].strip()

            # --- 3. Handle Integer and Float Quantities ---
            quantity = 1.0 # Default quantity
            # This new regex finds numbers that can be integers (e.g., "1") or floats (e.g., "0.920", "1,5").
            numeric_parts = re.findall(r'\b\d+[\.,]?\d*\b', product_part)
            
            if numeric_parts:
                # The quantity is assumed to be the last number found before the price section.
                # We use .replace(',', '.') to handle OCR interpreting a decimal point as a comma.
                quantity = float(numeric_parts[-1].replace(',', '.'))
                product_name = product_part.rsplit(numeric_parts[-1], 1)[0].strip()
            else:
                product_name = product_part

            # --- End of Quantity Logic ---

            standardized_size = np.nan
            size_match = re.search(r'(\d+[\.,]?\d*)\s*(g|kg|l|ml)\b', product_name, re.IGNORECASE)
            if size_match:
                value = float(size_match.group(1).replace(',', '.'))
                unit = size_match.group(2).lower()
                if unit in ['g', 'ml']:
                    standardized_size = value / 1000.0
                else:
                    standardized_size = value
                product_name = product_name.replace(size_match.group(0), '').strip()

            product_name = re.sub(r'\s+', ' ', product_name).strip()

            if not product_name or len(product_name) < 2:
                continue

            # Apply correction from the map
            correction_rule = corrections_map.get(product_name, {})
            corrected_name = correction_rule.get('name', product_name)

            # --- UPDATE: If correction map says to correct name to "-", remove whole record. ---
            if corrected_name == "-":
                continue  # Skip this record entirely and move to the next line
            # --- END UPDATE ---
            
            product_name = corrected_name

            # Check if size is missing and add to missing_names set
            if np.isnan(standardized_size):
                if product_name not in size_map:
                    missing_names.add((product_name, "migros"))
                else:
                    standardized_size = size_map[product_name]
            
            # Calculate price_per_one (price per kilogram or liter)
            if not np.isnan(standardized_size) and standardized_size > 0:
                price_per_one = price / standardized_size
            else:
                price_per_one = np.nan
            
            status_flag = "!" if np.isnan(price) or np.isnan(standardized_size) else ""
            
            # Create ReceiptItem with CORRECT field names matching data_models.py
            item = ReceiptItem(
                pid=0,                      # ✅ Correct: pid (not id) - will be assigned later
                status_flag=status_flag,    # ✅ Correct: status_flag (not status)
                date=date_str,              # ✅ Correct: date
                order_number=0,             # ✅ Correct: order_number (not quantity) - will be assigned later
                name=product_name,          # ✅ Correct: name (not product_name)
                size=standardized_size,     # ✅ Correct: size (not standardized_size)
                price=price,                # ✅ Correct: price
                price_per_one=price_per_one,# ✅ Correct: price_per_one (not price_per_unit)
                shop="migros",              # ✅ Correct: shop (not store)
                discount=discount           # ✅ Correct: discount (not discount_percentage)
            )
            found_items.append(item)
            
        return found_items, missing_names