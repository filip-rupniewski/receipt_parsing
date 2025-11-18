#!/usr/bin/env python3

import cv2
import pytesseract
import re
import argparse
import os
import numpy as np
import csv
from datetime import datetime

def preprocess_image(image_path):
    # ... (this function is unchanged) ...
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None: return None
    if np.mean(image) < 128: image = cv2.bitwise_not(image)
    _, processed_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return processed_image

def extract_text_from_image(processed_image):
    # ... (this function is unchanged) ...
    custom_config = r'--oem 3 --psm 4'
    try:
        return pytesseract.image_to_string(processed_image, lang='deu+eng', config=custom_config)
    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract is not installed or not in your PATH.")
        return None

def parse_date_from_text(text):
    """
    Parses the date from raw text using a robust, heuristic-based approach
    that is resilient to OCR errors. Returns date in DD.MM.YYYY format or None.
    """
    lines = text.split('\n')
    # Anchor the search on a 4-digit year (e.g., 2025)
    year_pattern = re.compile(r'\b(20\d{2})\b') 

    # Search the last 10 lines of the receipt from the bottom up
    for line in reversed(lines[-10:]):
        year_match = year_pattern.search(line)
        if year_match:
            year_str = year_match.group(1)
            
            # Find all numbers on the same line as the year
            all_numbers_on_line = re.findall(r'\b\d+\b', line)
            
            try:
                # Find the position of our year in the list of numbers
                year_index = all_numbers_on_line.index(year_str)
                
                # The day and month should be the two numbers right before the year
                if year_index >= 2:
                    month_str = all_numbers_on_line[year_index - 1]
                    day_str = all_numbers_on_line[year_index - 2]
                    
                    # --- Validation Step ---
                    month = int(month_str)
                    day = int(day_str)
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        # Success! We found a valid date structure.
                        # Format it with leading zeros.
                        return f"{day:02d}.{month:02d}.{int(year_str)}"
            except (ValueError, IndexError):
                # This can happen if the year is found but the structure isn't as expected.
                # We just continue to the next line.
                continue
    
    # If the new logic fails, fallback to the old, stricter regex as a backup
    date_pattern_strict = r'(\d{2}[-.]\d{2}[-.,]\s?\d{4})'
    for line in lines[-10:]:
        match = re.search(date_pattern_strict, line)
        if match:
            nums = re.findall(r'\d+', match.group(1))
            if len(nums) == 3 and len(nums[2]) == 4:
                return f"{nums[0]}.{nums[1]}.{nums[2]}"

    return None # Return None if no date could be confidently parsed

def parse_items_from_text(text, date_str):
    """
    Parses all items from the raw text, using the provided date string.
    """
    # ... (This entire function remains unchanged from the previous correct version) ...
    lines = text.split('\n')
    found_items = []
    IGNORE_KEYWORDS = ['mwst', 'rabatt', 'bargeld', 'rückgeld', 'bezeichnung']
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.lower().startswith('total'):
            break

        if not re.match(r'^[A-Z]\s', line) or any(kw in line.lower() for kw in IGNORE_KEYWORDS):
            i += 1
            continue

        processed_lines = 1
        product_name_raw = line
        price, price_per_one, standardized_size = np.nan, np.nan, np.nan
        size_volume_str = "N/A"
        discount = 0.0

        prices_on_line = re.findall(r'(\d+[\.,]\d{1,2})', line)
        
        if line.endswith('#') and i + 1 < len(lines):
            next_line = lines[i+1]
            if 'verbilligung' in next_line.lower() and prices_on_line:
                original_price = float(prices_on_line[-1].replace(',', '.'))
                discount_amount_match = re.search(r'(\d+[\.,]\d+)', next_line)
                
                if discount_amount_match:
                    discount_amount = float(discount_amount_match.group(1).replace(',', '.'))
                    price = original_price - discount_amount
                    price_per_one = original_price
                    discount = (discount_amount / original_price) * 100 if original_price > 0 else 0
                    product_name_raw = line.rsplit(prices_on_line[-1], 1)[0]
                    processed_lines = 2
        
        elif not prices_on_line and i + 1 < len(lines):
            next_line = lines[i+1]
            prices_on_next_line = re.findall(r'(\d+[\.,]\d{1,2})', next_line)
            if prices_on_next_line:
                price = float(prices_on_next_line[-1].replace(',', '.'))
                price_per_one = price
                
                qty_match = re.search(r'^(\d+)\s*[xX%]', next_line)
                if qty_match:
                    quantity = int(qty_match.group(1))
                    size_volume_str = f"{quantity} pcs"
                elif 'kg' in next_line.lower() or 'ko' in next_line.lower():
                    size_volume_str = f"{prices_on_next_line[0]} Kg"
                
                processed_lines = 2

        if np.isnan(price) and prices_on_line:
            price = float(prices_on_line[-1].replace(',', '.'))
            price_per_one = price
            product_name_raw = line.rsplit(prices_on_line[-1], 1)[0]

        product_name = re.sub(r'^[A-Z]\s', '', product_name_raw).strip().rstrip('#').strip()
        
        words = product_name.split()
        if len(words) == 1 and len(words[0]) < 3:
            i += processed_lines
            continue

        if size_volume_str == "N/A":
            size_match = re.search(r'(\d+[\.,]?\d*\s*(?:g|kg|l|ml|Stk|Pet|Ikg)\b)', product_name, re.IGNORECASE)
            if size_match: size_volume_str = size_match.group(1)
        
        if 'pcs' in str(size_volume_str):
            qty_match = re.search(r'(\d+)', str(size_volume_str))
            unit_match = re.search(r'(\d+[\.,]?\d*)\s*(g|kg|l|ml)\b', product_name, re.IGNORECASE)
            if qty_match and unit_match:
                quantity = int(qty_match.group(1))
                per_item_value = float(unit_match.group(1).replace(',', '.'))
                unit = unit_match.group(2).lower()
                total_value = quantity * per_item_value
                final_unit = 'kg' if unit in ['g', 'kg'] else 'l'
                final_value = total_value / 1000.0 if unit in ['g', 'ml'] else total_value
                size_volume_str = f"{final_value:.3f} {final_unit}"
        
        size_match = re.search(r'(\d+[\.,]?\d*)\s*(g|kg|l|ml)\b', str(size_volume_str), re.IGNORECASE)
        if size_match:
            value = float(size_match.group(1).replace(',', '.'))
            unit = size_match.group(2).lower()
            standardized_size = value / 1000.0 if unit in ['g', 'ml'] else value

        size_match_in_name = re.search(r'(\d+[\.,]?\d*\s*(?:g|kg|l|ml|Stk|Pet|Ikg)\b)', product_name, re.IGNORECASE)
        if size_match_in_name:
            product_name = product_name.replace(size_match_in_name.group(1), '').strip()
        
        if not np.isnan(price) and not np.isnan(standardized_size) and standardized_size > 0:
            price_per_one = price / standardized_size
        elif 'pcs' not in str(size_volume_str):
            price_per_one = price

        status_flag = "!" if (np.isnan(price) or np.isnan(standardized_size)) else ""

        found_items.append({
            "status_flag": status_flag, "date": date_str, "name": product_name.strip(),
            "size": standardized_size, "price": price, "price_per_one": price_per_one,
            "shop": "denner", "discount": discount
        })
        
        i += processed_lines
            
    return found_items

def main():
    # ... (The main function remains unchanged from the previous correct version) ...
    parser = argparse.ArgumentParser(description="Process a Denner shop receipt image to extract itemized data.")
    parser.add_argument("image_path", help="The path to the ORIGINAL receipt image file.")
    parser.add_argument("--debug", action="store_true", help="Print the raw OCR text and save the processed image for debugging.")
    parser.add_argument("-o", "--output_dir", help="Directory to save preprocessed images.", default="preprocessed_images")
    args = parser.parse_args()
    if not os.path.exists(args.image_path):
        print(f"Error: File not found at {args.image_path}")
        return
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"1. Processing image: {args.image_path}")
    processed_image = preprocess_image(args.image_path)
    if processed_image is None: return
    print("2. Performing OCR to extract text...")
    raw_text = extract_text_from_image(processed_image)
    if raw_text is None: return
    
    parsed_date_str = parse_date_from_text(raw_text)

    if args.debug:
        base_filename = os.path.basename(args.image_path)
        name, _ = os.path.splitext(base_filename)
        date_for_filename = "NODATE"
        if parsed_date_str:
            try:
                dt_object = datetime.strptime(parsed_date_str, '%d.%m.%Y')
                date_for_filename = dt_object.strftime('%Y%m%d')
            except (ValueError, TypeError): pass
        
        debug_filename = os.path.join(args.output_dir, f"{name}_debug_{date_for_filename}.png")
        cv2.imwrite(debug_filename, processed_image)
        print(f"\n--- Preprocessed image saved as {debug_filename} ---")
        
        print("\n--- Raw OCR Text ---\n")
        print(raw_text)
        print("\n--- End Raw OCR Text ---\n")

    print("3. Parsing text to find items...")
    items = parse_items_from_text(raw_text, date_str=parsed_date_str or "N/A")

    if not items:
        print("Could not find any items.")
        return
    
    print("\n--- Processed Receipt Data ---\n")
    header_console = ['status', 'date', 'product name', 'size/volume', 'price', 'price per 1', 'shop', 'discount']
    print('\t'.join(header_console))
    for item in items:
        price_str = f"{item['price']:.2f}" if not np.isnan(item['price']) else "NaN"
        ppo_str = f"{item['price_per_one']:.2f}" if not np.isnan(item['price_per_one']) else "NaN"
        size_str = f"{item['size']:.3f}" if not np.isnan(item['size']) else "NaN"
        discount_str = f"{item['discount']:.0f}%" if item['discount'] > 0 else ""
        print(f"{item['status_flag']}\t{item['date']}\t{item['name']}\t{size_str}\t{price_str}\t{ppo_str}\t{item['shop']}\t{discount_str}")

    output_data_dir = "output_data"
    os.makedirs(output_data_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_filename = os.path.join(output_data_dir, f"receipt_data_{timestamp}.csv")
    print(f"\n4. Saving data to {output_filename}...")
    header_csv = ['status', 'date', 'product name', 'size/volume', 'price', 'price per 1', 'shop', 'discount']
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(header_csv)
        for item in items:
            price_str = f"{item['price']:.2f}" if not np.isnan(item['price']) else "NaN"
            ppo_str = f"{item['price_per_one']:.2f}" if not np.isnan(item['price_per_one']) else "NaN"
            size_str = f"{item['size']:.3f}".replace('.', ',') if not np.isnan(item['size']) else "NaN"
            discount_str = f"{item['discount']:.0f}%" if item['discount'] > 0 else ""
            row_data = [
                item['status_flag'], item['date'], item['name'],
                size_str, price_str, ppo_str, item['shop'], discount_str
            ]
            writer.writerow(row_data)
    print(f"--- Successfully saved data to {output_filename} ---")

if __name__ == "__main__":
    main()