#!/usr/bin/env python3

import cv2
import pytesseract
import re
import argparse
import os
import numpy as np

def preprocess_image(image_path):
    """
    Loads an image and applies a robust pre-processing pipeline to prepare it for OCR.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not read image from path: {image_path}")
        return None

    if np.mean(image) < 128:
        image = cv2.bitwise_not(image)

    _, processed_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return processed_image

def extract_text_from_image(processed_image):
    """
    Uses Tesseract to perform OCR on the processed image.
    """
    custom_config = r'--oem 3 --psm 4'
    try:
        text = pytesseract.image_to_string(processed_image, lang='deu+eng', config=custom_config)
        return text
    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract is not installed or not in your PATH.")
        return None

def parse_receipt_text(text):
    """
    Parses raw OCR text with an advanced, stateful, index-skipping logic to handle
    complex receipts, including priceless items and aggressive garbage filtering.
    """
    lines = text.split('\n')
    
    date_str = "N/A"
    date_pattern = r'(\d{2}[-.]\d{2}[-.,]\s?\d{4})'
    for line in lines[-10:]:
        match = re.search(date_pattern, line)
        if match:
            # Extract all numbers and join with a hyphen for a clean format
            nums = re.findall(r'\d+', match.group(1))
            if len(nums) == 3:
                date_str = '-'.join(nums)
            break

    found_items = []
    IGNORE_KEYWORDS = ['verbilligung', 'mwst', 'rabatt', 'total', 'bargeld', 'rückgeld', 'bezeichnung']
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # --- Aggressive Filtering of Non-Item Lines ---
        # A valid item MUST start with a tax code (e.g., "B ", "D ")
        if not re.match(r'^[A-Z]\s', line):
            i += 1
            continue
        if any(kw in line.lower() for kw in IGNORE_KEYWORDS):
            i += 1
            continue
        if re.search(r'-\d+[\.,]\d+', line): # Filter discounts
            i += 1
            continue
        
        product_name_raw = line
        price = np.nan
        price_per_one = np.nan
        size_volume = "N/A"
        
        prices_on_line = re.findall(r'\d+[\.,]\d{1,2}', line)
        
        # Assume we will process 1 line, unless we find a 2-line item
        processed_lines = 1

        # CASE 1: Two-line item (item name has NO price, details are on next line)
        if not prices_on_line and i + 1 < len(lines):
            next_line = lines[i+1]
            prices_on_next_line = re.findall(r'\d+[\.,]\d{1,2}', next_line)
            
            if prices_on_next_line:
                price_str = prices_on_next_line[-1] # Total price is always last
                price = float(price_str.replace(',', '.'))
                price_per_one = price # Default to total price

                qty_match = re.search(r'^(\d+)\s*[xX%]', next_line)
                if qty_match:
                    quantity = int(qty_match.group(1))
                    if quantity > 0:
                        price_per_one = price / quantity
                        size_volume = f"{quantity} pcs"
                elif 'kg' in next_line.lower() or 'ko' in next_line.lower():
                    if prices_on_next_line:
                        size_volume = f"{prices_on_next_line[0]} Kg"
                elif len(prices_on_next_line) >= 2:
                    potential_ppo = float(prices_on_next_line[0].replace(',', '.'))
                    if potential_ppo > price:
                        price_per_one = price / 2
                        size_volume = "2 pcs"
                    else:
                        price_per_one = potential_ppo
                
                processed_lines = 2 # We processed two lines, so skip both.
        
        # CASE 2: Standard one-line item
        elif prices_on_line:
            price_str = prices_on_line[-1]
            price = float(price_str.replace(',', '.'))
            price_per_one = price
            product_name_raw = line.rsplit(price_str, 1)[0]
        
        # --- Process and Finalize the Found Item (even if it's priceless) ---
        product_name = re.sub(r'^[A-Z#]\s', '', product_name_raw).strip()

        # NEW, MORE AGGRESSIVE GARBAGE FILTER
        if len(product_name.split(' ')[0]) < 3 and len(product_name) < 5:
            i += processed_lines
            continue

        if size_volume == "N/A":
            size_match = re.search(r'(\d+[\.,]?\d*\s*(?:g|kg|l|ml|Stk|Pet|Ikg)\b)', product_name, re.IGNORECASE)
            if size_match:
                size_volume = size_match.group(1)
                product_name = product_name.replace(size_volume, '').strip()

        found_items.append({
            "date": date_str,
            "name": product_name.strip(),
            "size": size_volume,
            "price": price,
            "price_per_one": price_per_one
        })
        
        i += processed_lines
            
    return found_items


def main():
    # ... main function remains exactly the same ...
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
    if processed_image is None:
        return

    print("2. Performing OCR to extract text...")
    raw_text = extract_text_from_image(processed_image)
    if raw_text is None:
        return
        
    if args.debug:
        base_filename = os.path.basename(args.image_path)
        name, ext = os.path.splitext(base_filename)
        debug_filename = os.path.join(args.output_dir, f"{name}_debug_preprocessed_image.png")
        cv2.imwrite(debug_filename, processed_image)
        print(f"\n--- Preprocessed image saved as {debug_filename} ---")
        
        print("\n--- Raw OCR Text ---\n")
        print(raw_text)
        print("\n--- End Raw OCR Text ---\n")

    print("3. Parsing text to find items...")
    items = parse_receipt_text(raw_text)

    if not items:
        print("Could not find any items.")
        return
    
    print("\n--- Processed Receipt Data ---\n")
    print("date\tproduct name\tsize/nr of pieces/volume\tprice\tprice per 1")
    for item in items:
        # Custom formatting to handle NaN values gracefully
        price_str = f"{item['price']:.2f}" if not np.isnan(item['price']) else "NaN"
        ppo_str = f"{item['price_per_one']:.2f}" if not np.isnan(item['price_per_one']) else "NaN"
        
        print(
            f"{item['date']}\t"
            f"{item['name']}\t"
            f"{item['size']}\t"
            f"{price_str}\t"
            f"{ppo_str}"
        )

if __name__ == "__main__":
    main()