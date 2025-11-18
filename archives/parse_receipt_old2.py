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
        print("On Ubuntu, install it with: sudo apt install tesseract-ocr tesseract-ocr-deu")
        return None

def parse_receipt_text(text):
    """
    Parses raw OCR text with advanced logic to handle multi-buy items
    where the price is on a separate line.
    """
    lines = text.split('\n')
    
    date_str = "N/A"
    date_pattern = r'(\d{2}[-.]\d{2}[-.]\d{4})'
    for line in lines[-10:]:
        match = re.search(date_pattern, line)
        if match:
            date_str = match.group(1).replace('.', '-')
            break

    found_items = []
    # This regex now captures: (1) quantity, (2) price-per-one, (3) total price
    multi_buy_pattern = re.compile(r'^\s*(\d+)\s*[xX]\s*(\d+[\.,]\d{2}).*?(\d+[\.,]\d{2})\b')
    price_pattern = re.compile(r'(\d+[\.,]\d{1,2})\b')

    for i, line in enumerate(lines):
        line = line.strip()

        if multi_buy_pattern.match(line):
            continue

        if not (line and re.match(r'^([A-Z]|\d)\s', line)):
            continue
            
        price = None
        price_per_one = None
        quantity = None
        product_name_raw = line
        is_multi_buy = False

        # CASE 1: Multi-buy item (price is on the NEXT line)
        if i + 1 < len(lines):
            multi_match = multi_buy_pattern.search(lines[i+1])
            if multi_match:
                is_multi_buy = True
                quantity = int(multi_match.group(1))
                price_per_one = float(multi_match.group(2).replace(',', '.'))
                price = float(multi_match.group(3).replace(',', '.'))
                product_name_raw = line

        # CASE 2: Standard item (price is on the CURRENT line)
        if not is_multi_buy:
            price_match = price_pattern.search(line)
            if price_match:
                price = float(price_match.group(1).replace(',', '.'))
                price_per_one = price
                product_name_raw = line[:price_match.start()].strip()

        # If we found a valid item (either standard or multi-buy), process it
        if price is not None:
            product_name_raw = re.sub(r'^([A-Z]|\d)\s', '', product_name_raw).strip()
            size_volume = "N/A"

            if is_multi_buy:
                size_volume = f"{quantity} pcs"
            
            size_match = re.search(r'(\d+[\.,]?\d*\s*(g|kg|l|ml|Stk)\b)', product_name_raw, re.IGNORECASE)
            if size_match:
                # For multi-buy, we prefer "X pcs", otherwise use the extracted size
                if not is_multi_buy:
                    size_volume = size_match.group(1)
                # Clean the size from the product name regardless
                product_name_raw = product_name_raw.replace(size_match.group(1), '').strip()
            
            product_name = product_name_raw.strip()

            if any(keyword in product_name.lower() for keyword in ['mwst', 'total']) or not product_name:
                continue

            found_items.append({
                "date": date_str,
                "name": product_name,
                "size": size_volume,
                "price": price,
                "price_per_one": price_per_one
            })

    return found_items


def main():
    parser = argparse.ArgumentParser(description="Process a Denner shop receipt image to extract itemized data.")
    parser.add_argument("image_path", help="The path to the ORIGINAL receipt image file.")
    parser.add_argument("--debug", action="store_true", help="Print the raw OCR text and save the processed image for debugging.")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: File not found at {args.image_path}")
        return

    print(f"1. Processing image: {args.image_path}")
    processed_image = preprocess_image(args.image_path)
    if processed_image is None:
        return

    print("2. Performing OCR to extract text...")
    raw_text = extract_text_from_image(processed_image)
    if raw_text is None:
        return
        
    if args.debug:
        file_name_without_extension = os.path.splitext(os.path.basename(args.image_path))[0]
        debug_filename = f"{file_name_without_extension}_debug_preprocessed_image.png"
        #find path to the parent folder of the image
        image_path = os.path.dirname(args.image_path)   
        parent_path = os.path.dirname(image_path)     
        #change folder to preprocessed_images
        debug_filename = os.path.join("preprocessed_images", debug_filename)
        debug_filename = os.path.join(parent_path, debug_filename)
        cv2.imwrite(debug_filename, processed_image)
        print(f"\n--- Preprocessed image saved as {debug_filename} ---")
        
        print("\n--- Raw OCR Text ---\n")
        print(raw_text)
        print("\n--- End Raw OCR Text ---\n")

    print("3. Parsing text to find items...")
    items = parse_receipt_text(raw_text)

    if not items:
        print("Could not find any items. The OCR quality may still be too low.")
        print("Check the debug_preprocessed_image.png and the raw text output.")
        return
    
    print("\n--- Processed Receipt Data ---\n")
    print("date\tproduct name\tsize/nr of pieces/volume\tprice\tprice per 1")
    for item in items:
        print(
            f"{item['date']}\t"
            f"{item['name']}\t"
            f"{item['size']}\t"
            f"{item['price']:.2f}\t"
            f"{item['price_per_one']:.2f}"
        )

if __name__ == "__main__":
    main()