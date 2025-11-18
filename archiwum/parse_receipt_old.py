#!/usr/bin/env python3

import cv2
import pytesseract
import re
import argparse
import os
from datetime import datetime

def preprocess_image(image_path):
    """
    Loads an image, converts it to grayscale, and applies a threshold
    to make it binary (black and white) to improve OCR accuracy.
    """
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from path: {image_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to binarize the image.
    # This is often better than a simple global threshold for photos with uneven lighting.
    processed_image = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    return processed_image

def extract_text_from_image(processed_image):
    """
    Uses Tesseract to perform OCR on the processed image.
    Specifies English and German as potential languages.
    """
    # Use pytesseract to extract text. Add language codes as needed.
    # 'deu' for German, 'eng' for English.
    custom_config = r'--oem 3 --psm 6'
    try:
        text = pytesseract.image_to_string(processed_image, lang='deu+eng', config=custom_config)
        return text
    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract is not installed or not in your PATH.")
        print("On Ubuntu, install it with: sudo apt install tesseract-ocr")
        return None

def parse_receipt_text(text):
    """
    Parses the raw OCR text to find the date and individual product lines.
    This is the most crucial and store-specific part of the script.
    """
    lines = text.split('\n')
    
    # --- Date Extraction ---
    # Try to find a date in common formats (DD.MM.YYYY, DD/MM/YYYY, etc.)
    date_str = "N/A"
    date_pattern = r'\d{2}[./-]\d{2}[./-]\d{4}'
    for line in lines:
        match = re.search(date_pattern, line)
        if match:
            # Try to parse the date to ensure it's valid
            try:
                # Normalize separators to '-' for consistent parsing
                normalized_date = match.group(0).replace('.', '-').replace('/', '-')
                datetime.strptime(normalized_date, '%d-%m-%Y')
                date_str = normalized_date
                break # Stop after finding the first valid date
            except ValueError:
                continue

    # --- Item Extraction ---
    # This regex is the core of the item parsing. It looks for lines that
    # likely contain a product name followed by a price.
    #
    # Explanation of the regex:
    # ^(.*?)             - (Group 1: Product Name) Non-greedily capture any characters at the start of the line.
    # \s+                - One or more whitespace characters.
    # (\d+[\.,]\d{2})     - (Group 2: Price) Capture a number in the format X.XX or X,XX.
    # \s*CHF?            - Optional whitespace and currency symbol (e.g., CHF)
    # \s*$               - Optional whitespace at the end of the line.
    #
    # **This regex is a good starting point but may need to be adjusted for your specific receipts.**
    item_pattern = re.compile(r'^(.*?)\s+(\d+[\.,]\d{2})\s*CHF?$', re.MULTILINE)

    found_items = []
    
    for match in item_pattern.finditer(text):
        product_name_raw = match.group(1).strip()
        price_str = match.group(2).replace(',', '.')
        price = float(price_str)

        # --- Sub-parsing for Size/Quantity and Price per unit ---
        size_volume = "N/A"
        price_per_one = price # Default to the line price
        
        # Check for patterns like "2 x 1.50" or "3 Stk"
        quantity_match = re.search(r'^(\d+)\s*[xX]\s+', product_name_raw)
        if quantity_match:
            quantity = int(quantity_match.group(1))
            # Price on the line is often the total price, so divide to get per-item price
            price_per_one = price / quantity
            # Clean the quantity from the product name
            product_name = re.sub(r'^\d+\s*[xX]\s+', '', product_name_raw).strip()
            size_volume = f"{quantity} pcs"
        else:
            product_name = product_name_raw
            # Look for weight/volume patterns like 500g, 1.5L, etc.
            size_match = re.search(r'(\d+[\.,]?\d*\s*(?:g|kg|l|ml|Stk)\b)', product_name, re.IGNORECASE)
            if size_match:
                size_volume = size_match.group(1)
                # Remove the size from the product name for cleanliness
                product_name = product_name.replace(size_volume, '').strip()

        # Filter out lines that are likely totals or discounts
        if any(keyword in product_name.lower() for keyword in ['total', 'subtotal', 'rabatt', 'card', 'mwst']):
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
    parser = argparse.ArgumentParser(description="Process a shop receipt image to extract itemized data.")
    parser.add_argument("image_path", help="The path to the receipt image file.")
    parser.add_argument("--debug", action="store_true", help="Print the raw OCR text for debugging.")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: File not found at {args.image_path}")
        return

    print("1. Pre-processing image...")
    processed_image = preprocess_image(args.image_path)
    if processed_image is None:
        return

    print("2. Performing OCR to extract text...")
    raw_text = extract_text_from_image(processed_image)
    if raw_text is None:
        return
        
    if args.debug:
        print("\n--- Raw OCR Text ---\n")
        print(raw_text)
        print("\n--- End Raw OCR Text ---\n")

    print("3. Parsing text to find items...")
    items = parse_receipt_text(raw_text)

    if not items:
        print("Could not find any items. The receipt format might not be recognized.")
        print("Try running with --debug to see the raw text and adjust the regex patterns in the script.")
        return
    
    print("\n--- Processed Receipt Data ---\n")
    # Print header
    print("date\tproduct name\tsize/nr of pieces/volume\tprice\tprice per 1")
    # Print item data
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