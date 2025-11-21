# main.py

import argparse
import time
import collections
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Set

from paddleocr import PaddleOCR  # <<< ADDED: Import PaddleOCR

# Import from our new modules
from config import (
    load_configuration, Configuration,
    OCR_CACHE_DIR, OUTPUT_DIR, CORRECTION_FILES_DIR, DEBUG_IMG_DIR,
    MANUAL_INPUT_FILENAME
)
from data_models import ReceiptItem
from processing.receipt_processor import ReceiptProcessor
from parsers.base_parser import BaseParser
from parsers.denner_parser import DennerParser
from parsers.lidl_parser import LidlParser
from parsers.migros_parser import MigrosParser # <<< ADDED: Import your new Migros parser
import data_utils
import reporting

# --- Helper Functions for Main Execution ---

def get_parser(shop_name: str) -> BaseParser:
    """Factory function to get the correct parser class based on shop name."""
    if shop_name == 'denner':
        return DennerParser()
    elif shop_name == 'lidl':
        return LidlParser()
    elif shop_name == 'migros': # <<< ADDED: Case for Migros
        return MigrosParser()
    else:
        raise ValueError(f"Unknown or unsupported shop: {shop_name}")

def get_files_to_process(input_path: Path) -> List[Path]:
    """Scans a directory or validates a single file for processing."""
    VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    if input_path.is_dir():
        print(f"Input is a directory. Scanning for images in: {input_path}")
        return sorted([p for p in input_path.iterdir() if p.suffix.lower() in VALID_EXTENSIONS])
    elif input_path.is_file() and input_path.suffix.lower() in VALID_EXTENSIONS:
        print(f"Input is a single file: {input_path}")
        return [input_path]
    else:
        print(f"Error: Path not found or is not a supported image file/directory: {input_path}")
        return []

def setup_environment(debug_mode: bool):
    """Create all necessary directories for the application to run."""
    print("\n" + "="*20 + " Setting Up Environment " + "="*20)
    OCR_CACHE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    CORRECTION_FILES_DIR.mkdir(exist_ok=True)
    if debug_mode:
        DEBUG_IMG_DIR.mkdir(exist_ok=True)
    print("All necessary directories are present.")


# --- Main Execution Pipeline ---

# <<< CHANGED: Function now accepts the paddle_engine
def process_receipts(files: List[Path], config: Configuration, args: argparse.Namespace, paddle_engine: PaddleOCR) -> Tuple[List[ReceiptItem], float, Set[Tuple[str, str]]]:
    """Process all receipt images and collect the initial data."""
    all_items = []
    total_ocr_time = 0.0
    all_missing_names = set()
    parser = get_parser(args.shop)

    for image_path in files:
        # <<< CHANGED: Pass the paddle_ocr_engine to the processor
        processor = ReceiptProcessor(
            image_path,
            parser=parser,
            shop_name=args.shop,
            paddle_ocr_engine=paddle_engine, # Pass the initialized engine
            debug=args.debug
        )
        items, ocr_duration, missing = processor.process(config)
        all_items.extend(items)
        total_ocr_time += ocr_duration
        all_missing_names.update(missing)
    return all_items, total_ocr_time, all_missing_names

def finalize_data(items: List[ReceiptItem], config: Configuration) -> List[ReceiptItem]:
    """Apply all post-processing steps: corrections, manual data, sorting, and final assignments."""
    print("\n" + "="*20 + " Finalizing Data Set " + "="*21)
    
    manual_items = data_utils.load_manual_inputs(CORRECTION_FILES_DIR / MANUAL_INPUT_FILENAME)
    if manual_items:
        items = data_utils.insert_manual_items(items, manual_items)

    items = data_utils.apply_corrections(items, config.corrections_map)
    
    items.sort()
    
    final_items = []
    grouped_by_date = collections.defaultdict(list)
    for item in items:
        grouped_by_date[item.date].append(item)
        
    for date_key in sorted(grouped_by_date.keys(), key=lambda d: datetime.strptime(d, '%d.%m.%Y') if d != "N/A" else datetime.max):
        date_items = grouped_by_date[date_key]
        for i, item in enumerate(date_items):
            item.order_number = i + 1
        final_items.extend(date_items)
        
    for i, item in enumerate(final_items):
        item.pid = i + 1

    data_utils.apply_naive_corrections(final_items)
    
    return final_items

def generate_reports(items: List[ReceiptItem], config: Configuration, missing_names: Set, total_duration: float, ocr_duration: float):
    """Generate all console and file-based outputs."""
    if not items and not missing_names:
        print("\n--- Finished. No items found and no new products to correct. ---")
        return
        
    print(f"\n--- Finished processing. Found a total of {len(items)} items. ---")
    reporting.display_console_preview(items)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_filepath = OUTPUT_DIR / f"receipt_data_{timestamp}.csv"
    reporting.save_csv(items, output_filepath)

    if config.polish_map:
        polish_filepath = OUTPUT_DIR / f"polish_receipt_data_{timestamp}.csv"
        reporting.save_csv(items, polish_filepath, polish_map=config.polish_map)
        
    reporting.display_summary(items)
    reporting.save_correction_template(items, config, missing_names)
    reporting.display_performance_metrics(total_duration, len(items), ocr_duration)

def main():
    parser = argparse.ArgumentParser(description="Process receipt images to extract itemized data.")
    parser.add_argument("input_path", help="Path to a receipt image file OR a directory of images.")
    # <<< CHANGED: Added 'migros' to the available choices
    parser.add_argument("--shop", choices=['denner', 'lidl', 'migros'], required=True, help="The shop the receipt is from.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save processed images.")
    args = parser.parse_args()
    
    start_time = time.perf_counter()
    
    # 1. Setup
    setup_environment(args.debug)
    config = load_configuration()
    files_to_process = get_files_to_process(Path(args.input_path))
    if not files_to_process:
        return
        
    # <<< ADDED: Initialize PaddleOCR engine once at the start
    print("\n" + "="*20 + " Initializing OCR Engines " + "="*19)
    # The engine is only loaded if needed, but we define the variable.
    paddle_engine = None
    if args.shop == 'migros':
        print("Initializing PaddleOCR engine (this may take a moment)...")
        # Use lang='de' for German, add use_textline_orientation for better results
        paddle_engine = PaddleOCR(lang='de', use_textline_orientation=True)
        print("PaddleOCR engine ready.")
    else:
        print("Tesseract will be used for OCR.")

    # 2. Process
    # <<< CHANGED: Pass the engine to the processing function
    all_items, total_ocr_time, all_missing_names = process_receipts(files_to_process, config, args, paddle_engine)
    
    # 3. Finalize
    final_items = finalize_data(all_items, config)
    
    # 4. Report
    total_duration = time.perf_counter() - start_time
    generate_reports(final_items, config, all_missing_names, total_duration, total_ocr_time)

if __name__ == "__main__":
    main()