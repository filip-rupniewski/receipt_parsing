# processing/receipt_processor.py

import cv2
import pytesseract
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Set

# Import from our new modules
from config import OCR_CACHE_DIR, DEBUG_IMG_DIR, Configuration
from data_models import ReceiptItem
from parsers.base_parser import BaseParser

class ReceiptProcessor:
    """Handles preprocessing, OCR, and parsing for a single receipt image."""
    def __init__(self, image_path: Path, parser: BaseParser, shop_name: str, debug: bool = False):
        self.image_path = image_path
        self.parser = parser
        self.shop_name = shop_name
        self.debug = debug
        self.raw_text: Optional[str] = None

    def process(self, config: Configuration) -> Tuple[List[ReceiptItem], float, Set[Tuple[str, str]]]:
        """
        Main processing pipeline for a single image.
        Returns a list of found items, the OCR duration, and any names needing size definitions.
        """
        print(f"\n--- Processing: {self.image_path.name} (Shop: {self.shop_name.capitalize()}) ---")
        
        print("1. Preprocessing image...")
        processed_image = self._preprocess_image()
        if processed_image is None:
            print(f"Warning: Could not read or process image {self.image_path.name}. Skipping.")
            return [], 0.0, set()

        print("2. Extracting text (from cache or OCR)...")
        self.raw_text, ocr_duration = self._extract_text_from_image(processed_image)
        if not self.raw_text:
            print(f"Warning: OCR returned no text for {self.image_path.name}. Skipping.")
            return [], ocr_duration, set()

        date_str = config.manual_dates.get(self.image_path.name)
        if date_str:
            print(f"   -> Using manually provided date: {date_str}")
        else:
            # DELEGATE to the specific parser
            date_str = self.parser.parse_date(self.raw_text)
        
        self._update_ocr_cache_filename(date_str)

        if self.debug:
            self._save_debug_output(processed_image, date_str)
            
        print("3. Parsing text to find items...")
        # DELEGATE to the specific parser
        items, missing_names = self.parser.parse_items(self.raw_text, date_str or "N/A", config.corrections_map, config.size_map)
        
        if items:
            print(f"   -> Found {len(items)} items.")
        else:
            print("   -> Could not find any items in this image.")
        return items, ocr_duration, missing_names

    def _preprocess_image(self) -> Optional[np.ndarray]:
        image = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        if np.mean(image) < 128:
            image = cv2.bitwise_not(image)
        _, processed_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return processed_image
    
    def _extract_text_from_image(self, processed_image: np.ndarray) -> Tuple[Optional[str], float]:
        cache_file_glob = OCR_CACHE_DIR.glob(f"*_{self.image_path.stem}.txt")
        simple_cache_path = OCR_CACHE_DIR / f"{self.image_path.name}.txt"
        
        cache_files = list(cache_file_glob)
        if not cache_files and simple_cache_path.exists():
            cache_files.append(simple_cache_path)

        if cache_files:
            cache_filepath = cache_files[0]
            print(f"   -> Found cached OCR text. Loading from '{cache_filepath.name}'.")
            return cache_filepath.read_text(encoding='utf-8'), 0.0

        print("   -> No cache found. Performing OCR...")
        custom_config = r'--oem 3 --psm 4'
        start_ocr_time = time.perf_counter()
        text = None
        try:
            text = pytesseract.image_to_string(processed_image, lang='deu+eng', config=custom_config)
        except pytesseract.TesseractNotFoundError:
            print("Error: Tesseract is not installed or not in your PATH.")
        
        ocr_duration = time.perf_counter() - start_ocr_time

        if text:
            simple_cache_path.write_text(text, encoding='utf-8')
            print(f"   -> OCR text saved to temporary cache: {simple_cache_path.name}")
        
        return text, ocr_duration

    def _update_ocr_cache_filename(self, date_str: Optional[str]):
        simple_cache_path = OCR_CACHE_DIR / f"{self.image_path.name}.txt"
        if not simple_cache_path.exists():
            return
            
        date_for_filename = datetime.strptime(date_str, '%d.%m.%Y').strftime('%Y%m%d') if date_str else "NODATE"
        final_name = f"ocr_debug_{date_for_filename}_{self.image_path.stem}.txt"
        final_filepath = OCR_CACHE_DIR / final_name

        if simple_cache_path.name != final_filepath.name:
            simple_cache_path.rename(final_filepath)
            print(f"   -> Cache file updated to: {final_filepath.name}")
            
    def _save_debug_output(self, image: np.ndarray, date_str: Optional[str]):
        date_for_filename = datetime.strptime(date_str, '%d.%m.%Y').strftime('%Y%m%d') if date_str else "NODATE"
        debug_image_path = DEBUG_IMG_DIR / f"debug_{date_for_filename}_{self.image_path.stem}.png"
        cv2.imwrite(str(debug_image_path), image)
        print(f"   -> Preprocessed image saved as {debug_image_path.name}")
        preview = (self.raw_text[:250] + "...") if self.raw_text and len(self.raw_text) > 250 else self.raw_text
        print(f"   -> Raw OCR Text Preview:\n---\n{preview}\n---")