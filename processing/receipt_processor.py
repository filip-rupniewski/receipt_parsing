# processing/receipt_processor.py

import cv2
import pytesseract
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Set

# Import for PaddleOCR
from paddleocr import PaddleOCR

# Import from our new modules
from config import OCR_CACHE_DIR, DEBUG_IMG_DIR, Configuration
from data_models import ReceiptItem
from parsers.base_parser import BaseParser

# --- Helper Function for PaddleOCR (from your provided script) ---
def format_receipt_structure(texts: list, bboxes: list) -> str:
    """Formats PaddleOCR results into a structured layout preserving spatial positions."""
    if not texts or not bboxes or len(texts) != len(bboxes):
        return "\n".join(texts)
    
    items = []
    for text, bbox in zip(texts, bboxes):
        bbox_array = np.array(bbox)
        center_y = int(np.mean(bbox_array[:, 1]))
        min_x = int(np.min(bbox_array[:, 0]))
        max_x = int(np.max(bbox_array[:, 0]))
        items.append({'text': text, 'y': center_y, 'min_x': min_x, 'max_x': max_x})
    
    items.sort(key=lambda x: (x['y'], x['min_x']))
    
    rows, current_row, y_threshold = [], [], 20
    if items:
        current_row.append(items[0])
        for item in items[1:]:
            if abs(item['y'] - current_row[-1]['y']) < y_threshold:
                current_row.append(item)
            else:
                rows.append(sorted(current_row, key=lambda x: x['min_x']))
                current_row = [item]
        rows.append(sorted(current_row, key=lambda x: x['min_x']))

    output_lines = []
    for row in rows:
        line, last_x = "", 0
        for item in row:
            gap = item['min_x'] - last_x
            if last_x > 0:
                if gap > 80: line += '\t'
                elif gap > 15: line += '   '
                else: line += ' '
            line += item['text']
            last_x = item['max_x']
        output_lines.append(line)
    
    return "\n".join(output_lines)


class ReceiptProcessor:
    """Handles preprocessing, OCR, and parsing for a single receipt image."""
    def __init__(self, image_path: Path, parser: BaseParser, shop_name: str, 
                 paddle_ocr_engine: Optional[PaddleOCR] = None, debug: bool = False):
        self.image_path = image_path
        self.parser = parser
        self.shop_name = shop_name
        self.debug = debug
        self.paddle_ocr_engine = paddle_ocr_engine
        self.raw_text: Optional[str] = None
        self.processed_image: Optional[np.ndarray] = None # To store the image for debugging

    def process(self, config: Configuration) -> Tuple[List[ReceiptItem], float, Set[Tuple[str, str]]]:
        """
        Main processing pipeline for a single image.
        Returns a list of found items, the OCR duration, and any names needing size definitions.
        """
        print(f"\n--- Processing: {self.image_path.name} (Shop: {self.shop_name.capitalize()}) ---")
        
        # Step 1 and 2 are now combined in _extract_text_from_image
        print("1. Preprocessing image and extracting text (from cache or OCR)...")
        self.raw_text, ocr_duration = self._extract_text_from_image()
        
        if not self.raw_text:
            print(f"Warning: OCR returned no text for {self.image_path.name}. Skipping.")
            return [], ocr_duration, set()

        date_str = config.manual_dates.get(self.image_path.name)
        if date_str:
            print(f"   -> Using manually provided date: {date_str}")
        else:
            date_str = self.parser.parse_date(self.raw_text)
        
        self._update_ocr_cache_filename(date_str)

        if self.debug and self.processed_image is not None:
            self._save_debug_output(self.processed_image, date_str)
            
        print("2. Parsing text to find items...")
        items, missing_names = self.parser.parse_items(self.raw_text, date_str or "N/A", config.corrections_map, config.size_map)
        
        if items:
            print(f"   -> Found {len(items)} items.")
        else:
            print("   -> Could not find any items in this image.")
        return items, ocr_duration, missing_names

    # --- Tesseract-specific Preprocessing ---
    def _preprocess_image_tesseract(self) -> Optional[np.ndarray]:
        image = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        if np.mean(image) < 128:
            image = cv2.bitwise_not(image)
        _, processed_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return processed_image
    
    # --- PaddleOCR-specific Preprocessing ---
    def _preprocess_image_paddle(self) -> Optional[np.ndarray]:
        original_img = cv2.imread(str(self.image_path))
        if original_img is None:
            print(f"Error: Could not read image at {self.image_path}")
            return None

        max_dimension = 2000
        height, width = original_img.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_size = (int(width * scale), int(height * scale))
            original_img = cv2.resize(original_img, new_size, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
        blurred = cv2.medianBlur(sharpened, 3)
        return blurred

    # --- PaddleOCR-specific OCR execution ---
    def _perform_ocr_paddle(self, image_data: np.ndarray) -> str:
        """Performs OCR using PaddleOCR and formats the output."""
        try:
            # Ensure image is BGR for PaddleOCR
            if len(image_data.shape) == 2:
                image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
            
            result = self.paddle_ocr_engine.predict(image_data)
            if not result or not result[0]:
                return ""

            page_result = result[0]
            # Extract texts and bboxes directly from the result dictionary
            texts = page_result.get('rec_texts', [])
            bboxes = page_result.get('dt_polys', []) or page_result.get('rec_polys', [])

            if not texts or not bboxes:
                return ""

            structured_text = format_receipt_structure(texts, bboxes)
            return structured_text
        except Exception as e:
            import traceback
            print(f"An error occurred during PaddleOCR processing: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")
            return ""

    def _extract_text_from_image(self) -> Tuple[Optional[str], float]:
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
        start_ocr_time = time.perf_counter()
        text = None

        # === OCR ENGINE ROUTER ===
        if self.shop_name == 'migros':
            print("   -> Using PaddleOCR engine for Migros receipt.")
            if not self.paddle_ocr_engine:
                print("Error: Migros shop detected but PaddleOCR engine was not provided.")
                return None, 0.0
            
            self.processed_image = self._preprocess_image_paddle()
            if self.processed_image is not None:
                text = self._perform_ocr_paddle(self.processed_image)
        
        else: # Default to Tesseract for all other shops
            print("   -> Using Tesseract engine.")
            self.processed_image = self._preprocess_image_tesseract()
            if self.processed_image is not None:
                try:
                    custom_config = r'--oem 3 --psm 4'
                    text = pytesseract.image_to_string(self.processed_image, lang='deu+eng', config=custom_config)
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