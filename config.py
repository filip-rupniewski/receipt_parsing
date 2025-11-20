#!/usr/bin/env python3

"""
Central configuration module for the receipt processor application.

This module contains:
- All static file and directory paths.
- The Configuration dataclass to hold loaded settings.
- Functions to load all configuration data from CSV files.
"""

import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict

# --- 1. Constants & File Paths ---

# Directories
CORRECTION_FILES_DIR = Path("correction_files")
OCR_CACHE_DIR = Path("ocr_cache")
OUTPUT_DIR = Path("output_data")
DEBUG_IMG_DIR = Path("preprocessed_images")

# Filenames
CORRECTIONS_FILENAME = "product_name_corrections.csv"
MANUAL_INPUT_FILENAME = "manual_input.csv"
MANUAL_CHANGES_FILENAME = "manual_changes.csv"
MANUAL_DATES_FILENAME = "manual_dates.csv"
STANDARD_SIZES_FILENAME = "standard_sizes.csv"
POLISH_TRANSLATIONS_FILENAME = "polish_translations.csv"
CORRECTION_TEMPLATE_FILENAME = "product_name_correction_template.csv"

# CSV Headers
CSV_HEADER = ['PID', 'status', 'date', 'order number', 'product name', 'size/volume', 'price', 'price per 1', 'shop', 'discount']


# --- 2. Configuration Data Structure ---

@dataclass
class Configuration:
    """Holds all loaded configuration data from CSV files."""
    corrections_map: Dict[str, dict] = field(default_factory=dict)
    size_map: Dict[str, float] = field(default_factory=dict)
    manual_dates: Dict[str, str] = field(default_factory=dict)
    polish_map: Dict[str, str] = field(default_factory=dict)


# --- 3. Configuration Loading Functions ---

def _load_csv_map(filepath: Path, key_col: int, val_col: int, has_header: bool = True) -> Dict[str, str]:
    """Helper to load a 2-column CSV into a dictionary."""
    data_map = {}
    if not filepath.exists():
        print(f"\nInfo: Data file not found at '{filepath}'. This may be expected.")
        return data_map
    try:
        with filepath.open('r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            if has_header:
                next(reader, None)
            for row in reader:
                if len(row) > max(key_col, val_col) and row[key_col].strip():
                    key = row[key_col].strip()
                    val = row[val_col].strip()
                    data_map[key] = val
        print(f"\nLoaded {len(data_map)} entries from '{filepath.name}'.")
    except Exception as e:
        print(f"\nError reading file '{filepath}': {e}. Skipping.")
    return data_map

def load_configuration() -> Configuration:
    """Loads all necessary configuration files into a single object."""
    print("\n" + "="*20 + " Loading Configuration " + "="*20)
    config = Configuration()
    config.manual_dates = _load_csv_map(CORRECTION_FILES_DIR / MANUAL_DATES_FILENAME, key_col=0, val_col=1, has_header=False)
    config.polish_map = _load_csv_map(CORRECTION_FILES_DIR / POLISH_TRANSLATIONS_FILENAME, key_col=1, val_col=2)
    
    # Load size map
    size_map_str = _load_csv_map(CORRECTION_FILES_DIR / STANDARD_SIZES_FILENAME, key_col=1, val_col=2)
    for name, size_str in size_map_str.items():
        try:
            config.size_map[name] = float(size_str.replace(',', '.'))
        except ValueError:
             print(f"Warning: Invalid size for '{name}' in '{STANDARD_SIZES_FILENAME}'. Skipping.")

    # Load corrections map (more complex structure)
    corrections_path = CORRECTION_FILES_DIR / CORRECTIONS_FILENAME
    if corrections_path.exists():
        try:
            with corrections_path.open('r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=';')
                next(reader, None)
                for row in reader:
                    if len(row) < 4:
                        continue
                    _, ocr_name, correct_name, correct_size_str = [field.strip() for field in row]
                    if not ocr_name:
                        continue
                    rule = {}
                    if correct_name:
                        rule['name'] = correct_name
                    if correct_size_str:
                        try:
                            rule['size'] = float(correct_size_str.replace(',', '.'))
                        except ValueError:
                            print(f"Warning: Invalid size '{correct_size_str}' for '{ocr_name}' in corrections file.")
                    if rule:
                        config.corrections_map[ocr_name] = rule
            print(f"Loaded {len(config.corrections_map)} rules from '{corrections_path.name}'.")
        except Exception as e:
            print(f"\nError reading corrections file '{corrections_path}': {e}. Skipping.")
    
    return config