# Receipt OCR Processor

A Python script to automatically extract itemized data from shopping receipt images using Optical Character Recognition (OCR). It processes images from different shops (Denner, Lidl), parses the text for products and prices, applies a multi-layered correction system, and outputs the structured data into a single CSV file.

Diagram showing the flow: 

Image files -> Preprocessing -> OCR (with caching) -> Text Parsing -> Data Correction -> Final CSV Output

## Key Features

-   **Multi-Shop Support:** Includes dedicated parsing logic for different supermarket receipt formats (currently **Denner** and **Lidl**).
-   **Image Pre-processing:** Automatically enhances receipt images for better OCR accuracy (grayscale, inversion for dark backgrounds, Otsu's binarization).
-   **OCR with Caching:** Uses the Tesseract engine to extract text and saves the result in an `ocr_cache` directory. Subsequent runs on the same image are instantaneous, avoiding re-processing.
-   **Intelligent Parsing:** Uses shop-specific regular expressions to identify individual items, prices, sizes, and discounts from raw text.
-   **Advanced Data Correction:**
    -   **Product Corrections (`product_name_corrections.csv`):** Fixes common OCR errors in product names and fills in missing sizes.
    -   **Standard Sizes (`standard_sizes.csv`):** Defines default sizes for products, enabling the script to calculate total volume for multi-buy items (e.g., `3 x Milk`).
    -   **Manual Overrides (`manual_dates.csv`, `manual_input.csv`):** Allows for manual entry of entire items or overriding incorrect dates for specific receipts.
-   **Heuristic Data Enrichment:** "Naively" fills in missing price or size data for an item by looking up the price-per-unit from historical purchases of the same item.
-   **Structured CSV Output:** Saves all extracted data into a clean, timestamped CSV file, sorted chronologically. An optional translated version can also be generated.
-   **Correction Template Generation:** Automatically creates a CSV template (`product_name_correction_template.csv`) with all unique product names to make adding correction rules easy.
-   **Debug Mode:** An optional `--debug` flag saves the pre-processed images and raw OCR text for troubleshooting.

## Technologies Used

-   **Python 3**
-   **OpenCV (`opencv-python`):** For all image processing tasks.
-   **Tesseract OCR Engine:** The core OCR engine.
-   **Pytesseract (`pytesseract`):** A Python wrapper for Tesseract.
-   **NumPy:** For numerical operations and image data handling.

## Setup and Installation

### 1. Prerequisites

You must have the Tesseract OCR engine installed on your system. It is a separate program that `pytesseract` calls.

-   **Windows:** Download and run the installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). Make sure to add the Tesseract installation directory to your system's `PATH` environment variable.
-   **macOS:** Install via Homebrew: `brew install tesseract`
-   **Linux (Debian/Ubuntu):** The script uses German and English language packs. Install them with Tesseract:
    `sudo apt update && sudo apt install tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng`

### 2. Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required Python packages:**
    *(Assuming you have a `requirements.txt` file)*
    ```bash
    pip install opencv-python pytesseract numpy
    ```

## How to Use

### 1. Directory Structure

The script is designed to work with a specific directory structure. It will create these directories automatically if they don't exist.

```
your_project_folder/
├── your_receipts/          <-- Put your receipt images here
│   ├── receipt1.jpg
│   └── receipt2.png
├── correction_files/       <-- All manual data and correction CSV files
│   ├── product_name_corrections.csv
│   ├── standard_sizes.csv
│   ├── manual_dates.csv
│   ├── manual_input.csv
│   └── polish_translations.csv
├── ocr_cache/              <-- Cached raw text from OCR is stored here
├── output_data/            <-- The final CSV outputs will be saved here
├── preprocessed_images/    <-- Saved here if you use --debug mode
└── process_receipts.py     <-- The main script
```

### 2. Running the Script

Execute the script from your terminal. You must specify the input path (a single image or a directory) and the shop it's from.

**To process a single image from Denner:**
```bash
python process_receipts.py path/to/your_receipts/receipt1.jpg --shop denner
```

**To process an entire directory of Lidl receipts:**
```bash
python process_receipts.py path/to/your_receipts/ --shop lidl
```

**To run in debug mode:**
```bash
python process_receipts.py path/to/your_receipts/ --shop denner --debug
```

### 3. The Correction & Data Workflow

The script's power comes from its ability to learn from corrections via several CSV files in the `correction_files/` directory.

1.  **First Run:** Run the script on your images. It may have errors in product names or fail to find dates.
2.  **Generate Template:** The script will create `correction_files/product_name_correction_template.csv`. This file contains every unique product name it found.
3.  **Add Your Corrections:**
    - **`product_name_corrections.csv`**: Rename the template to this. Open it and fill in the `correct product name` and `correct size` columns for any misidentified items. A `-` in the `correct product name` column will cause the item to be deleted from the output.
    - **`standard_sizes.csv`**: To help parse multi-buy items (e.g., `3 x 2.50`), add the product's correct name and its standard size (in kg or L) here.
    - **`manual_dates.csv`**: If the script can't find a date for an image, add an entry here mapping the filename to the correct date (e.g., `receipt1.jpg;24.12.2023`).
    - **`manual_input.csv`**: If an item was missed entirely, you can add it manually to this file, and it will be merged into the final output.
4.  **Re-run the Script:** Run the script again. It will now load all your rules and overrides, resulting in much cleaner and more complete data.

## Output

-   **Primary CSV File:** A file named `receipt_data_YYYY-MM-DD_HH-MM-SS.csv` will be created in the `output_data` directory. It contains all the processed items, sorted by date.
-   **Translated CSV File (Optional):** If you provide a `polish_translations.csv` file, a second output `polish_receipt_data_... .csv` will be created with product names translated.
-   **Console Output:** You will see a progress log, a preview of the first 20 processed items, a final summary of product counts, and performance metrics.

## Future Improvements (TODO)

-   [x] Add a mechanism for purely manual entries (`manual_input.csv`).
-   [ ] Expand support for additional supermarket receipt formats (e.g., Coop, Migros).