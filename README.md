# Receipt OCR Processor

A Python script to automatically extract itemized data from shopping receipt images using Optical Character Recognition (OCR). It processes images, parses the text for products, prices, and sizes, applies corrections, and outputs the structured data into a single CSV file.

 <!-- This is a placeholder image. You could create a diagram showing the flow. -->

## Key Features

-   **Image Pre-processing:** Automatically enhances receipt images for better OCR accuracy (grayscale, inversion, Otsu's binarization).
-   **OCR Text Extraction:** Uses the Tesseract engine to extract text from images.
-   **Intelligent Parsing:** Uses regular expressions to identify individual items, prices, sizes, and discounts from raw text.
-   **Rule-Based Data Correction:** Fix common OCR errors by applying rules from a user-maintained CSV file.
-   **Heuristic Data Enrichment:** "Naively" fills in missing price or size data for an item by using information from historical purchases of the same item.
-   **Structured CSV Output:** Saves all extracted data into a clean, timestamped CSV file, sorted chronologically.
-   **Correction Template Generation:** Automatically creates a CSV template with all unique product names to make adding correction rules easy.
-   **Console Reporting:** Provides a data preview, a final summary of product counts, and performance metrics after execution.
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
-   **Linux (Debian/Ubuntu):** Install via apt: `sudo apt update && sudo apt install tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng`

### 2. Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/filip-rupniewski/receipt_parsing.git
    cd receipt-parsing
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
    ```bash
    pip install -r requirements.txt
    ```

## How to Use

### 1. Directory Structure

Place your receipt images (e.g., `.jpg`, `.png`) in a directory. The script will create the other directories it needs.

```
receipt_parsing_/
├── receipts_pictures/        <-- Put your receipt images here
│   ├── receipt1.jpg
│   └── receipt2.png
├── correction_files/     <-- Correction CSV files go here
├── output_data/          <-- The final CSV output will be saved here
├── preprocessed_images/  <-- images after prepreocessing are saved here
└── process_receipts.py   <-- The main script
```

### 2. Running the Script

Execute the script from your terminal, pointing it to your image file or directory.

**To process a single image:**
```bash
python process_receipts.py path/to/your_receipts/receipt1.jpg
```

**To process an entire directory of images:**
```bash
python process_receipts.py path/to/your_receipts/
```

**To run in debug mode:**
```bash
python process_receipts.py path/to/your_receipts/ --debug
```

### 3. The Correction Workflow

The script's power comes from its ability to learn from corrections.

1.  **First Run:** Run the script on your images for the first time. It may have errors in product names or sizes.
2.  **Generate Template:** The script will create a file at `correction_files/product_name_correction_template.csv`. This file contains all the unique product names it found.
3.  **Add Your Corrections:** Open the `...template.csv` file.
    - In the `correct product name` column, type the correct name for any product that was misidentified by OCR.
    - In the `correct size` column, enter the correct standardized size (in `kg` or `l`) if it was missed or incorrect.

    **Example `product_name_correction_template.csv`:**
    ```csv
    nr;product name;correct product name;correct size
    1;APFEL GOLDEN;Apfel Golden Delicious;1
    2;MlLCH DRINK;Milch Drink;0,5
    3;Tomaten;Tomaten;
    ```
4.  **Activate Corrections:** Rename the template file from `product_name_correction_template.csv` to `product_name_corrections.csv` within the `correction_files` directory.
5.  **Re-run the Script:** Run the script again. This time, it will load your rules and automatically apply the corrections, resulting in much cleaner data.

## Output

-   **Primary CSV File:** A file named `receipt_data_YYYY-MM-DD_HH-MM-SS.csv` will be created in the `output_data` directory. It contains all the processed items, sorted by date.
-   **Console Output:** You will see a progress log, a preview of the first 20 processed items, a final summary of product counts, and performance metrics.

## Future Improvements (TODO)

-   [ ] Add a mechanism for purely manual entries (e.g., from a separate CSV file).
-   [ ] Implement a more advanced correction system for one-off manual fixes.
-   [ ] Create a historical price library to better flag price anomalies or estimate missing prices.
-   [ ] Support for additional supermarket receipt formats.
