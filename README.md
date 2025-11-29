# Bajaj Finserv HackRx Datathon - Bill Extraction API

**Author:** Keshav Kumar Singh | BITS Pilani, Hyderabad Campus

A production-ready API that extracts structured bill data from multi-page PDFs and images. Built for the Bajaj Finserv HackRx Datathon challenge.

---

## Overview

This solution processes medical bills through a straightforward two-step pipeline:

1. **OCR Step**: Converts PDF pages to images and extracts raw text using Tesseract OCR
2. **Extraction Step**: Parses the OCR text with rule-based heuristics to identify bill items, quantities, rates, and amounts

The approach prioritizes reliability and accuracy over complexity. Instead of relying on heavy ML models, I built a deterministic parser that handles the common patterns found in medical bills—pharmacy items, line items with quantities and rates, and page-level classifications.

---

## How It Works

### Step 1: OCR Processing

When a bill URL is submitted, the API:
- Downloads the file (PDF or image) to a temporary location
- If it's a PDF, converts each page to a high-resolution PNG using Poppler (300 DPI)
- Runs Tesseract OCR on each page to extract text
- Passes the raw OCR text to the extraction engine

### Step 2: Data Extraction

The extraction engine (`extract_items_from_text`) applies strict validation rules:

- **Currency Detection**: Only accepts amounts with explicit currency symbols (₹, Rs, INR), comma-formatted numbers (1,234.56), or numbers with more than 2 digits before the decimal
- **Filtering**: Rejects dates, years (19xx/20xx), and invoice numbers that might be misclassified as money
- **Item Parsing**: Extracts item names, quantities, rates, and amounts from each line
- **Page Classification**: Categorizes pages as "Bill Detail", "Final Bill", or "Pharmacy" based on content keywords
- **Confidence Scoring**: Assigns a confidence score (0.0-1.0) to each extracted item based on validation signals

The result is a structured JSON that matches the HackRx evaluation schema exactly.

---

## API Endpoints

### Health Check
```
GET /
```
Returns `{"status": "OK"}` to verify the API is running.

### Bill Extraction (HackRx Webhook)
```
POST /api/v1/hackrx/run
```

**Request Body:**
```json
{
  "document": "https://example.com/bill.pdf"
}
```

**Response Schema:**
```json
{
  "is_success": true,
  "token_usage": {
    "total_tokens": 0,
    "input_tokens": 0,
    "output_tokens": 0
  },
  "data": {
    "pagewise_line_items": [
      {
        "page_no": "1",
        "page_type": "Bill Detail",
        "bill_items": [
          {
            "item_name": "Paracetamol 650mg",
            "item_amount": 54.0,
            "item_rate": 18.0,
            "item_quantity": 3.0,
            "confidence": 0.95
          }
        ]
      },
      {
        "page_no": "2",
        "page_type": "Final Bill",
        "bill_items": []
      }
    ],
    "total_item_count": 1,
    "reconciled_amount": 54.0
  }
}
```

---

## Setup Instructions (Windows)

### Prerequisites

1. **Python 3.8+** (tested with Python 3.12)
2. **Tesseract OCR** - Download from [GitHub releases](https://github.com/tesseract-ocr/tesseract/wiki) and add to PATH
3. **Poppler** - Download from [Poppler releases](https://github.com/oschwartz10612/poppler-windows/releases) and add `bin` folder to PATH

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Kshaw17-web/KeshavKumarSingh_BITSHyderabad.git
cd KeshavKumarSingh_BITSHyderabad
```

2. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Verify Tesseract and Poppler are accessible:
```bash
tesseract --version
pdftoppm -h
```

5. Start the API server:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

---

## How to Deploy

### Option 1: Railway (Recommended)

1. Push your code to GitHub
2. Sign up at [Railway.app](https://railway.app) and create a new project
3. Connect your GitHub repository
4. Add environment variables:
   - `TESSERACT_CMD`: Path to tesseract executable (if custom)
   - `POPPLER_PATH`: Path to poppler bin directory (if custom)
5. Railway will automatically detect FastAPI and deploy
6. Your API will be available at `https://your-project.railway.app`

### Option 2: ngrok (For Local Testing)

If you're running the API locally and need a public URL for HackRx testing:

1. Install ngrok: `choco install ngrok` or download from [ngrok.com](https://ngrok.com)
2. Start your API: `uvicorn src.api:app --host 0.0.0.0 --port 8000`
3. In another terminal, run: `ngrok http 8000`
4. Copy the HTTPS URL (e.g., `https://abc123.ngrok-free.app`)
5. Your submission endpoint will be: `https://abc123.ngrok-free.app/api/v1/hackrx/run`

**Note**: Free ngrok URLs expire after 2 hours. For production, use Railway or another hosting service.

---

## Testing Instructions

### Test with cURL

```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Content-Type: application/json" \
  -d '{"document": "https://example.com/sample-bill.pdf"}'
```

### Test with PowerShell

```powershell
$payload = @{
    document = "https://example.com/sample-bill.pdf"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/hackrx/run" `
  -Method Post `
  -Body $payload `
  -ContentType "application/json"
```

### Test with Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/hackrx/run",
    json={"document": "https://example.com/sample-bill.pdf"}
)
print(response.json())
```

---

## Folder Structure

```
.
├── src/
│   ├── api.py                 # FastAPI application and endpoints
│   ├── schemas.py             # Pydantic models for request/response validation
│   └── extractor/
│       ├── bill_extractor.py # Main extraction logic (OCR text → structured data)
│       ├── pipeline.py       # File I/O wrapper (file path → OCR text)
│       ├── parse_lines.py    # Line-by-line parsing utilities
│       └── validation_filters.py # Currency/date/invoice detection patterns
├── data/
│   └── raw/                   # Training samples and test bills
├── ocr_outputs/               # OCR results (images + JSON) from baseline scripts
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation script
└── README.md                  # This file
```

---

## LayoutLMv3 Integration (Accuracy Enhancement)

### Overview

The extraction pipeline now includes an **ensemble approach** combining:
1. **Heuristic extraction** (fast, rule-based)
2. **LayoutLMv3 model** (deep learning for table/line item parsing)

The ensemble reconciler merges results using fuzzy matching, preferring model predictions when available.

### Model Configuration

- **Model ID**: `microsoft/layoutlmv3-base`
- **Device**: CPU (with GPU fallback if available)
- **Batch Size**: 4 pages
- **Max Sequence Length**: 512 tokens
- **Installation**: `pip install transformers>=4.30.0 torch`

### Fallback Behavior

If LayoutLMv3 is not available or fails:
- System automatically falls back to heuristic extraction
- No breaking changes to API output
- Graceful degradation ensures reliability

### Usage

The ensemble is enabled by default. To disable (heuristic only):
- Set environment variable: `USE_LAYOUTLMV3=false`
- Or modify `src/extractor/bill_extractor.py` to set `LAYOUTLMV3_AVAILABLE = False`

---

## Dataset-Specific Tuning Guide

### Tuning Parameters

The extraction pipeline includes several parameters that can be tuned for specific datasets:

#### 1. OCR Configuration (`src/utils/ocr_runner.py`)

```python
# PaddleOCR settings
rec_algorithm = "SVTR_LCNet"  # Options: "SVTR_LCNet", "CRNN", "Rosetta"
det_db_unclip_ratio = 2.3     # Range: 1.5-3.0 (higher = more text detected)
max_text_length = 200         # Range: 100-500 (longer = better for complex layouts)
```

**Tuning Tips:**
- **Low-quality scans**: Increase `det_db_unclip_ratio` to 2.5-3.0
- **Dense tables**: Increase `max_text_length` to 300-500
- **Handwritten text**: Use `rec_algorithm = "CRNN"` (better for handwriting)

#### 2. Ensemble Reconciliation (`src/extractor/ensemble_reconciler.py`)

```python
# Fuzzy matching thresholds
name_threshold = 0.8          # Range: 0.6-0.95 (higher = stricter matching)
amount_tolerance = 0.01       # Range: 0.001-0.05 (relative tolerance)
prefer_model = True           # Prefer model results over heuristic
```

**Tuning Tips:**
- **High OCR accuracy**: Increase `name_threshold` to 0.85-0.9
- **Noisy OCR**: Decrease `name_threshold` to 0.7-0.75
- **Model confidence low**: Set `prefer_model = False`

#### 3. Item Extraction Patterns (`src/extractor/bill_extractor.py`)

```python
# Currency detection
CURRENCY_SYMBOLS = r'[₹Rs\.INR]'  # Add more symbols if needed
AMOUNT_PATTERN = r'(\d{1,6}(?:[,\s]\d{3})*(?:\.\d{1,2})?)'
```

**Tuning Tips:**
- **Different currency**: Update `CURRENCY_SYMBOLS` regex
- **Very large amounts**: Increase `\d{1,6}` to `\d{1,9}` in `AMOUNT_PATTERN`
- **Different decimal format**: Modify pattern (e.g., `1.234,56` for European format)

#### 4. LayoutLMv3 Model (`src/extractor/layoutlmv3_wrapper.py`)

```python
LAYOUTLMV3_MODEL_ID = "microsoft/layoutlmv3-base"
MAX_SEQ_LENGTH = 512          # Range: 256-1024 (higher = more context)
BATCH_SIZE = 4                # Range: 1-8 (adjust based on memory)
```

**Tuning Tips:**
- **Fine-tuned model**: Replace `LAYOUTLMV3_MODEL_ID` with your fine-tuned model path
- **Long documents**: Increase `MAX_SEQ_LENGTH` to 768-1024
- **Memory constraints**: Decrease `BATCH_SIZE` to 1-2

#### 5. Preprocessing (`src/preprocessing/image_utils.py`)

```python
max_side = 1024               # Range: 512-2048 (higher = better quality, slower)
fast_mode = False             # Skip expensive ops for speed
```

**Tuning Tips:**
- **High-resolution bills**: Increase `max_side` to 1536-2048
- **Speed priority**: Enable `fast_mode = True` for large PDFs
- **Low-quality scans**: Disable `fast_mode` to enable denoising/deskew

### Performance vs Accuracy Trade-offs

| Configuration | Speed | Accuracy | Use Case |
|--------------|-------|----------|----------|
| Heuristic only | ⚡⚡⚡ | ⭐⭐ | Fast processing, simple bills |
| Ensemble (default) | ⚡⚡ | ⭐⭐⭐ | Balanced, production use |
| Ensemble + fine-tuned model | ⚡ | ⭐⭐⭐⭐ | Maximum accuracy, custom dataset |

### Testing and Validation

Use the provided test script to measure accuracy:

```bash
# Test single PDF
python test_extraction_accuracy.py --pdf data/raw/training_samples/train_sample_1.pdf

# Test all PDFs in directory
python test_extraction_accuracy.py --pdf-dir data/raw/training_samples

# Ground truth format: {pdf_name}_ground_truth.json
```

**Ground Truth Format:**
```json
{
  "pages": [
    {
      "page_no": "1",
      "items": [
        {
          "item_name": "Paracetamol 650mg",
          "item_amount": 54.0,
          "item_rate": 18.0,
          "item_quantity": 3.0
        }
      ]
    }
  ]
}
```

### Recommended Tuning Workflow

1. **Baseline**: Run test script with default parameters
2. **Identify issues**: Review precision/recall metrics
3. **Adjust parameters**: Modify thresholds based on error patterns
4. **Re-test**: Validate improvements
5. **Fine-tune model** (optional): Train LayoutLMv3 on your dataset

---

## Notes for Evaluators

### Key Features

- **Ensemble Extraction**: Combines heuristic + LayoutLMv3 for improved accuracy
- **Strict Validation**: The extraction engine rejects dates, years, and invoice numbers to avoid false positives
- **Multi-page Support**: Handles bills with multiple pages, classifying each page appropriately
- **Error Handling**: Gracefully handles OCR failures, missing files, and malformed input
- **Defensive Fallback**: Falls back to heuristic if models unavailable

### Performance Considerations

- OCR processing time scales with page count (typically 1-3 seconds per page)
- The API uses temporary files for PDF conversion, which are automatically cleaned up
- Memory usage is minimal since pages are processed sequentially

### Known Limitations

- Handwritten text may not be extracted accurately (Tesseract limitation)
- Very low-quality scans may produce incomplete results
- Complex table layouts might require manual post-processing

### Testing Recommendations

The API has been tested with the provided training samples. For best results:
- Use high-resolution PDFs or images (300+ DPI)
- Ensure bills have clear text (not handwritten)
- Test with bills that have standard formatting

---

## Repository

**GitHub**: [https://github.com/Kshaw17-web/KeshavKumarSingh_BITSHyderabad](https://github.com/Kshaw17-web/KeshavKumarSingh_BITSHyderabad)

**Collaborator**: `hackrxbot` (added for evaluation)

---

## Contact

For questions or issues, please open an issue on GitHub or contact me at the email associated with my GitHub account.
