# Debug Logging Documentation

## Overview

The system now includes comprehensive debug logging for every request, enabling detailed audit of leaderboard failures. All debug artifacts are saved per-request in organized folders.

## Debug File Structure

For each request (identified by `request_id`), the following files are saved in `logs/{request_id}/`:

### 1. Raw OCR Text
- **Files**: `{request_id}_p{page_no}_ocr.json` and `{request_id}_p{page_no}_ocr.tsv`
- **Description**: Raw OCR output from pytesseract in both JSON (structured) and TSV (tab-separated) formats
- **Contains**: Token-level data including text, bounding boxes, confidence scores, and hierarchy (level, block, paragraph, line, word)
- **Saved by**: `ocr_image_to_tsv()` in `src/utils/ocr_runner.py`

### 2. Preprocessed Images
- **Files**: `{request_id}_p{page_no}_pre.png`
- **Description**: Preprocessed images optimized for OCR (denoised, deskewed, CLAHE-enhanced, resized)
- **Saved by**: `preprocess_image_for_ocr()` in `src/preprocessing/image_utils.py`

### 3. Parser Diagnostics
- **Files**: `{request_id}_p{page_no}_parser_diagnostic.json`
- **Description**: Detailed parsing information for each page
- **Contains**:
  - `page_no`: Page number
  - `raw_lines_count`: Number of lines detected
  - `column_centers`: Detected column x-coordinates
  - `parsed_items_before_dedup`: Items extracted before deduplication
  - `deduped_items`: Items after deduplication
  - `reported_total`: Total extracted from OCR text (if found)
  - `final_total`: Calculated total from items
  - `reconciliation_method`: Method used for reconciliation
- **Saved by**: `extract_bill_data_with_tsv()` in `src/extractor/bill_extractor.py`

### 4. Final Response
- **Files**: `last_response.json`
- **Description**: Complete extraction result in HackRx schema format
- **Contains**: Full response with `is_success`, `token_usage`, `data.pagewise_line_items`, `bill_items`, `total_item_count`
- **Saved by**: `extract_bill_data_with_tsv()` in `src/extractor/bill_extractor.py`

## Batch Processing

When running `run_batch.py`:

1. **Per-Request Debug Logs**: Each PDF is processed with its filename as `request_id`, and all debug files are saved to `logs/{pdf_stem}/`

2. **Copied to Output Directory**: After processing, debug logs are copied to `local_test_outputs/debug_logs/{pdf_stem}/` for easy review

3. **Archive Creation**: After all PDFs are processed, a zip archive `local_test_outputs/debug_logs_archive.zip` is created containing all debug logs

## Usage

### Single File Processing
```python
from src.extractor.bill_extractor import extract_bill_data_with_tsv
from src.utils.pdf_loader import load_pdf_to_images

images = load_pdf_to_images("path/to/file.pdf")
result = extract_bill_data_with_tsv(images, request_id="my_request_123")
# Debug files saved to: logs/my_request_123/
```

### Batch Processing
```bash
python run_batch.py
# Debug logs copied to: local_test_outputs/debug_logs/
# Archive created: local_test_outputs/debug_logs_archive.zip
```

## Reviewing Leaderboard Failures

When a leaderboard sample fails:

1. **Locate the request_id** from the leaderboard submission or error logs
2. **Navigate to** `logs/{request_id}/` or `local_test_outputs/debug_logs/{request_id}/`
3. **Review files in order**:
   - `*_pre.png`: Check if preprocessing is correct
   - `*_ocr.json`: Check if OCR detected text correctly
   - `*_parser_diagnostic.json`: Check parsing logic and column detection
   - `last_response.json`: Check final output structure

## File Naming Convention

- **OCR files**: `{request_id}_p{page_no}_ocr.{json|tsv}`
- **Preprocessed images**: `{request_id}_p{page_no}_pre.png`
- **Parser diagnostics**: `{request_id}_p{page_no}_parser_diagnostic.json`
- **Final response**: `last_response.json` (one per request, not per page)

## Notes

- All debug files are saved with UTF-8 encoding
- JSON files are pretty-printed with 2-space indentation
- Images are saved as PNG format
- Debug directories are created automatically if they don't exist
- If `request_id` is not provided, files use "default" or "page" as fallback

