# HackRx Pipeline Fix & Upgrade Summary

## ✅ All Fixes Completed

### 1. **API Fixed to Use TSV-Based Extractor** ✅
- **File**: `src/api.py`
- **Change**: Switched from `extract_bill_data` to `extract_bill_data_with_tsv`
- **Impact**: Now uses the geometry-first parsing pipeline with better accuracy
- **Added**: Request ID generation for debug logging

### 2. **Adaptive Preprocessing** ✅
- **File**: `src/preprocessing/image_utils.py`
- **Changes**:
  - Added OTSU threshold with fallback to adaptive threshold
  - Added auto-binary for faded text detection
  - Added minimum width enforcement (1200px) for better OCR
  - Improved CLAHE application with fallback chain
- **Impact**: Better handling of low-contrast, faded, and complex invoices

### 3. **OCR Numeric Refinement** ✅
- **File**: `src/extractor/bill_extractor.py`
- **Change**: Lowered threshold from 80 to 60 (as per requirements)
- **Impact**: More aggressive re-OCR for low-confidence numeric tokens

### 4. **Improved Fallback Extraction** ✅
- **File**: `src/extractor/bill_extractor.py`
- **Change**: Fallback now triggers if <3 items (instead of 0)
- **Impact**: Better extraction on difficult invoices

### 5. **Heuristic Rules for Structured Rows** ✅
- **File**: `src/extractor/bill_extractor.py`
- **Added Functions**:
  - `_merge_split_lines()`: Merges lines that appear split across OCR lines
  - `_parse_structured_row()`: Parses rows like "CANNULA 22G 1 105.00 0.00 105.00"
  - `_detect_page_type_from_text()`: Auto-detects Pharmacy/Bill Detail/Final Bill
- **Impact**: Better handling of structured medical/pharmacy invoices

### 6. **Batch Debug Tool** ✅
- **File**: `src/tools/batch_debug.py`
- **Changes**:
  - Now uses `extract_bill_data_with_tsv` for consistency
  - Improved CSV output with proper field names
  - Added table summary with success indicators
  - Better error handling
- **Usage**: `python -m src.tools.batch_debug --input <folder> --out <output_folder>`

### 7. **Error Handling - No 5xx Errors** ✅
- **File**: `src/api.py`
- **Changes**:
  - All exceptions wrapped in try-catch
  - Returns safe error responses with `is_success: false`
  - HTTP exceptions properly re-raised
  - Extractor failures return safe schema
- **Impact**: API never crashes with 5xx errors

### 8. **Column Detection Improvements** ✅
- **File**: `src/extractor/parsers.py`
- **Already Implemented**:
  - K-means clustering for column detection
  - Projection-based fallback when <2 centers detected
  - Image-based projection profile analysis

## 📋 Files Modified

1. `src/api.py` - Fixed to use TSV extractor, added error handling
2. `src/extractor/bill_extractor.py` - Added heuristic rules, improved fallbacks
3. `src/preprocessing/image_utils.py` - Added adaptive preprocessing
4. `src/tools/batch_debug.py` - Updated to use full pipeline, improved output

## 🚀 How to Test

### 1. Test API Locally
```bash
uvicorn src.api:app --reload
```

### 2. Test Batch Processing
```bash
python -m src.tools.batch_debug --input data/raw/training_samples/TRAINING_SAMPLES --out local_test_outputs
```

### 3. Check Debug Logs
- Debug files saved to `logs/{request_id}/`
- Includes: OCR JSON, preprocessed images, parser diagnostics, final response

## 🔍 Key Improvements

1. **Better Accuracy**: TSV-based parser with geometry-first approach
2. **Robust Preprocessing**: Adaptive methods handle various invoice types
3. **Smart Fallbacks**: Multiple fallback strategies ensure items are extracted
4. **Structured Row Detection**: Handles medical/pharmacy invoices better
5. **No Crashes**: All errors return safe responses, no 5xx errors
6. **Better Debugging**: Comprehensive logging for troubleshooting

## 📊 Expected Results

- **Before**: Empty item lists on training samples
- **After**: Items extracted using TSV pipeline with fallbacks
- **Leaderboard**: Improved scores due to better extraction accuracy

## ⚠️ Notes

- The pipeline now uses `extract_bill_data_with_tsv` which is more accurate
- All preprocessing is adaptive and handles edge cases
- Fallback extraction ensures at least some items are found
- Error handling ensures API stability

