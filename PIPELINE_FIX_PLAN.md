# HackRx Pipeline Fix Plan

## Issues Identified

1. **API calls wrong extractor**: Uses `extract_bill_data` instead of `extract_bill_data_with_tsv`
2. **Preprocessing not adaptive**: Missing OTSU, threshold fallback, auto-binary
3. **OCR numeric refinement**: Threshold is 80, should be 60
4. **Final fallback extraction**: Only triggers if 0 items, should trigger if <3 items
5. **Column detection**: Needs better fallback
6. **Batch debug tool**: Exists but needs improvements
7. **Heuristic rules**: Missing structured row detection and line merging
8. **Error handling**: Need to ensure no 5xx errors

## Fix Implementation Order

1. Fix API to use extract_bill_data_with_tsv
2. Improve preprocessing with adaptive methods
3. Fix OCR numeric refinement threshold
4. Improve fallback extraction (<3 items)
5. Add heuristic rules for structured rows
6. Update batch_debug.py
7. Ensure all error handling returns safe responses

