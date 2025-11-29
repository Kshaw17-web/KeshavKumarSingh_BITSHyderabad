"""
Diagnostic tool to debug why extraction is returning 0 items.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
from src.utils.pdf_loader import load_pdf_to_images
from src.extractor.bill_extractor import extract_bill_data_with_tsv
from src.utils.ocr_runner import ocr_image_to_tsv
from src.extractor.parsers import group_words_to_lines, detect_column_centers, map_tokens_to_columns, parse_row_from_columns, is_probable_item
from src.preprocessing.image_utils import preprocess_image_for_ocr
import numpy as np
from PIL import Image

def diagnose_pdf(pdf_path: Path):
    """Diagnose a single PDF to see where extraction is failing."""
    print(f"\n{'='*80}")
    print(f"DIAGNOSING: {pdf_path.name}")
    print(f"{'='*80}\n")
    
    # Load PDF
    try:
        pages = load_pdf_to_images(pdf_path, dpi=300)
        print(f"[OK] Loaded {len(pages)} pages")
    except Exception as e:
            print(f"[ERROR] Failed to load PDF: {e}")
        return
    
    # Process first page in detail
    if pages:
        page = pages[0]
        print(f"\n--- Processing Page 1 ---")
        
        # Preprocess
        try:
            cv2_img = preprocess_image_for_ocr(
                page,
                max_side=2000,
                target_dpi=300,
                fast_mode=False,
                return_cv2=True,
                enhance_for_multilingual=True,
                enhance_for_handwritten=True
            )
            print(f"[OK] Preprocessed image: shape={cv2_img.shape if isinstance(cv2_img, np.ndarray) else 'PIL'}")
        except Exception as e:
            print(f"[ERROR] Preprocessing failed: {e}")
            return
        
        # OCR
        try:
            ocr = ocr_image_to_tsv(cv2_img, request_id="diagnose", page_no=1, save_debug_dir="logs/diagnose")
            text_tokens = ocr.get('text', [])
            print(f"[OK] OCR completed: {len(text_tokens)} text tokens")
            if text_tokens:
                print(f"  First 10 tokens: {text_tokens[:10]}")
            else:
                print(f"  [WARNING] No text tokens found!")
        except Exception as e:
            print(f"[ERROR] OCR failed: {e}")
            return
        
        # Group to lines
        try:
            lines = group_words_to_lines(ocr)
            print(f"[OK] Grouped to {len(lines)} lines")
            if lines:
                print(f"  First line has {len(lines[0])} tokens")
                print(f"  First line tokens: {[t.get('text', '') for t in lines[0][:5]]}")
        except Exception as e:
            print(f"[ERROR] Line grouping failed: {e}")
            return
        
        # Detect columns
        try:
            col_centers = detect_column_centers(lines, max_columns=6, gray_img_array=cv2_img)
            print(f"[OK] Detected {len(col_centers)} column centers: {col_centers}")
        except Exception as e:
            print(f"[ERROR] Column detection failed: {e}")
            return
        
        # Parse lines
        parsed_items = []
        for i, ln in enumerate(lines[:10]):  # Check first 10 lines
            try:
                cols = map_tokens_to_columns(ln, col_centers)
                parsed = parse_row_from_columns(cols)
                is_item = is_probable_item(parsed)
                print(f"\n  Line {i+1}:")
                print(f"    Tokens: {[t.get('text', '') for t in ln[:5]]}")
                print(f"    Parsed: name='{parsed.get('item_name')}', amount={parsed.get('item_amount')}, qty={parsed.get('item_quantity')}")
                print(f"    Is probable item: {is_item}")
                if is_item:
                    parsed_items.append(parsed)
            except Exception as e:
                print(f"    [ERROR] Parsing failed: {e}")
        
        print(f"\n[OK] Parsed {len(parsed_items)} items from first 10 lines")
        
        # Run full extraction
        print(f"\n--- Running Full Extraction ---")
        try:
            result = extract_bill_data_with_tsv(pages, request_id="diagnose")
            data = result.get("data", {})
            pagewise = data.get("pagewise_line_items", [])
            total_items = data.get("total_item_count", 0)
            reconciled = data.get("reconciled_amount", 0.0)
            
            print(f"[OK] Full extraction completed")
            print(f"  Total pages processed: {len(pagewise)}")
            print(f"  Total items extracted: {total_items}")
            print(f"  Reconciled amount: {reconciled}")
            
            for page_data in pagewise:
                page_no = page_data.get("page_no", "?")
                items = page_data.get("bill_items", [])
                fraud_flags = page_data.get("fraud_flags", [])
                print(f"\n  Page {page_no}:")
                print(f"    Items: {len(items)}")
                print(f"    Fraud flags: {len(fraud_flags)}")
                if items:
                    print(f"    First item: {items[0]}")
        except Exception as e:
            print(f"[ERROR] Full extraction failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="PDF file to diagnose")
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)
    
    diagnose_pdf(pdf_path)

