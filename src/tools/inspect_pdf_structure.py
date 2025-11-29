"""
Inspect PDF structure to understand bill format and identify item patterns.
"""
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.pdf_loader import load_pdf_to_images
from src.preprocessing.image_utils import preprocess_image_for_ocr
from src.utils.ocr_runner import ocr_image_to_tsv
from src.extractor.parsers import group_words_to_lines
import numpy as np

def inspect_pdf(pdf_path: Path):
    """Inspect PDF structure to understand bill format."""
    print(f"\n{'='*80}")
    print(f"INSPECTING: {pdf_path.name}")
    print(f"{'='*80}\n")
    
    # Load PDF
    pages = load_pdf_to_images(pdf_path, dpi=300)
    print(f"Loaded {len(pages)} pages\n")
    
    # Process each page
    for page_idx, page in enumerate(pages, 1):
        print(f"\n{'='*60}")
        print(f"PAGE {page_idx}")
        print(f"{'='*60}\n")
        
        # Preprocess
        cv2_img = preprocess_image_for_ocr(
            page,
            max_side=2000,
            target_dpi=300,
            fast_mode=False,
            return_cv2=True,
            enhance_for_multilingual=True,
            enhance_for_handwritten=True
        )
        
        # OCR
        ocr = ocr_image_to_tsv(
            cv2_img,
            request_id="inspect",
            page_no=page_idx,
            save_debug_dir="logs/inspect"
        )
        
        # Group to lines
        lines = group_words_to_lines(ocr)
        print(f"Total lines detected: {len(lines)}\n")
        
        # Analyze lines
        print("LINE ANALYSIS:")
        print("-" * 60)
        print(f"{'Line':<6} {'Tokens':<8} {'Text Preview':<40} {'Has Amount':<12}")
        print("-" * 60)
        
        for i, line in enumerate(lines[:30], 1):  # First 30 lines
            tokens = [t.get('text', '') for t in line if t.get('text', '').strip()]
            line_text = ' '.join(tokens[:5])  # First 5 tokens
            if len(line_text) > 40:
                line_text = line_text[:37] + "..."
            
            # Check for amounts (numbers with decimals or currency)
            has_amount = False
            amounts = []
            for token in tokens:
                # Look for numeric patterns
                import re
                if re.search(r'\d+[.,]\d+', token) or (token.replace(',', '').replace('.', '').isdigit() and len(token) > 1):
                    try:
                        clean = token.replace('₹', '').replace('Rs', '').replace(',', '').replace('$', '')
                        val = float(''.join(c for c in clean if c.isdigit() or c in '.-'))
                        if 1 <= val <= 1000000:  # Reasonable amount range
                            has_amount = True
                            amounts.append(val)
                    except:
                        pass
            
            amount_str = f"{amounts[0]:.2f}" if amounts else "No"
            print(f"{i:<6} {len(tokens):<8} {line_text:<40} {amount_str:<12}")
            
            # Show full line if it has an amount
            if has_amount and len(tokens) > 2:
                print(f"       Full line: {' '.join(tokens)}")
        
        print("\n" + "="*60)
        print("POTENTIAL ITEM ROWS (lines with amounts):")
        print("="*60)
        
        item_candidates = []
        for i, line in enumerate(lines, 1):
            tokens = [t.get('text', '') for t in line if t.get('text', '').strip()]
            line_text = ' '.join(tokens)
            
            # Check for amounts
            amounts = []
            for token in tokens:
                import re
                if re.search(r'\d+[.,]\d+', token) or (token.replace(',', '').replace('.', '').isdigit() and len(token) > 1):
                    try:
                        clean = token.replace('₹', '').replace('Rs', '').replace(',', '').replace('$', '')
                        val = float(''.join(c for c in clean if c.isdigit() or c in '.-'))
                        if 10 <= val <= 1000000:  # Reasonable item amount
                            amounts.append((token, val))
                    except:
                        pass
            
            if amounts:
                # This might be an item row
                item_candidates.append({
                    'line_no': i,
                    'tokens': tokens,
                    'text': line_text,
                    'amounts': amounts
                })
        
        # Display item candidates
        for candidate in item_candidates[:20]:  # First 20
            print(f"\nLine {candidate['line_no']}:")
            print(f"  Text: {candidate['text']}")
            print(f"  Amounts found: {[f'{a[0]} ({a[1]:.2f})' for a in candidate['amounts']]}")
            print(f"  Token count: {len(candidate['tokens'])}")
        
        print(f"\nTotal potential item rows: {len(item_candidates)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="PDF file to inspect")
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)
    
    inspect_pdf(pdf_path)

