"""
Test script for extract_bill_data_with_tsv function.

Usage:
    python test_extract_single_file.py <pdf_path> [request_id]
    
Example:
    python test_extract_single_file.py data/raw/training_samples/TRAINING_SAMPLES/train_sample_13.pdf req123
"""

import sys
from pathlib import Path

try:
    from src.utils.pdf_loader import load_pdf_to_images
    from src.extractor.bill_extractor import extract_bill_data_with_tsv
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root and all dependencies are installed.")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    request_id = sys.argv[2] if len(sys.argv) > 2 else f"test_{Path(pdf_path).stem}"
    
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"Loading PDF: {pdf_path}")
    print(f"Request ID: {request_id}")
    print()
    
    try:
        # Load PDF to images
        images = load_pdf_to_images(
            pdf_path,
            dpi=150,
            poppler_path=None  # Will use system default
        )
        
        print(f"Loaded {len(images)} pages")
        print()
        
        # Extract bill data
        print("Running extraction pipeline...")
        result = extract_bill_data_with_tsv(images, request_id=request_id)
        
        # Print results
        print()
        print("=" * 60)
        print("EXTRACTION RESULT")
        print("=" * 60)
        print(f"Success: {result.get('is_success')}")
        
        if not result.get('is_success'):
            print(f"Error: {result.get('error')}")
            sys.exit(1)
        
        data = result.get('data', {})
        print(f"Total pages: {len(data.get('pagewise_line_items', []))}")
        print(f"Total items: {data.get('total_item_count', 0)}")
        print(f"Reconciled amount: ₹{data.get('reconciled_amount', 0.0)}")
        print()
        
        # Show items per page
        for page in data.get('pagewise_line_items', []):
            page_no = page.get('page_no', '?')
            items = page.get('bill_items', [])
            print(f"Page {page_no}: {len(items)} items")
            for item in items[:3]:  # Show first 3 items
                name = item.get('item_name', 'UNKNOWN')[:40]
                amount = item.get('item_amount', 0.0)
                print(f"  - {name}: ₹{amount}")
            if len(items) > 3:
                print(f"  ... and {len(items) - 3} more items")
        
        print()
        print(f"Debug files saved to: logs/{request_id}/")
        print(f"Final output: logs/{request_id}/last_response.json")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

