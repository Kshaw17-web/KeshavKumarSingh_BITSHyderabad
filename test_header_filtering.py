"""Test header filtering on train_sample_14."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.extractor.bill_extractor import extract_bill_data_with_tsv
from src.utils.pdf_loader import load_pdf_to_images

pages = load_pdf_to_images('data/raw/training_samples/TRAINING_SAMPLES/train_sample_14.pdf', dpi=300)
result = extract_bill_data_with_tsv(pages, request_id='test_header_filter')

data = result.get('data', {})
pagewise = data.get('pagewise_line_items', [])
all_items = [item for page in pagewise for item in page.get('bill_items', [])]

print(f"Total Items: {len(all_items)}")
print(f"Reconciled Amount: {data.get('reconciled_amount', 0):.2f}\n")

# Check for header-like items
header_keywords = ['ipno', 'bmno', 'age/bex', 'age bex', 'doctorname', 'discharge date', 
                   'discharge time', 'dischargedatetime', 'amount in words']

header_items = []
for item in all_items:
    name_lower = (item.get('item_name') or '').lower()
    for keyword in header_keywords:
        if keyword in name_lower:
            header_items.append(item)
            break

print(f"Header-like items found: {len(header_items)}")
if header_items:
    print("\nHeader items that should be filtered:")
    for item in header_items[:10]:
        print(f"  - {item.get('item_name')} (Amount: {item.get('item_amount')})")
else:
    print("\n[SUCCESS] No header items found - filtering is working!")

# Check for items with very small amounts that might be headers
small_amount_items = [item for item in all_items if item.get('item_amount', 0) < 10 and len((item.get('item_name') or '').split()) <= 3]
print(f"\nItems with amount < 10 and <= 3 words: {len(small_amount_items)}")
if small_amount_items:
    print("Sample small amount items (might be headers):")
    for item in small_amount_items[:5]:
        print(f"  - {item.get('item_name')} (Amount: {item.get('item_amount')})")

print(f"\n[SUMMARY]")
print(f"  Total Items: {len(all_items)}")
print(f"  Header-like items: {len(header_items)}")
print(f"  Filtering effectiveness: {((len(all_items) - len(header_items)) / len(all_items) * 100) if all_items else 0:.1f}%")

