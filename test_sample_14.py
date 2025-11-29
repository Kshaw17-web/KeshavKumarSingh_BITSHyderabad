"""Test extraction on train_sample_14.pdf and show detailed results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.extractor.bill_extractor import extract_bill_data_with_tsv
from src.utils.pdf_loader import load_pdf_to_images
import json

pdf_path = 'data/raw/training_samples/TRAINING_SAMPLES/train_sample_14.pdf'
print(f"Testing: {pdf_path}\n{'='*80}\n")

pages = load_pdf_to_images(pdf_path, dpi=300)
print(f"Loaded {len(pages)} pages\n")

result = extract_bill_data_with_tsv(pages, request_id='test_sample_14')

data = result.get('data', {})
print(f"{'='*80}")
print(f"EXTRACTION RESULTS")
print(f"{'='*80}")
print(f"Success: {result.get('is_success', False)}")
print(f"Total Items: {data.get('total_item_count', 0)}")
print(f"Reconciled Amount: {data.get('reconciled_amount', 0):.2f}")
print(f"\nPages: {len(data.get('pagewise_line_items', []))}")

pagewise = data.get('pagewise_line_items', [])
print(f"\n{'='*80}")
print(f"DETAILED ITEM BREAKDOWN")
print(f"{'='*80}")

total_items = 0
null_count = 0
null_items = []

for p_idx, page in enumerate(pagewise, 1):
    items = page.get('bill_items', [])
    total_items += len(items)
    print(f"\n--- Page {page.get('page_no')} ({page.get('page_type')}) ---")
    print(f"Items: {len(items)}")
    print(f"Reported Total: {page.get('reported_total')}")
    print(f"Fraud Flags: {len(page.get('fraud_flags', []))}")
    
    for i, item in enumerate(items[:10], 1):  # Show first 10 items per page
        name = item.get('item_name', 'N/A')
        amount = item.get('item_amount')
        rate = item.get('item_rate')
        qty = item.get('item_quantity')
        
        # Check for null/None values
        has_null = False
        null_fields = []
        if name is None or name == '':
            has_null = True
            null_fields.append('name')
        if amount is None:
            has_null = True
            null_fields.append('amount')
        if rate is None:
            has_null = True
            null_fields.append('rate')
        if qty is None:
            has_null = True
            null_fields.append('quantity')
        
        if has_null:
            null_count += 1
            null_items.append({
                'page': p_idx,
                'item': i,
                'name': name,
                'null_fields': null_fields
            })
        
        print(f"  {i}. {name}")
        print(f"     Amount: {amount}, Rate: {rate}, Qty: {qty}")
        if has_null:
            print(f"     [WARNING] NULL VALUES: {', '.join(null_fields)}")
    
    if len(items) > 10:
        print(f"  ... and {len(items) - 10} more items")

print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"Total Items Extracted: {total_items}")
print(f"Items with NULL values: {null_count}")
print(f"Items without NULL values: {total_items - null_count}")
print(f"Accuracy (no nulls): {((total_items - null_count) / total_items * 100) if total_items > 0 else 0:.1f}%")

if null_items:
    print(f"\n⚠️  Items with NULL values:")
    for null_item in null_items[:10]:  # Show first 10
        print(f"  Page {null_item['page']}, Item: {null_item['name']} - Missing: {', '.join(null_item['null_fields'])}")
    if len(null_items) > 10:
        print(f"  ... and {len(null_items) - 10} more items with nulls")

# Check if accuracy is good (threshold: >90% items have no nulls, and amount is never null)
amount_null_count = sum(1 for page in pagewise for item in page.get('bill_items', []) if item.get('item_amount') is None)
print(f"\nItems with NULL amount: {amount_null_count}")
print(f"Amount accuracy: {((total_items - amount_null_count) / total_items * 100) if total_items > 0 else 0:.1f}%")

if null_count == 0 and amount_null_count == 0:
    print(f"\n[SUCCESS] EXCELLENT: No NULL values found!")
elif null_count / total_items < 0.1 and amount_null_count == 0:
    print(f"\n[SUCCESS] GOOD: Less than 10% items have nulls, and all amounts are present")
else:
    print(f"\n[WARNING] NEEDS IMPROVEMENT: Some items have NULL values")
