"""Test full extraction on train_sample_1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.extractor.bill_extractor import extract_bill_data_with_tsv
from src.utils.pdf_loader import load_pdf_to_images

pages = load_pdf_to_images('data/raw/training_samples/TRAINING_SAMPLES/train_sample_1.pdf', dpi=300)
result = extract_bill_data_with_tsv(pages, request_id='test_final2')

data = result.get('data', {})
print(f'Items: {data.get("total_item_count", 0)}')
print(f'Amount: {data.get("reconciled_amount", 0)}')

pagewise = data.get('pagewise_line_items', [])
print(f'Pages: {len(pagewise)}')

for p in pagewise[:2]:
    items = p.get('bill_items', [])
    print(f'\nPage {p.get("page_no")}: {len(items)} items')
    if items:
        for i, item in enumerate(items[:5], 1):
            print(f'  {i}. {item.get("item_name")} - Amount: {item.get("item_amount")}, Rate: {item.get("item_rate")}, Qty: {item.get("item_quantity")}')

