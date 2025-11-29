"""Debug extraction to see where items are being lost."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.extractor.bill_extractor import extract_bill_data_with_tsv
from src.utils.pdf_loader import load_pdf_to_images
import json

pages = load_pdf_to_images('data/raw/training_samples/TRAINING_SAMPLES/train_sample_1.pdf', dpi=300)
result = extract_bill_data_with_tsv(pages, request_id='debug_detailed')

# Check diagnostic files
p = Path('logs/debug_detailed')
diag_files = list(p.rglob('*diagnostic*.json'))
print(f'Diagnostic files: {len(diag_files)}')

if diag_files:
    for diag_file in diag_files[:2]:
        d = json.loads(diag_file.read_text(encoding='utf-8'))
        print(f'\n{diag_file.name}:')
        print(f'  Parsed before dedup: {len(d.get("parsed_items_before_dedup", []))}')
        print(f'  Deduped: {len(d.get("deduped_items", []))}')
        if d.get("parsed_items_before_dedup"):
            print(f'  First parsed item: {d["parsed_items_before_dedup"][0]}')

data = result.get('data', {})
print(f'\nFinal result:')
print(f'  Items: {data.get("total_item_count", 0)}')
print(f'  Amount: {data.get("reconciled_amount", 0)}')

