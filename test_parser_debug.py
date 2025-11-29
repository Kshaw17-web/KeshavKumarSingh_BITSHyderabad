"""Debug parser on actual PDF lines."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.pdf_loader import load_pdf_to_images
from src.preprocessing.image_utils import preprocess_image_for_ocr
from src.utils.ocr_runner import ocr_image_to_tsv
from src.extractor.parsers import group_words_to_lines, detect_column_centers, map_tokens_to_columns, parse_row_from_columns, is_probable_item
import numpy as np

# Load and process first page
pages = load_pdf_to_images('data/raw/training_samples/TRAINING_SAMPLES/train_sample_1.pdf', dpi=300)
cv2_img = preprocess_image_for_ocr(
    pages[0], 
    max_side=2000, 
    target_dpi=300, 
    fast_mode=False, 
    return_cv2=True, 
    enhance_for_multilingual=True, 
    enhance_for_handwritten=True
)

ocr = ocr_image_to_tsv(cv2_img, request_id='debug', page_no=1, save_debug_dir='logs/debug_final')
lines = group_words_to_lines(ocr)
print(f'Total lines: {len(lines)}')

col_centers = detect_column_centers(lines, max_columns=6, gray_img_array=cv2_img)
print(f'Column centers: {col_centers}')

# Test lines 10-20 (where items should be)
parsed_count = 0
for i, ln in enumerate(lines[10:20], 10):
    tokens = [t.get('text', '') for t in ln if t.get('text', '').strip()]
    cols = map_tokens_to_columns(ln, col_centers)
    parsed = parse_row_from_columns(cols)
    is_item = is_probable_item(parsed)
    
    print(f'\nLine {i}:')
    print(f'  Tokens: {tokens[:8]}')
    print(f'  Parsed: name="{parsed.get("item_name")}", amount={parsed.get("item_amount")}, rate={parsed.get("item_rate")}, qty={parsed.get("item_quantity")}')
    print(f'  Is item: {is_item}')
    
    if is_item:
        parsed_count += 1

print(f'\nTotal items found in lines 10-20: {parsed_count}')

