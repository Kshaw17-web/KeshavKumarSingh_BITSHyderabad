"""Simple OCR test to verify Tesseract is working."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.pdf_loader import load_pdf_to_images
from src.preprocessing.image_utils import preprocess_image_for_ocr
from src.utils.ocr_runner import ocr_image_to_tsv
import numpy as np

# Test on first PDF
pdf_path = Path("data/raw/training_samples/TRAINING_SAMPLES/train_sample_1.pdf")
print(f"Testing OCR on: {pdf_path}")

# Load PDF
pages = load_pdf_to_images(pdf_path, dpi=300)
print(f"Loaded {len(pages)} pages")

# Process first page
if pages:
    page = pages[0]
    print("Preprocessing...")
    cv2_img = preprocess_image_for_ocr(
        page,
        max_side=2000,
        target_dpi=300,
        fast_mode=False,
        return_cv2=True,
        enhance_for_multilingual=True,
        enhance_for_handwritten=True
    )
    print(f"Preprocessed: shape={cv2_img.shape if isinstance(cv2_img, np.ndarray) else 'PIL'}")
    
    # Run OCR
    print("Running OCR...")
    try:
        ocr = ocr_image_to_tsv(
            cv2_img,
            request_id="test_ocr",
            page_no=1,
            save_debug_dir="logs/test_ocr"
        )
        
        text_tokens = [t for t in ocr.get('text', []) if t.strip()]
        print(f"OCR completed: {len(text_tokens)} non-empty tokens")
        print(f"First 20 tokens: {text_tokens[:20]}")
        
        # Check if file was saved
        ocr_file = Path("logs/test_ocr/test_ocr_p1_ocr.json")
        if ocr_file.exists():
            print(f"✓ OCR file saved: {ocr_file}")
        else:
            print(f"✗ OCR file NOT saved: {ocr_file}")
            
    except Exception as e:
        print(f"OCR failed: {e}")
        import traceback
        traceback.print_exc()

