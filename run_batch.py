# run_batch.py
import sys
import os
import json
import uuid
import tempfile
from pathlib import Path
from typing import List, Optional, Any
from datetime import datetime

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

try:
    from src.extractor.bill_extractor import extract_bill_data
except Exception:
    try:
        from extractor.bill_extractor import extract_bill_data
    except Exception:
        extract_bill_data = None

try:
    from src.preprocessing_helpers import preprocess_image_local, detect_whiteout_and_lowconf
    has_preproc_helpers = True
except Exception as e:
    print(f"ERROR: Could not import preprocessing_helpers: {e}")
    print("Make sure src/preprocessing_helpers.py exists and is importable.")
    sys.exit(1)

from pdf2image import convert_from_path
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import requests

DATA_RAW = ROOT / "data" / "raw"
TRAINING_GLOB = ["training_samples", "training_samples/*", "training_samples/**"]
OUTPUT_DIR = ROOT / "local_test_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DEBUG_MASKS_DIR = OUTPUT_DIR / "debug_masks"
DEBUG_MASKS_DIR.mkdir(exist_ok=True)
FRAUD_CSV = OUTPUT_DIR / "fraud_report.csv"

POPPLER_PATH = os.environ.get("POPPLER_PATH", r"C:\poppler-25.11.0\Library\bin")
TESSERACT_CMD_ENV = os.environ.get("TESSERACT_CMD", "")
if TESSERACT_CMD_ENV:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_ENV

DEFAULT_DPI = int(os.environ.get("OCR_DPI", "300"))

def find_training_pdfs(base: Path) -> List[Path]:
    candidates = []
    if (base / "training_samples").exists():
        folder = base / "training_samples"
        for f in folder.rglob("*.pdf"):
            candidates.append(f)
    # also fallback: any pdf under data/raw
    for f in base.rglob("*.pdf"):
        if "training_samples" in str(f).lower():
            pass
        else:
            continue
    return sorted(candidates)

def convert_pdf_to_images(pdf_path: Path, dpi: int = DEFAULT_DPI) -> List[Image.Image]:
    images = convert_from_path(str(pdf_path), dpi=dpi, poppler_path=(POPPLER_PATH if Path(POPPLER_PATH).exists() else None))
    return images

def simple_preprocess_image(img: Image.Image) -> Image.Image:
    if img.mode != "L":
        img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    return img

def run_ocr_on_images(images: List[Image.Image]) -> List[str]:
    pages = []
    for img in images:
        try:
            text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
        except Exception:
            text = ""
        pages.append(text)
    return pages

def save_json_output(output_path: Path, payload: Any):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

def safe_extract_call_with_fallback(file_path: Path, ocr_pages: List[str]):
    if extract_bill_data is None:
        raise RuntimeError("extract_bill_data function not importable")
    try:
        result = extract_bill_data(str(file_path))
        if isinstance(result, dict) and ("pagewise_line_items" in result or "total_item_count" in result):
            return result
        # if result is not in expected form, continue to try with ocr_pages
    except Exception:
        pass
    try:
        result = extract_bill_data(ocr_pages)
        return result
    except Exception as e:
        raise RuntimeError(f"Both attempts to call extract_bill_data failed: {e}")

def write_fraud_row(filename: str, page_no: int, flag_type: str, flag_score: float):
    from datetime import timezone
    row = f"{filename},{page_no},{flag_type},{flag_score},{datetime.now(timezone.utc).isoformat()}\n"
    with open(FRAUD_CSV, "a", encoding="utf8") as fh:
        fh.write(row)

def process_file(pdf_path: Path):
    print(f"Processing {pdf_path.name}")
    try:
        images = convert_pdf_to_images(pdf_path)
    except Exception as e:
        print("PDF->images conversion failed:", e)
        return

    preproc_images = []
    for i, img in enumerate(images, start=1):
        if has_preproc_helpers:
            try:
                pre_img = preprocess_image_local(img)
            except Exception as e:
                print(f"Warning: preprocessing failed for page {i}, using simple preprocessing: {e}")
                pre_img = simple_preprocess_image(img)
        else:
            pre_img = simple_preprocess_image(img)
        preproc_images.append(pre_img)
        try:
            debug_mask_path = DEBUG_MASKS_DIR / f"{pdf_path.stem}__p{i:03d}.png"
            pre_img.save(debug_mask_path)
        except Exception:
            pass

    try:
        ocr_pages = run_ocr_on_images(preproc_images)
    except Exception as e:
        ocr_pages = [""] * len(preproc_images)
        print("OCR failed:", e)

    # Detect whiteout/fraud using helper
    if has_preproc_helpers:
        try:
            flags = detect_whiteout_and_lowconf(preproc_images, ocr_pages)
            # flags expected as list of tuples (page_no, flag_type, score)
            for page_no, flag_type, score in flags:
                write_fraud_row(pdf_path.name, page_no, flag_type, score)
        except Exception as e:
            print(f"Warning: fraud detection failed for {pdf_path.name}: {e}")
            # Fallback: simple heuristic
            for idx, (img, text) in enumerate(zip(preproc_images, ocr_pages), start=1):
                txt_len = len(text.strip())
                if txt_len < 10:
                    write_fraud_row(pdf_path.name, idx, "low_ocr_text", 0.5)
    else:
        # simple heuristic: if a page's OCR text length is tiny while image has large white area
        for idx, (img, text) in enumerate(zip(preproc_images, ocr_pages), start=1):
            txt_len = len(text.strip())
            if txt_len < 10:
                write_fraud_row(pdf_path.name, idx, "low_ocr_text", 0.5)

    try:
        result = safe_extract_call_with_fallback(pdf_path, ocr_pages)
    except Exception as e:
        print("Extraction failed for", pdf_path.name, ":", e)
        return

    output_json_path = OUTPUT_DIR / f"{pdf_path.stem}_output.json"
    save_json_output(output_json_path, result)
    print("Saved output for", pdf_path.name, "->", output_json_path.name)

def main():
    # Initialize fraud CSV with header
    if FRAUD_CSV.exists():
        FRAUD_CSV.unlink()
    with open(FRAUD_CSV, "w", encoding="utf8") as fh:
        fh.write("file,page_no,flag_type,flag_score,timestamp\n")
    
    pdfs = []
    training_dir = DATA_RAW / "training_samples" / "TRAINING_SAMPLES"
    if training_dir.exists():
        for f in training_dir.rglob("*.pdf"):
            pdfs.append(f)
    else:
        training_dir = DATA_RAW / "training_samples"
        if training_dir.exists():
            for f in training_dir.rglob("*.pdf"):
                pdfs.append(f)
        else:
            for f in DATA_RAW.rglob("*.pdf"):
                if "train_sample" in f.name.lower():
                    pdfs.append(f)
    pdfs = sorted(set(pdfs))
    if not pdfs:
        print("No training PDFs found under:", DATA_RAW)
        return
    
    print(f"Found {len(pdfs)} PDF files to process")
    for p in pdfs:
        try:
            process_file(p)
        except Exception as e:
            print(f"Error while processing {p.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Batch finished. Outputs in: {OUTPUT_DIR}")
    print(f"Fraud report: {FRAUD_CSV}")

if __name__ == "__main__":
    main()
