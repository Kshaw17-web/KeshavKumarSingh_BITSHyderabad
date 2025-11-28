# baseline_ocr.py
# Robust baseline OCR: search recursively for train_sample_*.pdf under data/raw,
# convert pages to PNG (using explicit poppler_path), run pytesseract, save per-doc JSON.
#
# Usage:
#   python baseline_ocr.py
#
import json
import sys
from pathlib import Path
from pdf2image import convert_from_path
import pytesseract
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------- CONFIG ----------
# Root folder to search for training PDFs (recursive search)
RAW_ROOT = Path(r"C:\Users\ksr20\OneDrive\Desktop\BAJAJ FINSERV DATATHON\data\raw")

# Output folder for images and OCR JSONs
OUTPUT_DIR = Path(r"C:\Users\ksr20\OneDrive\Desktop\BAJAJ FINSERV DATATHON\ocr_outputs")

# Poppler bin folder (must contain pdfinfo.exe / pdftoppm.exe)
POPPLER_BIN = r"C:\poppler-25.11.0\Library\bin"

# DPI for conversion
DPI = 200
# -----------------------------

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def find_pdfs(root: Path):
    # recursive search for train_sample_*.pdf files
    pattern = "train_sample_*.pdf"
    files = sorted(root.rglob(pattern))
    return files

def process_pdf(pdf_path: Path):
    print(f"\nProcessing PDF: {pdf_path}")
    try:
        pages = convert_from_path(str(pdf_path), dpi=DPI, poppler_path=POPPLER_BIN)
    except Exception as e:
        print(f"ERROR: convert_from_path failed for {pdf_path.name}: {e}")
        return False

    doc_out = {"file": str(pdf_path), "pages": []}
    for i, page in enumerate(pages, start=1):
        img_name = f"{pdf_path.stem}_p{i}.png"
        img_path = OUTPUT_DIR / img_name
        try:
            page.save(img_path, "PNG")
        except Exception as e:
            print(f"ERROR: could not save image {img_path}: {e}")
            continue

        try:
            text = pytesseract.image_to_string(str(img_path), lang="eng")
        except Exception as e:
            print(f"ERROR: pytesseract failed on {img_path}: {e}")
            text = ""

        doc_out["pages"].append({
            "page_no": i,
            "image_path": str(img_path),
            "text_preview": text[:500],
            "full_text": text
        })
        print(f"  OCR page {i}: {len(text)} chars -> {img_path.name}")

    out_json = OUTPUT_DIR / f"{pdf_path.stem}_ocr.json"
    try:
        with open(out_json, "w", encoding="utf8") as f:
            json.dump(doc_out, f, indent=2, ensure_ascii=False)
        print(f"Saved OCR JSON: {out_json}")
    except Exception as e:
        print(f"ERROR: failed to write OCR JSON for {pdf_path.name}: {e}")
    return True

def main():
    ensure_dirs()

    if not RAW_ROOT.exists():
        print("ERROR: RAW_ROOT does not exist:", RAW_ROOT)
        sys.exit(1)

    pdfs = find_pdfs(RAW_ROOT)
    print(f"Found {len(pdfs)} PDF(s) matching 'train_sample_*.pdf' under: {RAW_ROOT}")
    if len(pdfs) == 0:
        print("No matching PDFs found. Here are files under RAW_ROOT (top-level):")
        for p in sorted(RAW_ROOT.iterdir()):
            print("  -", p.name)
        sys.exit(1)

    for pdf in pdfs:
        process_pdf(pdf)

    print("\nAll done. OCR outputs are in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
