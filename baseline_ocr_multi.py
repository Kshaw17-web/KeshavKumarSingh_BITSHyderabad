# baseline_ocr_multi.py
# Robust multi-page OCR pipeline: ensures every page is saved and OCR'd.
# Uses explicit poppler_path and explicit tesseract_cmd to avoid PATH issues.

import json, sys
from pathlib import Path
from pdf2image import convert_from_path, convert_from_bytes
import pytesseract

# ---------- CONFIG ----------
RAW_ROOT = Path(r"C:\Users\ksr20\OneDrive\Desktop\BAJAJ FINSERV DATATHON\data\raw")
OUTPUT_DIR = Path(r"C:\Users\ksr20\OneDrive\Desktop\BAJAJ FINSERV DATATHON\ocr_outputs")
POPPLER_BIN = r"C:\poppler-25.11.0\Library\bin"   # your poppler path
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # your tesseract path
DPI = 200
# -----------------------------

# ensure pytesseract knows where tesseract.exe is
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def find_pdfs(root: Path):
    return sorted(root.rglob("train_sample_*.pdf"))

def pages_from_pdf(pdf_path: Path):
    """
    Try convert_from_path first; if it returns only 1 page unexpectedly, fall back to convert_from_bytes.
    Return list of PIL Image objects.
    """
    try:
        pages = convert_from_path(str(pdf_path), dpi=DPI, poppler_path=POPPLER_BIN)
    except Exception as e:
        print(f"convert_from_path failed for {pdf_path} with error: {e}. Trying convert_from_bytes fallback.")
        try:
            raw = pdf_path.read_bytes()
            pages = convert_from_bytes(raw, dpi=DPI, poppler_path=POPPLER_BIN)
        except Exception as e2:
            print(f"convert_from_bytes also failed for {pdf_path}: {e2}")
            return []
    return pages

def process_pdf(pdf_path: Path):
    print(f"\n=== Processing PDF: {pdf_path.name} ===")
    pages = pages_from_pdf(pdf_path)
    n_pages = len(pages)
    print(f"Pages found: {n_pages}")

    if n_pages == 0:
        print("No pages extracted, skipping.")
        return False

    doc_out = {"file": str(pdf_path), "pages": []}
    for i, page in enumerate(pages, start=1):
        # create a unique filename using stem + page index
        img_filename = f"{pdf_path.stem}__p{i:03d}.png"
        img_path = OUTPUT_DIR / img_filename
        try:
            page.save(img_path, "PNG")
        except Exception as e:
            print(f"ERROR saving page {i} of {pdf_path.name}: {e}")
            continue

        # run OCR on saved image
        try:
            text = pytesseract.image_to_string(str(img_path), lang="eng")
        except Exception as e:
            print(f"ERROR pytesseract on {img_path.name}: {e}")
            text = ""

        print(f"  OCR page {i}/{n_pages}: {len(text)} chars -> {img_path.name}")

        doc_out["pages"].append({
            "page_no": i,
            "image_path": str(img_path),
            "text_preview": text[:600],
            "full_text": text
        })

    out_json = OUTPUT_DIR / f"{pdf_path.stem}_ocr.json"
    try:
        with open(out_json, "w", encoding="utf8") as f:
            json.dump(doc_out, f, indent=2, ensure_ascii=False)
        print(f"Saved OCR JSON: {out_json}")
    except Exception as e:
        print(f"ERROR writing JSON for {pdf_path.name}: {e}")

    return True

def main():
    ensure_dirs()

    if not RAW_ROOT.exists():
        print("ERROR: RAW_ROOT does not exist:", RAW_ROOT)
        sys.exit(1)

    pdfs = find_pdfs(RAW_ROOT)
    print(f"Found {len(pdfs)} PDF(s) matching 'train_sample_*.pdf' under: {RAW_ROOT}")

    if not pdfs:
        print("No PDFs found, exiting.")
        sys.exit(1)

    for pdf in pdfs:
        process_pdf(pdf)

    print("\nDone. All OCR outputs (images + JSON) are in:", OUTPUT_DIR)
    print("\nQuick verification commands (PowerShell):")
    print(f"  Get-ChildItem \"{OUTPUT_DIR}\" -Filter \"*_p*.png\" | Select-Object Name,Length")
    print(f"  Get-ChildItem \"{OUTPUT_DIR}\" -Filter \"*_ocr.json\" | Select-Object Name,Length")

if __name__ == "__main__":
    main()
