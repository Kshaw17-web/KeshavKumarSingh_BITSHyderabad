# ocr_from_images.py
# Takes all saved page images in ocr_outputs (pattern: <pdf_stem>__pNNN.png)
# Runs pytesseract on each image (explicit tesseract path), and writes per-document JSON
# with page_no, image_path, text_preview, full_text.

import json
from pathlib import Path
import pytesseract
from PIL import Image

# ---------- CONFIG ----------
OCR_IMAGES_DIR = Path(r"C:\Users\ksr20\OneDrive\Desktop\BAJAJ FINSERV DATATHON\ocr_outputs")
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # your tesseract path
OUT_DIR = OCR_IMAGES_DIR  # write JSONs next to images
# -----------------------------

pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

def list_images(dirpath: Path):
    # expect names like train_sample_1__p001.png
    return sorted(dirpath.glob("*__p*.png"))

def group_by_stem(images):
    groups = {}
    for img in images:
        stem = img.name.split("__p")[0]
        groups.setdefault(stem, []).append(img)
    # sort pages by page number extracted from filename
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda p: int(p.name.split("__p")[1].split(".")[0]))
    return groups

def ocr_image(img_path: Path):
    try:
        text = pytesseract.image_to_string(str(img_path), lang="eng")
    except Exception as e:
        print(f"ERROR OCR {img_path.name}: {e}")
        text = ""
    return text

def main():
    images = list_images(OCR_IMAGES_DIR)
    if not images:
        print("No page images found in:", OCR_IMAGES_DIR)
        return

    groups = group_by_stem(images)
    print(f"Found {len(groups)} documents to OCR (based on image stems).")

    for stem, imgs in groups.items():
        doc = {"file_stem": stem, "pages": []}
        print(f"\nProcessing document: {stem} ({len(imgs)} pages)")
        for i, img in enumerate(imgs, start=1):
            print(f"  OCR -> {img.name}", end=" ... ")
            text = ocr_image(img)
            print(f"{len(text)} chars")
            doc["pages"].append({
                "page_no": i,
                "image_path": str(img),
                "text_preview": text[:600],
                "full_text": text
            })
        out_json = OUT_DIR / f"{stem}_ocr.json"
        with open(out_json, "w", encoding="utf8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print("  Saved:", out_json)

    print("\nAll done. OCR JSONs written to:", OUT_DIR)

if __name__ == "__main__":
    main()
