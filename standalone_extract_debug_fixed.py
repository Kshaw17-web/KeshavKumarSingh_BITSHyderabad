import json, os, sys
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import cv2

# import your helpers
from src.preprocessing.image_utils import preprocess_image_for_ocr
from src.utils.ocr_runner import ocr_image_to_tsv, ocr_numeric_region
from src.extractor.parsers import group_words_to_lines, detect_column_centers, map_tokens_to_columns, parse_row_from_columns, is_probable_item
from src.extractor.cleanup import dedupe_items, reconcile_totals

PDF_PATH = r"C:\\Users\\ksr20\\OneDrive\\Desktop\\Bajaj Finserv Datathon\\data\\raw\\training_samples\\TRAINING_SAMPLES\\train_sample_2.pdf"
OUT_DIR = Path("logs") / "standalone_debug_fixed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
request_id = "standalone_debug_fixed"

pages = convert_from_path(PDF_PATH, dpi=300)
page_objs = []
totals_sum = 0.0

def pil_to_gray_cv2(pil_img):
    """Convert PIL image to grayscale cv2 ndarray."""
    arr = np.array(pil_img)
    if arr.ndim == 3:
        # PIL gives RGB; convert to GRAY BGR-like for CV2 usage
        try:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        except Exception:
            # fallback: average
            return np.mean(arr, axis=2).astype(np.uint8)
    return arr

for pi, pil_page in enumerate(pages, start=1):
    # Call your preprocess API with signature (img, max_side=..., target_dpi=..., fast_mode=...)
    try:
        prepped_pil = preprocess_image_for_ocr(pil_page, max_side=1024, target_dpi=300, fast_mode=False)
        # ensure we have a PIL image
        if not hasattr(prepped_pil, "convert"):
            # if the function returned a numpy array, convert back to PIL
            prepped_pil = Image.fromarray(prepped_pil)
    except Exception as e:
        # fallback: use the original PIL converted to L
        print("preprocess error:", e)
        prepped_pil = pil_page.convert("L")

    # Convert to cv2 grayscale ndarray for ocr runner (if ocr_runner expects cv2)
    try:
        proc_cv2 = pil_to_gray_cv2(prepped_pil)
    except Exception as e:
        print("convert-to-cv2 error:", e)
        proc_cv2 = np.array(prepped_pil.convert("L"))

    # Save debug preprocessed image
    try:
        Image.fromarray(proc_cv2 if proc_cv2.ndim==2 else cv2.cvtColor(proc_cv2, cv2.COLOR_BGR2RGB)).save(OUT_DIR / f"p{pi}_pre.png")
    except Exception:
        pass

    # OCR -> TSV/dict using your ocr runner (it may accept cv2 array)
    try:
        ocr = ocr_image_to_tsv(proc_cv2, request_id=request_id, page_no=pi, save_debug_dir=str(OUT_DIR))
    except Exception as e:
        # If ocr_image_to_tsv expects PIL, pass prepped_pil
        try:
            ocr = ocr_image_to_tsv(prepped_pil, request_id=request_id, page_no=pi, save_debug_dir=str(OUT_DIR))
        except Exception as e2:
            print("OCR runner error:", e2)
            # build a minimal empty ocr structure so downstream scripts don't crash
            ocr = {"text": [], "conf": [], "left": [], "top": [], "width": [], "height": [], "line_num": [], "block_num": [], "par_num": []}

    # Group and parse
    lines = group_words_to_lines(ocr)
    centers = detect_column_centers(lines)
    parsed_items = []
    for ln in lines:
        cols = map_tokens_to_columns(ln, centers)
        parsed = parse_row_from_columns(cols)

        # try numeric reocr for rightmost numeric token if conf low
        try:
            right_numeric = None
            for t in reversed(ln):
                if any(ch.isdigit() for ch in t.get("text","")):
                    right_numeric = t; break
            if right_numeric and isinstance(right_numeric.get("conf"), (int,float)) and right_numeric.get("conf") < 60:
                l = max(0, int(right_numeric.get("left",0)) - 4)
                ttop = max(0, int(right_numeric.get("top",0)) - 4)
                r = int(right_numeric.get("left",0) + right_numeric.get("width",0) + 4)
                b = int(right_numeric.get("top",0) + right_numeric.get("height",0) + 4)
                # crop from proc_cv2 (grayscale ndarray)
                crop = proc_cv2[ttop:b, l:r]
                val = ocr_numeric_region(crop)
                if val is not None:
                    parsed["item_amount"] = val
        except Exception:
            pass

        if is_probable_item(parsed):
            parsed_items.append(parsed)

    # fallback if nothing found
    if len(parsed_items) == 0:
        for ln in lines:
            tokens = [t['text'] for t in ln if t.get('text')]
            amt = None; idx = None
            for i in range(len(tokens)-1,-1,-1):
                s = tokens[i]
                if any(ch.isdigit() for ch in s):
                    sc = s.replace('₹','').replace(',','').replace('$','')
                    sc = ''.join(ch for ch in sc if (ch.isdigit() or ch in '.-'))
                    try:
                        amt = float(sc); idx = i; break
                    except:
                        amt = None
            if amt is not None:
                name = " ".join(tokens[:idx]) if idx else " ".join(tokens)
                parsed_items.append({"item_name": name, "item_amount": amt, "item_rate": None, "item_quantity": None})

    deduped = dedupe_items(parsed_items, name_threshold=88)
    ocr_text = " ".join([t for t in ocr.get('text',[]) if t])
    import re
    m = re.search(r"(?:grand total|net amt|net amount|net total|total amount|balance amt|balance)\s*[:\s]*([₹\d,.\s]+)", ocr_text, flags=re.I)
    reported = None
    if m:
        s = m.group(1).replace('₹','').replace(',','')
        try:
            reported = float("".join(ch for ch in s if (ch.isdigit() or ch in ".-")))
        except:
            reported = None

    final_total, method = reconcile_totals(deduped, reported)
    page_obj = {"page_no": pi, "page_type": "Bill Detail", "bill_items": deduped, "reported_total": reported, "final_total": final_total}
    page_objs.append(page_obj)
    totals_sum += final_total or 0.0

out = {"source_file": PDF_PATH, "extraction_result": {"pagewise_line_items": page_objs, "total_item_count": sum(len(p['bill_items']) for p in page_objs), "reconciled_amount": totals_sum}, "n_pages": len(pages)}
Path(OUT_DIR / "last_response.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("Wrote", OUT_DIR / "last_response.json")
