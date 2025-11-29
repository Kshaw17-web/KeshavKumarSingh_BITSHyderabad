import json, os, sys
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import numpy as np

from src.preprocessing.image_utils import preprocess_image_for_ocr
from src.utils.ocr_runner import ocr_image_to_tsv, ocr_numeric_region
from src.extractor.parsers import group_words_to_lines, detect_column_centers, map_tokens_to_columns, parse_row_from_columns, is_probable_item
from src.extractor.cleanup import dedupe_items, reconcile_totals

PDF_PATH = r"C:\\Users\\ksr20\\OneDrive\\Desktop\\Bajaj Finserv Datathon\\data\\raw\\training_samples\\TRAINING_SAMPLES\\train_sample_2.pdf"
OUT_DIR = Path("logs") / "standalone_debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)
request_id = "standalone_debug"

pages = convert_from_path(PDF_PATH, dpi=300)
page_objs = []
totals_sum = 0.0
for pi, pil_page in enumerate(pages, start=1):
    try:
        proc = preprocess_image_for_ocr(pil_page, return_cv2=True, apply_adaptive_threshold=True, ensure_min_width=1200)
    except Exception as e:
        print("preprocess error:", e)
        proc = np.array(pil_page.convert("L"))

    ocr = ocr_image_to_tsv(proc, request_id=request_id, page_no=pi, save_debug_dir=str(OUT_DIR))
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
                l = max(0, int(right_numeric["left"]) - 4)
                ttop = max(0, int(right_numeric["top"]) - 4)
                r = int(right_numeric["left"] + right_numeric["width"] + 4)
                b = int(right_numeric["top"] + right_numeric["height"] + 4)
                crop = proc[ttop:b, l:r]
                val = ocr_numeric_region(crop)
                if val is not None:
                    parsed["item_amount"] = val
        except Exception:
            pass

        if is_probable_item(parsed):
            parsed_items.append(parsed)

    if len(parsed_items) == 0:
        for ln in lines:
            tokens = [t['text'] for t in ln if t.get('text')]
            amt = None; idx = None
            for i in range(len(tokens)-1,-1,-1):
                s = tokens[i]
                if any(ch.isdigit() for ch in s):
                    sc = s.replace('₹','').replace(',','').replace('$','')
                    sc = ''.join(ch for ch in sc if (ch.isdigit() or ch in '.-'))
                    try: amt = float(sc); idx = i; break
                    except: amt = None
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
        try: reported = float("".join(ch for ch in s if (ch.isdigit() or ch in ".-")))
        except: reported = None

    final_total, method = reconcile_totals(deduped, reported)
    page_obj = {"page_no": pi, "page_type": "Bill Detail", "bill_items": deduped, "reported_total": reported, "final_total": final_total}
    page_objs.append(page_obj)
    totals_sum += final_total or 0.0

out = {"source_file": PDF_PATH, "extraction_result": {"pagewise_line_items": page_objs, "total_item_count": sum(len(p['bill_items']) for p in page_objs), "reconciled_amount": totals_sum}, "n_pages": len(pages)}
Path(OUT_DIR / "last_response.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("Wrote", OUT_DIR / "last_response.json")
