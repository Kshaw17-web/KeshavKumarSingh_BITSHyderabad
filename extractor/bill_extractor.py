# extractor/bill_extractor.py
# Minimal extraction function used by src/api.py
# Exposes: extract_bill_data(file_path) -> (extracted_data_dict, token_usage_dict)
#
# The function:
# - converts PDF -> images (uses poppler if available)
# - runs Tesseract OCR on each page
# - applies heuristic parsing to find line-items (name, qty, rate, amount)
# - returns data in the datathon schema and a token_usage placeholder

import re
import os
from pathlib import Path
from pdf2image import convert_from_path, convert_from_bytes
import pytesseract
import json

# Config - update if your poppler/tesseract live elsewhere
POPPLER_BIN = r"C:\poppler-25.11.0\Library\bin"    # keep as-is if you used this earlier
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

# number regex
RE_NUMBER = re.compile(r'(?<!\d)(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{1,2})?(?!\d)')

def parse_number(s):
    if s is None:
        return 0.0
    s = str(s).replace(',', '').replace('₹','').replace('Rs.','').strip()
    s = re.sub(r'[^\d\.\-]','', s)
    try:
        return float(s)
    except Exception:
        return 0.0

def is_item_line(line):
    low = line.lower()
    if not line.strip():
        return False
    if any(w in low for w in ("total", "subtotal", "grand total", "net amount", "amount payable",
                              "rupees", "balance", "tax", "discount", "invoice", "bill no", "patient", "insurance")):
        return False
    if not RE_NUMBER.search(line):
        return False
    if len(line.strip()) < 6:
        return False
    return True

def parse_items_from_text(text):
    items = []
    if not text:
        return items
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i+1 < len(lines) and re.match(r'^[\d\(\)\-]', lines[i+1]):
            merged.append(line + " " + lines[i+1]); i += 2; continue
        if line.endswith('-') and i+1 < len(lines):
            merged.append(line[:-1] + lines[i+1]); i += 2; continue
        merged.append(line); i += 1

    for line in merged:
        if not is_item_line(line):
            continue
        nums = RE_NUMBER.findall(line)
        nums_parsed = [parse_number(n) for n in nums]
        item = {"item_name": None, "item_quantity": 1.0, "item_rate": 0.0, "item_amount": 0.0}
        if nums:
            m = RE_NUMBER.search(line)
            name_part = line[:m.start()].strip() if m else re.sub(RE_NUMBER, '', line).strip()
            item["item_name"] = name_part or line
            if len(nums_parsed) >= 3:
                item["item_amount"] = nums_parsed[-1]
                item["item_rate"] = nums_parsed[-2]
                item["item_quantity"] = nums_parsed[-3] if nums_parsed[-3] != 0 else 1.0
            elif len(nums_parsed) == 2:
                a, b = nums_parsed[0], nums_parsed[1]
                item["item_amount"] = b
                if float(a).is_integer() and abs(a) < 100:
                    item["item_quantity"] = a
                    if item["item_quantity"] != 0:
                        item["item_rate"] = round(item["item_amount"] / item["item_quantity"], 2)
                else:
                    item["item_rate"] = a
                    item["item_quantity"] = 1.0
            else:
                item["item_amount"] = nums_parsed[-1]
                item["item_rate"] = nums_parsed[-1]
                item["item_quantity"] = 1.0
        else:
            item["item_name"] = line
        item["item_name"] = re.sub(r'[\s]{2,}', ' ', item["item_name"]).strip(' -:')
        try:
            item["item_amount"] = round(float(item["item_amount"]), 2)
            item["item_rate"] = round(float(item["item_rate"]), 2)
            item["item_quantity"] = round(float(item["item_quantity"]), 2)
        except Exception:
            item["item_amount"] = float(item.get("item_amount") or 0.0)
            item["item_rate"] = float(item.get("item_rate") or 0.0)
            item["item_quantity"] = float(item.get("item_quantity") or 0.0)
        if item["item_amount"] == 0 and item["item_rate"] == 0:
            continue
        items.append(item)
    return items

def convert_pdf_to_images(pdf_path):
    pages = []
    try:
        pages = convert_from_path(str(pdf_path), dpi=200, poppler_path=POPPLER_BIN)
    except Exception:
        try:
            raw = Path(pdf_path).read_bytes()
            pages = convert_from_bytes(raw, dpi=200, poppler_path=POPPLER_BIN)
        except Exception:
            pages = []
    return pages

def extract_bill_data(file_path):
    """
    Input: file_path (PDF or image bytes saved to disk)
    Output: (extracted_data_dict, token_usage_dict)
    extracted_data_dict follows datathon schema:
      { "pagewise_line_items": [...], "total_item_count": int }
    token_usage_dict is placeholder: {"total_tokens":0,"input_tokens":0,"output_tokens":0}
    """
    fp = Path(file_path)
    pages = convert_pdf_to_images(fp)
    # If convert failed and file is an image, try opening as image
    from PIL import Image
    if not pages:
        try:
            pages = [Image.open(fp)]
        except Exception:
            pages = []

    pagewise = []
    all_items = []
    for pno, page in enumerate(pages, start=1):
        try:
            text = pytesseract.image_to_string(page, lang='eng')
        except Exception:
            text = ""
        items = parse_items_from_text(text)
        pagewise.append({
            "page_no": str(pno),
            "page_type": "Bill Detail" if len(items) > 0 else "Final Bill",
            "bill_items": [
                {
                    "item_name": it["item_name"],
                    "item_amount": float(it["item_amount"]),
                    "item_rate": float(it["item_rate"]),
                    "item_quantity": float(it["item_quantity"])
                } for it in items
            ]
        })
        all_items.extend(items)

    # simple dedupe by exact name+amount (fast)
    unique = []
    seen = set()
    for it in all_items:
        key = (it["item_name"].strip().lower(), round(float(it["item_amount"]),2))
        if key in seen:
            continue
        seen.add(key)
        unique.append(it)

    extracted = {
        "pagewise_line_items": pagewise,
        "total_item_count": len(unique)
    }

    token_usage = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
    return extracted, token_usage
