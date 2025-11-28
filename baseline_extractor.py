"""
baseline_extractor.py

- Input: OCR JSON files produced earlier (ocr_outputs/*_ocr.json).
- Output: datathon-style extraction JSONs in baseline_results/
  Each output: <stem>_extracted.json with schema:
  {
    "is_success": true,
    "token_usage": {"total_tokens":0,"input_tokens":0,"output_tokens":0},
    "data": {
      "pagewise_line_items": [
        {"page_no":"1","page_type":"Bill Detail","bill_items":[{...}, ...]},
        ...
      ],
      "total_item_count": int
    }
  }

Notes:
- This is a hybrid-first implementation: strong regex rules + heuristics.
- An optional LLM fallback (commented) is provided if you want to improve
  tricky documents later (you will need an API key and small changes).
"""

import json
import re
from pathlib import Path
from decimal import Decimal, InvalidOperation
from collections import defaultdict

# Optional fuzzy matching for dedupe. If not installed, code will fall back.
try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False

# ---------- CONFIG ----------
OCR_DIR = Path(r"C:\Users\ksr20\OneDrive\Desktop\BAJAJ FINSERV DATATHON\ocr_outputs")
OUT_DIR = Path(r"C:\Users\ksr20\OneDrive\Desktop\BAJAJ FINSERV DATATHON\baseline_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tolerance settings
AMOUNT_TOL = 0.02  # 2% tolerance when comparing amounts
NAME_FUZZY_THRESH = 88  # if using rapidfuzz

# Regex patterns to find decimal numbers (rates/amounts)
RE_NUMBER = re.compile(r'(?<!\d)(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{1,2})?(?!\d)')
# Possible patterns for a full item line: name ... qty ... rate ... amount
# We'll heuristically pick last two numbers as (rate, amount) OR (amount) depending on context.

# Helper: convert string to float / decimal safely
def parse_number(s):
    if s is None:
        return 0.0
    # remove common noise
    s = s.replace(',', '').replace('₹', '').replace('Rs.', '').strip()
    # remove stray non-digit except dot/minus
    s = re.sub(r'[^\d\.\-]', '', s)
    if s == '':
        return 0.0
    try:
        return float(s)
    except Exception:
        try:
            return float(Decimal(s))
        except (InvalidOperation, Exception):
            return 0.0

# Heuristic: determine if a line is likely an item line
def is_item_line(line):
    # ignore blank, headings or lines with words like Total, Invoice, Bill, Sub-Total etc
    low = line.lower()
    if not line.strip():
        return False
    if any(w in low for w in ["total", "subtotal", "grand total", "net amount", "amount payable", "rupees", "balance", "tax", "discount", "invoice", "bill no", "patient", "insurance"]):
        return False
    # require at least one numeric token (rate or amount)
    if not RE_NUMBER.search(line):
        return False
    # length check — avoid super short tokens like page numbers
    if len(line.strip()) < 6:
        return False
    return True

# Primary rule-based parser for a page text -> list of item dicts
def parse_items_from_text(text):
    """
    Given a full page OCR text, split into lines, apply heuristics and regex
    to extract candidate items. Returns list of dicts:
    { "item_name": str, "item_quantity": float, "item_rate": float, "item_amount": float }
    """
    items = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Merge lines that are hyphen-continued or short name lines followed by numeric line
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # merge if the next line starts with a number then next line probably contains qty/rate/amount
        if i+1 < len(lines) and re.match(r'^[\d\(\)\-]', lines[i+1]):
            merged.append(line + " " + lines[i+1])
            i += 2
            continue
        # If current line ends with '-' (hyphen) join with next
        if line.endswith('-') and i+1 < len(lines):
            merged.append(line[:-1] + lines[i+1])
            i += 2
            continue
        merged.append(line)
        i += 1

    for line in merged:
        if not is_item_line(line):
            continue

        # find all numeric tokens in line
        nums = RE_NUMBER.findall(line)
        nums_parsed = [parse_number(n) for n in nums]

        item = {"item_name": None, "item_quantity": 1.0, "item_rate": 0.0, "item_amount": 0.0}

        # Strategy:
        # - If there are 3 or more numbers: assume last = amount, second-last = rate, third-last = quantity OR vice versa.
        # - If 2 numbers: assume last=amount, second=rate OR last=amount and rate unknown.
        # - If 1 number: treat as amount.
        # We'll also attempt to extract item_name as the left part before the first numeric token.

        if nums:
            # get indices of matches to split name vs numbers
            m = RE_NUMBER.search(line)
            name_part = line[:m.start()].strip()
            if not name_part:
                # fallback: everything upto last number minus numbers
                # attempt to remove numeric tokens from end
                name_part = re.sub(RE_NUMBER, '', line).strip()
            item["item_name"] = name_part or line

            # choose numbers for qty/rate/amount
            if len(nums_parsed) >= 3:
                # common pharmacy pattern: qty rate amount OR rate qty amount (ambiguous)
                # We'll assume: last = amount, second-last = rate, third-last = qty
                item["item_amount"] = nums_parsed[-1]
                item["item_rate"] = nums_parsed[-2]
                item["item_quantity"] = nums_parsed[-3] if nums_parsed[-3] != 0 else 1.0
            elif len(nums_parsed) == 2:
                # decide by context: if second is much larger than first, treat as amount
                a, b = nums_parsed[0], nums_parsed[1]
                # heuristics: amount often larger than rate or equals rate*qty
                item["item_amount"] = b
                # if a is small int, might be qty
                if float(a).is_integer() and abs(a) < 100 and a <= 1000:
                    item["item_quantity"] = a
                    # try compute rate
                    if item["item_quantity"] != 0:
                        item["item_rate"] = round(item["item_amount"] / item["item_quantity"], 2)
                else:
                    item["item_rate"] = a
                    item["item_quantity"] = 1.0
            else:
                # single number -> assume it's amount
                item["item_amount"] = nums_parsed[-1]
                item["item_rate"] = nums_parsed[-1]
                item["item_quantity"] = 1.0
        else:
            # No numbers (shouldn't happen due to is_item_line), fallback
            item["item_name"] = line
            item["item_quantity"] = 1.0
            item["item_rate"] = 0.0
            item["item_amount"] = 0.0

        # Clean up name: remove stray separators and multiple spaces
        name_clean = re.sub(r'[\s]{2,}', ' ', item["item_name"]).strip(' -:')
        item["item_name"] = name_clean

        # Round numeric values reasonably
        try:
            item["item_amount"] = round(float(item["item_amount"]), 2)
            item["item_rate"] = round(float(item["item_rate"]), 2)
            item["item_quantity"] = round(float(item["item_quantity"]), 2)
        except Exception:
            # fallback to zeros
            item["item_amount"] = float(item.get("item_amount") or 0.0)
            item["item_rate"] = float(item.get("item_rate") or 0.0)
            item["item_quantity"] = float(item.get("item_quantity") or 0.0)

        # Filter obviously bad rows
        if item["item_amount"] == 0 and item["item_rate"] == 0:
            # if both zero, skip
            continue

        items.append(item)

    return items

# Dedupe items across pages using fuzzy name + amount closeness
def dedupe_items(item_list):
    kept = []
    for it in item_list:
        dup = False
        for k in kept:
            # compare name
            if HAS_RAPIDFUZZ:
                score = fuzz.token_sort_ratio(it["item_name"], k["item_name"])
            else:
                score = 100 if it["item_name"].strip().lower() == k["item_name"].strip().lower() else 0
            # compare amount closeness
            a, b = it["item_amount"], k["item_amount"]
            amt_close = (abs(a - b) <= max(1.0, abs(b)) * AMOUNT_TOL)
            if score >= NAME_FUZZY_THRESH and amt_close:
                dup = True
                break
        if not dup:
            kept.append(it)
    return kept

# Optional LLM fallback (pseudocode)
def llm_postprocess_block(text_block):
    """
    Placeholder: send a prompt to a model to extract tabular rows from a text block.
    If you enable this, populate token usage and replace the heuristic output for low-confidence pages.
    This function is intentionally left as a stub. You can add an OpenAI call here later.
    """
    # Example return: list of items in same dict format
    return []

# Main pipeline
def process_all():
    results = []
    ocr_files = sorted(OCR_DIR.glob("*_ocr.json"))
    if not ocr_files:
        print("No OCR JSON files found in", OCR_DIR)
        return

    for jf in ocr_files:
        print("Processing OCR file:", jf.name)
        doc = json.loads(jf.read_text(encoding='utf8'))
        pagewise = []
        all_items = []
        pages = doc.get("pages", [])
        for p in pages:
            page_no = str(p.get("page_no") or p.get("page_no", 0))
            text = p.get("full_text") or p.get("text_preview") or ""
            # parse items on this page
            items = parse_items_from_text(text)
            # if page extraction seems empty or suspicious, you can call llm_postprocess_block(text)
            # fallback = llm_postprocess_block(text)
            # if fallback: items = fallback
            pagewise.append({
                "page_no": page_no,
                "page_type": "Bill Detail" if len(items)>0 else "Final Bill",
                "bill_items": [
                    {
                        "item_name": it["item_name"],
                        "item_amount": float(it["item_amount"]),
                        "item_rate": float(it["item_rate"]),
                        "item_quantity": float(it["item_quantity"])
                    } for it in items
                ]
            })
            all_items.extend([it for it in items])

        # dedupe across all_items
        unique_items = dedupe_items(all_items)
        total_item_count = len(unique_items)

        # Build datathon-style output
        out = {
            "is_success": True,
            "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
            "data": {
                "pagewise_line_items": pagewise,
                "total_item_count": total_item_count
            }
        }

        out_path = OUT_DIR / f"{jf.stem.replace('_ocr','')}_extracted.json"
        with open(out_path, "w", encoding='utf8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("Wrote extracted:", out_path)

    print("All done. Extracted files are in:", OUT_DIR)

if __name__ == "__main__":
    process_all()
