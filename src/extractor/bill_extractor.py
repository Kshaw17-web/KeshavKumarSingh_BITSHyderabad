# src/extractor/bill_extractor.py
import re
from typing import List, Dict, Any

# A small helper to detect currency-like tokens conservatively
CURRENCY_RE = re.compile(r'(?:₹|Rs\.?|INR)?\s*([0-9]{1,3}(?:[,\s][0-9]{3})*(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2}))')

def _clean_amount_str(s: str) -> float:
    """Remove commas and whitespace and convert to float."""
    try:
        return float(s.replace(',', '').replace(' ', ''))
    except Exception:
        # fallback
        try:
            return float(re.sub(r'[^\d.]', '', s))
        except Exception:
            return 0.0

def _detect_page_type(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("pharmacy", "drug", "tablet", "capsule", "syrup")):
        return "Pharmacy"
    if any(k in t for k in ("final bill", "grand total", "net payable", "amount payable", "invoice total", "total payable", "subtotal", "grand total")):
        return "Final Bill"
    return "Bill Detail"

def extract_bill_data(ocr_pages: List[str]) -> Dict[str, Any]:
    """
    Minimal extractor that takes a list of OCR page texts and returns
    a dict with structure matching the datathon response 'data' field.
    This is intentionally conservative and meant to give valid structured output
    so your API runs while you iterate on a richer extractor.
    """
    pagewise_line_items = []
    all_amounts = []
    total_item_count = 0

    for i, page_text in enumerate(ocr_pages, start=1):
        page_no = str(i)
        page_type = _detect_page_type(page_text)

        # Split into lines and examine lines containing currency-like tokens
        lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
        bill_items = []
        seen_spans = set()

        for ln in lines:
            # find all currency-like matches in the line
            for m in CURRENCY_RE.finditer(ln):
                raw_amount = m.group(1)
                span = m.span()
                # avoid repeating same span
                if span in seen_spans:
                    continue
                seen_spans.add(span)

                amt_val = _clean_amount_str(raw_amount)
                # Heuristic filter: ignore extremely small numbers that look like page numbers or years
                if amt_val == 0.0:
                    continue
                if amt_val < 0.01:
                    continue

                # Try to extract an item name candidate: the text before amount (up to 60 chars)
                before = ln[:m.start()].strip(" -:|,")
                after = ln[m.end():].strip(" -:|,")
                # prefer before if it looks like a name, else after, else whole line
                if before and len(before) > 0 and not re.search(r'\b(?:no|id|date|invoice|qty|page)\b', before.lower()):
                    item_name = before[:80]
                elif after and len(after) > 0 and not re.search(r'\b(?:no|id|date|invoice|qty|page)\b', after.lower()):
                    item_name = after[:80]
                else:
                    item_name = ln[:120]

                # Try to extract qty x rate patterns (simple)
                qty = None
                rate = None
                qxr = re.search(r'(\d+(?:\.\d+)?)\s*[xX\*]\s*(\d+(?:\.\d+)?)', ln)
                if qxr:
                    try:
                        qty = float(qxr.group(1))
                        rate = float(qxr.group(2))
                    except Exception:
                        qty = None
                        rate = None

                item = {
                    "item_name": item_name.strip() if item_name else "UNKNOWN",
                    "item_amount": round(float(amt_val), 2),
                    "item_rate": float(round(rate, 2)) if rate is not None else None,
                    "item_quantity": float(qty) if qty is not None else None
                }
                bill_items.append(item)
                all_amounts.append(float(amt_val))
                total_item_count += 1

        pagewise_line_items.append({
            "page_no": page_no,
            "page_type": page_type,
            "bill_items": bill_items
        })

    # reconciled amount: naive sum of unique amounts (avoid accidental double-counting by using positions is hard here,
    # so we sum all discovered amounts - it's a reasonable baseline)
    reconciled_amount = round(sum(all_amounts), 2) if all_amounts else 0.0

    return {
        "pagewise_line_items": pagewise_line_items,
        "total_item_count": total_item_count,
        "reconciled_amount": reconciled_amount
    }
