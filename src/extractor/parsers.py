"""
Geometry-first parsing utilities for bill extraction.

Functions:
- group_words_to_lines(ocr_dict)
- detect_column_centers(all_lines)
- map_tokens_to_columns(line_tokens, col_centers)
- parse_row_from_columns(columns)
- is_probable_item(parsed_row)

Intended to be fed with pytesseract image_to_data(..., Output.DICT) output.
"""

from typing import List, Dict, Any, Optional

import re

# optional dependency for clustering; if sklearn not available, fallback to heuristic
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def group_words_to_lines(ocr_dict: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """
    Convert pytesseract Output.DICT into list of lines.
    
    Each line is a list of token dicts: text,left,top,width,height,conf
    
    Args:
        ocr_dict: Dictionary from pytesseract.image_to_data(..., output_type=Output.DICT)
                  Expected keys: 'text', 'left', 'top', 'width', 'height', 'conf',
                                'block_num', 'par_num', 'line_num'
    
    Returns:
        List of lines, where each line is a list of token dictionaries with keys:
        - text: str - The OCR text
        - left: int - X coordinate of left edge
        - top: int - Y coordinate of top edge
        - width: int - Width of bounding box
        - height: int - Height of bounding box
        - conf: int - Confidence score (-1 if unavailable)
    
    Example:
        >>> ocr_dict = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        >>> lines = group_words_to_lines(ocr_dict)
        >>> print(f"Found {len(lines)} lines")
    """
    lines = {}
    n = len(ocr_dict.get('text', []))
    
    for i in range(n):
        text = str(ocr_dict['text'][i]).strip()
        if text == "" or text.lower() in (" ",):
            continue
        
        key = (ocr_dict.get('block_num', [0])[i], ocr_dict.get('par_num', [0])[i], ocr_dict.get('line_num', [0])[i])
        
        token = {
            "text": text,
            "left": int(ocr_dict.get('left', [0])[i]),
            "top": int(ocr_dict.get('top', [0])[i]),
            "width": int(ocr_dict.get('width', [0])[i]),
            "height": int(ocr_dict.get('height', [0])[i]),
            "conf": int(float(ocr_dict.get('conf', [-1])[i])) if str(ocr_dict.get('conf', ['-1'])[i]).strip() not in ("-1", "") else -1
        }
        
        lines.setdefault(key, []).append(token)
    
    # Sort by page/block/para/line order
    processed = []
    for k, tokens in sorted(lines.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        tokens = sorted(tokens, key=lambda t: t['left'])
        processed.append(tokens)
    
    return processed


def detect_column_centers(all_lines: List[List[Dict[str, Any]]], max_columns: int = 5) -> List[float]:
    """
    Return sorted x-centers for columns detected across all lines.
    
    Uses k-means clustering if sklearn is available; otherwise uses quantile-based heuristics.
    This function analyzes the horizontal positions of all tokens across all lines to identify
    column boundaries, which is essential for parsing tabular data.
    
    Args:
        all_lines: List of lines, where each line is a list of token dictionaries
        max_columns: Maximum number of columns to detect (default: 5)
    
    Returns:
        Sorted list of x-coordinates representing column centers (left to right)
    
    Example:
        >>> lines = group_words_to_lines(ocr_dict)
        >>> col_centers = detect_column_centers(lines, max_columns=4)
        >>> print(f"Detected {len(col_centers)} columns at x-positions: {col_centers}")
    """
    centers = []
    for line in all_lines:
        for t in line:
            centers.append(t['left'] + t['width'] / 2.0)
    
    if len(centers) < 3:
        return []
    
    if SKLEARN_AVAILABLE:
        k = min(max_columns, max(2, len(set([int(round(c)) for c in centers])) // 5))
        k = max(2, k)
        kmeans = KMeans(n_clusters=k, random_state=0).fit([[c] for c in centers])
        col_centers = sorted([float(c[0]) for c in kmeans.cluster_centers_])
        return col_centers
    
    # fallback heuristic: use quantiles
    import numpy as np
    arr = np.array(centers)
    k = min(max_columns, max(2, len(arr) // 10))
    quantiles = np.quantile(arr, np.linspace(0, 1, k + 1)[1:-1])
    return sorted(list(quantiles))


def map_tokens_to_columns(line_tokens: List[Dict[str, Any]], col_centers: List[float]) -> List[str]:
    """
    Map a line's tokens to a list of column strings based on nearest column center.
    
    This function assigns each token in a line to its nearest column based on the
    horizontal position of the token's center. Tokens are then grouped by column
    and joined into strings.
    
    Args:
        line_tokens: List of token dictionaries for a single line
        col_centers: List of x-coordinates representing column centers (from detect_column_centers)
    
    Returns:
        Ordered list of column texts (left to right), where each element is a space-separated
        string of tokens assigned to that column
    
    Example:
        >>> line = [{"text": "Item", "left": 10, ...}, {"text": "100.00", "left": 200, ...}]
        >>> col_centers = [50.0, 250.0]
        >>> columns = map_tokens_to_columns(line, col_centers)
        >>> # Returns: ["Item", "100.00"]
    """
    if not col_centers:
        # no columns detected: return single joined string
        return [" ".join([t['text'] for t in line_tokens])]
    
    cols = {i: [] for i in range(len(col_centers))}
    
    for t in line_tokens:
        cx = t['left'] + t['width'] / 2.0
        distances = [abs(cx - c) for c in col_centers]
        idx = int(min(range(len(distances)), key=lambda i: distances[i]))
        cols.setdefault(idx, []).append(t['text'])
    
    ordered = [" ".join(cols[i]).strip() for i in sorted(cols.keys())]
    return ordered


def _clean_num_str(s: Optional[str]) -> Optional[float]:
    """
    Clean numeric-like string and convert to float, handling common OCR mistakes.
    
    This helper function normalizes OCR text that may contain numeric values by:
    - Removing currency symbols (₹, Rs)
    - Removing thousands separators (commas)
    - Fixing common OCR character misrecognitions (O→0, l/I/|→1)
    - Removing non-numeric characters
    
    Args:
        s: String that may contain a number
    
    Returns:
        Float value if conversion succeeds, None otherwise
    
    Example:
        >>> _clean_num_str("₹1,234.56")  # Returns: 1234.56
        >>> _clean_num_str("lO0.5O")     # Returns: 100.50 (after OCR fixes)
    """
    if not s:
        return None
    
    s0 = s.strip()
    s0 = s0.replace('₹', '').replace('Rs', '').replace(',', '')
    s0 = re.sub(r'[Oo]', '0', s0)  # O -> 0
    s0 = re.sub(r'[lI\|]', '1', s0)  # l/I/| -> 1
    s0 = re.sub(r'[^\d.\-]', '', s0)
    
    if s0 in ("", ".", "-", "--"):
        return None
    
    try:
        return float(s0)
    except Exception:
        return None


def parse_row_from_columns(columns: List[str]) -> Dict[str, Optional[object]]:
    """
    Improved heuristic extraction from column strings.

    Strategy:
    - Flatten tokens and collect numeric-like tokens in order (left->right).
    - Prefer the rightmost *decimal-like* numeric (contains '.' or > 99 or has 2+ digits after decimal)
      as the item_amount. This skips trailing small integers (GST% like 5,12).
    - If decimal-like amount not found, use the last numeric that looks like an amount.
    - For qty: select a small integer (<= 1000) appearing *before* amount.
    - For rate: prefer a numeric with decimal or that fits amount/qty relationship.
    - If rate missing and qty & amount present: compute rate = amount / qty
    - Return a stable dict with item_name, item_amount, item_rate, item_quantity, and optional gst_percent.
    """
    tokens = []
    for c in columns:
        if c:
            tokens.extend([t for t in c.split() if t.strip() != ""])

    # Helper to clean numeric string to float or None
    def _clean_num_str_raw(s):
        if s is None:
            return None
        s0 = str(s).strip()
        s0 = s0.replace('₹', '').replace('Rs', '').replace(',', '').replace('$','')
        s0 = re.sub(r'[^\d.\-]', '', s0)
        if s0 in ("", ".", "-", "--"):
            return None
        try:
            return float(s0)
        except:
            return None

    # gather numeric tokens with original string + cleaned float if possible
    numerics = []
    for t in tokens:
        if re.search(r'\d', t):
            cleaned = _clean_num_str_raw(t)
            numerics.append({"raw": t, "float": cleaned})

    # Function to decide if a numeric looks decimal-like / amount-like
    def looks_amount_like(n):
        if n is None:
            return False
        s = str(n)
        if '.' in s:
            # has decimal point: likely amount/rate
            return True
        # if >= 100 -> likely an amount (some items have amounts <100 though)
        try:
            if float(n) >= 100:
                return True
        except:
            pass
        # two or more digits (>=10) is more likely to be amount than GST% or small count
        try:
            if abs(float(n)) >= 10:
                return True
        except:
            pass
        return False

    # Default outputs
    item_amount = None
    item_rate = None
    item_qty = None
    gst_percent = None

    # If there are numeric tokens, scan from right to left to pick amount, gst, qty, rate
    if numerics:
        # find candidate amount: prefer rightmost numeric where looks_amount_like == True
        amount_idx = None
        for i in range(len(numerics)-1, -1, -1):
            if numerics[i]["float"] is not None and looks_amount_like(numerics[i]["float"]):
                amount_idx = i
                break
        # If not found, fallback to last numeric with a float
        if amount_idx is None:
            for i in range(len(numerics)-1, -1, -1):
                if numerics[i]["float"] is not None:
                    amount_idx = i
                    break

        if amount_idx is not None:
            item_amount = numerics[amount_idx]["float"]

            # check if there's a small integer after amount -> likely GST%
            if amount_idx + 1 < len(numerics):
                nxt = numerics[amount_idx + 1]["float"]
                if nxt is not None and 0 < nxt <= 30 and float(nxt).is_integer():
                    # treat as GST%
                    gst_percent = int(float(nxt))

            # pick qty: scan left of amount, looking for a small integer
            # Conservative: only accept if token has no decimals, length <= 3, and < 100
            for j in range(amount_idx - 1, -1, -1):
                v = numerics[j]["float"]
                raw_token = numerics[j]["raw"]
                if v is None:
                    continue
                # Check if token contains decimal places or .00 (treat as amount, not qty)
                if '.' in raw_token or '.00' in raw_token:
                    continue  # Skip decimals - these are amounts/rates, not quantities
                # Only accept small integer quantities: length <= 3 and < 100
                if float(v).is_integer():
                    v_int = int(float(v))
                    token_len = len(raw_token.replace(',', '').replace('₹', '').replace('Rs', '').strip())
                    if 0 < v_int < 100 and token_len <= 3:
                        item_qty = v_int
                        break

            # pick rate: look for a decimal-like numeric just left of amount (or any decimal-like)
            rate_idx = None
            # prefer the numeric immediately left of amount if decimal-like
            if amount_idx - 1 >= 0 and numerics[amount_idx - 1]["float"] is not None:
                cand = numerics[amount_idx - 1]["float"]
                if looks_amount_like(cand):
                    rate_idx = amount_idx - 1
            # otherwise scan left for decimal-like
            if rate_idx is None:
                for j in range(amount_idx - 1, -1, -1):
                    v = numerics[j]["float"]
                    if v is not None and looks_amount_like(v):
                        rate_idx = j
                        break
            if rate_idx is not None:
                item_rate = numerics[rate_idx]["float"]

            # if amount exists but rate missing and qty present, compute rate
            if item_amount is not None and item_rate is None and item_qty:
                try:
                    if item_qty != 0:
                        item_rate = round(item_amount / item_qty, 4)
                except Exception:
                    pass

    # derive item_name: exclude the numeric tokens that are clearly qty/rate/amount/gst at the right side
    name_tokens = tokens.copy()
    # remove trailing numeric-like tokens up to 4 (qty, rate, disc, amount, gst)
    remove_count = 0
    for _ in range(5):
        if name_tokens and re.search(r'\d', name_tokens[-1]):
            name_tokens.pop()
            remove_count += 1
        else:
            break
    item_name = " ".join([t for t in name_tokens if t.lower() not in ("no", "qty", "pcs")]).strip() or None

    # Final normalization: ensure floats are numeric types
    # If item_amount looks tiny (<=30) but gst_percent is None and a preceding decimal exists, attempt correction
    if item_amount is not None and item_amount <= 30 and gst_percent is None:
        # try to recover: if there exists any decimal-like numeric left of this small number, pick that instead
        if numerics:
            for i in range(len(numerics)-1, -1, -1):
                v = numerics[i]["float"]
                if looks_amount_like(v):
                    # choose the first decimal-like from the right that is not equal to current small value
                    if v != item_amount:
                        item_amount = v
                        # recompute qty/rate heuristics could be re-run, but skip for speed
                        break

    # Post-process quantity: strict validation
    # If rightmost token includes ".00" or has decimals, treat as amount not qty
    if item_qty is not None:
        # Check if the quantity came from a token with decimals
        qty_token_found = False
        for num_info in numerics:
            if num_info["float"] == item_qty:
                raw_token = num_info["raw"]
                # If token contains decimal places or .00, it's an amount, not quantity
                if '.' in raw_token or '.00' in raw_token:
                    item_qty = None
                    break
                qty_token_found = True
                break
        
        # Only accept small integer quantities: <= 0 or > 100 are invalid
        if item_qty is not None:
            if isinstance(item_qty, (int, float)):
                if item_qty <= 0 or item_qty > 100:
                    item_qty = None

    return {
        "item_name": item_name,
        "item_amount": item_amount,
        "item_rate": item_rate,
        "item_quantity": item_qty,
        "gst_percent": gst_percent
    }


def is_probable_item(parsed_row: Dict[str, Optional[object]], conf_threshold: int = 40) -> bool:
    """
    Consider a row an item if it has:
    - a valid item_amount (float and > 0), OR
    - a valid quantity (int > 0) and a rate (float > 0).
    
    Avoid rows that look like totals/headers.
    """
    name = (parsed_row.get("item_name") or "").lower()
    non_item_keywords = ["subtotal", "total", "discount", "gst", "tax", "invoice", "net amount", "amount due", "mrp", "balance", "paid"]
    if any(k in name for k in non_item_keywords):
        return False

    amt = parsed_row.get("item_amount")
    qty = parsed_row.get("item_quantity")
    rate = parsed_row.get("item_rate")

    if amt is not None and isinstance(amt, (int, float)) and amt > 0:
        return True
    if qty and rate and isinstance(rate, (int, float)) and rate > 0:
        return True

    return False


if __name__ == "__main__":
    """
    CLI usage example for testing the parsers module.
    
    Usage:
        python -m src.extractor.parsers tests/fixtures/sample_ocr.json
    
    The input JSON file should contain pytesseract Output.DICT format:
    {
        "text": [...],
        "left": [...],
        "top": [...],
        "width": [...],
        "height": [...],
        "conf": [...],
        "block_num": [...],
        "par_num": [...],
        "line_num": [...]
    }
    """
    import json
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.extractor.parsers <ocr_json_file>")
        sys.exit(0)
    
    path = sys.argv[1]
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            ocr = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {path}: {e}")
        sys.exit(1)
    
    lines = group_words_to_lines(ocr)
    centers = detect_column_centers(lines)
    
    print(f"Found {len(lines)} lines")
    print(f"Detected {len(centers)} column centers: {centers}")
    print("\n" + "="*60)
    print("Parsing first 20 lines:")
    print("="*60 + "\n")
    
    for ln in lines[:20]:
        cols = map_tokens_to_columns(ln, centers)
        parsed = parse_row_from_columns(cols)
        is_item = is_probable_item(parsed)
        
        print("COLUMNS:", cols)
        print("PARSED:", parsed)
        print("IS_ITEM:", is_item)
        print("---")
