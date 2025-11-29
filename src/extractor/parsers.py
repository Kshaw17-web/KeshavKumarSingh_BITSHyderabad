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
            "conf": int(float(ocr_dict.get('conf', [ -1 ])[i])) if str(ocr_dict.get('conf', ['-1'])[i]).strip() not in ("-1","") else -1
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
    Uses k-means if sklearn available; otherwise uses quantile-based heuristics.
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
    Returns ordered list of column texts (left->right).
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
    """Clean numeric-like string and convert to float, handling common OCR mistakes."""
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
    Heuristic extraction from column strings.
    columns: list of strings (left->right).
    Returns dict: item_name, item_amount (float), item_rate (float), item_quantity (int/float)
    """
    tokens = []
    for c in columns:
        if c:
            tokens.extend([t for t in c.split() if t.strip() != ""])
    numeric_tokens = [t for t in tokens if re.search(r'\d', t)]
    cleaned_nums = [_clean_num_str(t) for t in numeric_tokens]

    item_amount = None
    item_rate = None
    item_quantity = None

    if cleaned_nums:
        # pick last non-None as amount
        for v in reversed(cleaned_nums):
            if v is not None:
                item_amount = v
                break
        # attempt to assign qty & rate from previous numerical values
        if len(cleaned_nums) >= 2:
            second_last = cleaned_nums[-2]
            if second_last is not None and float(second_last).is_integer() and abs(second_last) < 1000:
                item_quantity = int(second_last)
                if len(cleaned_nums) >= 3:
                    item_rate = cleaned_nums[-3]
            else:
                item_rate = second_last

    # derive item_name by removing trailing numeric-like tokens (up to 3) from tokens
    name_tokens = tokens.copy()
    for _ in range(3):
        if name_tokens and re.search(r'\d', name_tokens[-1]):
            name_tokens.pop()
        else:
            break
    item_name = " ".join([t for t in name_tokens if t.lower() not in ("no", "qty", "pcs")]).strip() or None

    return {
        "item_name": item_name,
        "item_amount": item_amount,
        "item_rate": item_rate,
        "item_quantity": item_quantity
    }


def is_probable_item(parsed_row: Dict[str, Optional[object]]) -> bool:
    """
    Basic filter to decide if a parsed_row is an item row.
    Prefer rows with item_amount; exclude common non-item keywords.
    """
    if parsed_row.get("item_amount") is not None:
        return True
    name = (parsed_row.get("item_name") or "").lower()
    if not name:
        return False
    non_item_keywords = ["subtotal", "total", "discount", "gst", "tax", "invoice", "net amount", "amount due", "mrp"]
    if any(k in name for k in non_item_keywords):
        return False
    return True


if __name__ == "__main__":
    import json, sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.extractor.parsers <ocr_json_file>")
        sys.exit(0)
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        ocr = json.load(f)
    lines = group_words_to_lines(ocr)
    centers = detect_column_centers(lines)
    for ln in lines[:20]:
        cols = map_tokens_to_columns(ln, centers)
        parsed = parse_row_from_columns(cols)
        print("COLUMNS:", cols)
        print("PARSED:", parsed)
        print("---")

