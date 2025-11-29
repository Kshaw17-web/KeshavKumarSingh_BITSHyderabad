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
import numpy as np

# optional dependency for clustering; if sklearn not available, fallback to heuristic
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# optional dependency for peak detection
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    find_peaks = None


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


def detect_column_centers(all_lines: List[List[Dict[str, Any]]], max_columns: int = 5, gray_img_array: Optional[np.ndarray] = None) -> List[float]:
    """
    Return sorted x-centers for columns detected across all lines.
    
    Uses k-means clustering if sklearn is available, with fallback to projection profile
    and peak detection for complex layouts. This reduces columns collapsing on complex layouts.
    
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
    # Collect token centers
    centers = []
    for line in all_lines:
        for t in line:
            centers.append(t['left'] + t['width'] / 2.0)
    
    if len(centers) < 3:
        return []
    
    # Try k-means first if available
    if SKLEARN_AVAILABLE:
        try:
            k = min(max_columns, max(2, len(set([int(round(c)) for c in centers])) // 5))
            k = max(2, k)
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit([[c] for c in centers])
            col_centers = sorted([float(c[0]) for c in kmeans.cluster_centers_])
            # Validate k-means result (check if clusters are reasonable)
            if len(col_centers) >= 2 and col_centers[-1] - col_centers[0] > 50:
                return col_centers
        except Exception:
            # k-means failed, fall through to projection profile
            pass
    
    # Fallback: Projection profile + peak detection
    # Compute vertical projection profile (sum of token widths per x-position)
    if not centers:
        return []
    
    # Find the range of x positions
    min_x = min(centers) - 50
    max_x = max(centers) + 50
    width = int(max_x - min_x) + 1
    
    # Create projection profile: for each x position, sum up token "ink" (width contribution)
    projection = np.zeros(width, dtype=np.float32)
    
    for line in all_lines:
        for t in line:
            left = t['left']
            token_width = t.get('width', 10)
            # Add contribution to projection profile
            start_idx = max(0, int(left - min_x))
            end_idx = min(width, int(left - min_x + token_width))
            if start_idx < end_idx:
                projection[start_idx:end_idx] += 1.0
    
    # Smooth the projection profile slightly
    if len(projection) > 10:
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        projection = np.convolve(projection, kernel, mode='same')
    
    # Detect peaks in projection profile
    peaks = []
    
    if SCIPY_AVAILABLE and find_peaks is not None:
        try:
            # Use scipy peak detection
            # height: minimum peak height (at least 10% of max)
            # distance: minimum distance between peaks (at least 50 pixels)
            min_height = np.max(projection) * 0.1
            min_distance = max(50, width // (max_columns * 2))
            
            detected_peaks, properties = find_peaks(
                projection,
                height=min_height,
                distance=min_distance,
                prominence=np.max(projection) * 0.05
            )
            
            peaks = [float(p + min_x) for p in detected_peaks]
        except Exception:
            # scipy failed, use simple local maxima
            pass
    
    # Fallback: Simple local maxima detection
    if not peaks:
        # Find local maxima manually
        min_height = np.max(projection) * 0.1
        min_distance = max(50, width // (max_columns * 2))
        
        for i in range(min_distance, len(projection) - min_distance):
            if projection[i] >= min_height:
                # Check if it's a local maximum
                is_peak = True
                for j in range(max(0, i - min_distance // 2), min(len(projection), i + min_distance // 2)):
                    if j != i and projection[j] > projection[i]:
                        is_peak = False
                        break
                
                if is_peak:
                    # Check if we already have a peak too close
                    x_pos = float(i + min_x)
                    too_close = False
                    for existing_peak in peaks:
                        if abs(x_pos - existing_peak) < min_distance:
                            too_close = True
                            break
                    
                    if not too_close:
                        peaks.append(x_pos)
                        if len(peaks) >= max_columns:
                            break
    
    # Sort and limit to max_columns
    peaks = sorted(peaks)
    if len(peaks) > max_columns:
        # Keep the strongest peaks
        peak_strengths = [projection[int(p - min_x)] for p in peaks]
        sorted_indices = sorted(range(len(peaks)), key=lambda i: peak_strengths[i], reverse=True)
        peaks = [peaks[i] for i in sorted_indices[:max_columns]]
        peaks = sorted(peaks)
    
    if peaks:
        return peaks
    
    # Final fallback: use quantiles
    arr = np.array(centers)
    k = min(max_columns, max(2, len(arr) // 10))
    quantiles = np.quantile(arr, np.linspace(0, 1, k + 1)[1:-1])
    result = sorted(list(quantiles))
    
    # If we still have too few centers (< 2), try projection-based method if image provided
    if len(result) < 2 and gray_img_array is not None:
        try:
            projection_centers = detect_columns_by_projection(gray_img_array, min_peak_distance=40)
            if len(projection_centers) >= 2:
                return projection_centers
        except Exception:
            # Projection method failed, return quantile result
            pass
    
    # Return what we have
    return result


def detect_columns_by_projection(gray_img_array, min_peak_distance=40):
    """
    Detect column centers using vertical projection profile on grayscale image.
    
    This function uses image projection to find column boundaries by detecting
    peaks in the vertical projection profile (sum of dark pixels per x-coordinate).
    
    Args:
        gray_img_array: numpy 2D array (dtype uint8, 0..255) representing grayscale image
        min_peak_distance: Minimum distance between peaks in pixels (default: 40)
    
    Returns:
        List of x-coordinates representing detected column centers (peaks)
    
    Example:
        >>> import cv2
        >>> img = cv2.imread("invoice.png", cv2.IMREAD_GRAYSCALE)
        >>> col_centers = detect_columns_by_projection(img, min_peak_distance=50)
        >>> print(f"Detected {len(col_centers)} columns")
    """
    if gray_img_array is None or gray_img_array.size == 0:
        return []
    
    # Ensure it's a 2D array
    if len(gray_img_array.shape) != 2:
        return []
    
    # Compute vertical projection profile: sum of dark pixels per x-coordinate
    # Invert so dark pixels (text) have higher values
    col_profile = (255 - gray_img_array).sum(axis=0)
    
    # Normalize & find peaks
    if SCIPY_AVAILABLE and find_peaks is not None:
        try:
            # Use percentile-based height threshold (50th percentile)
            height_threshold = np.percentile(col_profile, 50)
            peaks, _ = find_peaks(
                col_profile,
                distance=min_peak_distance,
                height=height_threshold
            )
            return list(peaks.astype(float))
        except Exception:
            # scipy failed, fall through to simple peak detection
            pass
    
    # Fallback: Simple peak detection
    peaks = []
    height_threshold = np.percentile(col_profile, 50)
    
    for i in range(min_peak_distance, len(col_profile) - min_peak_distance):
        if col_profile[i] >= height_threshold:
            # Check if it's a local maximum
            is_peak = True
            for j in range(max(0, i - min_peak_distance // 2), 
                          min(len(col_profile), i + min_peak_distance // 2)):
                if j != i and col_profile[j] > col_profile[i]:
                    is_peak = False
                    break
            
            if is_peak:
                # Check if we already have a peak too close
                too_close = False
                for existing_peak in peaks:
                    if abs(i - existing_peak) < min_peak_distance:
                        too_close = True
                        break
                
                if not too_close:
                    peaks.append(float(i))
    
    return sorted(peaks)


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
    Optimized for leaderboard: Capture ALL line items while avoiding false positives.
    
    CRITICAL REQUIREMENT: Don't miss any line-item entries!
    
    Strategy:
    - Be LESS conservative to avoid missing items (critical requirement)
    - Accept items with amount > 0 OR items with name + quantity/rate
    - Reject only obvious non-items (exact summary keywords, invoice IDs, pure totals)
    
    Returns True if row is likely a bill line item.
    """
    name = (parsed_row.get("item_name") or "").lower().strip()
    
    # Reject obvious non-items (but be careful - only exact matches)
    # Only reject if name is EXACTLY a summary keyword (not if it contains it)
    exact_non_items = ["subtotal", "sub total", "total", "grand total", "net amount", 
                       "amount due", "balance", "paid", "invoice no", "invoice number",
                       "gst", "cgst", "sgst", "igst", "tax", "vat", "discount"]
    if name in exact_non_items:
        return False
    
    # Reject if name is just a number or empty
    if not name or name.replace(".", "").replace(",", "").strip().isdigit():
        return False

    # Require at least one numeric amount > 0 OR quantity+rate combo
    amt = parsed_row.get("item_amount")
    qty = parsed_row.get("item_quantity")
    rate = parsed_row.get("item_rate")
    
    # Accept if has valid amount
    if amt is not None and isinstance(amt, (int, float)) and amt > 0:
        # Reject if amount looks like an invoice-id (very large integer > 100000)
        if amt > 100000 and float(amt).is_integer():
            return False
        # Accept if amount is reasonable
        return True
    
    # Accept if has quantity AND rate (even without amount, might be valid)
    if qty and rate and isinstance(qty, (int, float)) and isinstance(rate, (int, float)):
        if qty > 0 and rate > 0:
            return True
    
    # If no amount and no qty+rate, but has a substantial name, be lenient
    # (might be item with missing amount due to OCR error)
    if name and len(name.split()) >= 2:  # At least 2 words
        # Only accept if name doesn't look like a summary row
        if not any(exact_kw in name for exact_kw in exact_non_items):
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
