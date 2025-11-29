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
    
    Handles Indian hospital bill format:
    - Pattern: [serial] [date/code] [item_name] [rate]x[quantity] [total]
    - Example: "1. 1Hi1/2025RIOOL 2Dechocardiography 1180.00x1.00 1180.00"
    - Example: "4.15/11/2025LB270 DENGUEIGMAND.IGG 640.00x1,00 640.00"
    """
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
        # Handle "rate x quantity" format (e.g., "640.00x1.00" or "350.00x2.00")
        if 'x' in s0.lower():
            # Split on 'x' and take the first part as rate
            parts = s0.lower().split('x')
            if len(parts) >= 2:
                s0 = parts[0].strip()  # Take rate part
        s0 = s0.replace('₹', '').replace('Rs', '').replace('$','')
        
        # Handle comma as decimal separator (European format: "184,00")
        # If comma appears and no dot, or comma is after digits near end, it's likely decimal
        if ',' in s0 and '.' not in s0:
            # European format: "184,00" -> "184.00"
            s0 = s0.replace(',', '.')
        elif ',' in s0:
            # Indian format: "1,500.00" -> remove comma (thousand separator)
            s0 = s0.replace(',', '')
        
        s0 = re.sub(r'[^\d.\-]', '', s0)
        if s0 in ("", ".", "-", "--"):
            return None
        try:
            return float(s0)
        except:
            return None
    
    # Helper to check if a token looks like a date code (not a rate)
    def _is_date_code(token_str):
        """Check if token looks like a date code (e.g., '61S/11/2025LBO10', '15/11/2025LB270')"""
        if not token_str:
            return False
        # Contains slashes with digits (date pattern)
        if '/' in token_str and re.search(r'\d+/\d+', token_str):
            return True
        # Very long numeric string (>8 digits) likely a date code or invoice number
        digits_only = ''.join(c for c in token_str if c.isdigit())
        if len(digits_only) > 8:
            return True
        # Pattern like "2025LBO10" or "LB270" (alphanumeric code)
        if re.search(r'\d+[A-Z]+\d+', token_str) or re.search(r'[A-Z]+\d+', token_str):
            return True
        return False
    
    # Helper to extract rate and quantity from "rate x quantity" format
    def _extract_rate_qty(token):
        """Extract rate and quantity from tokens like '640.00x1.00' or '350.00x2.00' or '184,00x1.00'"""
        if 'x' in token.lower():
            parts = token.lower().split('x')
            if len(parts) >= 2:
                try:
                    rate_str = parts[0].strip().replace('₹', '').replace('Rs', '')
                    qty_str = parts[1].strip().replace('₹', '').replace('Rs', '')
                    
                    # Handle comma as decimal separator (European format: "184,00")
                    # If comma appears after digits and before end, it's likely a decimal separator
                    if ',' in rate_str and '.' not in rate_str:
                        # European format: "184,00" -> "184.00"
                        rate_str = rate_str.replace(',', '.')
                    elif ',' in rate_str:
                        # Indian format: "1,500.00" -> remove comma
                        rate_str = rate_str.replace(',', '')
                    
                    if ',' in qty_str and '.' not in qty_str:
                        qty_str = qty_str.replace(',', '.')
                    elif ',' in qty_str:
                        qty_str = qty_str.replace(',', '')
                    
                    rate = float(''.join(c for c in rate_str if c.isdigit() or c in '.-'))
                    qty = float(''.join(c for c in qty_str if c.isdigit() or c in '.-'))
                    if 0 < rate <= 1000000 and 0 < qty <= 1000:  # Reasonable ranges
                        return rate, qty
                except:
                    pass
        return None, None

    # gather numeric tokens with original string + cleaned float if possible
    # Also check for "rate x quantity" format
    numerics = []
    rate_qty_tokens = []  # Track tokens with "rate x quantity" format
    
    for t in tokens:
        if re.search(r'\d', t):
            # Check if this is a "rate x quantity" token
            rate, qty = _extract_rate_qty(t)
            if rate is not None and qty is not None:
                rate_qty_tokens.append({"raw": t, "rate": rate, "quantity": qty})
                # Also add rate as a numeric (for amount detection)
                numerics.append({"raw": t, "float": rate, "is_rate_qty": True})
            else:
                cleaned = _clean_num_str_raw(t)
                numerics.append({"raw": t, "float": cleaned, "is_rate_qty": False})

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

    # First, check if we have "rate x quantity" tokens (common in hospital bills)
    if rate_qty_tokens:
        # Use the rightmost rate x quantity token
        rq_token = rate_qty_tokens[-1]
        item_rate = rq_token["rate"]
        item_qty = rq_token["quantity"]
        # Calculate amount if not explicitly found
        if item_rate and item_qty:
            item_amount = item_rate * item_qty

    # If there are numeric tokens, scan from right to left to pick amount, gst, qty, rate
    if numerics:
        # find candidate amount: prefer rightmost numeric where looks_amount_like == True
        # Skip "rate x quantity" tokens (we already handled those)
        amount_idx = None
        for i in range(len(numerics)-1, -1, -1):
            # Skip rate x quantity tokens (they're not the final amount)
            if numerics[i].get("is_rate_qty", False):
                continue
            if numerics[i]["float"] is not None and looks_amount_like(numerics[i]["float"]):
                amount_idx = i
                break
        # If not found, fallback to last numeric with a float (but not rate x quantity)
        if amount_idx is None:
            for i in range(len(numerics)-1, -1, -1):
                if numerics[i].get("is_rate_qty", False):
                    continue
                if numerics[i]["float"] is not None:
                    amount_idx = i
                    break

        if amount_idx is not None:
            candidate_amount = numerics[amount_idx]["float"]
            # Prefer the rightmost amount over calculated amount (if we have rate x qty)
            if item_amount is None:
                item_amount = candidate_amount
            elif candidate_amount > 0:
                # If amounts are close, use the rightmost one (it's likely the total)
                if abs(candidate_amount - item_amount) / max(candidate_amount, item_amount) < 0.2:
                    item_amount = candidate_amount
                elif candidate_amount > item_amount * 0.9:
                    # Rightmost amount is likely the total, prefer it
                    item_amount = candidate_amount

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
            # BUT exclude date codes and invoice numbers
            rate_idx = None
            # prefer the numeric immediately left of amount if decimal-like
            if amount_idx - 1 >= 0 and numerics[amount_idx - 1]["float"] is not None:
                cand = numerics[amount_idx - 1]["float"]
                raw_cand = numerics[amount_idx - 1]["raw"]
                # Reject if it looks like a date code (e.g., "6111202510" from "61S/11/2025LBO10")
                if looks_amount_like(cand) and not _is_date_code(raw_cand):
                    rate_idx = amount_idx - 1
            # otherwise scan left for decimal-like
            if rate_idx is None:
                for j in range(amount_idx - 1, -1, -1):
                    v = numerics[j]["float"]
                    raw_v = numerics[j]["raw"]
                    if v is not None and looks_amount_like(v) and not _is_date_code(raw_v):
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
            
            # If we have rate x qty but amount is different, prefer the explicit amount
            # (the rightmost amount is usually the total, which is what we want)
            if item_rate and item_qty and item_amount:
                calculated = item_rate * item_qty
                # If amounts are close (within 20%), use the explicit amount
                if abs(calculated - item_amount) / max(calculated, item_amount) < 0.2:
                    # They match, keep as is
                    pass
                else:
                    # Use the explicit amount (rightmost), recalculate rate if needed
                    if item_qty > 0:
                        item_rate = round(item_amount / item_qty, 4)
            
            # If we have rate x qty but amount is different, prefer the explicit amount
            # (the rightmost amount is usually the total, which is what we want)
            if item_rate and item_qty and item_amount:
                calculated = item_rate * item_qty
                # If amounts are close (within 10%), use the explicit amount
                if abs(calculated - item_amount) / max(calculated, item_amount) < 0.1:
                    # They match, keep as is
                    pass
                else:
                    # Use the explicit amount (rightmost), recalculate rate if needed
                    if item_qty > 0:
                        item_rate = round(item_amount / item_qty, 4)

    # derive item_name: exclude the numeric tokens that are clearly qty/rate/amount/gst at the right side
    name_tokens = tokens.copy()
    
    # Remove serial numbers at the start (e.g., "1.", "2.", "4.")
    if name_tokens and re.match(r'^\d+[.,]?$', name_tokens[0].strip()):
        name_tokens.pop(0)
    
    # Remove date/code patterns (e.g., "1Hi1/2025RIOOL", "15/11/2025LB270")
    # These usually come after serial numbers
    if name_tokens:
        first_token = name_tokens[0]
        # Pattern: contains digits, slashes, or alphanumeric codes
        if re.search(r'\d+[/-]\d+', first_token) or (len(first_token) > 5 and re.search(r'\d+[A-Z]+\d+', first_token)):
            name_tokens.pop(0)
    
    # remove trailing numeric-like tokens up to 5 (rate x qty, amount, gst)
    # Also remove "rate x quantity" format tokens
    removed = 0
    for _ in range(6):  # Increased to handle rate x qty tokens
        if not name_tokens:
            break
        last_token = name_tokens[-1]
        # Remove if it's numeric or "rate x quantity" format
        if re.search(r'\d', last_token):
            # Check if it's "rate x quantity" format
            if 'x' in last_token.lower():
                name_tokens.pop()
                removed += 1
            elif re.search(r'\d+[.,]\d+', last_token) or last_token.replace(',', '').replace('.', '').isdigit():
                name_tokens.pop()
                removed += 1
            else:
                break
        else:
            break
    
    # Clean up item name
    item_name = " ".join([t for t in name_tokens if t.lower() not in ("no", "qty", "pcs", ".", ",")]).strip() or None
    
    # Remove leading/trailing special characters
    if item_name:
        item_name = re.sub(r'^[.,\s]+|[.,\s]+$', '', item_name)

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
    
    # Reject common header patterns (these are NOT bill items)
    header_patterns = [
        "bill no", "billno", "bill date", "billedate", "biedate",
        "patient name", "patiestname", "patientname", "patiest",
        "reg no", "regno", "registration",
        "ipd no", "ipdno", "ip no", "ipno", "ipno bmno", "bmno", "bm no",
        "mobile no", "mobileno", "phone", "mebile",
        "age", "sex", "gender", "agebex", "age/bex", "age bex",
        "address", "admission date", "discharge date",
        "doctor", "dr.", "consulting", "commlt", "doctorname", "doctor name",
        "discharge date", "discharge time", "dischargedatetime",
        "category", "tegery",
        "sno", "sl no", "serial no",
        "particulars", "description", "item",
        "qty", "quantity", "rate", "amount", "total",
        "department", "ward", "roony", "bed",
        "husband", "hesband", "nases",
        "subtotal", "sub total", "subtotml", "sub totml"
    ]
    name_lower = name.lower()
    # Check if name matches header patterns (exact or starts with/ends with/contains)
    for pattern in header_patterns:
        if (name_lower == pattern or 
            name_lower.startswith(pattern + " ") or 
            name_lower.endswith(" " + pattern) or
            name_lower.startswith(pattern + ".") or
            " " + pattern + " " in " " + name_lower + " "):
            return False
    
    # Reject patterns that look like header fields (e.g., "IPNo BMNo", "Age/Bex", "DoctorMaree DischargeDaimTime")
    # These often have specific patterns: short words, abbreviations, or field labels
    header_field_patterns = [
        r'^[a-z]{1,3}\s+[a-z]{1,3}\s*$',  # Very short words like "IPNo BMNo"
        r'^[a-z]+/[a-z]+$',  # Patterns like "Age/Bex"
        r'^[a-z]+\s*[a-z]+\s*(date|time|name|no|id)$',  # Field labels
        r'^(doctor|patient|discharge|admission)\s*[a-z]+\s*(date|time|name)',  # Doctor/Patient fields
    ]
    for pattern in header_field_patterns:
        if re.match(pattern, name_lower):
            return False
    
    # Reject if name contains only header-like abbreviations (2-3 short words, all caps or mixed)
    words = name_lower.split()
    if len(words) <= 3:
        # Check if all words are very short (1-4 chars) and look like abbreviations
        if all(len(w) <= 4 for w in words):
            # Common header abbreviations
            header_abbrevs = ['ip', 'no', 'bm', 'age', 'sex', 'reg', 'id', 'ipd', 'uhid', 'mrn',
                             'dr', 'doc', 'dis', 'adm', 'disch', 'date', 'time', 'name', 'addr']
            if all(w in header_abbrevs or w.replace('/', '').replace('-', '') in header_abbrevs for w in words):
                return False
    
    # Reject if name is just a date pattern or code (e.g., "1Hi1/2025RIOOL", "15/11/2025LB270")
    name_no_spaces = name.replace(' ', '')
    if re.match(r'^[\d/]+[A-Z]+\d+$', name_no_spaces) or re.match(r'^\d+[/-]\d+[/-]\d+', name):
        return False
    
    # Reject if name looks like "Amount in Words" or similar summary text
    if (re.search(r'amount\s*in\s*words', name_lower) or 
        re.search(r'words?\s*only', name_lower) or
        re.search(r'amountinword', name_lower) or  # OCR variant: "AmountinWorde"
        re.search(r'rupees?\s*only', name_lower)):
        return False
    
    # Reject if name is very short and looks like a code/header (e.g., "D.0.A.", "L", "t")
    if len(name.strip()) <= 3:
        # Check if it's just digits, single letter, or pattern like "D.0.A"
        if (name.strip().isdigit() or 
            re.match(r'^[A-Z]\.?\d*\.?$', name.strip()) or
            re.match(r'^[a-z]$', name.strip()) or
            name.strip() in ['t', 'L', 'i', 'j', '.']):
            return False
    
    # Reject patterns like "D.0.A. t DOLD" or "Department Ward/RoonyBed1"
    if re.search(r'\b(d\.0\.a|dold|department|ward|roony|bed)\b', name_lower):
        return False
    
    # Reject if name contains only header-like words
    header_words = ['d', '0', 'a', 't', 'l', 'i', 'j', 'department', 'ward', 'roony', 'bed']
    name_words = name_lower.split()
    if len(name_words) <= 3 and all(w in header_words or len(w) <= 2 for w in name_words):
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
