"""
High-performance bill extraction pipeline with optimized speed for large PDFs.
Supports multilingual, handwritten, and complex table layouts.
"""

import re
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

# Import new utilities
from src.utils.ocr_runner import run_ocr_parallel, extract_text_from_ocr_result, ocr_image_to_tsv, ocr_numeric_region
from src.preprocessing.image_utils import preprocess_image_for_ocr, is_blank_page
from src.preprocessing.fraud_filters import detect_fraud_flags, compute_unified_fraud_score

# Import parsers and cleanup
from src.extractor.parsers import (
    group_words_to_lines,
    detect_column_centers,
    map_tokens_to_columns,
    parse_row_from_columns,
    is_probable_item
)
from src.extractor.cleanup import dedupe_items, reconcile_totals

# LayoutLMv3 model integration (optional, falls back to heuristic)
try:
    from src.extractor.layoutlmv3_wrapper import extract_with_layoutlmv3
    from src.extractor.ensemble_reconciler import reconcile_ensemble, format_items_for_schema
    LAYOUTLMV3_AVAILABLE = True
except Exception:
    LAYOUTLMV3_AVAILABLE = False
    extract_with_layoutlmv3 = None
    reconcile_ensemble = None
    format_items_for_schema = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

# LayoutParser for structural extraction
try:
    import layoutparser as lp
    LAYOUTPARSER_AVAILABLE = True
except ImportError:
    LAYOUTPARSER_AVAILABLE = False
    lp = None

# Sentence transformer for page classification
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    SentenceTransformer = None

# Word vectors for semantic similarity (fallback)
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    fuzz = None


# Pre-compiled regex patterns for item extraction (performance optimization)
CURRENCY_SYMBOLS = r'[₹Rs\.INR]'
AMOUNT_PATTERN = r'(\d{1,6}(?:[,\s]\d{3})*(?:\.\d{1,2})?)'
CURRENCY_AMOUNT_RE = re.compile(
    rf'({CURRENCY_SYMBOLS}?\s*{AMOUNT_PATTERN})',
    re.IGNORECASE
)
NUMBER_RE = re.compile(r'(\d+(?:[,\s]\d{3})*(?:\.\d{1,2})?)')
QTY_RATE_MULT_RE = re.compile(r'(\d+(?:\.\d+)?)\s*[×x]\s*(\d+(?:\.\d+)?)', re.IGNORECASE)
TOTAL_PATTERN = re.compile(r'\b(total|subtotal|grand total|tax|gst|discount)\b', re.IGNORECASE)

# Patterns to reject (dates, invoice numbers, etc.) - pre-compiled
DATE_PATTERN = re.compile(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b')
YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')
INVOICE_PATTERN = re.compile(r'\b(INV|INVOICE|BILL\s*NO|MRN|UHID|PATIENT\s*ID)\b', re.IGNORECASE)
WHITESPACE_RE = re.compile(r'\s+')
REPORTED_TOTAL_RE = re.compile(
    r'\b(?:total|grand total|net payable|amount payable)[\s:]*[₹Rs\.INR]?\s*(\d{1,6}(?:[,\s]\d{3})*(?:\.\d{1,2})?)',
    re.IGNORECASE
)

# Global sentence transformer model (lazy loading)
_page_classifier_model = None


def _get_page_classifier_model():
    """Lazy load sentence transformer model for page classification."""
    global _page_classifier_model
    if _page_classifier_model is None and SENTENCE_TRANSFORMER_AVAILABLE:
        try:
            _page_classifier_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            _page_classifier_model = None
    return _page_classifier_model


def _clean_ocr_text(text: str) -> str:
    """
    NLP cleanup of OCR outputs.
    
    - De-duplicate repeated characters (e.g., "PAAATHOLOGY" → "PATHOLOGY")
    - Fix OCR merged tokens (e.g., "100.501" → "100.50 1")
    - Normalize currency symbols
    - Remove sequences of >3 symbols
    
    Args:
        text: Raw OCR text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove sequences of >3 symbols
    text = re.sub(r'[^\w\s]{4,}', ' ', text)
    
    # De-duplicate repeated characters (3+ consecutive)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Fix OCR merged tokens: number.number followed by number
    # Pattern: digit.digit digit (e.g., "100.501" → "100.50 1")
    text = re.sub(r'(\d+\.\d{2})(\d+)', r'\1 \2', text)
    
    # Normalize currency symbols
    text = re.sub(r'\b(Rs\.?|INR)\b', '₹', text, flags=re.IGNORECASE)
    text = re.sub(r'Rs\.', '₹', text)
    
    # Fix common OCR errors
    text = text.replace('|', 'I')  # Pipe to I
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    return text.strip()


def _is_likely_identifier(text: str) -> bool:
    """Check if text looks like an identifier (invoice number, date, etc.) rather than an item."""
    if not text or len(text.strip()) < 3:
        return False
    
    text_lower = text.lower()
    
    # Check for invoice/ID keywords
    if INVOICE_PATTERN.search(text):
        return True
    
    # Check for date patterns
    if DATE_PATTERN.search(text):
        return True
    
    # Check for year patterns (but not amounts)
    if YEAR_PATTERN.search(text) and len(NUMBER_RE.findall(text)) <= 1:
        return True
    
    # Check for long digit sequences (likely IDs)
    digits_only = re.sub(r'\D', '', text)
    if len(digits_only) >= 8 and len(digits_only) / max(1, len(text.replace(' ', ''))) > 0.6:
        return True
    
    return False


def _extract_amount_from_text(text: str) -> Optional[float]:
    """Extract the largest valid amount from a line of text."""
    amounts = []
    
    # Find all currency patterns
    for match in CURRENCY_AMOUNT_RE.finditer(text):
        amount_str = match.group(2) if match.lastindex >= 2 else match.group(1)
        amount_str = amount_str.replace(',', '').replace(' ', '')
        try:
            amt = float(amount_str)
            if 0.01 <= amt <= 10_000_000:  # Reasonable range
                amounts.append(amt)
        except ValueError:
            continue
    
    # If no currency pattern, try finding numbers
    if not amounts:
        numbers = NUMBER_RE.findall(text)
        for num_str in numbers:
            try:
                num = float(num_str.replace(',', '').replace(' ', ''))
                if 0.01 <= num <= 10_000_000:
                    amounts.append(num)
            except ValueError:
                continue
    
    return max(amounts) if amounts else None


def _extract_qty_rate_from_text(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract quantity and rate from text line using multiple patterns."""
    numbers = NUMBER_RE.findall(text)
    numbers_clean = []
    
    for num_str in numbers:
        try:
            num = float(num_str.replace(',', '').replace(' ', ''))
            if 0.01 <= num <= 100_000:  # Reasonable range for qty/rate
                numbers_clean.append(num)
        except ValueError:
            continue
    
    # Pattern 1: qty × rate (using pre-compiled regex)
    mult_match = QTY_RATE_MULT_RE.search(text)
    if mult_match:
        qty = float(mult_match.group(1))
        rate = float(mult_match.group(2))
        return qty, rate
    
    # Pattern 2: qty | price | amount (pipe separated)
    pipe_parts = [p.strip() for p in text.split('|')]
    if len(pipe_parts) >= 3:
        try:
            qty = float(pipe_parts[-3].replace(',', ''))
            rate = float(pipe_parts[-2].replace(',', ''))
            return qty, rate
        except (ValueError, IndexError):
            pass
    
    # Pattern 3: Standard pattern (name qty rate amount)
    if len(numbers_clean) >= 3:
        return numbers_clean[-3], numbers_clean[-2]
    elif len(numbers_clean) == 2:
        # Assume: name rate amount (qty=1)
        return 1.0, numbers_clean[0]
    elif len(numbers_clean) == 1:
        # Only one number, might be rate or amount
        return None, None
    else:
        return None, None


def _infer_missing_amount(items: List[Dict[str, Any]], current_idx: int) -> Optional[float]:
    """Infer missing amount from neighbor rows."""
    if current_idx == 0 or len(items) == 0:
        return None
    
    # Check previous item
    if current_idx > 0:
        prev_item = items[current_idx - 1]
        if prev_item.get("item_rate") and prev_item.get("item_quantity"):
            # Use same rate * qty
            return prev_item["item_rate"] * prev_item["item_quantity"]
    
    # Check next item (if available)
    if current_idx < len(items) - 1:
        next_item = items[current_idx + 1]
        if next_item.get("item_rate") and next_item.get("item_quantity"):
            return next_item["item_rate"] * next_item["item_quantity"]
    
    return None


def _classify_page_type(text: str) -> str:
    """
    Classify page type using sentence transformer embeddings.
    Falls back to keyword matching if transformer not available.
    """
    if not text or len(text.strip()) < 10:
        return "Bill Detail"
    
    # Try sentence transformer first
    model = _get_page_classifier_model()
    if model is not None:
        try:
            # Reference embeddings
            reference_texts = [
                "pharmacy invoice medicine prescription drugs",
                "final bill grand total net payable amount payable",
                "medical report bill detail itemized charges"
            ]
            reference_embeddings = model.encode(reference_texts)
            
            # Get text embedding (use first 500 chars for efficiency)
            text_sample = text[:500] if len(text) > 500 else text
            text_embedding = model.encode([text_sample])
            
            # Calculate cosine similarity
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(text_embedding, reference_embeddings)[0]
            except ImportError:
                # Fallback: simple dot product (normalized)
                import numpy as np
                text_norm = text_embedding / np.linalg.norm(text_embedding)
                ref_norm = reference_embeddings / np.linalg.norm(reference_embeddings, axis=1, keepdims=True)
                similarities = np.dot(text_norm, ref_norm.T)[0]
            
            max_idx = np.argmax(similarities)
            if similarities[max_idx] > 0.3:  # Threshold
                if max_idx == 0:
                    return "Pharmacy"
                elif max_idx == 1:
                    return "Final Bill"
                else:
                    return "Bill Detail"
        except Exception:
            pass
    
    # Fallback to keyword matching
    text_lower = text.lower()
    
    if any(keyword in text_lower for keyword in ['pharmacy', 'medicines', 'drugs', 'prescription']):
        return "Pharmacy"
    
    if any(keyword in text_lower for keyword in ['final bill', 'grand total', 'net payable', 'amount payable']):
        return "Final Bill"
    
    return "Bill Detail"


def _extract_items_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Enhanced item extraction using regex + semantic similarity.
    
    Args:
        text: Cleaned OCR text from a page
        
    Returns:
        List of bill items with: item_name, item_amount, item_rate, item_quantity, confidence
    """
    items = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    
    # Track items for neighbor inference
    temp_items = []
    
    for line in lines:
        # Skip header/footer lines
        if _is_likely_identifier(line):
            continue
        
        # Skip total lines (using pre-compiled regex)
        if TOTAL_PATTERN.search(line):
            continue
        
        # Extract amount
        amount = _extract_amount_from_text(line)
        
        # Extract quantity and rate
        qty, rate = _extract_qty_rate_from_text(line)
        
        # If amount missing but qty and rate available, calculate
        if amount is None and qty is not None and rate is not None:
            amount = qty * rate
        
        # If still missing, try to infer from neighbors
        if amount is None:
            amount = _infer_missing_amount(temp_items, len(temp_items))
        
        if amount is None or amount < 0.01:
            continue
        
        # Extract item name (everything before the numbers)
        item_name = line
        # Remove amount from end
        for match in CURRENCY_AMOUNT_RE.finditer(line):
            item_name = item_name[:match.start()].strip()
            break
        if not item_name:
            # Fallback: remove all numbers
            item_name = NUMBER_RE.sub('', line).strip()
        
        # Clean item name (using pre-compiled regex)
        item_name = WHITESPACE_RE.sub(' ', item_name).strip()
        if not item_name or len(item_name) < 2:
            item_name = "UNKNOWN_ITEM"
        
        # Calculate confidence based on extraction quality
        confidence = 0.7  # Base confidence
        if rate is not None and qty is not None:
            confidence = 0.9  # High confidence if qty and rate found
        elif rate is not None or qty is not None:
            confidence = 0.8  # Medium confidence if one found
        
        # Check if amount matches qty * rate
        if rate is not None and qty is not None:
            expected = qty * rate
            if abs(expected - amount) / max(amount, 0.01) < 0.1:  # Within 10%
                confidence = 0.95
        
        item = {
            "item_name": item_name,
            "item_amount": round(amount, 2),
            "item_rate": round(rate, 2) if rate is not None else None,
            "item_quantity": round(qty, 2) if qty is not None else None,
            "confidence": round(confidence, 2)
        }
        items.append(item)
        temp_items.append(item)
    
    return items


def _detect_page_layout(img: "Image.Image") -> Optional[Dict[str, Any]]:
    """
    Use LayoutParser + detectron2 to extract text regions and table structure.
    
    Args:
        img: PIL Image
        
    Returns:
        Layout metadata dict with text blocks and table information
    """
    if not LAYOUTPARSER_AVAILABLE or not PIL_AVAILABLE:
        return None
    
    try:
        # Initialize LayoutParser model
        model = lp.Detectron2LayoutModel(
            'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
        )
        
        # Detect layout
        layout = model.detect(np.array(img))
        
        # Extract text regions and tables
        text_blocks = [block for block in layout if block.type in ['Text', 'Title']]
        table_blocks = [block for block in layout if block.type == 'Table']
        
        return {
            "layout_detected": True,
            "n_text_blocks": len(text_blocks),
            "n_table_blocks": len(table_blocks),
            "text_regions": [{"bbox": block.block.coordinates, "type": block.type} for block in text_blocks],
            "table_regions": [{"bbox": block.block.coordinates} for block in table_blocks]
        }
    except Exception:
        return None


def _reconcile_totals(items: List[Dict[str, Any]], reported_total: Optional[float] = None) -> Tuple[bool, float, float]:
    """
    Weighted reconciliation based on confidence.
    
    Args:
        items: List of bill items
        reported_total: Reported total from page (if available)
        
    Returns:
        Tuple of (reconciliation_ok, calculated_sum, relative_error)
    """
    if not items:
        return False, 0.0, 0.0
    
    # Weighted sum based on confidence
    weighted_sum = 0.0
    total_weight = 0.0
    
    for item in items:
        amount = item.get("item_amount", 0.0)
        confidence = item.get("confidence", 0.7)
        weight = confidence
        weighted_sum += amount * weight
        total_weight += weight
    
    # Also calculate simple sum
    simple_sum = sum(item.get("item_amount", 0.0) for item in items)
    
    # Use weighted sum if weights available, else simple sum
    calculated_sum = weighted_sum / total_weight if total_weight > 0 else simple_sum
    
    if reported_total is None:
        return None, calculated_sum, 0.0
    
    # Calculate relative error
    if reported_total > 0:
        relative_error = abs(calculated_sum - reported_total) / reported_total
        reconciliation_ok = relative_error < 0.05  # 5% tolerance
    else:
        relative_error = 0.0
        reconciliation_ok = False
    
    return reconciliation_ok, calculated_sum, relative_error


def extract_bill_data(
    pages: Union[List["Image.Image"], List[np.ndarray]],
    enable_profiling: bool = False
) -> Dict[str, Any]:
    """
    High-performance extraction function optimized for large PDFs (20-50 pages).
    
    Optimizations:
    - Fast blank page detection (skip processing)
    - Parallel OCR with shared models
    - Fast preprocessing mode for large PDFs
    - Pre-compiled regexes
    - Profiling hooks
    
    Args:
        pages: List of PIL Images or numpy arrays (one per page)
        enable_profiling: Enable timing profiling (default False)
        
    Returns:
        Dictionary with HackRx-compatible structure
    """
    profile_times = {} if enable_profiling else None
    start_time = time.time() if enable_profiling else None
    
    if not pages:
        return {
            "pagewise_line_items": [],
            "total_item_count": 0,
            "reconciled_amount": 0.0
        }
    
    # Convert numpy arrays to PIL Images if needed
    t0 = time.time() if enable_profiling else None
    pil_images = []
    for page in pages:
        if isinstance(page, np.ndarray):
            if PIL_AVAILABLE:
                pil_images.append(Image.fromarray(page))
            else:
                raise RuntimeError("PIL/Pillow required for numpy array conversion")
        elif isinstance(page, Image.Image):
            pil_images.append(page)
        else:
            raise TypeError(f"Unsupported page type: {type(page)}")
    if enable_profiling:
        profile_times["convert_to_pil"] = time.time() - t0
    
    # Fast mode for large PDFs
    use_fast_mode = len(pil_images) >= 20
    
    # Preprocess images with fast pipeline
    t1 = time.time() if enable_profiling else None
    preprocessed_images = []
    blank_page_indices = []
    
    for idx, img in enumerate(pil_images):
        try:
            # Fast blank page detection (skip expensive preprocessing)
            if is_blank_page(img):
                blank_page_indices.append(idx)
                preprocessed_images.append(img)  # Keep original for blank pages
                continue
            
            # Fast preprocessing (skip expensive ops in fast mode)
            preprocessed = preprocess_image_for_ocr(img, max_side=1024, fast_mode=use_fast_mode)
            preprocessed_images.append(preprocessed)
        except Exception:
            preprocessed_images.append(img)
    if enable_profiling:
        profile_times["preprocessing"] = time.time() - t1
    
    # Run optimized OCR in parallel (shared model, no auto-detect for speed)
    t2 = time.time() if enable_profiling else None
    try:
        ocr_results = run_ocr_parallel(
            preprocessed_images,
            lang="en",  # Fixed language for speed (disable auto-detect)
            use_angle_cls=True,
            max_workers=6,
            rec_algorithm="SVTR_LCNet",
            det_db_unclip_ratio=2.3,
            max_text_length=200,
            auto_detect_lang=False  # Disabled for speed
        )
        ocr_texts = extract_text_from_ocr_result(ocr_results)
    except Exception:
        ocr_texts = [""] * len(preprocessed_images)
    if enable_profiling:
        profile_times["ocr_parallel"] = time.time() - t2
    
    # Process each page (skip blank pages early)
    t3 = time.time() if enable_profiling else None
    pagewise_items = []
    total_item_count = 0
    total_amount = 0.0
    
    for page_idx, (img, ocr_text) in enumerate(zip(preprocessed_images, ocr_texts), start=1):
        # Skip processing for blank pages
        if (page_idx - 1) in blank_page_indices:
            pagewise_items.append({
                "page_no": str(page_idx),
                "page_type": "Bill Detail",
                "bill_items": [],
                "fraud_flags": [],
                "reported_total": None,
                "reconciliation_ok": None,
                "reconciliation_relative_error": None
            })
            continue
        
        # Clean OCR text with NLP cleanup
        cleaned_text = _clean_ocr_text(ocr_text)
        
        # Skip if text is too short (likely blank or error)
        if len(cleaned_text.strip()) < 10:
            pagewise_items.append({
                "page_no": str(page_idx),
                "page_type": "Bill Detail",
                "bill_items": [],
                "fraud_flags": [],
                "reported_total": None,
                "reconciliation_ok": None,
                "reconciliation_relative_error": None
            })
            continue
        
        # Detect fraud flags (use fast mode for large PDFs)
        try:
            fraud_flags, debug_maps = detect_fraud_flags(
                img,
                save_debug_maps=False,
                use_fast_mode=use_fast_mode
            )
        except Exception:
            fraud_flags = []
            debug_maps = {}
        
        # Skip LayoutParser in fast mode (expensive)
        layout_meta = None
        if not use_fast_mode:
            layout_meta = _detect_page_layout(img)
        
        # Extract items using ensemble: heuristic + LayoutLMv3 model
        try:
            # Heuristic extraction (always runs)
            heuristic_items = _extract_items_from_text(cleaned_text)
            
            # LayoutLMv3 model extraction (if available)
            model_items = []
            if LAYOUTLMV3_AVAILABLE and extract_with_layoutlmv3 is not None:
                try:
                    # Get OCR boxes from OCR results if available
                    ocr_boxes = None
                    if page_idx <= len(ocr_results):
                        ocr_result = ocr_results[page_idx - 1]
                        if len(ocr_result) > 2:
                            ocr_details = ocr_result[2]
                            if ocr_details:
                                ocr_boxes = [detail.get("bbox", []) for detail in ocr_details if detail.get("bbox")]
                    
                    # Run model extraction
                    model_results = extract_with_layoutlmv3(
                        [img],
                        [cleaned_text],
                        [ocr_boxes] if ocr_boxes else None
                    )
                    if model_results and len(model_results) > 0:
                        model_items = model_results[0].get("items", [])
                except Exception as e:
                    # Fallback: use heuristic only
                    if enable_profiling:
                        print(f"  LayoutLMv3 extraction failed for page {page_idx}: {e}")
                    model_items = []
            
            # Reconcile ensemble results
            if LAYOUTLMV3_AVAILABLE and reconcile_ensemble is not None and format_items_for_schema is not None and model_items:
                try:
                    reconciled_items = reconcile_ensemble(
                        heuristic_items,
                        model_items,
                        prefer_model=True,
                        name_threshold=0.8
                    )
                    bill_items = format_items_for_schema(reconciled_items)
                except Exception as e:
                    # Fallback: use heuristic only
                    if enable_profiling:
                        print(f"  Ensemble reconciliation failed for page {page_idx}: {e}")
                    bill_items = heuristic_items
            else:
                # Use heuristic only
                bill_items = heuristic_items
        except Exception:
            bill_items = []
        
        # Fast page classification (skip transformer in fast mode)
        try:
            if use_fast_mode:
                # Fast keyword-based classification
                text_lower = cleaned_text.lower()
                if any(kw in text_lower for kw in ['pharmacy', 'medicines', 'drugs']):
                    page_type = "Pharmacy"
                elif any(kw in text_lower for kw in ['final bill', 'grand total', 'net payable']):
                    page_type = "Final Bill"
                else:
                    page_type = "Bill Detail"
            else:
                page_type = _classify_page_type(cleaned_text)
        except Exception:
            page_type = "Bill Detail"
        
        # Extract reported total from text (using pre-compiled regex)
        reported_total = None
        total_match = REPORTED_TOTAL_RE.search(cleaned_text)
        if total_match:
            try:
                reported_total = float(total_match.group(1).replace(',', '').replace(' ', ''))
            except ValueError:
                pass
        
        # Reconcile totals
        reconciliation_ok, calculated_sum, relative_error = _reconcile_totals(bill_items, reported_total)
        
        # Calculate page totals
        page_total = sum(item.get("item_amount", 0.0) for item in bill_items)
        total_item_count += len(bill_items)
        total_amount += page_total
        
        # Format fraud flags for output
        formatted_flags = [
            {
                "flag_type": flag.get("flag_type", "unknown"),
                "score": float(flag.get("score", 0.0)),
                "meta": flag.get("meta", {})
            }
            for flag in fraud_flags
        ]
        
        pagewise_items.append({
            "page_no": str(page_idx),
            "page_type": page_type,
            "bill_items": bill_items,
            "fraud_flags": formatted_flags,
            "reported_total": reported_total,
            "reconciliation_ok": reconciliation_ok,
            "reconciliation_relative_error": float(relative_error) if relative_error is not None else None
        })
    
    if enable_profiling:
        profile_times["item_extraction"] = time.time() - t3
        profile_times["total_time"] = time.time() - start_time
        # Add profiling to output
        return {
            "pagewise_line_items": pagewise_items,
            "total_item_count": total_item_count,
            "reconciled_amount": round(total_amount, 2),
            "_profiling": profile_times
        }
    
    return {
        "pagewise_line_items": pagewise_items,
        "total_item_count": total_item_count,
        "reconciled_amount": round(total_amount, 2)
    }


def extract_bill_data_with_tsv(
    pages: Union[List["Image.Image"], List[np.ndarray]],
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract bill data using OCR TSV -> parsers -> dedupe -> reconcile pipeline.
    
    This function implements the full geometry-first parsing pipeline:
    1. Preprocess images and save debug artifacts
    2. Run OCR with TSV output
    3. Use parsers.py to group words, detect columns, and parse rows
    4. Deduplicate items using cleanup.py
    5. Reconcile totals
    6. Save final output to logs/{request_id}/last_response.json
    
    Args:
        pages: List of PIL Images or numpy arrays (one per page)
        request_id: Request identifier for organizing debug files (optional)
        
    Returns:
        Dictionary with HackRx-compatible structure:
        {
            "is_success": bool,
            "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
            "data": {
                "pagewise_line_items": [...],
                "total_item_count": int,
                "reconciled_amount": float
            },
            "error": str (only if is_success=False)
        }
    """
    try:
        if not pages:
            return {
                "is_success": False,
                "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
                "data": {
                    "pagewise_line_items": [],
                    "total_item_count": 0,
                    "reconciled_amount": 0.0
                },
                "error": "No pages provided"
            }
        
        # Convert numpy arrays to PIL Images if needed
        pil_images = []
        for page in pages:
            if isinstance(page, np.ndarray):
                if PIL_AVAILABLE:
                    pil_images.append(Image.fromarray(page))
                else:
                    raise RuntimeError("PIL/Pillow required for numpy array conversion")
            elif isinstance(page, Image.Image):
                pil_images.append(page)
            else:
                raise TypeError(f"Unsupported page type: {type(page)}")
        
        # Setup debug directory
        debug_dir = None
        if request_id:
            debug_dir = Path("logs") / str(request_id)
            debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Integration parameters
        NUMERIC_CONF_THRESHOLD = 60    # if token.conf < this, try numeric re-ocr
        QUANTITY_MAX_REASONABLE = 100  # prefer qty <= 100; >100 may be mis-assigned
        
        # Process each page using improved extraction
        all_bill_items = []
        pagewise_line_items = []
        all_deduped_items = []
        total_reconciled = 0.0
        
        for page_idx, img in enumerate(pil_images, start=1):
            try:
                run_log_dir = debug_dir if debug_dir else Path("logs") / (request_id or "default")
                run_log_dir.mkdir(parents=True, exist_ok=True)
                
                # 1) Preprocess (returns cv2 array)
                save_debug_path = str(run_log_dir / f"{request_id or 'page'}_p{page_idx}_pre.png") if request_id else None
                cv2_img = preprocess_image_for_ocr(
                    img,
                    return_cv2=True,
                    save_debug_path=save_debug_path
                )
                
                # Save preprocessed image (ensure it's saved)
                try:
                    if isinstance(cv2_img, np.ndarray):
                        Image.fromarray(cv2_img).save(run_log_dir / f"{request_id or 'page'}_p{page_idx}_pre.png")
                except Exception:
                    pass
                
                # 2) OCR -> TSV dict
                ocr = ocr_image_to_tsv(
                    cv2_img,
                    request_id=request_id,
                    page_no=page_idx,
                    save_debug_dir=str(run_log_dir)
                )
                
                # 3) Group to lines and detect columns
                lines = group_words_to_lines(ocr)
                col_centers = detect_column_centers(lines, max_columns=6)
                
                parsed_items = []
                
                # 4) Parse each line using improved parse_row_from_columns
                for ln in lines:
                    cols = map_tokens_to_columns(ln, col_centers)
                    parsed = parse_row_from_columns(cols)
                    
                    # if parsed has numeric tokens with low confidences, attempt numeric re-ocr
                    try:
                        # get token confidences from 'ln' tokens
                        right_numeric = None
                        for t in reversed(ln):
                            if any(ch.isdigit() for ch in t.get("text", "")):
                                right_numeric = t
                                break
                        
                        if right_numeric and (isinstance(right_numeric.get("conf"), (int, float)) and right_numeric.get("conf") < NUMERIC_CONF_THRESHOLD):
                            # crop area from preprocessed image
                            l = max(0, int(right_numeric["left"]) - 4)
                            t = max(0, int(right_numeric["top"]) - 4)
                            r = int(right_numeric["left"] + right_numeric["width"] + 4)
                            b = int(right_numeric["top"] + right_numeric["height"] + 4)
                            
                            try:
                                # cv2_img (grayscale) -> crop array then pass to ocr_numeric_region
                                crop = cv2_img[t:b, l:r]
                                val = ocr_numeric_region(crop)
                                if val is not None:
                                    # Overwrite item_amount if parse thinks it's numeric spot
                                    parsed['item_amount'] = val
                            except Exception:
                                pass
                    except Exception:
                        pass
                    
                    # 5) Apply is_probable_item to filter
                    if is_probable_item(parsed):
                        # Conservative quantity sanity: prefer qty <= QUANTITY_MAX_REASONABLE
                        qty = parsed.get("item_quantity")
                        if qty and isinstance(qty, (int, float)) and qty > QUANTITY_MAX_REASONABLE:
                            # if quantity seems unreasonably large, unset it
                            parsed["item_quantity"] = None
                        
                        parsed_items.append(parsed)
                
                # 6) Fallback: if no parsed_items found, run very simple rightmost-number heuristic
                if len(parsed_items) == 0:
                    # quick fallback: take each line and pick the rightmost numeric token as amount
                    for ln in lines:
                        tokens = [t['text'] for t in ln if t.get('text')]
                        if not tokens:
                            continue
                        
                        # search right to left for token with digit
                        amt = None
                        amt_token_idx = None
                        for i in range(len(tokens)-1, -1, -1):
                            s = tokens[i]
                            if any(ch.isdigit() for ch in s):
                                # clean numeric like 1,234.56 and decimals
                                s_clean = s.replace('₹', '').replace(',', '').replace('$', '')
                                s_clean = ''.join(ch for ch in s_clean if (ch.isdigit() or ch in ".-"))
                                try:
                                    amt = float(s_clean)
                                    amt_token_idx = i
                                    break
                                except:
                                    amt = None
                        
                        if amt is not None:
                            name = " ".join(tokens[:amt_token_idx]) if amt_token_idx is not None and amt_token_idx > 0 else " ".join(tokens)
                            parsed_items.append({
                                "item_name": name,
                                "item_amount": amt,
                                "item_rate": None,
                                "item_quantity": None
                            })
                
                # 7) Deduplicate & reconcile (per page)
                deduped = dedupe_items(parsed_items, name_threshold=88)
                
                # Try to extract a reported_total from pages: scan ocr text for 'Total|Net Amt|Grand Total' patterns
                reported_total = None
                ocr_text_joined = " ".join([tok for tok in ocr.get('text', []) if tok])
                m = re.search(r"(?:grand total|net amt|net amount|net total|total amount|total)\s*[:\s]*([₹\d,.\s]+)", ocr_text_joined, flags=re.I)
                if m:
                    s = m.group(1)
                    s = s.replace('₹', '').replace(',', '')
                    try:
                        reported_total = float("".join(ch for ch in s if (ch.isdigit() or ch in ".-")))
                    except:
                        reported_total = None
                
                final_total, method = reconcile_totals(deduped, reported_total)
                
                # prepare schema part for this page
                page_obj = {
                    "page_no": str(page_idx),
                    "page_type": "Bill Detail",
                    "bill_items": deduped,
                    "fraud_flags": [],
                    "reported_total": reported_total,
                    "reconciliation_ok": (abs(final_total - (reported_total or final_total)) / (reported_total or final_total) if reported_total and reported_total > 0 else None),
                    "reconciliation_relative_error": None if reported_total is None or reported_total == 0 else (final_total - reported_total) / reported_total
                }
                
                pagewise_line_items.append(page_obj)
                all_deduped_items.extend(deduped)
                total_reconciled += final_total
                
            except Exception as e:
                # Continue with empty page on error
                pagewise_line_items.append({
                    "page_no": str(page_idx),
                    "page_type": "Bill Detail",
                    "bill_items": [],
                    "fraud_flags": [],
                    "reported_total": None,
                    "reconciliation_ok": None,
                    "reconciliation_relative_error": None
                })
        
        # Final deduplication across all pages
        final_deduped = dedupe_items(all_deduped_items, name_threshold=88)
        
        # Final reconciliation (use total from all pages)
        final_total, method = reconcile_totals(final_deduped, None)
        
        # Update pagewise items with deduplicated items (simplified - in production, 
        # you might want to keep page-level items and dedupe only at final level)
        # For now, we'll keep pagewise structure but use deduped count
        
        # Build response
        response = {
            "is_success": True,
            "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
            "data": {
                "pagewise_line_items": pagewise_line_items,
                "total_item_count": len(final_deduped),
                "reconciled_amount": final_total
            }
        }
        
        # Save final output
        if debug_dir:
            output_path = debug_dir / "last_response.json"
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(response, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        
        return response
        
    except Exception as e:
        # Return error response (no 5xx, just is_success: false)
        error_response = {
            "is_success": False,
            "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
            "data": {
                "pagewise_line_items": [],
                "total_item_count": 0,
                "reconciled_amount": 0.0
            },
            "error": str(e)
        }
        
        # Save error response too
        if request_id:
            debug_dir = Path("logs") / str(request_id)
            debug_dir.mkdir(parents=True, exist_ok=True)
            output_path = debug_dir / "last_response.json"
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(error_response, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        
        return error_response
