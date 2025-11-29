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
from src.utils.text_utils import clean_amount, clean_item_name

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

# PaddleOCR import
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None

# OpenCV import for image conversion
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

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
            
            # High-quality preprocessing for better OCR accuracy
            preprocessed = preprocess_image_for_ocr(img, max_side=2000, target_dpi=300, fast_mode=False)
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
        
        # Validate pages are valid images
        if not PIL_AVAILABLE:
            return {
                "is_success": False,
                "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
                "data": {
                    "pagewise_line_items": [],
                    "total_item_count": 0,
                    "reconciled_amount": 0.0
                },
                "error": "PIL/Pillow not available"
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
        NUMERIC_CONF_THRESHOLD = 60    # if token.conf < this, try numeric re-ocr (as per requirements)
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
                
                # 1) Preprocess (returns cv2 array) - High quality mode for leaderboard
                # Enhanced for multilingual and handwritten bills (differentiator)
                save_debug_path = str(run_log_dir / f"{request_id or 'page'}_p{page_idx}_pre.png") if request_id else None
                cv2_img = preprocess_image_for_ocr(
                    img,
                    max_side=2000,  # Higher resolution for complex bills
                    target_dpi=300,
                    fast_mode=False,
                    return_cv2=True,
                    save_debug_path=save_debug_path,
                    enhance_for_multilingual=True,  # Differentiator: multilingual support
                    enhance_for_handwritten=True   # Differentiator: handwritten support
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
                # Pass cv2_img for projection-based fallback if needed
                col_centers = detect_column_centers(lines, max_columns=6, gray_img_array=cv2_img)
                
                parsed_items = []
                
                # 4) Parse each line using improved parse_row_from_columns
                # First, try to detect and merge split lines (heuristic rule)
                merged_lines = _merge_split_lines(lines)
                
                for ln in merged_lines:
                    cols = map_tokens_to_columns(ln, col_centers)
                    parsed = parse_row_from_columns(cols)
                    
                    # Heuristic: Detect structured rows like "CANNULA 22G 1 105.00 0.00 105.00"
                    if not parsed.get("item_amount") or parsed.get("item_amount") == 0:
                        structured_parsed = _parse_structured_row(ln)
                        if structured_parsed and structured_parsed.get("item_amount"):
                            parsed = structured_parsed
                    
                    # if parsed has numeric tokens with low confidences, attempt numeric re-ocr
                    try:
                        # get token confidences from 'ln' tokens
                        right_numeric = None
                        for t in reversed(ln):
                            if any(ch.isdigit() for ch in t.get("text", "")):
                                right_numeric = t
                                break
                        
                        if right_numeric and right_numeric.get("conf", 100) < NUMERIC_CONF_THRESHOLD:
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
                        # Conservative quantity validation (already handled in parse_row_from_columns)
                        # Additional check: ensure qty is valid integer < 100
                        qty = parsed.get("item_quantity")
                        if qty is not None:
                            if isinstance(qty, (int, float)):
                                if qty <= 0 or qty > 100:
                                    parsed["item_quantity"] = None
                                # If rightmost token includes ".00" treat as amount not qty
                                # (This is already checked in parse_row_from_columns, but double-check here)
                                if not float(qty).is_integer():
                                    parsed["item_quantity"] = None
                        
                        parsed_items.append(parsed)
                
                # 6) Fallback: if <3 parsed_items found, run very simple rightmost-number heuristic
                if len(parsed_items) < 3:
                    # quick fallback: take each line and pick the rightmost numeric token as amount
                    for ln in lines:
                        tokens = [t['text'] for t in ln if t.get('text')]
                        if not tokens:
                            continue
                        
                        # Skip if line looks like a header (contains common header keywords)
                        line_text = " ".join(tokens).lower()
                        header_keywords = ["bill no", "patient", "reg no", "ipd", "mobile", "age", "sex", 
                                          "address", "doctor", "category", "sno", "sl no", "particulars",
                                          "qty", "quantity", "rate", "amount", "total", "date"]
                        if any(kw in line_text for kw in header_keywords):
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
                                    # Only accept reasonable amounts (not invoice IDs or small numbers)
                                    if 1.0 <= amt <= 1000000:  # Between 1 and 1 million
                                        amt_token_idx = i
                                        break
                                except:
                                    amt = None
                        
                        if amt is not None and amt_token_idx is not None:
                            name = " ".join(tokens[:amt_token_idx]) if amt_token_idx > 0 else " ".join(tokens)
                            # Only add if name is substantial (not just a number or single character)
                            if name and len(name.strip()) > 2 and not name.strip().isdigit():
                                # Check if it's a probable item before adding
                                fallback_item = {
                                    "item_name": name.strip(),
                                    "item_amount": amt,
                                    "item_rate": None,
                                    "item_quantity": None
                                }
                                if is_probable_item(fallback_item):
                                    parsed_items.append(fallback_item)
                
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
                
                # Save parser diagnostic for this page
                if run_log_dir:
                    try:
                        parser_diagnostic = {
                            "page_no": page_idx,
                            "raw_lines_count": len(lines),
                            "column_centers": col_centers,
                            "parsed_items_before_dedup": parsed_items,
                            "deduped_items": deduped,
                            "reported_total": reported_total,
                            "final_total": final_total,
                            "reconciliation_method": method
                        }
                        diagnostic_path = run_log_dir / f"{request_id or 'page'}_p{page_idx}_parser_diagnostic.json"
                        with open(diagnostic_path, 'w', encoding='utf-8') as f:
                            json.dump(parser_diagnostic, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                
                # Auto-detect page_type
                page_type = _detect_page_type_from_text(ocr_text_joined)
                
                # Run fraud detection on this page (Differentiator)
                page_fraud_flags = []
                try:
                    from src.preprocessing.fraud_filters import detect_fraud_flags, compute_unified_fraud_score
                    flags, _ = detect_fraud_flags(cv2_img, use_fast_mode=True)
                    score = compute_unified_fraud_score(flags)
                    if score > 0.1:  # Only include significant fraud indicators
                        page_fraud_flags = [
                            {
                                "flag_type": flag.get("flag_type", "unknown"),
                                "score": float(flag.get("score", 0.0)),
                                "description": flag.get("description", "")
                            }
                            for flag in flags if flag.get("score", 0) > 0.1
                        ]
                except Exception:
                    pass  # Fraud detection is optional
                
                # prepare schema part for this page
                page_obj = {
                    "page_no": str(page_idx),
                    "page_type": page_type,
                    "bill_items": deduped,
                    "fraud_flags": page_fraud_flags,  # Differentiator: fraud detection
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
        
        # Final deduplication across all pages (prevent double counting)
        # Use stricter threshold to avoid merging different items
        final_deduped = dedupe_items(all_deduped_items, name_threshold=92)
        
        # CRITICAL: Final Total = sum of ALL individual line items (per problem statement)
        # Do NOT include sub-totals, taxes, or grand totals
        # Calculate from deduplicated items
        final_total = sum(item.get("item_amount", 0.0) for item in final_deduped if item.get("item_amount", 0.0) > 0)
        
        # Note: We don't use reconcile_totals here because:
        # - Problem statement says: Final Total = sum of all individual line items
        # - We should NOT adjust based on reported_total (that's for validation only)
        method = "sum_of_all_line_items"
        
        # Update pagewise items with deduplicated items (simplified - in production, 
        # you might want to keep page-level items and dedupe only at final level)
        # For now, we'll keep pagewise structure but use deduped count
        
        # Build response with differentiator metadata
        response = {
            "is_success": True,
            "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
            "data": {
                "pagewise_line_items": pagewise_line_items,
                "total_item_count": len(final_deduped),
                "reconciled_amount": round(final_total, 2)  # Final Total = sum of all line items
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


def extract_bill_data_paddleocr(
    pages: Union[List["Image.Image"], List[np.ndarray]],
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract bill data using PaddleOCR with line clustering and row parsing.
    
    This function uses PaddleOCR for better accuracy on Indian invoices:
    1. Initialize PaddleOCR with angle classification
    2. Run OCR on each page to get bounding boxes and text
    3. Group detected text boxes into rows based on Y-coordinates
    4. Parse each row to extract item name (left) and amount (right)
    5. Apply forensic regex cleaning to fix OCR number errors
    6. Return structured data matching HackRx schema
    
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
        if not PADDLEOCR_AVAILABLE:
            raise RuntimeError("PaddleOCR is not installed. Install with: pip install paddleocr")
        
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
        
        # Initialize PaddleOCR (singleton pattern for efficiency)
        ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
        # Setup debug directory
        debug_dir = None
        if request_id:
            debug_dir = Path("logs") / str(request_id)
            debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to PIL Images if needed, then to numpy for PaddleOCR
        pagewise_line_items = []
        all_bill_items = []
        
        for page_idx, page in enumerate(pages, start=1):
            try:
                # Convert to numpy array for PaddleOCR
                if isinstance(page, np.ndarray):
                    img_array = page.copy()
                    # Ensure BGR format for PaddleOCR
                    if len(img_array.shape) == 2:
                        # Grayscale to BGR
                        if CV2_AVAILABLE:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                        else:
                            img_array = np.stack([img_array] * 3, axis=-1)
                    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
                        # RGB to BGR
                        img_array = img_array[:, :, ::-1]
                elif isinstance(page, Image.Image):
                    img_array = np.array(page)
                    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                        # RGB to BGR
                        img_array = img_array[:, :, ::-1]
                    elif len(img_array.shape) == 2:
                        # Grayscale to BGR
                        if CV2_AVAILABLE:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                        else:
                            img_array = np.stack([img_array] * 3, axis=-1)
                else:
                    raise TypeError(f"Unsupported page type: {type(page)}")
                
                # Run PaddleOCR
                ocr_result = ocr_engine.ocr(img_array, cls=True)
                
                if not ocr_result or not ocr_result[0]:
                    # Empty page
                    pagewise_line_items.append({
                        "page_no": str(page_idx),
                        "bill_items": [],
                        "page_type": "Bill Detail",
                        "fraud_flags": [],
                        "reported_total": None,
                        "reconciliation_ok": None,
                        "reconciliation_relative_error": None
                    })
                    continue
                
                # Extract OCR data: each item is [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)]
                ocr_boxes = []
                for line in ocr_result[0]:
                    if line and len(line) >= 2:
                        box_coords = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        text_info = line[1]  # (text, confidence)
                        
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                            text = text_info[0]
                            confidence = text_info[1] if len(text_info) > 1 else 1.0
                        else:
                            text = str(text_info)
                            confidence = 1.0
                        
                        # Calculate bounding box properties
                        x_coords = [pt[0] for pt in box_coords]
                        y_coords = [pt[1] for pt in box_coords]
                        left = min(x_coords)
                        top = min(y_coords)
                        right = max(x_coords)
                        bottom = max(y_coords)
                        width = right - left
                        height = bottom - top
                        center_y = (top + bottom) / 2
                        
                        ocr_boxes.append({
                            "text": text.strip(),
                            "left": left,
                            "top": top,
                            "right": right,
                            "bottom": bottom,
                            "width": width,
                            "height": height,
                            "center_y": center_y,
                            "confidence": confidence
                        })
                
                if not ocr_boxes:
                    # No text detected
                    pagewise_line_items.append({
                        "page_no": str(page_idx),
                        "bill_items": [],
                        "page_type": "Bill Detail",
                        "fraud_flags": [],
                        "reported_total": None,
                        "reconciliation_ok": None,
                        "reconciliation_relative_error": None
                    })
                    continue
                
                # Step 2: Line Clustering - Group boxes into rows by Y-coordinates
                # Sort by center_y
                ocr_boxes.sort(key=lambda x: x["center_y"])
                
                # Cluster into rows with tolerance (default 10px)
                Y_TOLERANCE = 10
                rows = []
                current_row = [ocr_boxes[0]]
                
                for box in ocr_boxes[1:]:
                    # Check if box belongs to current row (within Y tolerance)
                    avg_y_current = sum(b["center_y"] for b in current_row) / len(current_row)
                    if abs(box["center_y"] - avg_y_current) <= Y_TOLERANCE:
                        current_row.append(box)
                    else:
                        # Start new row
                        rows.append(current_row)
                        current_row = [box]
                
                # Add last row
                if current_row:
                    rows.append(current_row)
                
                # Step 3: Row Parsing - Extract item name and amount from each row
                page_bill_items = []
                
                for row in rows:
                    # Sort boxes in row by x-coordinate (left to right)
                    row.sort(key=lambda x: x["left"])
                    
                    if not row:
                        continue
                    
                    # Detect if row looks like a bill item
                    # Usually has text on the left and a number on the far right
                    row_text = " ".join(box["text"] for box in row)
                    
                    # Skip obvious non-item rows (headers, totals, etc.)
                    row_text_lower = row_text.lower()
                    non_item_keywords = [
                        "subtotal", "total", "discount", "gst", "tax", "invoice",
                        "net amount", "amount due", "mrp", "balance", "paid",
                        "grand total", "net total", "total amount"
                    ]
                    if any(keyword in row_text_lower for keyword in non_item_keywords):
                        continue
                    
                    # Extract item name (left-most text) and amount (right-most numeric)
                    item_name_parts = []
                    amount_candidates = []
                    
                    for box in row:
                        text = box["text"]
                        # Check if text looks like a number/amount
                        # Remove currency symbols and check if it's numeric-like
                        text_clean = text.replace('₹', '').replace('Rs', '').replace('rs', '').replace(',', '').replace(' ', '')
                        # Check if it contains digits and looks like an amount
                        if re.search(r'\d', text_clean):
                            # Try to parse as amount
                            amount_value = clean_amount(text)
                            if amount_value > 0:
                                amount_candidates.append({
                                    "text": text,
                                    "value": amount_value,
                                    "left": box["left"],
                                    "right": box["right"]
                                })
                        else:
                            # Likely part of item name
                            item_name_parts.append(text)
                    
                    # If we found an amount, use the rightmost one
                    if amount_candidates:
                        # Sort by right position (rightmost first)
                        amount_candidates.sort(key=lambda x: x["right"], reverse=True)
                        amount_value = amount_candidates[0]["value"]
                        amount_text = amount_candidates[0]["text"]
                        
                        # Item name is everything except the amount
                        # Remove the amount text from item name parts
                        item_name_text = " ".join(item_name_parts)
                        # Clean item name
                        item_name = clean_item_name(item_name_text)
                        
                        # Only add if we have a valid item name and amount
                        if item_name and amount_value > 0:
                            page_bill_items.append({
                                "item_name": item_name,
                                "item_amount": round(amount_value, 2),
                                "item_rate": 0.0,  # Will be calculated if quantity available
                                "item_quantity": 1.0  # Default to 1
                            })
                    elif item_name_parts:
                        # No amount found, but we have text - might be item name only
                        # Only add if it looks substantial (not just a single word or number)
                        item_name_text = " ".join(item_name_parts)
                        item_name = clean_item_name(item_name_text)
                        if len(item_name.split()) >= 2:  # At least 2 words
                            page_bill_items.append({
                                "item_name": item_name,
                                "item_amount": 0.0,
                                "item_rate": 0.0,
                                "item_quantity": 1.0
                            })
                
                # Calculate page totals
                page_total = sum(item.get("item_amount", 0.0) for item in page_bill_items)
                
                # Try to extract reported total from OCR text
                all_text = " ".join(box["text"] for box in ocr_boxes)
                reported_total = None
                total_patterns = [
                    r"(?:grand total|net amt|net amount|net total|total amount|total)\s*[:\s]*([₹\d,.\s]+)",
                    r"total[:\s]+([₹\d,.\s]+)",
                ]
                for pattern in total_patterns:
                    m = re.search(pattern, all_text, flags=re.I)
                    if m:
                        total_text = m.group(1)
                        reported_total = clean_amount(total_text)
                        if reported_total > 0:
                            break
                
                # Add page to results
                pagewise_line_items.append({
                    "page_no": str(page_idx),
                    "bill_items": page_bill_items,
                    "page_type": "Bill Detail",
                    "fraud_flags": [],
                    "reported_total": reported_total,
                    "reconciliation_ok": None if reported_total is None else abs(page_total - reported_total) / reported_total < 0.05,
                    "reconciliation_relative_error": None if reported_total is None or reported_total == 0 else (page_total - reported_total) / reported_total
                })
                
                all_bill_items.extend(page_bill_items)
                
            except Exception as e:
                # Continue with empty page on error
                pagewise_line_items.append({
                    "page_no": str(page_idx),
                    "bill_items": [],
                    "page_type": "Bill Detail",
                    "fraud_flags": [],
                    "reported_total": None,
                    "reconciliation_ok": None,
                    "reconciliation_relative_error": None
                })
        
        # Calculate final totals
        total_item_count = len(all_bill_items)
        reconciled_amount = sum(item.get("item_amount", 0.0) for item in all_bill_items)
        
        # Build response
        response = {
            "is_success": True,
            "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
            "data": {
                "pagewise_line_items": pagewise_line_items,
                "total_item_count": total_item_count,
                "reconciled_amount": round(reconciled_amount, 2)
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
