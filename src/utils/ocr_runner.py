"""
High-performance parallel OCR processing using PaddleOCR with shared model loading.
"""

import os
import json
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PaddleOCR = None
    PADDLEOCR_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    from langdetect import detect, DetectorFactory
    LANGDETECT_AVAILABLE = True
    DetectorFactory.seed = 0
except ImportError:
    LANGDETECT_AVAILABLE = False
    detect = None

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    pytesseract = None


# Global OCR engine instances with thread lock for safe initialization
_ocr_engines: Dict[str, Optional[PaddleOCR]] = {}
_engine_lock = threading.Lock()


def _init_ocr_engine(
    lang: str = "en",
    use_angle_cls: bool = True,
    rec_algorithm: str = "SVTR_LCNet",
    det_db_unclip_ratio: float = 2.3,
    max_text_length: int = 200
) -> PaddleOCR:
    """
    Initialize PaddleOCR engine with thread-safe singleton pattern.
    Shared across all workers for memory efficiency.
    """
    global _ocr_engines
    
    # Use multi_lang if language is not English
    if lang != "en" and lang != "multi":
        lang_key = "multi"
    else:
        lang_key = lang
    
    # Thread-safe check and initialization
    with _engine_lock:
        # Double-check after acquiring lock
        if lang_key in _ocr_engines and _ocr_engines[lang_key] is not None:
            return _ocr_engines[lang_key]
        
        if not PADDLEOCR_AVAILABLE:
            raise RuntimeError("PaddleOCR is not installed. Install with: pip install paddleocr")
        
        # Initialize with enhanced configuration
        engine = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang_key,
            show_log=False,
            rec_algorithm=rec_algorithm,
            det_db_unclip_ratio=det_db_unclip_ratio,
            max_text_length=max_text_length
        )
        
        _ocr_engines[lang_key] = engine
        return engine


def _detect_language(text: str) -> str:
    """Fast language detection (cached)."""
    if not LANGDETECT_AVAILABLE or not text or len(text.strip()) < 10:
        return "en"
    
    try:
        detected = detect(text)
        lang_map = {
            "en": "en", "hi": "hi", "mr": "mr", "ta": "ta", "te": "te",
            "kn": "kn", "gu": "gu", "pa": "pa", "bn": "bn", "ur": "ur",
        }
        return lang_map.get(detected, "en")
    except Exception:
        return "en"


def _ocr_single_page(
    args: Tuple[int, "Image.Image", Optional[str], bool, str, float, int]
) -> Tuple[int, str, List[Dict[str, Any]], str]:
    """
    Optimized single page OCR processing.
    Uses pre-loaded shared engine for speed.
    """
    page_idx, img, lang, use_angle_cls, rec_algorithm, det_db_unclip_ratio, max_text_length = args
    
    try:
        # Fast image conversion
        if isinstance(img, Image.Image):
            img_array = np.array(img)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = img_array[:, :, ::-1]  # RGB to BGR
            elif len(img_array.shape) == 2 and CV2_AVAILABLE:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        else:
            img_array = img
        
        # Skip language detection in fast mode (use provided lang)
        detected_lang = lang if lang and lang != "auto" else "en"
        
        # Get shared engine (thread-safe)
        engine = _init_ocr_engine(detected_lang, use_angle_cls, rec_algorithm, det_db_unclip_ratio, max_text_length)
        
        # Run OCR (fast inference)
        result = engine.ocr(img_array, cls=use_angle_cls)
        
        # Fast text extraction
        text_lines = []
        ocr_details = []
        
        if result and len(result) > 0:
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    confidence = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 1.0
                    text_lines.append(text)
                    ocr_details.append({
                        "text": text,
                        "confidence": float(confidence),
                        "bbox": line[0] if line[0] else []
                    })
        
        ocr_text = "\n".join(text_lines)
        return (page_idx, ocr_text, ocr_details, detected_lang)
    
    except Exception:
        return (page_idx, "", [], "en")


def run_ocr_parallel(
    images: List["Image.Image"],
    lang: Optional[str] = "en",
    use_angle_cls: bool = True,
    max_workers: Optional[int] = None,  # Auto-detect CPU count if None
    rec_algorithm: str = "SVTR_LCNet",
    det_db_unclip_ratio: float = 2.3,
    max_text_length: int = 200,
    auto_detect_lang: bool = False  # Disabled by default for speed
) -> List[Tuple[int, str, List[Dict[str, Any]], str]]:
    """
    High-performance parallel OCR with shared model loading.
    
    Optimizations:
    - Shared PaddleOCR model across workers (memory efficient)
    - Thread-safe initialization
    - Parallel processing with ThreadPoolExecutor
    - Fast image conversion
    
    Args:
        images: List of PIL Images to process
        lang: Language code (default "en")
        use_angle_cls: Whether to use angle classification
        max_workers: Number of parallel workers (default 6)
        rec_algorithm: Recognition algorithm (default "SVTR_LCNet")
        det_db_unclip_ratio: Detection unclip ratio (default 2.3)
        max_text_length: Maximum text length (default 200)
        auto_detect_lang: Auto-detect language (default False for speed)
        
    Returns:
        List of tuples: (page_index, ocr_text, ocr_details, detected_lang)
    """
    if not PADDLEOCR_AVAILABLE:
        raise RuntimeError("PaddleOCR is not installed. Install with: pip install paddleocr")
    
    if not images:
        return []
    
    # Auto-detect max_workers based on CPU count if not specified
    if max_workers is None:
        import multiprocessing
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Pre-initialize engine in main thread (warm-up)
    _init_ocr_engine(lang or "en", use_angle_cls, rec_algorithm, det_db_unclip_ratio, max_text_length)
    
    # Prepare arguments
    args_list = [
        (idx, img, lang, use_angle_cls, rec_algorithm, det_db_unclip_ratio, max_text_length)
        for idx, img in enumerate(images)
    ]
    
    # Run OCR in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_page = {
            executor.submit(_ocr_single_page, args): args[0]
            for args in args_list
        }
        
        for future in as_completed(future_to_page):
            try:
                result = future.result()
                results.append(result)
            except Exception:
                page_idx = future_to_page[future]
                results.append((page_idx, "", [], "en"))
    
    # Sort by page index
    results.sort(key=lambda x: x[0])
    
    return results


def extract_text_from_ocr_result(ocr_result: List[Tuple[int, str, List[Dict[str, Any]], str]]) -> List[str]:
    """Extract just the text strings from OCR results."""
    return [text for _, text, _, _ in ocr_result]


def ocr_image_to_tsv(cv2_img, request_id=None, page_no=None, save_debug_dir=None) -> Dict[str, Any]:
    """
    Run pytesseract.image_to_data and save debug outputs (JSON and TSV).
    
    Args:
        cv2_img: OpenCV image (numpy array) or PIL Image
        request_id: Request identifier for organizing debug files (optional)
        page_no: Page number for filename (optional)
        save_debug_dir: Base directory for saving debug files (default: 'logs')
    
    Returns:
        Dictionary from pytesseract.image_to_data(..., output_type=Output.DICT)
    
    Example:
        >>> import cv2
        >>> img = cv2.imread('page.png')
        >>> ocr_dict = ocr_image_to_tsv(img, request_id='req123', page_no=1)
        >>> # Saves logs/req123/req123_p1_ocr.json and .tsv
    """
    if not PYTESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract is not installed. Install with: pip install pytesseract")
    
    # Convert cv2 image to PIL if needed
    if CV2_AVAILABLE and isinstance(cv2_img, np.ndarray):
        # OpenCV image (BGR) -> PIL Image (RGB)
        if len(cv2_img.shape) == 3:
            img_pil = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
        else:
            img_pil = Image.fromarray(cv2_img)
    elif isinstance(cv2_img, Image.Image):
        img_pil = cv2_img
    else:
        raise ValueError(f"Unsupported image type: {type(cv2_img)}")
    
    # Run pytesseract OCR
    ocr_dict = pytesseract.image_to_data(img_pil, output_type=pytesseract.Output.DICT)
    
    # Save debug files if request_id and page_no provided
    if request_id is not None and page_no is not None:
        if save_debug_dir is None:
            save_debug_dir = "logs"
        
        debug_dir = Path(save_debug_dir) / str(request_id)
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = debug_dir / f"{request_id}_p{page_no}_ocr.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(ocr_dict, f, ensure_ascii=False, indent=2)
        
        # Save TSV (tab-separated values)
        tsv_path = debug_dir / f"{request_id}_p{page_no}_ocr.tsv"
        with open(tsv_path, 'w', encoding='utf-8') as f:
            # Write header
            keys = list(ocr_dict.keys())
            f.write('\t'.join(keys) + '\n')
            
            # Write data rows
            n = len(ocr_dict.get('text', []))
            for i in range(n):
                row = [str(ocr_dict.get(k, [''])[i] if isinstance(ocr_dict.get(k), list) else ocr_dict.get(k, '')) for k in keys]
                f.write('\t'.join(row) + '\n')
    
    return ocr_dict


def ocr_numeric_region(region_img) -> Optional[float]:
    """
    OCR a numeric region using specialized PSM mode and whitelist.
    
    Uses '--psm 7' (single text line) with whitelist '0123456789.,₹'
    to improve accuracy for numeric values. Returns cleaned float.
    
    Args:
        region_img: OpenCV image (numpy array) or PIL Image containing numeric text
    
    Returns:
        Float value if successful, None otherwise
    
    Example:
        >>> import cv2
        >>> region = cv2.imread('amount_region.png')
        >>> amount = ocr_numeric_region(region)
        >>> print(f"Extracted amount: {amount}")
    """
    if not PYTESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract is not installed. Install with: pip install pytesseract")
    
    # Convert cv2 image to PIL if needed
    if CV2_AVAILABLE and isinstance(region_img, np.ndarray):
        if len(region_img.shape) == 3:
            img_pil = Image.fromarray(cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB))
        else:
            img_pil = Image.fromarray(region_img)
    elif isinstance(region_img, Image.Image):
        img_pil = region_img
    else:
        raise ValueError(f"Unsupported image type: {type(region_img)}")
    
    # Run OCR with numeric-specific config
    config = '--psm 7 -c tessedit_char_whitelist=0123456789.,₹'
    text = pytesseract.image_to_string(img_pil, config=config).strip()
    
    if not text:
        return None
    
    # Clean and convert to float
    # Remove currency symbols and commas
    cleaned = text.replace('₹', '').replace('Rs', '').replace(',', '').strip()
    
    # Remove any remaining non-numeric characters except decimal point
    cleaned = ''.join(c for c in cleaned if c.isdigit() or c == '.')
    
    if not cleaned or cleaned == '.':
        return None
    
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None
