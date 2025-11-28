"""
Preprocessing helpers for OCR and fraud detection.
Provides image preprocessing and whiteout/fraud detection capabilities.
"""

from typing import List, Tuple, Dict, Any
from PIL import Image, ImageOps, ImageFilter, ImageStat
import numpy as np
import cv2
import pytesseract


def preprocess_image_local(img: Image.Image) -> Image.Image:
    """
    Preprocess a single image for OCR with grayscale normalization,
    autocontrast, median filter, optional adaptive histogram equalization,
    and gaussian denoise.
    """
    # Convert to grayscale if needed
    if img.mode != "L":
        img = img.convert("L")
    
    # Apply autocontrast for better contrast
    img = ImageOps.autocontrast(img)
    
    # Apply median filter to reduce noise
    img = img.filter(ImageFilter.MedianFilter(size=3))
    
    # Convert to numpy array for advanced processing
    arr = np.array(img, dtype=np.float32)
    
    # Optional adaptive histogram equalization using skimage
    try:
        from skimage import exposure
        arr = exposure.equalize_adapthist(arr / 255.0, clip_limit=0.03)
        arr = (arr * 255).astype(np.uint8)
    except ImportError:
        # Fallback: use PIL's equalize
        arr = np.array(ImageOps.equalize(Image.fromarray(arr.astype(np.uint8))))
    
    # Convert back to PIL Image
    out = Image.fromarray(arr.astype(np.uint8))
    
    # Apply Gaussian blur for denoising
    out = out.filter(ImageFilter.GaussianBlur(radius=0.6))
    
    return out


def detect_whiteout_and_lowconf(images: List[Image.Image], ocr_pages: List[str]) -> List[Tuple[int, str, float]]:
    """
    Detect whiteout areas and low OCR confidence across multiple pages.
    
    Args:
        images: List of PIL Images (one per page)
        ocr_pages: List of OCR text strings (one per page)
        
    Returns:
        List of tuples: (page_no, flag_type, score)
    """
    flags = []
    
    for i, (img, text) in enumerate(zip(images, ocr_pages), start=1):
        w, h = img.size
        
        # Convert to numpy array for analysis
        if img.mode == "L":
            npimg = np.array(img)
        else:
            npimg = np.array(img.convert("L"))
        
        # Detect whiteout areas (very bright pixels)
        white_mask = (npimg >= 245)
        white_ratio = float(white_mask.sum()) / (w * h) if (w * h) > 0 else 0.0
        
        # Analyze OCR text
        text_len = len((text or "").strip())
        digit_ratio = sum(c.isdigit() for c in (text or "")) / max(1, text_len) if text_len > 0 else 0.0
        
        # Flag heavy whiteout (likely whiter/whiteout)
        if white_ratio > 0.55:
            flags.append((i, "whiteout_area", round(white_ratio, 3)))
        
        # Flag suspiciously low OCR output
        if text_len < 30:
            flags.append((i, "low_ocr_text", float(text_len) / 100.0))  # Normalize score
        
        # Flag digit-heavy low-length text (could be OCR garbage)
        if text_len > 0 and digit_ratio > 0.6 and text_len < 80:
            flags.append((i, "digit_heavy_lowlen", round(digit_ratio, 3)))
        
        # Additional check: use pytesseract confidence if available
        try:
            tsv = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            confs = [int(c) for c in tsv.get("conf", []) if c not in ("-1", "") and str(c).isdigit()]
            if confs:
                avg_conf = sum(confs) / len(confs)
                low_conf_pct = sum(1 for c in confs if c < 60) / len(confs)
                if low_conf_pct > 0.30:
                    flags.append((i, "low_ocr_confidence", round(low_conf_pct, 3)))
        except Exception:
            pass
    
    return flags


# Legacy function names for backward compatibility
def preprocess_full_pipeline(pil_img: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Legacy function that returns (preprocessed_image, diagnostics_dict).
    Uses preprocess_image_local internally.
    """
    prepped = preprocess_image_local(pil_img)
    w, h = pil_img.size
    diagnostics = {"width": w, "height": h}
    return prepped, diagnostics


def ela_map(pil_img: Image.Image, quality: int = 90, scale: int = 10) -> Image.Image:
    """
    Error Level Analysis (ELA) for detecting image tampering.
    """
    import io
    from PIL import ImageChops
    
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, "JPEG", quality=quality)
    buf.seek(0)
    reloaded = Image.open(buf).convert("RGB")
    ela = ImageChops.difference(pil_img.convert("RGB"), reloaded)
    
    def amplify(x):
        v = x * scale
        return 255 if v > 255 else int(v)
    
    ela = ela.point(amplify)
    return ela


def fix_whitener(pil_img: Image.Image) -> Tuple[Image.Image, np.ndarray]:
    """
    Detect and attempt to fix whiteout areas using inpainting.
    Returns (repaired_image, whiteout_mask).
    """
    arr = np.array(pil_img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    sat = hsv[:, :, 1]
    
    # Create mask for very bright, low-saturation areas (whiteout)
    mask = ((v > 220) & (sat < 20)).astype("uint8") * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    try:
        inpainted = cv2.inpaint(arr, mask, 5, cv2.INPAINT_TELEA)
        return Image.fromarray(inpainted), mask
    except Exception:
        return pil_img, mask
