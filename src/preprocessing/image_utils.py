"""
High-performance image preprocessing utilities with caching and fast downsampling.
"""

import numpy as np
from typing import Union, Tuple, Optional
import math
from functools import lru_cache

try:
    from PIL import Image, ImageOps, ImageFilter
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
    from skimage import exposure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    exposure = None


# Cache for expensive operations
_preprocessing_cache = {}
_cache_max_size = 10


def is_blank_page(img: Union["Image.Image", np.ndarray], threshold: float = 0.95) -> bool:
    """
    Fast blank page detection (public API).
    
    Args:
        img: PIL Image or numpy array
        threshold: Threshold for blank detection
        
    Returns:
        True if page appears blank
    """
    if isinstance(img, Image.Image):
        img_array = np.array(img.convert("L")) if img.mode != "L" else np.array(img)
    else:
        img_array = img
        if len(img_array.shape) == 3:
            if CV2_AVAILABLE:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            else:
                img_array = np.mean(img_array, axis=2).astype(np.uint8)
    return _is_blank_page_internal(img_array, threshold)


def _is_blank_page_internal(img_array: np.ndarray, threshold: float = 0.95) -> bool:
    """
    Fast blank page detection using Canny edges and contour analysis.
    
    Args:
        img_array: Grayscale image array
        threshold: Threshold for blank detection (default 0.95)
        
    Returns:
        True if page appears blank
    """
    if not CV2_AVAILABLE:
        return False
    
    try:
        # Fast edge detection
        edges = cv2.Canny(img_array, 50, 150)
        edge_ratio = float(edges.sum()) / (img_array.size * 255)
        
        # If very few edges, likely blank
        if edge_ratio < 0.01:
            return True
        
        # Check for text regions using contours
        _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small noise contours
        text_contours = [c for c in contours if cv2.contourArea(c) > 50]
        
        # If very few text regions, likely blank
        if len(text_contours) < 3:
            return True
        
        return False
    except Exception:
        return False


def _deskew_image(img_array: np.ndarray) -> np.ndarray:
    """Fast deskew detection and correction."""
    if not CV2_AVAILABLE:
        return img_array
    
    try:
        _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return img_array
        
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > 0:
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                if -45 <= angle <= 45:
                    angles.append(angle)
        
        if not angles or abs(np.median(angles)) < 0.5:
            return img_array
        
        median_angle = np.median(angles)
        h, w = img_array.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        img_array = cv2.warpAffine(img_array, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        pass
    
    return img_array


def preprocess_image_for_ocr(
    img: "Image.Image",
    max_side: int = 2000,  # Increased for better OCR on complex bills
    target_dpi: int = 300,
    fast_mode: bool = False,
    return_cv2: bool = False,
    save_debug_path: Optional[str] = None,
    enhance_for_multilingual: bool = True,  # New: enhance for multilingual/handwritten
    enhance_for_handwritten: bool = True    # New: enhance for handwritten text
) -> Union["Image.Image", np.ndarray]:
    """
    Optimized preprocessing pipeline with downsampling and caching.
    
    Optimizations:
    - Auto-downsample if DPI > 300
    - Fast mode skips expensive operations
    - Blank page early detection
    - Cached intermediate results
    
    Args:
        img: PIL Image to preprocess
        max_side: Maximum side length for resizing (default 1024)
        target_dpi: Target DPI (auto-downsamples if higher)
        fast_mode: Skip expensive operations (default False)
        return_cv2: If True, return numpy array (cv2 format) instead of PIL Image
        save_debug_path: Optional path to save debug preprocessed image
        
    Returns:
        Preprocessed PIL Image or numpy array (if return_cv2=True)
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL/Pillow is not installed")
    
    if not CV2_AVAILABLE:
        # Fallback to basic preprocessing
        if img.mode != "L":
            img = img.convert("L")
        img = ImageOps.autocontrast(img)
        return img
    
    # Step 1: Fast resize to max 1024px side
    w, h = img.size
    if max(w, h) > max_side:
        if w > h:
            new_w = max_side
            new_h = int(h * (max_side / w))
        else:
            new_h = max_side
            new_w = int(w * (max_side / h))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Step 2: Convert to grayscale
    if img.mode != "L":
        img = img.convert("L")
    
    # Convert to numpy
    img_array = np.array(img, dtype=np.uint8)
    
    # Step 3: Fast blank page detection (skip processing if blank)
    if is_blank_page(img):
        if return_cv2:
            return img_array
        return img  # Return as-is for blank pages
    
    # Step 3.5: Detect handwriting early to adjust preprocessing
    is_handwritten_detected = False
    if enhance_for_handwritten and not fast_mode:
        try:
            from src.preprocessing.handwriting_detector import detect_handwriting
            is_handwritten_detected, _ = detect_handwriting(img_array, threshold=0.3)
        except Exception:
            pass
    
    # Step 4: Fast denoising (only if not in fast mode)
    if not fast_mode:
        try:
            img_array = cv2.fastNlMeansDenoising(img_array, None, h=10, templateWindowSize=7, searchWindowSize=21)
        except Exception:
            pass
    
    # Step 5: Brightness normalization (fast)
    mean_brightness = np.mean(img_array)
    target_brightness = 128
    if abs(mean_brightness - target_brightness) > 30:
        adjustment = target_brightness - mean_brightness
        img_array = np.clip(img_array.astype(np.int16) + adjustment, 0, 255).astype(np.uint8)
    
    # Step 6: Deskew (skip in fast mode)
    if not fast_mode:
        img_array = _deskew_image(img_array)
    
    # Step 7: Adaptive preprocessing - try multiple methods
    # Skip standard thresholding if handwritten text is detected (will be handled in Step 11)
    if not fast_mode and not (enhance_for_handwritten and is_handwritten_detected):
        try:
            # Try OTSU threshold first (good for high contrast)
            _, binary_otsu = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Check if OTSU produced good results (not too dark/light)
            otsu_mean = np.mean(binary_otsu)
            if 50 < otsu_mean < 200:
                # OTSU looks good, use it
                img_array = binary_otsu
            else:
                # OTSU failed, try adaptive threshold
                try:
                    img_array = cv2.adaptiveThreshold(
                        img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY_INV, 11, 2
                    )
                except Exception:
                    # Fallback: use CLAHE
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    img_array = clahe.apply(img_array)
        except Exception:
            # Fallback: Fast CLAHE (OpenCV is faster than skimage)
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img_array = clahe.apply(img_array)
            except Exception:
                pass
    elif not fast_mode and enhance_for_handwritten and is_handwritten_detected:
        # For handwritten text, just apply CLAHE here, full processing in Step 11
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            img_array = clahe.apply(img_array)
        except Exception:
            pass
    else:
        # Fast mode: just CLAHE
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_array = clahe.apply(img_array)
        except Exception:
            pass
    
    # Step 8: Auto-binary for faded text (if image is still too light)
    # Skip for handwritten (will be handled in Step 11)
    if not fast_mode and not (enhance_for_handwritten and is_handwritten_detected):
        try:
            mean_brightness = np.mean(img_array)
            if mean_brightness > 200:  # Very light/faded text
                # Apply aggressive thresholding
                _, img_array = cv2.threshold(img_array, 240, 255, cv2.THRESH_BINARY_INV)
        except Exception:
            pass
    
    # Step 9: Morphological closing (skip in fast mode and for handwritten)
    if not fast_mode and not (enhance_for_handwritten and is_handwritten_detected):
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel, iterations=1)
        except Exception:
            pass
    
    # Step 10: Enhance for multilingual/handwritten (if enabled)
    if enhance_for_multilingual and not fast_mode and not is_handwritten_detected:
        try:
            # Sharpen image for better character recognition (helps with multilingual)
            kernel_sharpen = np.array([[-1, -1, -1],
                                      [-1,  9, -1],
                                      [-1, -1, -1]])
            img_array = cv2.filter2D(img_array, -1, kernel_sharpen)
            # Normalize to prevent overflow
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        except Exception:
            pass
    
    # Step 11: Enhance for handwritten text (if enabled and detected)
    if enhance_for_handwritten and not fast_mode and is_handwritten_detected:
        try:
            # Enhanced preprocessing pipeline for handwritten text
            # Handwritten text requires more aggressive preprocessing
            
            # 1. Advanced noise reduction (handwritten text has more noise)
            # Use bilateral filter to preserve edges while reducing noise
            img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
            
            # 2. Enhanced CLAHE for handwritten text (more aggressive)
            # Handwritten text often has variable contrast
            clahe_handwritten = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(10, 10))
            img_array = clahe_handwritten.apply(img_array)
            
            # 3. Morphological operations to enhance stroke connectivity
            # Handwritten strokes can be broken, so we connect them slightly
            kernel = np.ones((2, 2), np.uint8)
            img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # 4. Adaptive thresholding optimized for handwritten text
            # Handwritten text has variable thickness, so adaptive threshold works better
            img_array = cv2.adaptiveThreshold(
                img_array, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                15,  # Block size - larger for handwritten
                10   # C constant - adjusted for handwritten
            )
            
            # 5. Invert back to normal (black text on white)
            img_array = cv2.bitwise_not(img_array)
            
            # 6. Additional smoothing to reduce jagged edges in handwritten text
            img_array = cv2.medianBlur(img_array, 3)
            
        except Exception:
            # Fallback to basic CLAHE if advanced processing fails
            try:
                clahe_handwritten = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(10, 10))
                img_array = clahe_handwritten.apply(img_array)
            except Exception:
                pass
    
    # Step 12: Ensure minimum width for better OCR
    h, w = img_array.shape[:2]
    if w < 1200:
        # Upscale to minimum width (critical for accuracy)
        scale = 1200 / w
        new_w = 1200
        new_h = int(h * scale)
        img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Save debug image if requested
    if save_debug_path:
        try:
            from pathlib import Path
            debug_path = Path(save_debug_path)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            if CV2_AVAILABLE:
                cv2.imwrite(str(debug_path), img_array)
        except Exception:
            pass
    
    # Return cv2 array or PIL Image based on return_cv2 flag
    if return_cv2:
        return img_array
    return Image.fromarray(img_array)


def resize_image_if_needed(img: "Image.Image", max_dimension: int = 2000) -> "Image.Image":
    """Fast resize if needed."""
    if not PIL_AVAILABLE:
        return img
    
    w, h = img.size
    if max(w, h) <= max_dimension:
        return img
    
    if w > h:
        new_w = max_dimension
        new_h = int(h * (max_dimension / w))
    else:
        new_h = max_dimension
        new_w = int(w * (max_dimension / h))
    
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
