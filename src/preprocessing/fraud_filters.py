"""
Advanced document forensics fraud detection engine.
Refined for Bajaj Finserv Datathon to handle specific fraud types:
1. Whiteout/Whitener (Texture analysis with digital-file safeguards)
2. Overwriting (Ink saturation + Gradient analysis)
3. Font Inconsistency (Line height clustering)
4. Digital Tampering (ELA & Metadata)
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Union, Optional
from io import BytesIO
import math

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    from PIL import Image, ImageChops
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None

# Global OCR engine for font detection (lazy loaded)
_ocr_engine = None

def _get_ocr_engine():
    """Lazy load OCR engine for font detection."""
    global _ocr_engine
    if _ocr_engine is None and PADDLEOCR_AVAILABLE:
        try:
            # optimize for structure/box detection rather than pure text accuracy
            _ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except Exception:
            _ocr_engine = None
    return _ocr_engine

def _compute_ela_map(img: np.ndarray, quality: int = 85) -> np.ndarray:
    """Compute Error Level Analysis (ELA) map for tampering detection."""
    if not PIL_AVAILABLE:
        return np.zeros_like(img, dtype=np.float32)
    
    try:
        # Ensure we are working with RGB for ELA
        if len(img.shape) == 2:
            pil_img = Image.fromarray(img, mode="L").convert("RGB")
        else:
            # Convert BGR (OpenCV) to RGB (PIL)
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Save and reload with compression
        buf = BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")
        
        orig = pil_img
        
        # Compute difference
        diff = ImageChops.difference(orig, recompressed)
        diff_gray = diff.convert("L")
        diff_arr = np.array(diff_gray, dtype=np.float32)
        
        # Normalize to 0-1
        if diff_arr.max() > 0:
            return diff_arr / diff_arr.max()
        else:
            return diff_arr / 255.0
            
    except Exception:
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

def _detect_whiteout_regions(img_gray: np.ndarray) -> Tuple[float, np.ndarray, List]:
    """
    Refined Whitener Detection.
    Check: High Brightness + Low Texture.
    SAFEGUARD: Checks global background noise to ignore digital PDFs.
    """
    if not CV2_AVAILABLE:
        return 0.0, np.zeros_like(img_gray), []
    
    h, w = img_gray.shape[:2]
    total_pixels = w * h

    # SAFEGUARD: Detect "Digital PDF" vs "Scanned Paper"
    # Scanned paper has noise (std_dev > 5). Digital backgrounds are flat (std_dev < 2).
    bg_mean, bg_std = cv2.meanStdDev(img_gray)
    is_digital_file = bg_std < 5.0

    if is_digital_file:
        # If it's a digital file, "whitener" is impossible/irrelevant. 
        # Return 0 to prevent False Positives.
        return 0.0, np.zeros_like(img_gray), []

    # 1. High brightness detection (Whitener is usually brighter than the paper)
    # Adaptive thresholding based on background mean
    white_thresh = max(bg_mean[0][0] + 20, 240) # Ensure it's very bright
    bright_mask = (img_gray >= white_thresh).astype(np.uint8) * 255
    
    # 2. Low texture detection (Whitener creates a smooth patch on noisy paper)
    # Calculate local variance
    kernel_size = 15
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    mean = cv2.filter2D(img_gray.astype(np.float32), -1, kernel)
    sq_mean = cv2.filter2D(img_gray.astype(np.float32)**2, -1, kernel)
    variance = sq_mean - mean**2
    
    # Whitener patches are smooth (variance < 50 typically)
    low_texture_mask = (variance < 50).astype(np.uint8) * 255
    
    # Combine: Must be Bright AND Smooth
    combined_mask = cv2.bitwise_and(bright_mask, low_texture_mask)
    
    # 3. Geometric Filtering: Whitener marks are blobs, not lines
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = []
    
    valid_mask = np.zeros_like(img_gray)
    
    for c in contours:
        area = cv2.contourArea(c)
        # Filter noise (too small) and huge page borders (too big)
        if 100 < area < (total_pixels * 0.10): 
            # Check aspect ratio (whitener isn't usually a long thin line)
            x,y,cw,ch = cv2.boundingRect(c)
            aspect = float(cw)/ch
            if 0.2 < aspect < 5.0:
                large_contours.append(c)
                cv2.drawContours(valid_mask, [c], -1, 255, -1)

    # Calculate score based on area of suspected patches
    whitener_area = np.sum(valid_mask > 0)
    whitener_ratio = whitener_area / total_pixels
    
    # Heuristic score
    score = min(whitener_ratio * 100.0, 1.0) # 1% of page covered = 100% suspicion
    
    return score, valid_mask, large_contours

def _detect_overwriting(img_color: Optional[np.ndarray], img_gray: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Refined Overwriting Detection.
    1. Digital Edits: Sharp gradient detection.
    2. Physical Edits (Pen): HSV Saturation check (Ink vs Toner).
    """
    if not CV2_AVAILABLE:
        return 0.0, np.zeros_like(img_gray)
    
    h, w = img_gray.shape[:2]
    combined_mask = np.zeros_like(img_gray)

    # --- Method A: Gradient Magnitude (For Digital/Copy-Paste Edits) ---
    # Look for unnatural sharpness in dark regions
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Thresholds
    dark_mask = (img_gray < 150).astype(np.uint8) # Only look at text/ink
    # Top 5% sharpest edges
    high_grad_thresh = np.percentile(gradient_magnitude, 95) 
    sharp_edges = (gradient_magnitude > high_grad_thresh).astype(np.uint8)
    
    digital_edit_mask = cv2.bitwise_and(dark_mask, sharp_edges) * 255

    # --- Method B: Ink Saturation (For Pen Overwriting) ---
    ink_score = 0.0
    ink_mask = np.zeros_like(img_gray)
    
    if img_color is not None and len(img_color.shape) == 3:
        # Convert to HSV to separate Color from Intensity
        hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1] # Saturation
        
        # Black Toner/Carbon has near 0 saturation.
        # Blue/Black Pens have higher saturation (often > 20).
        # We detect "Colored" strokes on a "Grayscale" document.
        
        # Threshold for "Ink"
        _, potential_ink = cv2.threshold(s_channel, 25, 255, cv2.THRESH_BINARY)
        
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        ink_mask = cv2.morphologyEx(potential_ink, cv2.MORPH_OPEN, kernel)
        
        # Calculate ink score
        ink_pixels = cv2.countNonZero(ink_mask)
        if ink_pixels > 50: # Minimum threshold to ignore scanner noise
            ink_score = min(ink_pixels / (w * h * 0.005), 1.0)

    # Combine masks
    combined_mask = cv2.bitwise_or(digital_edit_mask, ink_mask)
    
    # Final Score: Max of either method
    grad_score = (np.sum(digital_edit_mask > 0) / (w * h)) * 100.0
    final_score = max(min(grad_score, 1.0), ink_score)
    
    return final_score, combined_mask

def _detect_font_inconsistency(img_array: np.ndarray) -> float:
    """
    Refined Font Inconsistency.
    Uses 'Line Height Consistency' instead of Aspect Ratio.
    Detects inserted lines that are slightly larger/smaller than the document standard.
    """
    if not CV2_AVAILABLE or not PADDLEOCR_AVAILABLE:
        return 0.0
    
    try:
        ocr_engine = _get_ocr_engine()
        if ocr_engine is None:
            return 0.0
        
        # Run OCR to get bounding boxes
        result = ocr_engine.ocr(img_array, cls=False)
        
        if not result or len(result) == 0 or len(result[0]) == 0:
            return 0.0
        
        # Extract heights of every text box
        box_heights = []
        for line in result[0]:
            # line structure: [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], (text, conf)]
            box = line[0] 
            if len(box) >= 4:
                # Calculate height: Average of left-height and right-height
                h_left = abs(box[3][1] - box[0][1])
                h_right = abs(box[2][1] - box[1][1])
                avg_h = (h_left + h_right) / 2.0
                if avg_h > 5: # Ignore tiny noise
                    box_heights.append(avg_h)
        
        if len(box_heights) < 10: 
            return 0.0 # Not enough text to judge consistency
            
        box_heights = np.array(box_heights)
        
        # Determine the "Dominant" font size (Median is robust to outliers)
        median_h = np.median(box_heights)
        
        # Define tolerance: 15% variation is normal scanning noise. 
        # Anything beyond 20% diff is suspicious.
        lower_bound = median_h * 0.80
        upper_bound = median_h * 1.20
        
        # Find outliers
        outliers = box_heights[(box_heights < lower_bound) | (box_heights > upper_bound)]
        
        # Scoring Logic:
        # If 100% are outliers? No, that just means variable font sizes (Header vs Body).
        # We look for a *minority* of text that is inconsistent (Inserts).
        # E.g., if 10-30% of text is a different size, it's suspicious.
        
        total_boxes = len(box_heights)
        outlier_count = len(outliers)
        outlier_ratio = outlier_count / total_boxes
        
        # Penalize if outliers exist but aren't dominant (0.05 < ratio < 0.40)
        if 0.05 < outlier_ratio < 0.40:
            return min(outlier_ratio * 3.0, 1.0)
            
        return 0.0
    
    except Exception:
        return 0.0

def _detect_jpeg_ghost(img_array: np.ndarray) -> float:
    """Detect JPEG ghost artifacts (double compression)."""
    if not CV2_AVAILABLE:
        return 0.0
    try:
        qualities = [95, 85, 75, 65]
        differences = []
        for q in qualities:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
            result, encimg = cv2.imencode('.jpg', img_array, encode_param)
            if result:
                decimg = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
                if decimg is not None:
                    diff = np.abs(img_array.astype(float) - decimg.astype(float))
                    differences.append(np.mean(diff))
        
        if differences:
            # High variance in re-compression error indicates mismatched quantization
            return min(np.std(differences) / 50.0, 1.0)
    except Exception:
        pass
    return 0.0

def detect_fraud_flags(
    img: Union["Image.Image", np.ndarray],
    save_debug_maps: bool = False,
    debug_output_dir: Optional[str] = None,
    use_fast_mode: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """
    Main Entry Point.
    Orchestrates all checks and returns a unified report.
    """
    flags = []
    debug_maps = {}
    
    if not CV2_AVAILABLE or not PIL_AVAILABLE:
        return flags, debug_maps
    
    try:
        # Preprocessing: Ensure we have both Color (for Ink) and Grayscale (for Texture)
        img_color = None
        img_gray = None
        
        if isinstance(img, Image.Image):
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_color = np.array(img)
            img_color = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR) # PIL is RGB, OpenCV is BGR
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        elif isinstance(img, np.ndarray):
            if len(img.shape) == 3:
                img_color = img
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img
                img_color = None # No color data available
        
        # Resize for performance if needed
        if use_fast_mode:
            h, w = img_gray.shape[:2]
            if max(h, w) > 1000:
                scale = 1000 / max(h, w)
                img_gray = cv2.resize(img_gray, None, fx=scale, fy=scale)
                if img_color is not None:
                    img_color = cv2.resize(img_color, None, fx=scale, fy=scale)

        # 1. Whitener Detection
        whiteout_score, whiteout_mask, _ = _detect_whiteout_regions(img_gray)
        if whiteout_score > 0.15:
            flags.append({
                "flag_type": "whiteout_regions",
                "score": whiteout_score,
                "meta": {"detection_method": "texture_variance"}
            })
            debug_maps["whiteout_mask"] = whiteout_mask

        # 2. Overwriting Detection (Ink + Digital)
        overwriting_score, overwriting_mask = _detect_overwriting(img_color, img_gray)
        if overwriting_score > 0.15:
            flags.append({
                "flag_type": "overwriting",
                "score": overwriting_score,
                "meta": {"detection_method": "ink_saturation_and_gradient"}
            })
            debug_maps["overwriting_mask"] = overwriting_mask

        # 3. ELA (Digital Tampering)
        if img_color is not None:
            ela_map = _compute_ela_map(img_color)
            ela_score = float(np.mean(ela_map)) * 10.0 # Scale up
            if ela_score > 0.2:
                flags.append({
                    "flag_type": "ela_anomaly",
                    "score": min(ela_score, 1.0),
                    "meta": {"method": "error_level_analysis"}
                })
                debug_maps["ela_heatmap"] = (ela_map * 255).astype(np.uint8)

        # 4. Font Inconsistency (Skip in fast mode)
        if not use_fast_mode:
            font_score = _detect_font_inconsistency(img_color if img_color is not None else img_gray)
            if font_score > 0.2:
                flags.append({
                    "flag_type": "font_inconsistency",
                    "score": font_score,
                    "meta": {"method": "line_height_clustering"}
                })

        # 5. JPEG Ghost
        ghost_score = _detect_jpeg_ghost(img_gray)
        if ghost_score > 0.3:
            flags.append({
                "flag_type": "jpeg_ghost",
                "score": ghost_score,
                "meta": {"method": "double_compression_artifacts"}
            })

    except Exception as e:
        flags.append({
            "flag_type": "error",
            "score": 0.0,
            "meta": {"error_msg": str(e)}
        })

    return flags, debug_maps

def compute_unified_fraud_score(flags: List[Dict[str, Any]]) -> float:
    """Weighted sum of all fraud indicators."""
    if not flags:
        return 0.0
        
    weights = {
        "whiteout_regions": 0.30,   # High priority (Visual tampering)
        "overwriting": 0.25,        # High priority (Fraud)
        "font_inconsistency": 0.20, # Medium priority (Inserts)
        "ela_anomaly": 0.15,        # Low priority (Digital only)
        "jpeg_ghost": 0.10          # Low priority (Hard to prove)
    }
    
    total_score = 0.0
    total_weight = 0.0
    
    for f in flags:
        w = weights.get(f["flag_type"], 0.05)
        total_score += f["score"] * w
        total_weight += w
        
    if total_weight == 0: return 0.0
    return min(total_score / total_weight, 1.0)