"""
Advanced document forensics fraud detection engine.
Implements whiteout detection, overwriting detection, digital tampering, font inconsistency, and geometry tampering.
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
    from PIL import Image, ImageChops, ImageFilter
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
            _ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except Exception:
            _ocr_engine = None
    return _ocr_engine


def _compute_ela_map(img: np.ndarray, quality: int = 85) -> np.ndarray:
    """
    Compute Error Level Analysis (ELA) map for tampering detection.
    
    Args:
        img: Input image as numpy array
        quality: JPEG compression quality for recompression
        
    Returns:
        Normalized ELA map (0-1 float array)
    """
    if not PIL_AVAILABLE:
        return np.zeros_like(img, dtype=np.float32)
    
    try:
        # Convert numpy to PIL
        if len(img.shape) == 2:
            pil_img = Image.fromarray(img, mode="L")
        else:
            pil_img = Image.fromarray(img)
        
        # Save and reload with compression
        buf = BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")
        
        # Convert original to RGB if needed
        if len(img.shape) == 2:
            orig = pil_img.convert("RGB")
        else:
            orig = Image.fromarray(img).convert("RGB")
        
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
        return np.zeros_like(img, dtype=np.float32)


def _detect_whiteout_regions(img_array: np.ndarray) -> Tuple[float, np.ndarray, List]:
    """
    Enhanced whiteout/whitener detection.
    
    Detects:
    - High brightness patches
    - Low texture regions
    - Contour gaps
    
    Returns:
        Tuple of (whiteout_score, whiteout_mask, large_contours)
    """
    if not CV2_AVAILABLE:
        return 0.0, np.zeros_like(img_array), []
    
    h, w = img_array.shape[:2]
    total_pixels = w * h
    
    # 1. High brightness detection
    white_threshold = 240
    bright_mask = (img_array >= white_threshold).astype(np.uint8) * 255
    bright_ratio = float(bright_mask.sum()) / (total_pixels * 255)
    
    # 2. Low texture detection (variance-based)
    kernel_size = 15
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    mean = cv2.filter2D(img_array.astype(np.float32), -1, kernel)
    variance = cv2.filter2D((img_array.astype(np.float32) - mean) ** 2, -1, kernel)
    low_texture_mask = (variance < 50).astype(np.uint8) * 255
    
    # Combine bright and low-texture regions
    combined_mask = cv2.bitwise_and(bright_mask, low_texture_mask)
    
    # 3. Find contours and detect gaps
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [c for c in contours if cv2.contourArea(c) > max(100, 0.002 * total_pixels)]
    
    # Calculate gap score (large white regions indicate whiteout)
    gap_score = 0.0
    if large_contours:
        total_contour_area = sum(cv2.contourArea(c) for c in large_contours)
        gap_score = total_contour_area / total_pixels
    
    # Combined whiteout score
    whiteout_score = min((bright_ratio * 0.4 + gap_score * 0.6) * 2.0, 1.0)
    
    return whiteout_score, combined_mask, large_contours


def _detect_overwriting(img_array: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Detect overwriting (sharp dark strokes over faded background).
    
    Detects:
    - Sharp dark strokes over faded background
    - Stroke-width inconsistency
    - Local contrast spikes
    
    Returns:
        Tuple of (overwriting_score, overwriting_mask)
    """
    if not CV2_AVAILABLE:
        return 0.0, np.zeros_like(img_array)
    
    h, w = img_array.shape[:2]
    
    # 1. Detect sharp dark strokes (high gradient magnitude in dark regions)
    grad_x = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Dark regions with high gradient indicate overwriting
    dark_mask = (img_array < 100).astype(np.uint8)
    high_gradient = (gradient_magnitude > np.percentile(gradient_magnitude, 90)).astype(np.uint8)
    overwriting_candidates = cv2.bitwise_and(dark_mask, high_gradient) * 255
    
    # 2. Stroke-width inconsistency detection
    # Use distance transform to estimate stroke width
    binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # Find regions with inconsistent stroke width
    stroke_width_mean = np.mean(dist_transform[dist_transform > 0]) if np.any(dist_transform > 0) else 0
    stroke_width_std = np.std(dist_transform[dist_transform > 0]) if np.any(dist_transform > 0) else 0
    
    # Regions with high std relative to mean indicate inconsistency
    if stroke_width_mean > 0:
        inconsistency_mask = (dist_transform > stroke_width_mean + 2 * stroke_width_std).astype(np.uint8) * 255
    else:
        inconsistency_mask = np.zeros_like(img_array, dtype=np.uint8)
    
    # 3. Local contrast spikes
    kernel = np.ones((5, 5), np.float32) / 25
    local_mean = cv2.filter2D(img_array.astype(np.float32), -1, kernel)
    local_contrast = np.abs(img_array.astype(np.float32) - local_mean)
    high_contrast = (local_contrast > np.percentile(local_contrast, 95)).astype(np.uint8) * 255
    
    # Combine all indicators
    combined_overwriting = cv2.bitwise_or(
        overwriting_candidates,
        cv2.bitwise_or(inconsistency_mask, high_contrast)
    )
    
    # Calculate score
    overwriting_ratio = float(combined_overwriting.sum()) / (w * h * 255)
    overwriting_score = min(overwriting_ratio * 10.0, 1.0)
    
    return overwriting_score, combined_overwriting


def _detect_jpeg_ghost(img_array: np.ndarray) -> float:
    """
    Detect JPEG ghost artifacts (double compression indicators).
    
    Returns:
        JPEG ghost score (0-1)
    """
    if not CV2_AVAILABLE:
        return 0.0
    
    try:
        # Test multiple compression qualities
        qualities = [95, 85, 75, 65]
        differences = []
        
        for q in qualities:
            # Compress and decompress
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
            result, encimg = cv2.imencode('.jpg', img_array, encode_param)
            if result:
                decimg = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
                if decimg is not None:
                    diff = np.abs(img_array.astype(float) - decimg.astype(float))
                    differences.append(np.mean(diff))
        
        if differences:
            # High variance in differences indicates ghost artifacts
            ghost_score = min(np.std(differences) / 50.0, 1.0)
            return ghost_score
    except Exception:
        pass
    
    return 0.0


def _detect_font_inconsistency(img_array: np.ndarray) -> float:
    """
    Detect font inconsistency by analyzing glyph shapes from OCR boxes.
    
    Returns:
        Font inconsistency score (0-1)
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
        
        # Extract bounding boxes and compute shape descriptors
        boxes = []
        for line in result[0]:
            if line and len(line) >= 1:
                bbox = line[0]
                if bbox and len(bbox) >= 4:
                    boxes.append(bbox)
        
        if len(boxes) < 5:  # Need at least 5 boxes for comparison
            return 0.0
        
        # Compute aspect ratios and areas for each box
        aspect_ratios = []
        areas = []
        
        for bbox in boxes:
            # Convert bbox to rectangle
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            
            if height > 0:
                aspect_ratios.append(width / height)
            if width > 0 and height > 0:
                areas.append(width * height)
        
        if len(aspect_ratios) < 3:
            return 0.0
        
        # High variance in aspect ratios or areas indicates font inconsistency
        aspect_std = np.std(aspect_ratios) if aspect_ratios else 0
        area_std = np.std(areas) if areas else 0
        aspect_mean = np.mean(aspect_ratios) if aspect_ratios else 1
        
        # Normalize by mean to get coefficient of variation
        if aspect_mean > 0:
            aspect_cv = aspect_std / aspect_mean
        else:
            aspect_cv = 0
        
        area_mean = np.mean(areas) if areas else 1
        if area_mean > 0:
            area_cv = area_std / area_mean
        else:
            area_cv = 0
        
        # High coefficient of variation indicates inconsistency
        inconsistency_score = min((aspect_cv + area_cv) / 2.0 * 2.0, 1.0)
        
        return inconsistency_score
    
    except Exception:
        return 0.0


def _detect_geometry_tampering(img_array: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Detect geometry tampering (warped text lines, unnatural skew).
    
    Returns:
        Tuple of (geometry_score, warping_mask)
    """
    if not CV2_AVAILABLE:
        return 0.0, np.zeros_like(img_array)
    
    h, w = img_array.shape[:2]
    
    # 1. Detect text lines using horizontal projections
    # Binarize image
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Horizontal projection
    horizontal_projection = np.sum(binary, axis=1)
    
    # Find text line regions (high projection values)
    threshold = np.mean(horizontal_projection) + np.std(horizontal_projection)
    text_line_mask = (horizontal_projection > threshold).astype(np.uint8)
    
    # 2. Detect warping by analyzing line straightness
    # Use Hough transform to detect lines
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 100, minLineLength=w // 10, maxLineGap=10)
    
    if lines is None or len(lines) == 0:
        return 0.0, np.zeros_like(img_array)
    
    # Calculate angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > 0:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if -45 <= angle <= 45:  # Horizontal-ish lines
                angles.append(angle)
    
    if len(angles) < 3:
        return 0.0, np.zeros_like(img_array)
    
    # High variance in angles indicates warping
    angle_std = np.std(angles)
    angle_mean = np.mean(angles)
    
    # 3. Detect local skew anomalies
    # Divide image into regions and check skew in each
    region_size = min(200, w // 4, h // 4)
    warping_regions = []
    
    for y in range(0, h, region_size):
        for x in range(0, w, region_size):
            region = binary[y:y+region_size, x:x+region_size]
            if region.size == 0:
                continue
            
            # Detect lines in region
            region_lines = cv2.HoughLinesP(region, 1, np.pi / 180, 50, minLineLength=region_size // 5, maxLineGap=5)
            if region_lines is not None and len(region_lines) > 0:
                region_angles = []
                for line in region_lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(x2 - x1) > 0:
                        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                        if -45 <= angle <= 45:
                            region_angles.append(angle)
                
                if region_angles:
                    region_angle_std = np.std(region_angles)
                    if region_angle_std > 5:  # High local variance
                        warping_regions.append((x, y, region_size, region_size))
    
    # Create warping mask
    warping_mask = np.zeros_like(img_array, dtype=np.uint8)
    for x, y, w_reg, h_reg in warping_regions:
        warping_mask[y:y+h_reg, x:x+w_reg] = 255
    
    # Calculate geometry tampering score
    warping_ratio = float(warping_mask.sum()) / (w * h * 255)
    angle_anomaly = min(angle_std / 10.0, 1.0) if angle_std > 0 else 0.0
    
    geometry_score = min((warping_ratio * 0.5 + angle_anomaly * 0.5) * 2.0, 1.0)
    
    return geometry_score, warping_mask


def detect_fraud_flags(
    img: Union["Image.Image", np.ndarray],
    save_debug_maps: bool = False,
    debug_output_dir: Optional[str] = None,
    use_fast_mode: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """
    Comprehensive fraud detection with unified scoring.
    
    Detects:
    - Whiteout/whitener regions
    - Overwriting
    - Digital tampering (ELA, JPEG ghost, recompression)
    - Font inconsistency
    - Geometry tampering
    
    Args:
        img: PIL Image or numpy array
        save_debug_maps: Whether to save visualization maps
        debug_output_dir: Directory to save debug maps (if save_debug_maps=True)
        use_fast_mode: Use low-res pre-scan for batch processing (default False)
        
    Returns:
        Tuple of (fraud_flags_list, debug_maps_dict)
        fraud_flags_list: List of dicts with keys: flag_type, score (0-1), meta
        debug_maps_dict: Dictionary of visualization maps (ELA, contours, masks, etc.)
    """
    flags = []
    debug_maps = {}
    
    if not CV2_AVAILABLE or not PIL_AVAILABLE:
        return flags, debug_maps
    
    try:
        # Convert to numpy array
        if isinstance(img, Image.Image):
            if img.mode != "L":
                img_gray = img.convert("L")
            else:
                img_gray = img
            img_array = np.array(img_gray)
        else:
            img_array = img
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # Fast mode: low-res pre-scan for batch processing
        original_array = img_array.copy()
        if use_fast_mode and img_array.shape[0] > 1000 or img_array.shape[1] > 1000:
            # Resize to max 800px for fast processing
            max_dim = 800
            h_orig, w_orig = img_array.shape[:2]
            if max(h_orig, w_orig) > max_dim:
                scale = max_dim / max(h_orig, w_orig)
                new_w = int(w_orig * scale)
                new_h = int(h_orig * scale)
                img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        h, w = img_array.shape[:2]
        total_pixels = w * h
        
        # 1. Whiteout detection
        whiteout_score, whiteout_mask, large_contours = _detect_whiteout_regions(img_array)
        if whiteout_score > 0.1:
            flags.append({
                "flag_type": "whiteout_regions",
                "score": whiteout_score,
                "meta": {
                    "n_large_regions": len(large_contours),
                    "whiteout_ratio": float(whiteout_mask.sum()) / (total_pixels * 255)
                }
            })
            debug_maps["whiteout_mask"] = whiteout_mask
        
        # 2. Overwriting detection
        overwriting_score, overwriting_mask = _detect_overwriting(img_array)
        if overwriting_score > 0.1:
            flags.append({
                "flag_type": "overwriting",
                "score": overwriting_score,
                "meta": {
                    "overwriting_ratio": float(overwriting_mask.sum()) / (total_pixels * 255)
                }
            })
            debug_maps["overwriting_mask"] = overwriting_mask
        
        # 3. Digital tampering - ELA
        ela_map = _compute_ela_map(img_array)
        ela_mean = float(np.mean(ela_map))
        ela_high_ratio = float((ela_map > 0.06).sum()) / total_pixels
        ela_std = float(np.std(ela_map))
        
        if ela_mean > 0.05 or ela_high_ratio > 0.015:
            flags.append({
                "flag_type": "ela_anomaly",
                "score": min(ela_mean * 10.0, 1.0),
                "meta": {
                    "ela_mean": ela_mean,
                    "ela_high_ratio": ela_high_ratio,
                    "ela_std": ela_std
                }
            })
            debug_maps["ela_heatmap"] = (ela_map * 255).astype(np.uint8)
        
        # 4. Digital tampering - JPEG ghost
        jpeg_ghost_score = _detect_jpeg_ghost(img_array)
        if jpeg_ghost_score > 0.2:
            flags.append({
                "flag_type": "jpeg_ghost",
                "score": jpeg_ghost_score,
                "meta": {"ghost_artifact_detected": True}
            })
        
        # 5. Compression inconsistency
        if ela_std > 0.08:
            flags.append({
                "flag_type": "compression_anomaly",
                "score": min(ela_std * 5.0, 1.0),
                "meta": {
                    "ela_std": ela_std,
                    "ela_mean": ela_mean
                }
            })
        
        # 6. Font inconsistency (skip in fast mode for speed)
        font_score = 0.0
        if not use_fast_mode:
            font_score = _detect_font_inconsistency(img_array)
            if font_score > 0.3:
                flags.append({
                    "flag_type": "font_inconsistency",
                    "score": font_score,
                    "meta": {"font_variation_detected": True}
                })
        
        # 7. Geometry tampering (skip in fast mode)
        geometry_score = 0.0
        warping_mask = np.zeros_like(img_array, dtype=np.uint8)
        if not use_fast_mode:
            geometry_score, warping_mask = _detect_geometry_tampering(img_array)
            if geometry_score > 0.2:
                flags.append({
                    "flag_type": "geometry_tampering",
                    "score": geometry_score,
                    "meta": {
                        "warping_ratio": float(warping_mask.sum()) / (total_pixels * 255)
                    }
                })
                debug_maps["warping_mask"] = warping_mask
        
        # 8. Contour gap detection (part of whiteout)
        if large_contours:
            # Draw contours for visualization
            contour_map = np.zeros_like(img_array, dtype=np.uint8)
            cv2.drawContours(contour_map, large_contours, -1, 255, 2)
            debug_maps["contour_map"] = contour_map
        
        # Save debug maps if requested
        if save_debug_maps and debug_output_dir:
            _save_debug_maps(debug_maps, debug_output_dir)
    
    except Exception as e:
        flags.append({
            "flag_type": "detection_error",
            "score": 0.0,
            "meta": {"error": str(e)}
        })
    
    return flags, debug_maps


def _save_debug_maps(debug_maps: Dict[str, np.ndarray], output_dir: str):
    """Save debug visualization maps to disk."""
    if not CV2_AVAILABLE:
        return
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for map_name, map_array in debug_maps.items():
            if map_array is not None and map_array.size > 0:
                file_path = output_path / f"{map_name}.png"
                cv2.imwrite(str(file_path), map_array)
    except Exception:
        pass


def compute_unified_fraud_score(flags: List[Dict[str, Any]]) -> float:
    """
    Compute unified fraud score using weighted sum.
    
    Weights:
    - whiteout_regions: 0.25
    - overwriting: 0.20
    - ela_anomaly: 0.20
    - jpeg_ghost: 0.15
    - font_inconsistency: 0.10
    - geometry_tampering: 0.10
    
    Args:
        flags: List of fraud flag dictionaries
        
    Returns:
        Unified fraud score (0-1)
    """
    weights = {
        "whiteout_regions": 0.25,
        "overwriting": 0.20,
        "ela_anomaly": 0.20,
        "jpeg_ghost": 0.15,
        "compression_anomaly": 0.10,
        "font_inconsistency": 0.10,
        "geometry_tampering": 0.10,
        "edge_anomaly": 0.05
    }
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for flag in flags:
        flag_type = flag.get("flag_type", "")
        score = flag.get("score", 0.0)
        weight = weights.get(flag_type, 0.05)
        
        weighted_sum += score * weight
        total_weight += weight
    
    if total_weight > 0:
        return min(weighted_sum / total_weight, 1.0)
    else:
        return 0.0
