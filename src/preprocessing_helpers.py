"""
Enhanced fraud detection with whiteout detection, ELA analysis, and structured flagging.
Upgraded with Gaussian + morphological analysis and robust scoring system.
"""

from typing import List, Tuple, Any, Dict, Optional
from PIL import Image, ImageChops, ImageFilter, ImageOps
import numpy as np
import io
import os
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False
    cv2 = None


def _compute_ela_map(pil_img: Image.Image, quality: int = 85) -> np.ndarray:
    """
    Compute Error Level Analysis (ELA) map for tampering detection.
    
    Args:
        pil_img: PIL Image
        quality: JPEG compression quality for recompression
        
    Returns:
        Normalized ELA map (0-1 float array)
    """
    try:
        # Save and reload with compression
        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")
        
        # Compute difference
        orig = pil_img.convert("RGB")
        diff = ImageChops.difference(orig, recompressed)
        diff_gray = diff.convert("L")
        diff_arr = np.array(diff_gray, dtype=np.float32)
        
        # Normalize to 0-1
        if diff_arr.max() > 0:
            return diff_arr / diff_arr.max()
        else:
            return diff_arr / 255.0
    except Exception:
        return np.zeros((pil_img.height, pil_img.width), dtype=np.float32)


def _gaussian_whiteout_analysis(img_array: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Enhanced whiteout detection using Gaussian filtering and morphological analysis.
    
    Args:
        img_array: Grayscale image array
        
    Returns:
        Tuple of (whiteout_score, whiteout_mask)
    """
    if not CV2_AVAILABLE:
        return 0.0, np.zeros_like(img_array, dtype=np.uint8)
    
    # Step 1: Gaussian blur to smooth noise
    blurred = cv2.GaussianBlur(img_array.astype(np.uint8), (5, 5), 0)
    
    # Step 2: Threshold for bright regions (potential whiteout)
    _, binary = cv2.threshold(blurred, 245, 255, cv2.THRESH_BINARY)
    
    # Step 3: Morphological operations to connect nearby white regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    # Closing: fills small gaps in whiteout regions
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Opening: removes small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Step 4: Find contours of whiteout regions
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 5: Filter large contours (likely whiteout patches)
    large_contours = [c for c in contours if cv2.contourArea(c) > 100]
    
    # Step 6: Create mask from large contours
    whiteout_mask = np.zeros_like(img_array, dtype=np.uint8)
    cv2.drawContours(whiteout_mask, large_contours, -1, 255, -1)
    
    # Step 7: Calculate whiteout score
    whiteout_ratio = float(whiteout_mask.sum()) / (img_array.size * 255)
    
    # Score based on ratio and number of large regions
    score = min(whiteout_ratio * 10.0 + len(large_contours) * 0.1, 1.0)
    
    return score, whiteout_mask


def _morphological_texture_analysis(img_array: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Analyze texture using morphological operations to detect whiteout.
    
    Args:
        img_array: Grayscale image array
        
    Returns:
        Tuple of (texture_score, texture_mask)
    """
    if not CV2_AVAILABLE:
        return 0.0, np.zeros_like(img_array, dtype=np.uint8)
    
    # Step 1: Compute gradient (texture)
    grad_x = cv2.Sobel(img_array.astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_array.astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Step 2: Low texture regions (whiteout areas have low texture)
    low_texture = gradient_magnitude < np.percentile(gradient_magnitude, 20)
    low_texture_mask = (low_texture * 255).astype(np.uint8)
    
    # Step 3: Morphological closing to connect regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(low_texture_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Step 4: Find large low-texture regions
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_low_texture = [c for c in contours if cv2.contourArea(c) > 200]
    
    texture_mask = np.zeros_like(img_array, dtype=np.uint8)
    cv2.drawContours(texture_mask, large_low_texture, -1, 255, -1)
    
    # Score based on low-texture region ratio
    texture_ratio = float(texture_mask.sum()) / (img_array.size * 255)
    score = min(texture_ratio * 8.0, 1.0)
    
    return score, texture_mask


def detect_whiteout_and_lowconf(
    pil_img: Image.Image,
    ocr_text: str = "",
    enable_ela: bool = True
) -> List[Dict[str, Any]]:
    """
    Enhanced whiteout detection with Gaussian + morphological analysis and ELA comparison.
    
    Args:
        pil_img: PIL Image to analyze
        ocr_text: Optional OCR text for low confidence detection
        enable_ela: Enable ELA-based tampering detection
        
    Returns:
        List of structured flags: [{"flag_type": str, "score": float, "meta": dict}, ...]
    """
    flags = []
    
    # Convert to grayscale array
    gray = pil_img.convert("L")
    img_array = np.array(gray, dtype=np.uint8)
    w, h = pil_img.size
    total_pixels = w * h
    
    # 1. Gaussian + Morphological whiteout detection
    if CV2_AVAILABLE:
        gaussian_score, gaussian_mask = _gaussian_whiteout_analysis(img_array)
        if gaussian_score > 0.1:
            flags.append({
                "flag_type": "whiteout_gaussian",
                "score": float(gaussian_score),
                "meta": {
                    "whiteout_ratio": float(gaussian_mask.sum()) / (total_pixels * 255),
                    "method": "gaussian_morphological"
                }
            })
        
        # 2. Texture-based whiteout detection
        texture_score, texture_mask = _morphological_texture_analysis(img_array)
        if texture_score > 0.15:
            flags.append({
                "flag_type": "whiteout_low_texture",
                "score": float(texture_score),
                "meta": {
                    "low_texture_ratio": float(texture_mask.sum()) / (total_pixels * 255),
                    "method": "morphological_texture"
                }
            })
    
    # 3. Simple brightness-based detection (fallback)
    white_thresh = 245
    white_mask = img_array >= white_thresh
    white_ratio = float(white_mask.sum()) / img_array.size
    if white_ratio > 0.08:
        flags.append({
            "flag_type": "whiteout_brightness",
            "score": min(white_ratio * 5.0, 1.0),
            "meta": {
                "white_ratio": white_ratio,
                "threshold": white_thresh
            }
        })
    
    # 4. Edge anomaly detection near white regions
    if CV2_AVAILABLE:
        edges = cv2.Canny(img_array, 100, 200)
        edge_density = float((edges > 0).sum()) / edges.size
        if edge_density > 0.12 and white_ratio > 0.03:
            flags.append({
                "flag_type": "edge_anomaly_near_white",
                "score": min(edge_density * 5.0, 1.0),
                "meta": {
                    "edge_density": edge_density,
                    "white_ratio": white_ratio
                }
            })
    
    # 5. ELA-based tampering detection
    if enable_ela:
        try:
            ela_map = _compute_ela_map(pil_img)
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
        except Exception:
            pass
    
    # 6. Low OCR density detection
    if ocr_text:
        words = [w for w in ocr_text.split() if len(w) > 1]
        expected_words = (w * h) / 50000.0
        if len(words) < expected_words * 0.5:
            flags.append({
                "flag_type": "low_ocr_density",
                "score": 1.0 - (len(words) / max(expected_words, 1.0)),
                "meta": {
                    "ocr_words": len(words),
                    "expected_words": expected_words
                }
            })
    
    return flags


def compute_combined_fraud_score(flags: List[Dict[str, Any]], threshold: float = 0.5) -> Tuple[float, bool]:
    """
    Compute combined fraud score and determine if page is suspicious.
    
    Args:
        flags: List of fraud flags
        threshold: Threshold for marking page as suspicious
        
    Returns:
        Tuple of (combined_score, is_suspicious)
    """
    if not flags:
        return 0.0, False
    
    # Weighted sum of scores
    weights = {
        "whiteout_gaussian": 0.30,
        "whiteout_low_texture": 0.25,
        "whiteout_brightness": 0.20,
        "ela_anomaly": 0.25,
        "edge_anomaly_near_white": 0.15,
        "low_ocr_density": 0.10
    }
    
    combined_score = 0.0
    for flag in flags:
        flag_type = flag.get("flag_type", "")
        score = flag.get("score", 0.0)
        weight = weights.get(flag_type, 0.10)
        combined_score += score * weight
    
    # Normalize to 0-1
    combined_score = min(combined_score, 1.0)
    is_suspicious = combined_score > threshold
    
    return combined_score, is_suspicious


def create_whiteout_mask(pil_img: Image.Image) -> Image.Image:
    """Create whiteout mask image."""
    gray = pil_img.convert("L")
    arr = np.array(gray)
    mask = (arr >= 245).astype("uint8") * 255
    mask_img = Image.fromarray(mask).convert("L")
    return mask_img


def inpaint_image(pil_img: Image.Image, mask_img: Image.Image) -> Image.Image:
    """
    Inpaint whiteout regions using OpenCV.
    
    Args:
        pil_img: Original image
        mask_img: Whiteout mask
        
    Returns:
        Inpainted image
    """
    if not CV2_AVAILABLE:
        return pil_img
    
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    mask = np.array(mask_img.convert("L"))
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Use TELEA algorithm for inpainting
    inpainted = cv2.inpaint(img, mask_bin, 3, cv2.INPAINT_TELEA)
    inpaint_pil = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
    return inpaint_pil


def create_debug_overlay(
    pil_img: Image.Image,
    flags: List[Dict[str, Any]],
    whiteout_mask: Optional[np.ndarray] = None,
    ela_map: Optional[np.ndarray] = None
) -> Image.Image:
    """
    Create debug overlay image with fraud detection visualizations.
    
    Args:
        pil_img: Original image
        flags: List of fraud flags
        whiteout_mask: Optional whiteout mask
        ela_map: Optional ELA map
        
    Returns:
        Overlay image with annotations
    """
    if not CV2_AVAILABLE:
        return pil_img
    
    # Convert to RGB
    overlay = pil_img.convert("RGB")
    overlay_arr = np.array(overlay)
    
    # Draw whiteout regions in red
    if whiteout_mask is not None:
        whiteout_mask_3d = np.stack([whiteout_mask] * 3, axis=2) / 255.0
        overlay_arr = (overlay_arr * (1 - whiteout_mask_3d * 0.5) + 
                      np.array([255, 0, 0]) * whiteout_mask_3d * 0.5).astype(np.uint8)
    
    # Draw ELA anomalies in yellow
    if ela_map is not None:
        ela_thresh = (ela_map > 0.06).astype(np.uint8) * 255
        ela_mask_3d = np.stack([ela_thresh] * 3, axis=2) / 255.0
        overlay_arr = (overlay_arr * (1 - ela_mask_3d * 0.3) + 
                      np.array([255, 255, 0]) * ela_mask_3d * 0.3).astype(np.uint8)
    
    overlay_img = Image.fromarray(overlay_arr)
    
    # Add text annotations
    if flags:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(overlay_img)
        
        # Try to use default font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        y_offset = 10
        for flag in flags[:5]:  # Show top 5 flags
            flag_type = flag.get("flag_type", "")
            score = flag.get("score", 0.0)
            text = f"{flag_type}: {score:.2f}"
            draw.text((10, y_offset), text, fill=(255, 0, 0), font=font)
            y_offset += 20
    
    return overlay_img


def save_debug_mask(out_dir: str, filename: str, mask_img: Image.Image):
    """Save debug mask image."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    mask_img.save(path)
    return path
