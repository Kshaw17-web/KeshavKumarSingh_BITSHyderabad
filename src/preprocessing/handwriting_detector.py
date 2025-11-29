"""
Handwriting detection utilities to automatically detect handwritten text in images.
"""

import numpy as np
from typing import Tuple, Optional

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


def detect_handwriting(img_array: np.ndarray, threshold: float = 0.3) -> Tuple[bool, float]:
    """
    Detect if an image contains handwritten text.
    
    Uses multiple heuristics:
    1. Stroke variability (handwritten has more variable stroke widths)
    2. Edge irregularity (handwritten has more irregular edges)
    3. Text alignment (handwritten is less aligned)
    4. Connected component analysis (handwritten has more irregular shapes)
    
    Args:
        img_array: Grayscale numpy array (cv2 format)
        threshold: Confidence threshold (0.0-1.0) for handwriting detection
        
    Returns:
        Tuple of (is_handwritten: bool, confidence: float)
    """
    if not CV2_AVAILABLE or img_array is None or img_array.size == 0:
        return False, 0.0
    
    try:
        # Convert to binary if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array.copy()
        
        # Ensure it's binary (black text on white)
        if gray.max() > 1:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            binary = (1 - gray) * 255
        
        scores = []
        
        # 1. Stroke width variability
        # Handwritten text has more variable stroke widths
        stroke_variability = _calculate_stroke_variability(binary)
        scores.append(stroke_variability)
        
        # 2. Edge irregularity
        # Handwritten text has more irregular edges
        edge_irregularity = _calculate_edge_irregularity(binary)
        scores.append(edge_irregularity)
        
        # 3. Text alignment
        # Handwritten text is less aligned
        alignment_score = _calculate_alignment_score(binary)
        scores.append(alignment_score)
        
        # 4. Connected component shape irregularity
        # Handwritten characters have more irregular shapes
        shape_irregularity = _calculate_shape_irregularity(binary)
        scores.append(shape_irregularity)
        
        # Average the scores
        confidence = np.mean(scores)
        
        is_handwritten = confidence >= threshold
        
        return is_handwritten, float(confidence)
    
    except Exception:
        return False, 0.0


def _calculate_stroke_variability(binary: np.ndarray) -> float:
    """Calculate stroke width variability (higher = more handwritten)."""
    try:
        # Use distance transform to estimate stroke widths
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Get stroke width values (non-zero pixels)
        stroke_widths = dist_transform[dist_transform > 0]
        
        if len(stroke_widths) < 10:
            return 0.0
        
        # Calculate coefficient of variation (std/mean)
        mean_width = np.mean(stroke_widths)
        std_width = np.std(stroke_widths)
        
        if mean_width == 0:
            return 0.0
        
        cv_score = std_width / mean_width
        # Normalize to 0-1 range (typical CV for handwritten: 0.3-0.8, printed: 0.1-0.3)
        return min(1.0, max(0.0, (cv_score - 0.1) / 0.7))
    
    except Exception:
        return 0.0


def _calculate_edge_irregularity(binary: np.ndarray) -> float:
    """Calculate edge irregularity (higher = more handwritten)."""
    try:
        # Find edges
        edges = cv2.Canny(binary, 50, 150)
        
        # Calculate edge density and distribution
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        
        if total_pixels == 0:
            return 0.0
        
        edge_density = edge_pixels / total_pixels
        
        # Handwritten text typically has higher edge density due to irregularity
        # Normalize (typical: handwritten 0.15-0.3, printed 0.05-0.15)
        return min(1.0, max(0.0, (edge_density - 0.05) / 0.25))
    
    except Exception:
        return 0.0


def _calculate_alignment_score(binary: np.ndarray) -> float:
    """Calculate text alignment score (lower alignment = more handwritten)."""
    try:
        # Use Hough lines to detect text lines
        lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
        
        if lines is None or len(lines) < 3:
            return 0.5  # Can't determine
        
        # Calculate horizontal alignment (y-coordinates should be similar for same line)
        y_coords = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            y_coords.append((y1 + y2) / 2)
        
        # Calculate variance in y-coordinates (higher variance = less aligned = more handwritten)
        y_variance = np.var(y_coords)
        
        # Normalize (typical: handwritten has higher variance)
        # Scale to 0-1 range
        normalized_variance = min(1.0, y_variance / 100.0)
        
        return normalized_variance
    
    except Exception:
        return 0.5


def _calculate_shape_irregularity(binary: np.ndarray) -> float:
    """Calculate shape irregularity of connected components (higher = more handwritten)."""
    try:
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels < 3:
            return 0.0
        
        irregularity_scores = []
        
        # Analyze each component (skip background, label 0)
        for label in range(1, min(num_labels, 100)):  # Limit to first 100 components
            # Get component mask
            component_mask = (labels == label).astype(np.uint8) * 255
            
            # Calculate area and perimeter
            area = stats[label, cv2.CC_STAT_AREA]
            if area < 10:  # Skip very small components
                continue
            
            # Get contour
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                continue
            
            # Circularity = 4π*area/perimeter²
            # Lower circularity = more irregular = more handwritten
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Invert and normalize (handwritten has lower circularity)
            irregularity = 1.0 - min(1.0, circularity)
            irregularity_scores.append(irregularity)
        
        if not irregularity_scores:
            return 0.0
        
        return np.mean(irregularity_scores)
    
    except Exception:
        return 0.0

