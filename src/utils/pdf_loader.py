"""
Optimized PDF to image conversion using pdf2image with performance enhancements.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

try:
    from pdf2image import convert_from_path
    from PIL import Image
except ImportError:
    convert_from_path = None
    Image = None


def load_pdf_to_images(
    pdf_path: Union[str, Path],
    dpi: int = 300,
    poppler_path: Optional[str] = None,
    use_grayscale: bool = False,
    thread_count: int = 4
) -> List["Image.Image"]:
    """
    Fast PDF to image conversion with optimizations.
    
    Optimizations:
    - Uses 'ppm' format for faster decode
    - Multi-threaded Poppler conversion (thread-count=4)
    - Optional grayscale mode for speed
    - Automatic downsampling for high DPI
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion (default 300, auto-downsampled if >300)
        poppler_path: Optional path to Poppler bin directory
        use_grayscale: Convert to grayscale directly (faster)
        thread_count: Number of threads for Poppler (default 4)
        
    Returns:
        List of PIL Image objects, one per page
        
    Raises:
        RuntimeError: If pdf2image is not installed or conversion fails
    """
    if convert_from_path is None:
        raise RuntimeError("pdf2image is not installed. Install with: pip install pdf2image")
    
    if Image is None:
        raise RuntimeError("PIL/Pillow is not installed. Install with: pip install pillow")
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Try to detect Poppler path from environment
    if poppler_path is None:
        poppler_path = os.getenv("POPPLER_PATH")
    
    # Validate Poppler path if provided
    if poppler_path:
        poppler_path = str(Path(poppler_path))
        if not Path(poppler_path).exists():
            poppler_path = None
    
    # Auto-downsample if DPI > 300 for speed
    effective_dpi = min(dpi, 300) if dpi > 300 else dpi
    
    try:
        # Build Poppler parameters for performance
        poppler_params = []
        if thread_count > 1:
            poppler_params.extend(["-thread-count", str(thread_count)])
        
        # Use PPM format for faster decode (faster than PNG)
        images = convert_from_path(
            str(pdf_path),
            dpi=effective_dpi,
            poppler_path=poppler_path,
            fmt='ppm',  # Faster format
            thread_count=thread_count if thread_count > 1 else None,
            grayscale=use_grayscale  # Direct grayscale conversion
        )
        
        # Convert to grayscale if requested but not done by Poppler
        if use_grayscale:
            images = [img.convert("L") if img.mode != "L" else img for img in images]
        
        return images
    except Exception as e:
        raise RuntimeError(f"PDF conversion failed: {e}. Ensure Poppler is installed and POPPLER_PATH is set correctly.")


def load_image_file(image_path: Union[str, Path]) -> "Image.Image":
    """
    Load a single image file (PNG, JPG, etc.) as PIL Image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image object
        
    Raises:
        RuntimeError: If PIL is not installed or file cannot be opened
    """
    if Image is None:
        raise RuntimeError("PIL/Pillow is not installed. Install with: pip install pillow")
    
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        return Image.open(image_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open image file: {e}")
