"""
Pipeline module for processing files directly.
Provides extract_bill_data_from_file function that handles file I/O.
"""

from pathlib import Path
from typing import Dict, Any
import os

# Import OCR conversion utilities
from .pdf_to_images import pdf_to_images

# Import extraction logic
from .bill_extractor import extract_bill_data


def extract_bill_data_from_file(file_path: str | Path) -> Dict[str, Any]:
    """
    Extract bill data from a file (PDF or image).
    
    This function:
    1. Converts PDF to images or uses image directly
    2. Runs OCR on each page
    3. Extracts bill data using extract_bill_data
    
    Args:
        file_path: Path to PDF or image file
        
    Returns:
        Dictionary with structure matching HackRx data schema:
        {
            "pagewise_line_items": [...],
            "total_item_count": int,
            "reconciled_amount": float
        }
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Convert file to images (handles both PDF and images)
    image_paths = pdf_to_images(str(file_path))
    
    # Run OCR on each image and collect text
    import pytesseract
    from PIL import Image
    
    ocr_pages: list[str] = []
    
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            # Convert to grayscale for better OCR
            if img.mode != "L":
                img = img.convert("L")
            
            # Run OCR
            text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
            ocr_pages.append(text)
        except Exception as e:
            # Continue with empty text if OCR fails
            ocr_pages.append("")
    
    # Extract bill data from OCR pages
    result = extract_bill_data(ocr_pages)
    
    # Cleanup temporary image files if they were created
    # (pdf_to_images creates temp files for PDF pages)
    for img_path in image_paths:
        img_path_obj = Path(img_path)
        # Only delete if it's in a temp directory (not the original file)
        if "datathon_pages_" in str(img_path_obj) and img_path_obj.exists():
            try:
                img_path_obj.unlink()
            except Exception:
                pass
    
    return result



