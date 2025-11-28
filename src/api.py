# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import uuid
import os
from pathlib import Path
from typing import List

# OCR / image libs
from pdf2image import convert_from_path
from PIL import Image, ImageFilter, ImageOps
import pytesseract

# Import your document-level extractor that expects List[str] (ocr pages)
# Try the src package first (when running from project root), fall back to root-level extractor
try:
    from src.extractor.bill_extractor import extract_bill_data
except Exception:
    from extractor.bill_extractor import extract_bill_data


# ---- Configuration: override via environment variables if needed ----
# Example:
#   setx TESSERACT_CMD "C:\Program Files\Tesseract-OCR\tesseract.exe"
#   setx POPPLER_PATH "C:\poppler-25.11.0\Library\bin"
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
POPPLER_PATH = os.getenv("POPPLER_PATH", r"C:\poppler-25.11.0\Library\bin")
DEFAULT_DPI = int(os.getenv("OCR_DPI", "300"))

# If tesseract not on PATH, point pytesseract to it
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

app = FastAPI(
    title="Bajaj Finserv Datathon API",
    description="Public API for extracting bill details",
    version="1.0"
)

# -------------------------------
#   Request Model
# -------------------------------
class BillRequest(BaseModel):
    document: str  # URL to bill image or pdf


# -------------------------------
#   Helper: Download document
# -------------------------------
def download_to_temp(url: str) -> str:
    """
    Download the file from URL and save locally as a temp file.
    Returns: path to downloaded file (string).
    """
    try:
        resp = requests.get(url, timeout=30)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")

    if resp.status_code != 200 or not resp.content:
        raise HTTPException(status_code=400, detail=f"Failed to download document, status: {resp.status_code}")

    tmp_dir = Path("temp_docs")
    tmp_dir.mkdir(exist_ok=True)
    ext = ".pdf"
    # try to infer extension from headers
    content_type = resp.headers.get("content-type", "").lower()
    if "image/png" in content_type:
        ext = ".png"
    elif "image/jpeg" in content_type or "image/jpg" in content_type:
        ext = ".jpg"
    # otherwise default .pdf

    tmp_path = tmp_dir / f"{uuid.uuid4()}{ext}"
    try:
        tmp_path.write_bytes(resp.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write temp file: {e}")

    return str(tmp_path)


# -------------------------------
#   OCR Helpers
# -------------------------------
def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    Basic preprocessing to improve OCR quality.
    Convert to grayscale, resize slightly, and sharpen.
    """
    try:
        img = img.convert("L")  # grayscale
        w, h = img.size
        # Resize moderately to help small fonts
        img = img.resize((int(w * 1.4), int(h * 1.4)), Image.LANCZOS)
        img = img.filter(ImageFilter.SHARPEN)
    except Exception:
        pass
    return img


def document_to_ocr_pages(local_path: str, poppler_path: str = POPPLER_PATH, dpi: int = DEFAULT_DPI) -> List[str]:
    """
    Convert a local PDF/image file into a list of page OCR texts.
    Works with PDF (via pdf2image) and single images (jpg/png).
    Returns list[str] where each element is one page's OCR text.
    """
    ocr_texts: List[str] = []
    try:
        suffix = Path(local_path).suffix.lower()
        images = []
        if suffix == ".pdf":
            # convert pdf -> list of PIL images
            # pdf2image will raise PDFInfoNotInstalledError if poppler not found
            images = convert_from_path(local_path, dpi=dpi, poppler_path=poppler_path)
        else:
            # single image file
            img = Image.open(local_path)
            images = [img]

        for img in images:
            img_proc = preprocess_image_for_ocr(img)
            # Using psm 6 (Assume a block of text). Tune if needed.
            text = pytesseract.image_to_string(img_proc, lang='eng', config='--psm 6')
            ocr_texts.append(text)
    except Exception as e:
        # raise a clearer HTTPException upstream
        raise RuntimeError(f"OCR conversion failed: {e}")

    return ocr_texts


# -------------------------------
#   Root & Health Check
# -------------------------------
@app.get("/")
def root():
    return {"status": "OK", "message": "Bajaj Datathon API running"}


@app.get("/health")
def health():
    return {"health": "alive"}


# -------------------------------
#   MAIN ENDPOINT (for students)
# -------------------------------
@app.post("/extract-bill-data")
def extract_bill(payload: BillRequest):
    if not payload.document:
        raise HTTPException(status_code=400, detail="Missing 'document' field")

    # Download file -> local temp path
    local_file = download_to_temp(payload.document)

    try:
        # Convert local file -> OCR per-page texts
        ocr_pages = document_to_ocr_pages(local_file)
    except Exception as e:
        if os.path.exists(local_file):
            os.remove(local_file)
        raise HTTPException(status_code=500, detail=str(e))

    try:
        # extract_bill_data expects List[str] (ocr pages)
        result = extract_bill_data(ocr_pages)
    except Exception as e:
        if os.path.exists(local_file):
            os.remove(local_file)
        raise HTTPException(status_code=500, detail=f"Bill extraction failed: {e}")

    # Cleanup temp file
    if os.path.exists(local_file):
        os.remove(local_file)

    return {
        "is_success": True,
        "token_usage": {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0
        },
        "data": result
    }


# -------------------------------
#   HACKRX WEBHOOK ENDPOINT (required)
# -------------------------------
@app.post("/api/v1/hackrx/run")
def hackrx_webhook(payload: BillRequest):
    if not payload.document:
        raise HTTPException(status_code=400, detail="Missing 'document' field")

    local_file = download_to_temp(payload.document)

    try:
        ocr_pages = document_to_ocr_pages(local_file)
    except Exception as e:
        if os.path.exists(local_file):
            os.remove(local_file)
        raise HTTPException(status_code=500, detail=str(e))

    try:
        result = extract_bill_data(ocr_pages)
    except Exception as e:
        if os.path.exists(local_file):
            os.remove(local_file)
        raise HTTPException(status_code=500, detail=f"Bill extraction failed: {e}")

    if os.path.exists(local_file):
        os.remove(local_file)

    return {
        "is_success": True,
        "token_usage": {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0
        },
        "data": result
    }
