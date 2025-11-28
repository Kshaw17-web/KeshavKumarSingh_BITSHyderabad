"""
src/api.py

Bajaj Finserv Datathon - Bill Extraction API
Enhanced FastAPI implementation: downloader, OCR preprocessing (OpenCV),
adaptive outlier filtering (IQR), identifier heuristics, and flexible extractor calling.
"""

import os
import tempfile
import uuid
import logging
import inspect
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# ---- Schemas: adjust names if your src/schemas.py uses different names ----
# Make sure src/schemas.py exports these Pydantic models:
# DocumentRequest, BillItem, PageItems, DataResponse, FullResponse
from src.schemas import (
    DocumentRequest,
    BillItem,
    PageItems,
    DataResponse,
    FullResponse
)
from src.preprocessing_helpers import preprocess_image_local, detect_whiteout_and_lowconf, preprocess_full_pipeline

# ---- Try to import extract_bill_data flexibly ----
extract_bill_data = None
_extract_info = {"source": None, "sig": None}
for try_path in ("src.extractor.bill_extractor", "extractor.bill_extractor", "extractor"):
    try:
        module = __import__(try_path, fromlist=["extract_bill_data"])
        extract_bill_data = getattr(module, "extract_bill_data", None)
        if extract_bill_data:
            _extract_info["source"] = try_path
            try:
                _extract_info["sig"] = inspect.signature(extract_bill_data)
            except Exception:
                _extract_info["sig"] = None
            break
    except Exception:
        continue

# Logging
LOG_PATH = Path("extract_debug.log")
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# FastAPI app
app = FastAPI(
    title="Bajaj Finserv Datathon API",
    description="Bill extraction API for HackRx challenge (with OCR preprocessing & heuristics)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # required for webhook usage
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# External tools configuration (override via env vars if needed)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
POPPLER_PATH = os.getenv("POPPLER_PATH", r"C:\poppler-25.11.0\Library\bin")
DEFAULT_DPI = int(os.getenv("OCR_DPI", "300"))

# Set tesseract cmd if present
try:
    import pytesseract  # type: ignore
    if TESSERACT_CMD and Path(TESSERACT_CMD).exists():
        pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_CMD)
except Exception:
    # Will raise a clear error if OCR is attempted without pytesseract
    pass

# ---------------------------------------------------------------------
# Helper: Ensure extractor is present
# ---------------------------------------------------------------------
def _ensure_extractor_available():
    if extract_bill_data is None:
        logging.error("extract_bill_data not found. _extract_info=%s", _extract_info)
        raise HTTPException(status_code=500, detail="Extractor implementation not found. Check src/extractor/bill_extractor.py")

# ---------------------------------------------------------------------
# Downloader (adds browser-like User-Agent to avoid 403/429)
# ---------------------------------------------------------------------
def download_file(url: str, timeout: int = 30) -> Path:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, timeout=timeout, stream=True, headers=headers)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.exception("Download failed for URL: %s", url)
        raise HTTPException(status_code=400, detail=f"Failed to download document from URL: {e}")

    content_type = (resp.headers.get("content-type") or "").lower()
    ext = ".pdf"
    if "image/png" in content_type:
        ext = ".png"
    elif "image/jpeg" in content_type or "image/jpg" in content_type:
        ext = ".jpg"
    else:
        p = Path(url.split("?")[0])
        if p.suffix:
            ext = p.suffix

    tmp = Path(tempfile.gettempdir()) / f"datathon_{uuid.uuid4().hex}{ext}"
    try:
        tmp.write_bytes(resp.content)
    except Exception as e:
        logging.exception("Failed to write temp file: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to save temporary file: {e}")

    logging.info("Downloaded URL -> %s", tmp)
    return tmp

# ---------------------------------------------------------------------
# Image preprocessing for OCR (OpenCV + Pillow)
# ---------------------------------------------------------------------
def _import_image_tools():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing image libs. Install: opencv-python numpy pillow") from e
    return cv2, np, Image

def preprocess_image_for_ocr(pil_img):
    cv2, np, Image = _import_image_tools()
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    if max(h, w) < 1200:
        gray = cv2.resize(gray, (int(w * 2.0), int(h * 2.0)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    from PIL import Image as PILImage  # local import
    return PILImage.fromarray(th)

# ---------------------------------------------------------------------
# Convert PDF/image -> list[str] (one OCR string per page) + preprocessing metadata
# ---------------------------------------------------------------------
def convert_file_to_ocr_pages(file_path: Path) -> Tuple[List[str], List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """
    Returns: (ocr_texts, preprocessing_metadata_list, fraud_flags_list)
    """
    try:
        from pdf2image import convert_from_path  # type: ignore
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except Exception as e:
        logging.exception("Missing OCR dependencies: %s", e)
        raise HTTPException(status_code=500, detail="OCR dependencies missing (pdf2image/pillow/pytesseract/opencv)")

    ocr_texts: List[str] = []
    preproc_metadata_list: List[Dict[str, Any]] = []
    fraud_flags_list: List[List[Dict[str, Any]]] = []
    suffix = file_path.suffix.lower()

    try:
        if suffix == ".pdf":
            poppler = str(POPPLER_PATH) if POPPLER_PATH and Path(POPPLER_PATH).exists() else None
            try:
                images = convert_from_path(str(file_path), dpi=DEFAULT_DPI, poppler_path=poppler)
            except Exception as e:
                logging.exception("PDF->image conversion failed: %s", e)
                raise HTTPException(status_code=500, detail=f"PDF conversion failed: {e}. Ensure Poppler installed.")
        else:
            from PIL import Image
            images = [Image.open(file_path)]

        preprocessed_images = []
        for idx, img in enumerate(images, start=1):
            try:
                # Use preprocessing pipeline
                prepped_img = preprocess_image_local(img)
                preprocessed_images.append(prepped_img)
                
                # Run OCR on preprocessed image
                text = pytesseract.image_to_string(prepped_img, lang="eng", config="--oem 3 --psm 6")
                ocr_texts.append(text or "")
                
                # Basic preprocessing metadata
                w, h = img.size
                preproc_meta = {
                    "width": w,
                    "height": h,
                    "preprocessed": True
                }
                preproc_metadata_list.append(preproc_meta)
                
                logging.info("OCR page %d len=%d", idx, len(text or ""))
                logging.info("OCR preview page %d: %s", idx, (text or "")[:300].replace("\n", " "))
            except Exception as e:
                logging.exception("OCR failed on page %d: %s", idx, e)
                ocr_texts.append("")
                preproc_metadata_list.append({})
                preprocessed_images.append(img)  # Fallback to original
        
        # Detect fraud flags across all pages at once
        try:
            fraud_tuples = detect_whiteout_and_lowconf(preprocessed_images, ocr_texts)
            # Convert tuples to dict format per page
            fraud_flags_by_page: Dict[int, List[Dict[str, Any]]] = {}
            for page_no, flag_type, score in fraud_tuples:
                if page_no not in fraud_flags_by_page:
                    fraud_flags_by_page[page_no] = []
                fraud_flags_by_page[page_no].append({
                    "type": flag_type,
                    "score": score
                })
            
            # Align fraud flags with pages
            for idx in range(1, len(ocr_texts) + 1):
                fraud_flags_list.append(fraud_flags_by_page.get(idx, []))
        except Exception as e:
            logging.exception("Fraud detection failed: %s", e)
            fraud_flags_list = [[] for _ in ocr_texts]
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("convert_file_to_ocr_pages error: %s", e)
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {e}")

    if not ocr_texts:
        raise HTTPException(status_code=500, detail="No text extracted by OCR")

    return ocr_texts, preproc_metadata_list, fraud_flags_list

# ---------------------------------------------------------------------
# Heuristics: identify likely identifiers (invoice ids, MRN, page numbers)
# ---------------------------------------------------------------------
import re
def looks_like_identifier(s: str) -> bool:
    if not s:
        return False
    s_low = s.lower()
    if re.search(r'\b(invoice|inv|id|ref|no\.?|bill no|mrn|uhid|patient id|page)\b', s_low):
        return True
    digits = re.sub(r'\D', '', s)
    # sequences of many digits (>=6) that dominate the line are likely identifiers not amounts
    if len(digits) >= 6 and (len(digits) / max(1, len(re.sub(r'\s', '', s)))) > 0.6:
        return True
    return False

# ---------------------------------------------------------------------
# Adaptive outlier removal (IQR + dynamic cap)
# ---------------------------------------------------------------------
def remove_outlier_amounts(amounts: List[float]) -> List[float]:
    if not amounts:
        return []
    import statistics
    vals = sorted([float(a) for a in amounts if a is not None and float(a) > 0])
    if not vals:
        return []
    n = len(vals)
    if n < 4:
        return vals
    q1 = statistics.median(vals[:n // 2])
    q3 = statistics.median(vals[(n + 1) // 2:])
    iqr = q3 - q1 if q3 > q1 else 0.0
    lower_bound = q1 - 1.5 * iqr if iqr > 0 else 0
    upper_bound = q3 + 1.5 * iqr if iqr > 0 else max(vals)
    med = statistics.median(vals)
    dynamic_upper = med * 20 if med > 0 else upper_bound
    ABSOLUTE_CAP = 10_000_000.0
    final = [v for v in vals if (v >= lower_bound and v <= upper_bound and v <= dynamic_upper and v <= ABSOLUTE_CAP)]
    if not final:
        final = [v for v in vals if v <= max(med * 50, upper_bound, 1_000_000)]
    return final

# ---------------------------------------------------------------------
# Decide how to call extractor: attempts to infer signature
# ---------------------------------------------------------------------
def extractor_accepts_file_path() -> Optional[bool]:
    """
    Inspect the signature of extract_bill_data to guess whether it expects
    a file path (str) or OCR pages (List[str]). Returns True=file_path, False=ocr_pages, None=unknown.
    """
    if extract_bill_data is None:
        return None
    sig = _extract_info.get("sig")
    if sig:
        params = list(sig.parameters.values())
        if not params:
            return None
        first = params[0]
        ann = first.annotation
        # heuristics on annotation or name
        name = first.name.lower()
        if ann in (str, "str") or "path" in name or "file" in name:
            return True
        if ann in (list, List, "List[str]", "list[str]") or "page" in name or "ocr" in name or "text" in name:
            return False
        return None
    # If we couldn't inspect signature, return None
    return None

def call_extractor_safely(file_path: Path, ocr_pages: List[str]) -> Dict[str, Any]:
    """
    Try calling extractor in a safe order:
      1) If signature suggests file path -> call with str(file_path)
      2) If signature suggests ocr pages -> call with ocr_pages
      3) Else: try ocr_pages then fallback to file_path
    """
    _ensure_extractor_available()
    prefer = extractor_accepts_file_path()
    # 1: signature prefers file
    if prefer is True:
        try:
            return extract_bill_data(str(file_path))
        except Exception as e:
            logging.exception("extractor(file_path) failed, will try ocr_pages fallback: %s", e)
            # fallback to ocr pages
            try:
                return extract_bill_data(ocr_pages)
            except Exception as e2:
                logging.exception("extractor(ocr_pages) also failed: %s", e2)
                raise
    # 2: signature prefers ocr pages
    if prefer is False:
        try:
            return extract_bill_data(ocr_pages)
        except Exception as e:
            logging.exception("extractor(ocr_pages) failed, will try file_path fallback: %s", e)
            try:
                return extract_bill_data(str(file_path))
            except Exception as e2:
                logging.exception("extractor(file_path) also failed: %s", e2)
                raise
    # 3: unknown signature -> try ocr_pages first (common in your earlier code), then file path
    try:
        return extract_bill_data(ocr_pages)
    except Exception as e:
        logging.exception("extractor(ocr_pages) failed; trying file path: %s", e)
        try:
            return extract_bill_data(str(file_path))
        except Exception as e2:
            logging.exception("extractor(file_path) failed as well: %s", e2)
            raise

# ---------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "OK", "message": "Bajaj Datathon API running"}

@app.post("/api/v1/hackrx/run", response_model=FullResponse)
def extract_bill_from_url(request: DocumentRequest) -> FullResponse:
    if not request or not getattr(request, "document", None):
        raise HTTPException(status_code=400, detail="Missing 'document' field in request")
    _ensure_extractor_available()

    temp_file: Optional[Path] = None
    try:
        temp_file = download_file(request.document)
        ocr_pages, preproc_metadata_list, fraud_flags_list = convert_file_to_ocr_pages(temp_file)

        # call extractor flexibly
        extraction_result = call_extractor_safely(temp_file, ocr_pages)
        if not isinstance(extraction_result, dict):
            raise RuntimeError("extract_bill_data returned non-dict result")

        # Validate and build Pydantic response objects
        page_objs: List[PageItems] = []
        all_amounts: List[float] = []

        for page_index, page in enumerate(extraction_result.get("pagewise_line_items", []), start=1):
            bills: List[BillItem] = []
            preproc_meta = None
            fraud_flags = None
            
            # Attach preprocessing and fraud metadata if available for this page
            try:
                page_idx = page_index - 1  # Convert to 0-based index
                if page_idx < len(preproc_metadata_list):
                    preproc_meta = preproc_metadata_list[page_idx]
                if page_idx < len(fraud_flags_list):
                    fraud_flags = fraud_flags_list[page_idx] if fraud_flags_list[page_idx] else None
            except Exception as e:
                logging.exception("Failed to attach preprocessing metadata for page %d: %s", page_index, e)
                preproc_meta = None
                fraud_flags = None

            for bi in page.get("bill_items", []):
                name = str(bi.get("item_name", "UNKNOWN")).strip()
                # skip obviously identifier-like lines
                if looks_like_identifier(name):
                    logging.info("Skipping identifier-like item name: %s", name)
                try:
                    amt_raw = bi.get("item_amount", 0.0)
                    amt = float(amt_raw) if amt_raw is not None and str(amt_raw).strip() != "" else 0.0
                except Exception:
                    amt = 0.0

                # only record amounts that pass a basic sanity test
                if amt and amt > 0 and amt < 10000000 and not looks_like_identifier(name):
                    all_amounts.append(amt)

                bills.append(
                    BillItem(
                        item_name=name,
                        item_amount=float(round(amt, 2)),
                        item_rate=(float(bi.get("item_rate")) if bi.get("item_rate") is not None else None),
                        item_quantity=(float(bi.get("item_quantity")) if bi.get("item_quantity") is not None else None),
                    )
                )

            page_objs.append(
                PageItems(
                    page_no=str(page.get("page_no", page_index)),
                    page_type=str(page.get("page_type", "Bill Detail")),
                    bill_items=bills,
                    reported_total=page.get("reported_total"),
                    reconciliation_ok=page.get("reconciliation_ok"),
                    reconciliation_relative_error=page.get("reconciliation_relative_error"),
                    preprocessing=preproc_meta,
                    fraud_flags=fraud_flags,
                )
            )

        filtered = remove_outlier_amounts(all_amounts)
        reconciled_amount = float(round(sum(filtered), 2)) if filtered else float(extraction_result.get("reconciled_amount", 0.0))

        data = DataResponse(
            pagewise_line_items=page_objs,
            total_item_count=int(extraction_result.get("total_item_count", sum(len(p.bill_items) for p in page_objs))),
            reconciled_amount=reconciled_amount
        )

        response = FullResponse(
            is_success=True,
            token_usage={"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
            data=data
        )
        logging.info("Extraction OK: items=%d reconciled=%.2f", data.total_item_count, data.reconciled_amount)
        return response

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("API extraction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")
    finally:
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass

@app.post("/extract-bill-file", response_model=FullResponse)
async def extract_bill_file(file: UploadFile = File(...)) -> FullResponse:
    _ensure_extractor_available()

    suffix = Path(file.filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        content = await file.read()
        tmp.write(content)

    try:
        ocr_pages, preproc_metadata_list, fraud_flags_list = convert_file_to_ocr_pages(tmp_path)
        extraction_result = call_extractor_safely(tmp_path, ocr_pages)
        if not isinstance(extraction_result, dict):
            raise RuntimeError("extract_bill_data returned non-dict result")

        page_objs: List[PageItems] = []
        all_amounts: List[float] = []

        for page_index, page in enumerate(extraction_result.get("pagewise_line_items", []), start=1):
            bills: List[BillItem] = []
            preproc_meta = None
            fraud_flags = None
            
            # Attach preprocessing and fraud metadata if available for this page
            try:
                page_idx = page_index - 1  # Convert to 0-based index
                if page_idx < len(preproc_metadata_list):
                    preproc_meta = preproc_metadata_list[page_idx]
                if page_idx < len(fraud_flags_list):
                    fraud_flags = fraud_flags_list[page_idx] if fraud_flags_list[page_idx] else None
            except Exception as e:
                logging.exception("Failed to attach preprocessing metadata for page %d (upload): %s", page_index, e)
                preproc_meta = None
                fraud_flags = None

            for bi in page.get("bill_items", []):
                name = str(bi.get("item_name", "UNKNOWN")).strip()
                if looks_like_identifier(name):
                    logging.info("Skipping identifier-like item name (upload): %s", name)
                try:
                    amt_raw = bi.get("item_amount", 0.0)
                    amt = float(amt_raw) if amt_raw is not None and str(amt_raw).strip() != "" else 0.0
                except Exception:
                    amt = 0.0

                if amt and amt > 0 and amt < 10000000 and not looks_like_identifier(name):
                    all_amounts.append(amt)

                bills.append(
                    BillItem(
                        item_name=name,
                        item_amount=float(round(amt, 2)),
                        item_rate=(float(bi.get("item_rate")) if bi.get("item_rate") is not None else None),
                        item_quantity=(float(bi.get("item_quantity")) if bi.get("item_quantity") is not None else None),
                    )
                )

            page_objs.append(
                PageItems(
                    page_no=str(page.get("page_no", page_index)),
                    page_type=str(page.get("page_type", "Bill Detail")),
                    bill_items=bills,
                    reported_total=page.get("reported_total"),
                    reconciliation_ok=page.get("reconciliation_ok"),
                    reconciliation_relative_error=page.get("reconciliation_relative_error"),
                    preprocessing=preproc_meta,
                    fraud_flags=fraud_flags,
                )
            )

        filtered = remove_outlier_amounts(all_amounts)
        reconciled_amount = float(round(sum(filtered), 2)) if filtered else float(extraction_result.get("reconciled_amount", 0.0))

        data = DataResponse(
            pagewise_line_items=page_objs,
            total_item_count=int(extraction_result.get("total_item_count", sum(len(p.bill_items) for p in page_objs))),
            reconciled_amount=reconciled_amount
        )

        response = FullResponse(
            is_success=True,
            token_usage={"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
            data=data
        )
        logging.info("Upload extraction OK: %s items=%d reconciled=%.2f", file.filename, data.total_item_count, data.reconciled_amount)
        return response

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Upload extraction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
