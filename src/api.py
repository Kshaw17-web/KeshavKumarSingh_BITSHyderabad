"""
Bajaj Finserv Datathon - Bill Extraction API
STRICT JSON COMPLIANCE MODE
"""

import os
import time
import logging
import traceback
import uuid
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Union

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- IMPORTS ---
# Ensure these match your folder structure
from src.schemas import (
    DocumentRequest,
    BillItem,
    PagewiseItem,
    ExtractionData,
    FullResponse
)

# Import Utilities (Assumed to exist based on your file tree)
try:
    from src.preprocessing.fraud_filters import detect_fraud_flags, compute_unified_fraud_score
    from src.extractor.bill_extractor import extract_bill_data_with_tsv
    from src.utils.pdf_loader import load_pdf_to_images
    FRAUD_ENGINE_AVAILABLE = True
except ImportError as e:
    FRAUD_ENGINE_AVAILABLE = False
    print(f"WARNING: Core modules missing: {e}. Running in skeleton mode.")

# --- SETUP ---
app = FastAPI(title="Bajaj Finserv Datathon API", version="Final-Strict")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn")
MAX_WORKERS = min(8, os.cpu_count() or 4)

# Set default POPPLER_PATH if not set
if "POPPLER_PATH" not in os.environ:
    default_poppler = r"C:\poppler-25.11.0\Library\bin"
    if Path(default_poppler).exists():
        os.environ["POPPLER_PATH"] = default_poppler
        logger.info(f"Set POPPLER_PATH to {default_poppler}")

# Check Tesseract availability
TESSERACT_AVAILABLE = shutil.which("tesseract") is not None
if not TESSERACT_AVAILABLE:
    logger.warning("Tesseract not found in PATH. OCR may fail. Install Tesseract or add to PATH.")

# === DEBUG EXCEPTION HANDLER (TEMPORARY - REMOVE AFTER FIXING) ===
@app.exception_handler(Exception)
async def debug_exceptions(request: Request, exc: Exception):
    """Temporary debug handler to capture all exceptions."""
    tb = traceback.format_exc()
    logger.error(f"=== DEBUG TRACEBACK ===\n{tb}\n=== END TRACEBACK ===")
    
    # Save traceback to file
    try:
        debug_dir = Path("logs") / "debug_errors"
        debug_dir.mkdir(parents=True, exist_ok=True)
        error_file = debug_dir / f"error_{uuid.uuid4().hex[:8]}.txt"
        error_file.write_text(tb, encoding="utf-8")
        logger.info(f"Traceback saved to: {error_file}")
    except Exception:
        pass
    
    # Return safe JSON response (HTTP 200, not 500)
    return JSONResponse(
        status_code=200,
        content={
            "is_success": False,
            "message": "Internal Server Error (debug). See 'traceback'.",
            "error": str(exc),
            "traceback": "\n".join(tb.splitlines()[-200:]) if len(tb.splitlines()) > 200 else tb,
            "traceback_file": str(error_file) if 'error_file' in locals() else None
        }
    )

# --- HELPER: ROW FILTERING ---
def is_summary_row(item_name: str) -> bool:
    """
    Prevents double counting.
    Returns True if the row is likely a Sub Total, Tax, or Grand Total.
    """
    item_name_lower = item_name.lower().replace(".", "").strip()
    
    # Strict keywords that indicate a summary row
    summary_keywords = [
        "total", "sub total", "subtotal", "net amount", "grand total", 
        "tax", "vat", "gst", "cgst", "sgst", "discount", "advance", 
        "balance", "net payable", "amount due"
    ]
    
    # Precise check: "Total" is bad. "Total Knee Replacement" is good.
    if item_name_lower in summary_keywords:
        return True
        
    for k in summary_keywords:
        # Check if line starts with keyword (e.g. "Total Amount")
        if item_name_lower.startswith(k + " ") or item_name_lower.endswith(" " + k):
             # Exception: Medical procedures starting with Total
             if "replacement" in item_name_lower or "surgery" in item_name_lower:
                 continue
             return True
    return False

# --- HELPER: FRAUD RUNNER (INTERNAL ONLY) ---
def run_internal_fraud_check(image, page_idx):
    """
    Runs fraud detection for internal logging and 'Differentiator' evidence.
    Does NOT affect JSON output to ensure schema compliance.
    """
    if not FRAUD_ENGINE_AVAILABLE:
        return
    try:
        flags, debug_maps = detect_fraud_flags(image, use_fast_mode=True)
        score = compute_unified_fraud_score(flags)
        if score > 0.1:
            logger.warning(f"FRAUD DETECTED on Page {page_idx}: Score {score:.2f} - Flags: {flags}")
    except Exception as e:
        logger.error(f"Fraud check error: {e}")

# --- HELPER: Handle file upload or URL ---
def _handle_document_input(document_input: Union[str, UploadFile, Path]) -> Path:
    """
    Handle document input: can be URL, local file path, or UploadFile.
    Returns Path to temporary file.
    """
    import tempfile
    
    # If it's an UploadFile
    if isinstance(document_input, UploadFile):
        # Save to temp file
        suffix = Path(document_input.filename).suffix if document_input.filename else ".pdf"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            shutil.copyfileobj(document_input.file, temp_file)
            temp_file.close()
            return Path(temp_file.name)
        except Exception as e:
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
            raise RuntimeError(f"Failed to save uploaded file: {e}")
    
    # If it's a string (URL or local path)
    if isinstance(document_input, str):
        # Check if it's a local file path
        local_path = Path(document_input)
        if local_path.exists() and local_path.is_file():
            return local_path
        
        # Otherwise, treat as URL and download
        try:
            import requests
            response = requests.get(document_input, timeout=30)
            response.raise_for_status()
            
            suffix = Path(document_input).suffix or ".pdf"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.write(response.content)
            temp_file.close()
            return Path(temp_file.name)
        except Exception as e:
            raise RuntimeError(f"Failed to download from URL: {e}")
    
    # If it's already a Path
    if isinstance(document_input, Path):
        if document_input.exists():
            return document_input
        raise FileNotFoundError(f"File not found: {document_input}")
    
    raise ValueError(f"Unsupported document input type: {type(document_input)}")


# --- MAIN ENDPOINT (File Upload) ---
@app.post("/api/v1/hackrx/run", response_model=FullResponse)
async def hackrx_run(
    request: Request,
    document: Optional[UploadFile] = File(None)
):
    """
    Extract bill data from uploaded PDF or URL.
    Supports both file upload (multipart/form-data) and JSON body with URL.
    """
    document_input = None
    
    # Check for file upload
    if document:
        document_input = document
    else:
        # Try JSON body
        try:
            body = await request.json()
            if body and "document" in body:
                document_input = body["document"]
        except:
            # Not JSON, might be form data
            try:
                form_data = await request.form()
                if "document" in form_data:
                    document_input = form_data["document"]
            except:
                pass
    
    if not document_input:
        return FullResponse(
            is_success=False,
            data=ExtractionData(
                pagewise_line_items=[],
                total_item_count=0,
                reconciled_amount=0.0
            )
        )

    temp_file = None
    request_id = f"api_{uuid.uuid4().hex[:8]}"
    error_log_dir = None
    
    try:
        # 1. Handle document input (file upload, URL, or local path)
        try:
            temp_file = _handle_document_input(document_input)
        except Exception as e:
            logger.error(f"Document handling error: {e}", exc_info=True)
            error_log_dir = Path("logs") / request_id
            error_log_dir.mkdir(parents=True, exist_ok=True)
            (error_log_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            return FullResponse(
                is_success=False,
                data=ExtractionData(
                    pagewise_line_items=[],
                    total_item_count=0,
                    reconciled_amount=0.0
                )
            )
        
        # 2. Load PDF to images
        try:
            images = load_pdf_to_images(temp_file, dpi=300)
        except Exception as e:
            logger.error(f"PDF loading error: {e}", exc_info=True)
            error_log_dir = Path("logs") / request_id
            error_log_dir.mkdir(parents=True, exist_ok=True)
            (error_log_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            return FullResponse(
                is_success=False,
                data=ExtractionData(
                    pagewise_line_items=[],
                    total_item_count=0,
                    reconciled_amount=0.0
                )
            )
        
        if not images:
            return FullResponse(
                is_success=False,
                data=ExtractionData(
                    pagewise_line_items=[],
                    total_item_count=0,
                    reconciled_amount=0.0
                )
            )

        # 3. Parallel Processing
        # We run Extraction (Critical) and Fraud Check (Background) in parallel
        ocr_future = None
        
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit OCR Task - Use TSV-based extractor for better accuracy
                if FRAUD_ENGINE_AVAILABLE:
                    ocr_future = executor.submit(extract_bill_data_with_tsv, images, request_id)
                
                # Submit Fraud Tasks (Fire and forget - purely for logging/console output)
                for i, img in enumerate(images):
                    executor.submit(run_internal_fraud_check, img, i + 1)
                
                # Wait for OCR (Timeout safe)
                ocr_result = ocr_future.result(timeout=45) if ocr_future else {}
        except Exception as e:
            logger.error(f"Extraction error: {e}", exc_info=True)
            error_log_dir = Path("logs") / request_id
            error_log_dir.mkdir(parents=True, exist_ok=True)
            (error_log_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            return FullResponse(
                is_success=False,
                data=ExtractionData(
                    pagewise_line_items=[],
                    total_item_count=0,
                    reconciled_amount=0.0
                )
            )

        # 4. Process & Format Data (Strict Schema)
        final_pagewise_items = []
        all_valid_items = []
        
        # Check if extraction was successful
        if not ocr_result.get("is_success", True):
            # Extraction failed, return safe error response
            return FullResponse(
                is_success=False,
                data=ExtractionData(
                    pagewise_line_items=[],
                    total_item_count=0,
                    reconciled_amount=0.0
                )
            )
        
        # Extract data from response
        data = ocr_result.get("data", {})
        raw_pages = data.get("pagewise_line_items", [])
        
        # If extractor failed or returned empty
        if not raw_pages and len(images) > 0:
            # Create dummy page to prevent 500 error
            raw_pages = [{"page_no": "1", "bill_items": []}]

        for page_data in raw_pages:
            page_no = str(page_data.get("page_no", "1"))
            raw_items = page_data.get("bill_items", [])
            
            clean_bill_items = []
            
            for item in raw_items:
                name = str(item.get("item_name", "Unknown")).strip()
                
                # Create strict BillItem
                # Handle None/Null values by defaulting to 0.0
                b_item = BillItem(
                    item_name=name,
                    item_amount=float(item.get("item_amount") or 0.0),
                    item_rate=float(item.get("item_rate") or 0.0),
                    item_quantity=float(item.get("item_quantity") or 1.0)
                )
                
                # Logic: Add to list, but check if it's a summary row
                clean_bill_items.append(b_item)
                
                if not is_summary_row(name):
                    all_valid_items.append(b_item)

            final_pagewise_items.append(PagewiseItem(
                page_no=page_no,
                bill_items=clean_bill_items
            ))

        # 5. Calculate Final Totals
        # "Total... without double counting" 
        reconciled_total = sum(item.item_amount for item in all_valid_items)
        # Use total_item_count from extractor if available, otherwise count all items
        total_count = data.get("total_item_count", len(clean_bill_items)) 

        # 6. Construct Response
        return FullResponse(
            is_success=True,
            data=ExtractionData(
                pagewise_line_items=final_pagewise_items,
                total_item_count=total_count,
                reconciled_amount=round(reconciled_total, 2)
            )
        )

    except Exception as e:
        logger.error(f"Fatal API Error: {e}", exc_info=True)
        # Save error traceback
        try:
            if not error_log_dir:
                error_log_dir = Path("logs") / request_id
                error_log_dir.mkdir(parents=True, exist_ok=True)
            (error_log_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass
        
        # Return a safe fallback response instead of 500 Crash
        return FullResponse(
            is_success=False,
            data=ExtractionData(
                pagewise_line_items=[],
                total_item_count=0,
                reconciled_amount=0.0
            )
        )
    finally:
        # Cleanup
        if temp_file and temp_file.exists():
            try: temp_file.unlink()
            except: pass

# Alias for backward compatibility if needed
@app.post("/extract-bill-data")
def simple_extract(payload: DocumentRequest):
    return hackrx_run(payload)