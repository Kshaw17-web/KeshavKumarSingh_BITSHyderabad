"""
Bajaj Finserv Datathon - Bill Extraction API
STRICT JSON COMPLIANCE MODE
"""

import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
    from src.utils.pdf_loader import download_to_temp, load_document_to_images
    FRAUD_ENGINE_AVAILABLE = True
except ImportError:
    FRAUD_ENGINE_AVAILABLE = False
    print("WARNING: Core modules missing. Running in skeleton mode.")

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

# --- MAIN ENDPOINT ---
@app.post("/api/v1/hackrx/run", response_model=FullResponse)
def hackrx_run(request: DocumentRequest):
    
    if not request.document:
        raise HTTPException(status_code=400, detail="Document URL missing")

    temp_file = None
    try:
        # 1. Download & Load
        temp_file = download_to_temp(request.document)
        images = load_document_to_images(temp_file)
        
        if not images:
            raise HTTPException(status_code=400, detail="Empty document")

        # 2. Parallel Processing
        # We run Extraction (Critical) and Fraud Check (Background) in parallel
        ocr_future = None
        
        # Generate request_id for debug logging
        import uuid
        request_id = f"api_{uuid.uuid4().hex[:8]}"
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit OCR Task - Use TSV-based extractor for better accuracy
            if FRAUD_ENGINE_AVAILABLE:
                ocr_future = executor.submit(extract_bill_data_with_tsv, images, request_id)
            
            # Submit Fraud Tasks (Fire and forget - purely for logging/console output)
            for i, img in enumerate(images):
                executor.submit(run_internal_fraud_check, img, i + 1)
            
            # Wait for OCR (Timeout safe)
            ocr_result = ocr_future.result(timeout=45) if ocr_future else {}

        # 3. Process & Format Data (Strict Schema)
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

            final_pagewise_items.append(PagewiseLineItem(
                page_no=page_no,
                bill_items=clean_bill_items
            ))

        # 4. Calculate Final Totals
        # "Total... without double counting" 
        reconciled_total = sum(item.item_amount for item in all_valid_items)
        # Use total_item_count from extractor if available, otherwise count all items
        total_count = data.get("total_item_count", len(clean_bill_items)) 

        # 5. Construct Response
        return FullResponse(
            is_success=True,
            data=ExtractionData(
                pagewise_line_items=final_pagewise_items,
                total_item_count=total_count,
                reconciled_amount=round(reconciled_total, 2)
            )
        )

    except Exception as e:
        logger.error(f"Fatal API Error: {e}")
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