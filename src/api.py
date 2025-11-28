from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import uuid
import os
from pathlib import Path

# IMPORTANT: your local extractor import
# Adjust if your folder is different (this is correct for src/extractor/bill_extractor.py)
from extractor.bill_extractor import extract_bill_data

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
    Returns: path to downloaded file.
    """
    try:
        response = requests.get(url, timeout=25)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")

    if response.status_code != 200:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to download document: HTTP {response.status_code}"
        )

    # Create temp directory
    temp_dir = Path("temp_docs")
    temp_dir.mkdir(exist_ok=True)

    # Create temp filename
    tmp_path = temp_dir / f"{uuid.uuid4()}.pdf"
    tmp_path.write_bytes(response.content)

    return str(tmp_path)


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

    # Download URL file → temp file
    local_file = download_to_temp(payload.document)

    try:
        result = extract_bill_data(local_file)
    except Exception as e:
        if os.path.exists(local_file):
            os.remove(local_file)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    # Cleanup
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
        result = extract_bill_data(local_file)
    except Exception as e:
        if os.path.exists(local_file):
            os.remove(local_file)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

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
