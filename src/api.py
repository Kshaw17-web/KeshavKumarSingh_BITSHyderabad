from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

from src.extractor.pdf_to_images import pdf_to_images
from src.extractor.parse_lines import extract_bill_lines
from src.utils.validator import build_stub_payload, validate_payload


class ExtractRequest(BaseModel):
    document: str


app = FastAPI(title="Datathon Extraction Prototype")


@app.post("/extract-bill-data")
def extract_bill_data(payload: ExtractRequest):
    """
    Extremely small FastAPI endpoint that routes a document path/URL through
    placeholder extractor utilities. Replace bodies with your real logic.
    """
    source = Path(payload.document)
    if not source.exists():
        raise HTTPException(status_code=404, detail="Document not found on disk")

    # Convert to images should the caller pass a PDF.
    page_images = pdf_to_images(source)

    # Parse text via OCR + heuristics (see parse_lines for TODO notes).
    parsed_items = extract_bill_lines(page_images)

    response_body = build_stub_payload(parsed_items)
    validate_payload(response_body)
    return response_body

