# BFHL Datathon - Starter Extractor (Keshav Kumar Singh)

This is a minimal starter implementation (Tesseract + simple heuristics) for the HackRx / BFHL datathon.

**IMPORTANT**: Edit and personalize these files (see ORIGIN_STATEMENT.md) before pushing.

## Quick start (Windows PowerShell)

1. Create & activate venv:

   python -m venv venv

   .\venv\Scripts\Activate.ps1

2. Install:

   pip install -r requirements.txt

3. Install Tesseract OCR for Windows (add to PATH):

   https://github.com/tesseract-ocr/tesseract/wiki

4. Start server:

   uvicorn src.api:app --reload --port 8000

5. Test locally (PowerShell):

   $payload = @{ document = "C:\\datathon_work\\data\\raw\\sample_test.pdf" } | ConvertTo-Json

   Invoke-RestMethod -Uri "http://127.0.0.1:8000/extract-bill-data" -Method Post -Body $payload -ContentType "application/json"

## Next steps (after you have a working pipeline)

- Improve OCR by adding PaddleOCR for multilingual/handwritten pages

- Add table detection and layout-aware model (e.g., LayoutLM/Donut)

- Add LLM-based post-processing to clean item names

- Implement fraud-detection (whitener / overwritten values)
