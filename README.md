# Bajaj Finserv Datathon – Bill Extraction API  

### Author: Keshav Kumar Singh (BITS Pilani, Hyderabad Campus)

This repository contains my implementation for the Bajaj Finserv HackRx Datathon challenge on automated multi-page bill extraction. The goal is to build a clean, reliable system that can take any bill (PDF or image), run OCR on every page, identify sections, extract item details, and return everything in a structured JSON format that matches the official evaluation spec.

I tried to keep the solution simple, modular, and easy to understand. The project uses FastAPI for the backend, Poppler to convert PDFs into images, and Tesseract OCR for extracting text. The parsing logic is rule-based and tuned to handle the training samples provided.

---

## 🔧 Technologies Used

- **FastAPI** – backend framework
- **Tesseract OCR** – text extraction
- **Poppler** – PDF → image conversion
- **Python 3.12**
- **ngrok** – to expose local API publicly for Datathon testing

---

## 📁 Project Structure

```
src/
├── api.py                    # FastAPI routes
├── extractor/
│   ├── bill_extractor.py      # Main extraction logic
│   ├── ocr_engine.py          # OCR processing
│   ├── utils.py               # Helper functions
│   └── __init__.py
ocr_outputs/                    # OCR PNGs + JSON output
data/raw/                       # Training samples
requirements.txt
README.md
```

---

##  How It Works

1. The API receives a **public file URL** (PDF or PNG/JPG).
2. The file is downloaded temporarily.
3. PDF files are converted to page-wise PNGs using Poppler.
4. Tesseract reads text from each page.
5. A lightweight parser extracts:
   - item names  
   - item quantities  
   - item rates  
   - final amounts  
6. Pages are classified as:
   - *Bill Detail*
   - *Final Bill*
   - *Pharmacy*
7. Output is assembled in the exact JSON schema required by HackRx.

---

##  API Endpoints

### Health Check  
**GET /** 

### Bill Extraction  
**POST /api/v1/hackrx/run**

**Request:**

```json
{
  "document": "<url-pointing-to-bill-file>"
}
```

**Response (sample structured output):**

```json
{
  "is_success": true,
  "token_usage": { "total_tokens": 0, "input_tokens": 0, "output_tokens": 0 },
  "data": {
    "pagewise_line_items": [
      {
        "page_no": 1,
        "page_type": "Bill Detail",
        "bill_items": [
          {
            "item_name": "Paracetamol 650mg",
            "item_amount": 54.0,
            "item_rate": 18.0,
            "item_quantity": 3
          }
        ]
      }
    ],
    "total_item_count": 1
  }
}
```

---

##  Running Locally

1. **Create virtual environment**

```bash
python -m venv venv
venv\Scripts\activate
```

2. **Install requirements**

```bash
pip install -r requirements.txt
```

3. **Install:**
   - Tesseract OCR
   - Poppler
   (and add both to PATH)

4. **Start API**

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

---

##  Making API Public for Datathon

Run:

```bash
ngrok http 8000
```

You will get a URL like:

```
https://dax-implemental-onie.ngrok-free.dev
```

Final submission endpoint:

```
https://dax-implemental-onie.ngrok-free.dev/api/v1/hackrx/run
```

---

##  Submission Notes (for portal)

```
This API processes multi-page bills using OCR and returns structured item data, totals, and page-level classifications as per the HackRx specification.

Repository: https://github.com/Kshaw17-web/KeshavKumarSingh_BITSHyderabad
Collaborator added: hackrxbot
```

---

##  About the Developer

I'm Keshav Kumar Singh, currently pursuing Mechanical Engineering at BITS Pilani, Hyderabad Campus.

I enjoy exploring ML, OCR systems, automation, and building practical API-based solutions. This project was a good mix of all of these and helped me understand document parsing in a real-world context.
