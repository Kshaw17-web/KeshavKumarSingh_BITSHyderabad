import os
import io
import uuid
import time
import tempfile
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response, FileResponse
from pydantic import BaseModel

try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

app = FastAPI(title="Bajaj Finserv Datathon API", version="1.0")

# Mount static directory (optional, useful for other static assets too)
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

POPPLER_PATH = os.getenv("POPPLER_PATH", r"C:\poppler-25.11.0\Library\bin")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
OCR_DPI = int(os.getenv("OCR_DPI", "200"))
MAX_WORKERS = int(os.getenv("OCR_WORKERS", "4"))
DEBUG_TRACE = os.getenv("DEBUG_TRACE", "0") == "1"

if pytesseract and Path(TESSERACT_CMD).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def json_error(detail: str, status: int = 500, exc: Optional[BaseException] = None):
    body = {"is_success": False, "message": detail}
    if DEBUG_TRACE and exc:
        body["traceback"] = traceback.format_exc()
    return JSONResponse(status_code=status, content=body)

def download_bytes(url: str, timeout: int = 25) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0"}
    session = requests.Session()
    try:
        r = session.get(url, headers=headers, timeout=timeout, stream=True)
        r.raise_for_status()
        return r.content
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Download failed: {e}")

def bytes_to_images(data: bytes, dpi: int = OCR_DPI) -> List[Any]:
    if convert_from_bytes is None:
        raise RuntimeError("pdf2image not installed")
    poppler = POPPLER_PATH if Path(POPPLER_PATH).exists() else None
    try:
        return convert_from_bytes(data, dpi=dpi, poppler_path=poppler)
    except Exception as e:
        raise RuntimeError(f"PDF->image conversion failed: {e}")

def image_to_text(img) -> str:
    if pytesseract is None:
        raise RuntimeError("pytesseract not installed")
    if Image is None:
        raise RuntimeError("Pillow not installed")
    if img.mode != "L":
        img = img.convert("L")
    try:
        text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")
        return text
    except Exception as e:
        raise RuntimeError(f"Tesseract failed: {e}")

def simple_page_extractor(page_text: str, page_no: int = 1) -> Dict:
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    page_type = "Bill Detail"
    lower = page_text.lower()
    if any(k in lower for k in ("pharmacy", "drug", "tablet", "syrup")):
        page_type = "Pharmacy"
    if any(k in lower for k in ("final bill", "grand total", "net payable", "amount payable", "invoice total")):
        page_type = "Final Bill"
    bill_items = []
    import re
    money_re = re.compile(r"((?:₹|Rs\.?|INR)?\s*\d{1,3}(?:[,\d]*)(?:\.\d{1,2})?)")
    qty_rate_re = re.compile(r"(\d+(?:\.\d+)?)\s*[xX*]\s*(\d+(?:\.\d+)?)")
    for ln in lines:
        qxr = qty_rate_re.search(ln)
        if qxr:
            qty = float(qxr.group(1))
            rate = float(qxr.group(2))
            amt = round(qty * rate, 2)
            name = qty_rate_re.sub("", ln).strip(" -:|,")
            bill_items.append({"item_name": name or "UNKNOWN", "item_quantity": qty, "item_rate": rate, "item_amount": amt, "confidence": "high"})
            continue
        m = money_re.findall(ln)
        if m:
            chosen = m[-1][0]
            num = re.sub(r"[^\d\.]", "", chosen)
            try:
                val = float(num) if num else 0.0
            except:
                val = 0.0
            name = ln.replace(chosen, "").strip(" -:|,")
            bill_items.append({"item_name": name or "UNKNOWN", "item_quantity": None, "item_rate": None, "item_amount": round(val, 2), "confidence": "medium"})
            continue
    total_item_count = len(bill_items)
    reconciled_amount = sum([bi.get("item_amount", 0.0) for bi in bill_items])
    return {"page_no": str(page_no), "page_type": page_type, "bill_items": bill_items, "total_item_count": total_item_count, "reconciled_amount": round(reconciled_amount, 2)}

def aggregate_pages(pages: List[Dict]) -> Dict:
    pagewise = []
    total_items = 0
    total_amount = 0.0
    for p in pages:
        page_dict = {"page_no": p.get("page_no", "1"), "page_type": p.get("page_type", "Bill Detail"), "bill_items": []}
        for bi in p.get("bill_items", []):
            page_dict["bill_items"].append({
                "item_name": bi.get("item_name"),
                "item_amount": float(bi.get("item_amount") or 0.0),
                "item_rate": float(bi.get("item_rate")) if bi.get("item_rate") is not None else None,
                "item_quantity": float(bi.get("item_quantity")) if bi.get("item_quantity") is not None else None
            })
            total_items += 1
            total_amount += float(bi.get("item_amount") or 0.0)
        if "fraud_flags" in p:
            page_dict["fraud_flags"] = p["fraud_flags"]
        pagewise.append(page_dict)
    return {"pagewise_line_items": pagewise, "total_item_count": total_items, "reconciled_amount": round(total_amount, 2)}

def ocr_and_extract_from_images(images) -> List[Dict]:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {}
        for i, img in enumerate(images, start=1):
            futures[ex.submit(process_single_image, img, i)] = i
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception:
                idx = futures[fut]
                results.append({"page_no": str(idx), "page_type": "Bill Detail", "bill_items": []})
    return sorted(results, key=lambda x: int(x.get("page_no", 1)))

def process_single_image(img, page_no):
    try:
        text = image_to_text(img)
    except Exception:
        text = ""
    try:
        from src import preprocessing_helpers as ph
        preproc_flags = ph.detect_whiteout_and_lowconf(img, text)
    except Exception:
        preproc_flags = []
    page_result = simple_page_extractor(text, page_no=page_no)
    if preproc_flags:
        page_result["fraud_flags"] = [{"flag_type": f[1], "score": f[2], "meta": f[3] if len(f) > 3 else {}} for f in preproc_flags]
    return page_result

def call_custom_extractor(ocr_texts: List[str]) -> Optional[Dict]:
    try:
        from src.extractor.bill_extractor import extract_bill_data
        res = extract_bill_data(ocr_texts)
        return res
    except Exception:
        return None

class DocumentRequest(BaseModel):
    document: Optional[str]

@app.get("/")
def root():
    return {"status": "OK", "message": "Bajaj Datathon API running"}

# Explicit favicon route
@app.get("/favicon.ico")
async def favicon():
    path = os.path.join("static", "favicon.ico")
    if os.path.exists(path):
        return FileResponse(path)
    return Response(content=b"", media_type="image/x-icon")

@app.post("/api/v1/hackrx/run")
async def hackrx_run(request: Request, document: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    corr = str(uuid.uuid4())
    start = time.time()
    try:
        if file:
            content = await file.read()
        elif document:
            try:
                content = download_bytes(document)
            except Exception as e:
                return json_error(f"Failed to download document from URL: {e}", status=400, exc=e)
        else:
            return json_error("Missing 'document' or file upload", status=400)
        images = []
        try:
            if b"%PDF" in content[:4]:
                images = bytes_to_images(content)
            else:
                if Image is None:
                    raise RuntimeError("Pillow missing")
                images = [Image.open(io.BytesIO(content))]
        except Exception as e:
            return json_error(f"Failed to convert input to images: {e}", status=400, exc=e)
        ocr_texts = []
        for img in images:
            try:
                txt = image_to_text(img)
            except Exception:
                txt = ""
            ocr_texts.append(txt)
        custom = call_custom_extractor(ocr_texts)
        if custom:
            data = custom
        else:
            page_results = ocr_and_extract_from_images(images)
            data = aggregate_pages(page_results)
        response = {"is_success": True, "token_usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}, "data": data}
        return JSONResponse(status_code=200, content=response)
    except HTTPException as he:
        raise he
    except Exception as e:
        return json_error(f"Unexpected server error: {e}", status=500, exc=e)
    finally:
        elapsed = time.time() - start

@app.post("/extract-bill-file")
async def extract_bill_file(file: UploadFile = File(...)):
    content = await file.read()
    class DummyRequest: pass
    dummy_req = DummyRequest()
    dummy_req._state = {}
    return await hackrx_run(dummy_req, None, file)

@app.post("/extract-bill-url")
def extract_bill_url(payload: DocumentRequest):
    return JSONResponse(status_code=200, content={"is_success": False, "message": "Use /api/v1/hackrx/run or upload a file"})

@app.post("/debug/trace")
async def debug_trace(document: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    try:
        if file:
            content = await file.read()
        elif document:
            content = download_bytes(document)
        else:
            raise HTTPException(status_code=400, detail="No file or document provided")
        pages = 0
        if b"%PDF" in content[:4]:
            pages = len(bytes_to_images(content))
        else:
            pages = 1
        return {"ok": True, "pages": pages}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})
