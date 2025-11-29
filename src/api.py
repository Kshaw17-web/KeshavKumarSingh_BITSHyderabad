"""
Bajaj Finserv Datathon - Bill Extraction API
Optimized for speed and predictable latency with timeouts, caching, and metrics.
"""

import os
import uuid
import tempfile
import time
import hashlib
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import multiprocessing

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

# Schemas
from src.schemas import (
    DocumentRequest,
    BillItem,
    PageItems,
    PagewiseLineItem,
    DataResponse,
    ExtractionData,
    FullResponse
)

# New extractor (expects list of PIL Images)
try:
    from src.extractor.bill_extractor import extract_bill_data
except Exception:
    try:
        from extractor.bill_extractor import extract_bill_data
    except Exception:
        extract_bill_data = None

# New utilities
try:
    from src.utils.pdf_loader import load_pdf_to_images, load_image_file
    PDF_LOADER_AVAILABLE = True
except Exception:
    PDF_LOADER_AVAILABLE = False
    load_pdf_to_images = None
    load_image_file = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

# FastAPI app
app = FastAPI(
    title="Bajaj Finserv Datathon API",
    description="Bill extraction API with PaddleOCR, multiprocessing, and fraud detection",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory (optional, useful for other static assets too)
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Configuration
POPPLER_PATH = os.getenv("POPPLER_PATH", r"C:\poppler-25.11.0\Library\bin")
DEFAULT_DPI = int(os.getenv("OCR_DPI", "150"))  # Lowered from 300 to 150
PAGE_TIMEOUT = float(os.getenv("PAGE_TIMEOUT", "7.0"))  # Page-level timeout in seconds
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # Cache TTL in seconds (1 hour)
MAX_WORKERS = int(os.getenv("MAX_WORKERS", str(max(1, multiprocessing.cpu_count() - 1))))  # CPU count - 1

# Metrics tracking
_metrics_lock = threading.Lock()
_metrics = {
    "requests_served": 0,
    "latencies": [],  # Store last 1000 latencies for percentile calculation
    "cache_hits": 0,
    "cache_misses": 0,
    "timeout_errors": 0,
    "start_time": time.time()
}

# URL cache (lightweight in-memory cache with TTL)
_cache_lock = threading.Lock()
_url_cache: Dict[str, Dict[str, Any]] = {}


def _get_cache_key(url: str) -> str:
    """Generate cache key from URL."""
    return hashlib.md5(url.encode()).hexdigest()


def _get_cached_file(url: str) -> Optional[Path]:
    """
    Get cached file if available and not expired.
    
    Returns:
        Path to cached file or None if not found/expired
    """
    if not ENABLE_CACHE:
        return None
    
    cache_key = _get_cache_key(url)
    
    with _cache_lock:
        if cache_key in _url_cache:
            cache_entry = _url_cache[cache_key]
            # Check if expired
            if datetime.now() - cache_entry["timestamp"] < timedelta(seconds=CACHE_TTL):
                _metrics["cache_hits"] += 1
                return Path(cache_entry["file_path"])
            else:
                # Expired, remove from cache
                try:
                    if Path(cache_entry["file_path"]).exists():
                        Path(cache_entry["file_path"]).unlink()
                except Exception:
                    pass
                del _url_cache[cache_key]
    
    _metrics["cache_misses"] += 1
    return None


def _cache_file(url: str, file_path: Path):
    """Cache file path for URL."""
    if not ENABLE_CACHE:
        return
    
    cache_key = _get_cache_key(url)
    
    with _cache_lock:
        # Clean up old cache entries if cache is too large
        if len(_url_cache) > 100:
            # Remove oldest 20% of entries
            sorted_entries = sorted(_url_cache.items(), key=lambda x: x[1]["timestamp"])
            for key, _ in sorted_entries[:20]:
                try:
                    if Path(_url_cache[key]["file_path"]).exists():
                        Path(_url_cache[key]["file_path"]).unlink()
                except Exception:
                    pass
                del _url_cache[key]
        
        _url_cache[cache_key] = {
            "file_path": str(file_path),
            "timestamp": datetime.now()
        }


def _update_metrics(latency: float, timeout: bool = False):
    """Update metrics with request latency."""
    with _metrics_lock:
        _metrics["requests_served"] += 1
        if timeout:
            _metrics["timeout_errors"] += 1
        
        # Keep only last 1000 latencies for percentile calculation
        _metrics["latencies"].append(latency)
        if len(_metrics["latencies"]) > 1000:
            _metrics["latencies"] = _metrics["latencies"][-1000:]


def download_to_temp(url: str, timeout: int = 30) -> Path:
    """
    Download file from URL to temporary location with caching support.
    
    Args:
        url: URL to download
        timeout: Request timeout in seconds
        
    Returns:
        Path to temporary file
        
    Raises:
        HTTPException: If download fails
    """
    # Check cache first
    cached_file = _get_cached_file(url)
    if cached_file and cached_file.exists():
        return cached_file
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        resp = requests.get(url, timeout=timeout, stream=True, headers=headers)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document from URL: {e}")

    # Determine file extension
    content_type = resp.headers.get("content-type", "").lower()
    ext = ".pdf"
    if "image/png" in content_type:
        ext = ".png"
    elif "image/jpeg" in content_type or "image/jpg" in content_type:
        ext = ".jpg"
    else:
        # Try to infer from URL
        url_lower = url.lower()
        for guess in (".pdf", ".png", ".jpg", ".jpeg", ".tiff"):
            if url_lower.endswith(guess):
                ext = guess
                break

    # Save to temp file
    temp_dir = Path(tempfile.gettempdir())
    tmp = temp_dir / f"datathon_{uuid.uuid4().hex}{ext}"
    try:
        tmp.write_bytes(resp.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save temporary file: {e}")
    
    # Cache the file
    _cache_file(url, tmp)
    
    return tmp


def load_document_to_images(file_path: Path) -> List["Image.Image"]:
    """
    Load document (PDF or image) and convert to list of PIL Images.
    
    Args:
        file_path: Path to PDF or image file
        
    Returns:
        List of PIL Images (one per page)
        
    Raises:
        HTTPException: If loading fails
    """
    if not PIL_AVAILABLE:
        raise HTTPException(status_code=500, detail="PIL/Pillow is not installed")
    
    if not PDF_LOADER_AVAILABLE:
        raise HTTPException(status_code=500, detail="PDF loader utilities not available")
    
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == ".pdf":
            # Use optimized PDF loading with configurable DPI
            images = load_pdf_to_images(
                file_path,
                dpi=DEFAULT_DPI,
                poppler_path=POPPLER_PATH if Path(POPPLER_PATH).exists() else None,
                use_grayscale=False,  # Keep color for better OCR
                thread_count=4  # Multi-threaded Poppler conversion
            )
            if not images:
                raise HTTPException(status_code=400, detail="PDF conversion produced no images")
            return images
        else:
            # Single image file
            img = load_image_file(file_path)
            return [img]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load document: {e}")


def extract_with_timeout(images: List["Image.Image"], timeout: float) -> Dict[str, Any]:
    """
    Extract bill data with page-level timeout and graceful fallback.
    
    Args:
        images: List of PIL Images
        timeout: Timeout per page in seconds
        
    Returns:
        Extraction result (may be partial if timeout occurs)
    """
    if not extract_bill_data:
        raise HTTPException(status_code=500, detail="Extractor not available")
    
    # Process pages with timeout
    total_timeout = timeout * len(images)
    
    executor = ThreadPoolExecutor(max_workers=1)  # Single worker for extraction
    future = executor.submit(extract_bill_data, images)
    
    try:
        result = future.result(timeout=total_timeout)
        return result
    except FutureTimeoutError:
        # Timeout occurred - try to get partial result
        # Process pages individually with timeout
        pagewise_items = []
        total_item_count = 0
        total_amount = 0.0
        
        for page_idx, img in enumerate(images, start=1):
            try:
                page_future = executor.submit(extract_bill_data, [img])
                page_result = page_future.result(timeout=timeout)
                
                if isinstance(page_result, dict):
                    page_items = page_result.get("pagewise_line_items", [])
                    if page_items:
                        pagewise_items.extend(page_items)
                        total_item_count += page_result.get("total_item_count", 0)
                        total_amount += page_result.get("reconciled_amount", 0.0)
            except FutureTimeoutError:
                # Page timed out - add empty page result
                pagewise_items.append({
                    "page_no": str(page_idx),
                    "page_type": "Bill Detail",
                    "bill_items": [],
                    "fraud_flags": [],
                    "reported_total": None,
                    "reconciliation_ok": None,
                    "reconciliation_relative_error": None
                })
            except Exception:
                # Error on page - add empty result
                pagewise_items.append({
                    "page_no": str(page_idx),
                    "page_type": "Bill Detail",
                    "bill_items": [],
                    "fraud_flags": [],
                    "reported_total": None,
                    "reconciliation_ok": None,
                    "reconciliation_relative_error": None
                })
        
        return {
            "pagewise_line_items": pagewise_items,
            "total_item_count": total_item_count,
            "reconciled_amount": round(total_amount, 2)
        }
    finally:
        executor.shutdown(wait=False)


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "OK",
        "message": "Bajaj Datathon API running",
        "version": "2.1.0",
        "extractor_available": extract_bill_data is not None,
        "config": {
            "default_dpi": DEFAULT_DPI,
            "page_timeout": PAGE_TIMEOUT,
            "cache_enabled": ENABLE_CACHE,
            "max_workers": MAX_WORKERS
        }
    }


@app.get("/favicon.ico")
async def favicon():
    """Explicit favicon route."""
    path = os.path.join("static", "favicon.ico")
    if os.path.exists(path):
        return FileResponse(path)
    return Response(content=b"", media_type="image/x-icon")


@app.get("/metrics")
def metrics():
    """
    Metrics endpoint returning average latency, 95th percentile latency, and requests served.
    """
    with _metrics_lock:
        latencies = _metrics["latencies"].copy()
        requests_served = _metrics["requests_served"]
        cache_hits = _metrics["cache_hits"]
        cache_misses = _metrics["cache_misses"]
        timeout_errors = _metrics["timeout_errors"]
        uptime = time.time() - _metrics["start_time"]
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        sorted_latencies = sorted(latencies)
        p95_index = int(len(sorted_latencies) * 0.95)
        p95_latency = sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else sorted_latencies[-1]
        p99_latency = sorted_latencies[int(len(sorted_latencies) * 0.99)] if len(sorted_latencies) > 0 else 0.0
    else:
        avg_latency = 0.0
        p95_latency = 0.0
        p99_latency = 0.0
    
    cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0.0
    
    return {
        "requests_served": requests_served,
        "avg_latency_seconds": round(avg_latency, 3),
        "p95_latency_seconds": round(p95_latency, 3),
        "p99_latency_seconds": round(p99_latency, 3),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": round(cache_hit_rate, 3),
        "timeout_errors": timeout_errors,
        "uptime_seconds": round(uptime, 1),
        "requests_per_second": round(requests_served / uptime, 2) if uptime > 0 else 0.0
    }


@app.post("/api/v1/hackrx/run", response_model=FullResponse)
def hackrx_run(request: DocumentRequest):
    """
    Main extraction endpoint. Accepts document URL and returns structured bill data.
    Optimized with timeouts, caching, and parallel processing.
    
    Args:
        request: DocumentRequest with document URL
        
    Returns:
        FullResponse with HackRx-compatible structure
    """
    start_time = time.time()
    timeout_occurred = False
    
    if not request.document or not request.document.strip():
        raise HTTPException(status_code=400, detail="Document URL is required")
    
    if extract_bill_data is None:
        raise HTTPException(
            status_code=500,
            detail="Extractor not available. Check that src/extractor/bill_extractor.py is properly configured."
        )
    
    temp_file: Optional[Path] = None
    
    try:
        # Download document (with caching)
        temp_file = download_to_temp(request.document)
        
        # Load document as images
        images = load_document_to_images(temp_file)
        
        if not images:
            raise HTTPException(status_code=400, detail="No images extracted from document")
        
        # Run extraction with timeout
        try:
            extraction_result = extract_with_timeout(images, PAGE_TIMEOUT)
        except Exception as e:
            # If extraction fails, try to return partial result
            timeout_occurred = True
            extraction_result = {
                "pagewise_line_items": [
                    {
                        "page_no": str(i + 1),
                        "page_type": "Bill Detail",
                        "bill_items": [],
                        "fraud_flags": [],
                        "reported_total": None,
                        "reconciliation_ok": None,
                        "reconciliation_relative_error": None
                    }
                    for i in range(len(images))
                ],
                "total_item_count": 0,
                "reconciled_amount": 0.0
            }
        
        if not isinstance(extraction_result, dict):
            raise HTTPException(status_code=500, detail="Extractor returned invalid result format")
        
        # Convert to Pydantic models
        pagewise_items = []
        all_amounts = []
        
        for page_data in extraction_result.get("pagewise_line_items", []):
            # Convert bill items
            bill_items = []
            for bi in page_data.get("bill_items", []):
                try:
                    bill_item = BillItem(
                        item_name=str(bi.get("item_name", "UNKNOWN")).strip(),
                        item_amount=float(bi.get("item_amount", 0.0)),
                        item_rate=float(bi.get("item_rate")) if bi.get("item_rate") is not None else None,
                        item_quantity=float(bi.get("item_quantity")) if bi.get("item_quantity") is not None else None,
                        confidence=str(bi.get("confidence", "medium")) if bi.get("confidence") else None
                    )
                    bill_items.append(bill_item)
                    all_amounts.append(bill_item.item_amount)
                except (ValueError, TypeError) as e:
                    # Skip invalid items
                    continue
            
            # Create page item
            page_item = PageItems(
                page_no=str(page_data.get("page_no", "1")),
                page_type=str(page_data.get("page_type", "Bill Detail")),
                bill_items=bill_items,
                reported_total=float(page_data.get("reported_total")) if page_data.get("reported_total") is not None else None,
                reconciliation_ok=page_data.get("reconciliation_ok"),
                reconciliation_relative_error=float(page_data.get("reconciliation_relative_error")) if page_data.get("reconciliation_relative_error") is not None else None,
                preprocessing=None,  # Can be added if needed
                fraud_flags=page_data.get("fraud_flags", [])
            )
            pagewise_items.append(page_item)
        
        # Calculate reconciled amount
        reconciled_amount = round(sum(all_amounts), 2) if all_amounts else float(extraction_result.get("reconciled_amount", 0.0))
        
        # Create response
        data_response = DataResponse(
            pagewise_line_items=pagewise_items,
            total_item_count=int(extraction_result.get("total_item_count", len(all_amounts))),
            reconciled_amount=reconciled_amount
        )
        
        response = FullResponse(
            is_success=True,
            token_usage={"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
            data=data_response
        )
        
        # Update metrics
        latency = time.time() - start_time
        _update_metrics(latency, timeout=timeout_occurred)
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        latency = time.time() - start_time
        _update_metrics(latency, timeout=False)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    finally:
        # Cleanup temporary file (only if not cached)
        if temp_file and temp_file.exists():
            cache_key = _get_cache_key(request.document)
            with _cache_lock:
                if cache_key not in _url_cache:
                    # Not in cache, safe to delete
                    try:
                        temp_file.unlink()
                    except Exception:
                        pass


@app.post("/extract-bill-data")
def simple_extract(payload: DocumentRequest):
    """
    Alias endpoint for backward compatibility.
    """
    return hackrx_run(payload)
