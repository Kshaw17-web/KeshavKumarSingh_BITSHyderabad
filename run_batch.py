"""
High-performance batch processing script with profiling and optimizations.
Uses optimized PaddleOCR pipeline with fraud detection.
"""

import sys
import os
import json
import csv
import time
import shutil
import zipfile
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(ROOT))

# Import new extractor
try:
    from src.extractor.bill_extractor import extract_bill_data, extract_bill_data_with_tsv
except Exception:
    try:
        from extractor.bill_extractor import extract_bill_data, extract_bill_data_with_tsv
    except Exception:
        print("ERROR: Could not import extract_bill_data")
        sys.exit(1)

# Import new utilities
try:
    from src.utils.pdf_loader import load_pdf_to_images
    PDF_LOADER_AVAILABLE = True
except Exception:
    print("ERROR: Could not import pdf_loader")
    PDF_LOADER_AVAILABLE = False
    sys.exit(1)

try:
    from src.preprocessing.fraud_filters import detect_fraud_flags, compute_unified_fraud_score
    FRAUD_DETECTION_AVAILABLE = True
except Exception:
    print("WARNING: Fraud detection not available")
    FRAUD_DETECTION_AVAILABLE = False
    def detect_fraud_flags(img, save_debug_maps=False, debug_output_dir=None):
        return [], {}
    def compute_unified_fraud_score(flags):
        return 0.0

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    print("ERROR: PIL/Pillow not available")
    PIL_AVAILABLE = False
    sys.exit(1)

# Configuration
DATA_RAW = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "local_test_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DEBUG_MASKS_DIR = OUTPUT_DIR / "debug_masks"
DEBUG_MASKS_DIR.mkdir(exist_ok=True)
FRAUD_CSV = OUTPUT_DIR / "fraud_report.csv"

POPPLER_PATH = os.environ.get("POPPLER_PATH", r"C:\poppler-25.11.0\Library\bin")
DEFAULT_DPI = int(os.environ.get("OCR_DPI", "300"))
DEBUG_MODE = os.environ.get("DEBUG", "False").lower() == "true"
ENABLE_PROFILING = os.environ.get("PROFILE", "True").lower() == "true"


def find_training_pdfs() -> List[Path]:
    """Find all training PDF files."""
    pdfs = []
    training_dir = DATA_RAW / "training_samples" / "TRAINING_SAMPLES"
    if training_dir.exists():
        for f in training_dir.rglob("*.pdf"):
            pdfs.append(f)
    else:
        training_dir = DATA_RAW / "training_samples"
        if training_dir.exists():
            for f in training_dir.rglob("*.pdf"):
                pdfs.append(f)
        else:
            for f in DATA_RAW.rglob("*.pdf"):
                if "train_sample" in f.name.lower():
                    pdfs.append(f)
    return sorted(set(pdfs))


def save_json_output(output_path: Path, payload: Any):
    """Save JSON output to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def save_debug_mask(mask_array, filename: str):
    """Save debug mask image."""
    try:
        import cv2
        mask_path = DEBUG_MASKS_DIR / filename
        cv2.imwrite(str(mask_path), mask_array)
        return str(mask_path)
    except Exception:
        return None


def write_fraud_row(writer, filename: str, page_no: int, flag_type: str, score: float):
    """Write a row to fraud CSV."""
    writer.writerow({
        "file": filename,
        "page_no": page_no,
        "flag_type": flag_type,
        "flag_score": score,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


def process_pdf(pdf_path: Path, csv_writer, total_pdfs: int = 1):
    """Process a single PDF file with profiling and comprehensive debug logging."""
    print(f"Processing: {pdf_path.name}")
    pdf_start_time = time.time()
    step_times = {}
    
    # Generate request_id from PDF name for debug folder organization
    request_id = pdf_path.stem
    
    try:
        # Step 1: Load PDF as images (optimized)
        t0 = time.time()
        use_grayscale = total_pdfs > 10  # Use grayscale for batch processing
        images = load_pdf_to_images(
            pdf_path,
            dpi=DEFAULT_DPI,
            poppler_path=POPPLER_PATH if Path(POPPLER_PATH).exists() else None,
            use_grayscale=use_grayscale,
            thread_count=4
        )
        step_times["pdf_to_images"] = time.time() - t0
        
        if not images:
            print(f"  WARNING: No images extracted from {pdf_path.name}")
            return
        
        print(f"  Loaded {len(images)} pages in {step_times['pdf_to_images']:.2f}s")
        
        # Step 2: Run extraction with TSV pipeline (includes debug logging)
        t1 = time.time()
        extraction_result = extract_bill_data_with_tsv(images, request_id=request_id)
        step_times["extraction"] = time.time() - t1
        
        # Log profiling if available
        if ENABLE_PROFILING and "_profiling" in extraction_result:
            prof = extraction_result.pop("_profiling")
            print(f"  Extraction timing:")
            for step, duration in prof.items():
                print(f"    {step}: {duration:.3f}s")
                step_times[f"extraction_{step}"] = duration
        
        # Step 3: Fraud detection (only if DEBUG mode or small PDFs)
        t2 = time.time()
        use_fast_mode = len(images) >= 20
        
        for page_idx, img in enumerate(images, start=1):
            if FRAUD_DETECTION_AVAILABLE:
                try:
                    # Only save debug maps in DEBUG mode
                    save_debug = DEBUG_MODE
                    debug_output_dir = None
                    if save_debug:
                        page_debug_dir = DEBUG_MASKS_DIR / f"{pdf_path.stem}_p{page_idx}"
                        page_debug_dir.mkdir(parents=True, exist_ok=True)
                        debug_output_dir = str(page_debug_dir)
                    
                    # Run fraud detection (fast mode for large PDFs)
                    flags, debug_maps = detect_fraud_flags(
                        img,
                        save_debug_maps=save_debug,
                        debug_output_dir=debug_output_dir,
                        use_fast_mode=use_fast_mode
                    )
                    
                    # Compute unified fraud score
                    unified_score = compute_unified_fraud_score(flags)
                    
                    # Write individual flags to CSV
                    for flag in flags:
                        write_fraud_row(
                            csv_writer,
                            pdf_path.name,
                            page_idx,
                            flag.get("flag_type", "unknown"),
                            float(flag.get("score", 0.0))
                        )
                    
                    # Write unified score
                    write_fraud_row(
                        csv_writer,
                        pdf_path.name,
                        page_idx,
                        "unified_fraud_score",
                        unified_score
                    )
                    
                    # Save visualization maps only in DEBUG mode
                    if save_debug and debug_maps:
                        try:
                            import cv2
                            for map_name, map_array in debug_maps.items():
                                if map_array is not None and map_array.size > 0:
                                    map_path = page_debug_dir / f"{map_name}.png"
                                    cv2.imwrite(str(map_path), map_array)
                        except Exception:
                            pass
                    
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"  WARNING: Fraud detection failed for page {page_idx}: {e}")
        
        step_times["fraud_detection"] = time.time() - t2
        
        # Step 4: Copy debug folder to output directory for easy review
        t3 = time.time()
        debug_source_dir = Path("logs") / request_id
        if debug_source_dir.exists():
            # Copy entire request folder to output directory
            debug_dest_dir = OUTPUT_DIR / "debug_logs" / request_id
            try:
                if debug_dest_dir.exists():
                    shutil.rmtree(debug_dest_dir)
                shutil.copytree(debug_source_dir, debug_dest_dir)
                print(f"  ✓ Debug logs copied to: {debug_dest_dir}")
            except Exception as e:
                if DEBUG_MODE:
                    print(f"  WARNING: Could not copy debug logs: {e}")
        
        # Step 5: Save output (in-memory buffer, write once)
        output_obj = {
            "source_file": str(pdf_path),
            "extraction_result": extraction_result,
            "n_pages": len(images),
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "debug_logs_path": str(debug_dest_dir) if debug_dest_dir else None
        }
        
        if ENABLE_PROFILING:
            output_obj["timing"] = {
                **step_times,
                "total_time": time.time() - pdf_start_time
            }
        
        # Save JSON output
        output_path = OUTPUT_DIR / f"{pdf_path.stem}_output.json"
        save_json_output(output_path, output_obj)
        step_times["file_io"] = time.time() - t3
        
        total_time = time.time() - pdf_start_time
        print(f"  ✓ Completed in {total_time:.2f}s ({len(images)} pages, {total_time/len(images):.2f}s/page avg)")
        if ENABLE_PROFILING:
            print(f"    Timing breakdown: PDF={step_times.get('pdf_to_images', 0):.2f}s, "
                  f"Extraction={step_times.get('extraction', 0):.2f}s, "
                  f"Fraud={step_times.get('fraud_detection', 0):.2f}s")
        
    except Exception as e:
        import traceback
        error_path = OUTPUT_DIR / f"{pdf_path.stem}_error.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf8")
        print(f"  ERROR processing {pdf_path.name}: {e}")
        print(f"  See: {error_path}")


def create_debug_archive():
    """Create a zip archive of all debug logs for easy review."""
    debug_logs_dir = OUTPUT_DIR / "debug_logs"
    if not debug_logs_dir.exists():
        return None
    
    archive_path = OUTPUT_DIR / "debug_logs_archive.zip"
    try:
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(debug_logs_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(OUTPUT_DIR)
                    zipf.write(file_path, arcname)
        print(f"  ✓ Created debug archive: {archive_path}")
        return str(archive_path)
    except Exception as e:
        if DEBUG_MODE:
            print(f"  WARNING: Could not create debug archive: {e}")
        return None


def main():
    """Main batch processing function with profiling and comprehensive debug logging."""
    global pdfs  # For use_grayscale check
    
    batch_start_time = time.time()
    
    # Initialize output directories
    (OUTPUT_DIR / "debug_logs").mkdir(exist_ok=True)
    
    # Initialize fraud CSV
    if FRAUD_CSV.exists():
        FRAUD_CSV.unlink()
    
    csv_file = open(FRAUD_CSV, "w", newline="", encoding="utf8")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=["file", "page_no", "flag_type", "flag_score", "timestamp"]
    )
    csv_writer.writeheader()
    
    # Find PDFs
    pdfs = find_training_pdfs()
    if not pdfs:
        print(f"No training PDFs found under: {DATA_RAW}")
        csv_file.close()
        return
    
    print(f"Found {len(pdfs)} PDF files to process")
    print(f"Profiling: {ENABLE_PROFILING}, Debug mode: {DEBUG_MODE}")
    print(f"Debug logs will be saved to: {OUTPUT_DIR / 'debug_logs'}")
    print("-" * 60)
    
    # Process each PDF
    for pdf in pdfs:
        try:
            process_pdf(pdf, csv_writer, total_pdfs=len(pdfs))
        except Exception as e:
            print(f"Fatal error processing {pdf.name}: {e}")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()
    
    csv_file.close()
    
    # Create debug archive for easy review
    archive_path = create_debug_archive()
    
    total_batch_time = time.time() - batch_start_time
    print("-" * 60)
    print(f"\nBatch processing complete!")
    print(f"Total time: {total_batch_time:.2f}s")
    print(f"Average per PDF: {total_batch_time/len(pdfs):.2f}s")
    print(f"Outputs: {OUTPUT_DIR}")
    print(f"Fraud report: {FRAUD_CSV}")
    print(f"Debug logs: {OUTPUT_DIR / 'debug_logs'}")
    if archive_path:
        print(f"Debug archive: {archive_path}")
    if DEBUG_MODE:
        print(f"Debug masks: {DEBUG_MASKS_DIR}")


if __name__ == "__main__":
    main()
