"""
Batch Debug Tool for BFHL/HackRx Bill Extraction
Runs the standalone_debug extractor on a folder of PDFs
and produces:
- Per-PDF debug folder
- last_response.json
- summary CSV with item counts & totals
"""

import argparse
from pathlib import Path
import json
import csv
import traceback

from pdf2image import convert_from_path
from PIL import Image
import numpy as np

# Your extractor utilities
from src.extractor.bill_extractor import extract_bill_data_with_tsv
from src.utils.pdf_loader import load_pdf_to_images

###########################################################
#            PROCESS A SINGLE FILE
###########################################################
def process_pdf(pdf_path: Path, out_dir: Path):
    pdf_name = pdf_path.stem
    req_dir = out_dir / pdf_name
    req_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "file": str(pdf_path),
        "page_results": [],
        "n_pages": 0,
        "total_item_count": 0,
        "reconciled_amount": 0.0,
        "error": None
    }

    try:
        # Use the full extractor pipeline
        try:
            pages = load_pdf_to_images(pdf_path, dpi=300)
            result["n_pages"] = len(pages)
        except Exception as e:
            result["error"] = f"PDF loading failed: {str(e)}"
            return result
        
        if not pages:
            result["error"] = "No pages extracted from PDF"
            return result
        
        # Run extraction using the TSV-based pipeline
        try:
            extraction_result = extract_bill_data_with_tsv(pages, request_id=pdf_name)
        except Exception as e:
            import traceback
            result["error"] = f"Extraction failed: {str(e)}"
            # Save error traceback
            error_file = req_dir / "error.txt"
            error_file.write_text(traceback.format_exc(), encoding="utf-8")
            return result
        
        if not extraction_result.get("is_success", True):
            result["error"] = extraction_result.get("error", "Extraction failed")
            return result
        
        # Extract data from response
        data = extraction_result.get("data", {})
        pagewise_items = data.get("pagewise_line_items", [])
        
        grand_total = 0.0
        for page_data in pagewise_items:
            page_no = page_data.get("page_no", "1")
            bill_items = page_data.get("bill_items", [])
            reported_total = page_data.get("reported_total")
            
            # Calculate page total
            page_total = sum(item.get("item_amount", 0.0) for item in bill_items)
            
            result["page_results"].append(
                {
                    "page_no": int(page_no) if page_no.isdigit() else 1,
                    "items": bill_items,
                    "reported_total": reported_total,
                    "final_total": page_total,
                }
            )
            result["total_item_count"] += len(bill_items)
            grand_total += page_total

        result["reconciled_amount"] = grand_total

    except Exception as e:
        result["error"] = str(e)
        traceback.print_exc()

    # write per-file JSON
    out_json = req_dir / "last_response.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return result


###########################################################
#            MAIN ENTRY
###########################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder containing PDFs")
    parser.add_argument("--out", required=True, help="Output folder")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(in_dir.glob("*.pdf"))

    summary_rows = []

    for pdf in pdf_files:
        print(f"Processing: {pdf.name}")
        res = process_pdf(pdf, out_dir)

        # Calculate reported_total from page results
        reported_total = None
        for page_res in res.get("page_results", []):
            if page_res.get("reported_total"):
                reported_total = page_res.get("reported_total")
                break
        
        summary_rows.append(
            {
                "filename": pdf.name,
                "item_count": res.get("total_item_count", 0),
                "reported_total": reported_total or 0.0,
                "reconciled_total": res.get("reconciled_amount", 0.0),
                "extraction_success": res.get("error") is None and res.get("total_item_count", 0) > 0,
            }
        )

    # write CSV
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "item_count",
                "reported_total",
                "reconciled_total",
                "extraction_success",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # Print table summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"{'Filename':<40} {'Items':<8} {'Reported':<12} {'Reconciled':<12} {'Success':<8}")
    print("-" * 80)
    for row in summary_rows:
        success_str = "✓" if row["extraction_success"] else "✗"
        print(f"{row['filename']:<40} {row['item_count']:<8} {row['reported_total']:<12.2f} {row['reconciled_total']:<12.2f} {success_str:<8}")
    print("=" * 80)
    print(f"\nTotal files processed: {len(summary_rows)}")
    print(f"Successful extractions: {sum(1 for r in summary_rows if r['extraction_success'])}")
    print(f"Total items extracted: {sum(r['item_count'] for r in summary_rows)}")
    print(f"\n=== SUMMARY WRITTEN TO {csv_path} ===")


if __name__ == "__main__":
    main()
