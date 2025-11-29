"""
CLI test script for fraud detection.
Usage: python test_fraud.py <pdf_path>
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from src.preprocessing_helpers import (
        detect_whiteout_and_lowconf,
        compute_combined_fraud_score,
        create_debug_overlay,
        _gaussian_whiteout_analysis,
        _compute_ela_map,
        inpaint_image,
        create_whiteout_mask
    )
    from src.utils.pdf_loader import load_pdf_to_images
    from src.utils.ocr_runner import run_ocr_parallel, extract_text_from_ocr_result
    FRAUD_DETECTION_AVAILABLE = True
except Exception as e:
    print(f"ERROR: Could not import fraud detection modules: {e}")
    FRAUD_DETECTION_AVAILABLE = False
    sys.exit(1)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    print("ERROR: PIL/Pillow not available")
    PIL_AVAILABLE = False
    sys.exit(1)

import numpy as np
import os


def process_pdf_for_fraud(
    pdf_path: Path,
    output_dir: Path,
    threshold: float = 0.5,
    enable_ocr: bool = True
) -> Dict[str, Any]:
    """
    Process PDF and detect fraud on each page.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save debug outputs
        threshold: Fraud score threshold
        enable_ocr: Enable OCR for low confidence detection
        
    Returns:
        Dictionary with results per page
    """
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path.name}")
    print(f"{'='*60}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_masks_dir = output_dir / "debug_masks"
    debug_masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Load PDF as images
    try:
        images = load_pdf_to_images(pdf_path, dpi=300)
        print(f"Loaded {len(images)} pages")
    except Exception as e:
        print(f"ERROR: Failed to load PDF: {e}")
        return {}
    
    # Run OCR if enabled
    ocr_texts = []
    if enable_ocr:
        try:
            print("Running OCR...")
            ocr_results = run_ocr_parallel(images, lang="en", max_workers=4)
            ocr_texts = extract_text_from_ocr_result(ocr_results)
            print(f"OCR completed for {len(ocr_texts)} pages")
        except Exception as e:
            print(f"WARNING: OCR failed: {e}")
            ocr_texts = [""] * len(images)
    else:
        ocr_texts = [""] * len(images)
    
    # Process each page
    results = {}
    suspicious_pages = []
    
    for page_idx, (img, ocr_text) in enumerate(zip(images, ocr_texts), start=1):
        print(f"\n--- Page {page_idx} ---")
        
        # Detect fraud
        flags = detect_whiteout_and_lowconf(img, ocr_text, enable_ela=True)
        
        # Compute combined score
        combined_score, is_suspicious = compute_combined_fraud_score(flags, threshold=threshold)
        
        # Print flags
        print(f"Flags detected: {len(flags)}")
        for flag in flags:
            flag_type = flag.get("flag_type", "")
            score = flag.get("score", 0.0)
            meta = flag.get("meta", {})
            print(f"  - {flag_type}: {score:.3f} {meta}")
        
        print(f"Combined score: {combined_score:.3f}")
        print(f"Suspicious: {'YES' if is_suspicious else 'NO'}")
        
        if is_suspicious:
            suspicious_pages.append(page_idx)
        
        # Generate debug visualizations
        try:
            # Get whiteout mask
            img_array = np.array(img.convert("L"))
            _, whiteout_mask = _gaussian_whiteout_analysis(img_array)
            
            # Get ELA map
            ela_map = _compute_ela_map(img)
            
            # Create overlay
            overlay = create_debug_overlay(img, flags, whiteout_mask, ela_map)
            
            # Save overlay
            overlay_path = debug_masks_dir / f"{pdf_path.stem}_p{page_idx}_overlay.png"
            overlay.save(overlay_path)
            print(f"  Saved overlay: {overlay_path.name}")
            
            # Save whiteout mask
            if whiteout_mask is not None and whiteout_mask.sum() > 0:
                mask_img = Image.fromarray(whiteout_mask)
                mask_path = debug_masks_dir / f"{pdf_path.stem}_p{page_idx}_whiteout_mask.png"
                mask_img.save(mask_path)
                print(f"  Saved whiteout mask: {mask_path.name}")
            
            # Save ELA heatmap
            if ela_map is not None:
                ela_uint8 = (ela_map * 255).astype(np.uint8)
                ela_img = Image.fromarray(ela_uint8)
                ela_path = debug_masks_dir / f"{pdf_path.stem}_p{page_idx}_ela_heatmap.png"
                ela_img.save(ela_path)
                print(f"  Saved ELA heatmap: {ela_path.name}")
            
            # Save inpainted image (if whiteout detected)
            if whiteout_mask is not None and whiteout_mask.sum() > 0:
                mask_pil = Image.fromarray(whiteout_mask)
                inpainted = inpaint_image(img, mask_pil)
                inpainted_path = debug_masks_dir / f"{pdf_path.stem}_p{page_idx}_inpainted.png"
                inpainted.save(inpainted_path)
                print(f"  Saved inpainted: {inpainted_path.name}")
        
        except Exception as e:
            print(f"  WARNING: Failed to generate debug images: {e}")
        
        # Store results
        results[page_idx] = {
            "flags": flags,
            "combined_score": combined_score,
            "is_suspicious": is_suspicious
        }
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total pages: {len(images)}")
    print(f"Suspicious pages: {len(suspicious_pages)}")
    if suspicious_pages:
        print(f"  Pages: {', '.join(map(str, suspicious_pages))}")
    
    avg_score = sum(r["combined_score"] for r in results.values()) / len(results) if results else 0.0
    print(f"Average fraud score: {avg_score:.3f}")
    
    return {
        "pdf_name": pdf_path.name,
        "total_pages": len(images),
        "suspicious_pages": suspicious_pages,
        "page_results": results,
        "output_dir": str(output_dir)
    }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Test fraud detection on PDF")
    parser.add_argument("pdf", type=str, help="Path to PDF file")
    parser.add_argument("--output-dir", type=str, default="local_test_outputs",
                       help="Output directory for debug images")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Fraud score threshold (default: 0.5)")
    parser.add_argument("--no-ocr", action="store_true",
                       help="Disable OCR (faster but less accurate)")
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    threshold = args.threshold
    enable_ocr = not args.no_ocr
    
    # Process PDF
    results = process_pdf_for_fraud(
        pdf_path,
        output_dir,
        threshold=threshold,
        enable_ocr=enable_ocr
    )
    
    print(f"\nDebug outputs saved to: {output_dir / 'debug_masks'}")
    print("Done!")


if __name__ == "__main__":
    main()


