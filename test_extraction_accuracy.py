"""
Unit tests for extraction accuracy with precision/recall metrics.
Compares heuristic vs ensemble (heuristic + LayoutLMv3) results.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from src.extractor.bill_extractor import extract_bill_data
    from src.utils.pdf_loader import load_pdf_to_images
    EXTRACTOR_AVAILABLE = True
except Exception as e:
    print(f"ERROR: Could not import extractor: {e}")
    EXTRACTOR_AVAILABLE = False
    sys.exit(1)

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    fuzz = None


def load_ground_truth(pdf_path: Path) -> Dict[str, Any]:
    """
    Load ground truth annotations for a PDF.
    Expected format: {pdf_name}_ground_truth.json
    {
        "pages": [
            {
                "page_no": "1",
                "items": [
                    {"item_name": "...", "item_amount": 123.45, ...}
                ]
            }
        ]
    }
    """
    gt_path = pdf_path.parent / f"{pdf_path.stem}_ground_truth.json"
    if not gt_path.exists():
        return None
    
    try:
        with open(gt_path, "r", encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return None


def normalize_item_name(name: str) -> str:
    """Normalize item name for comparison."""
    if not name:
        return ""
    # Lowercase, strip, remove extra spaces
    normalized = " ".join(name.lower().strip().split())
    # Remove common prefixes/suffixes
    normalized = normalized.replace("item:", "").replace("item", "").strip()
    return normalized


def match_item(
    predicted: Dict[str, Any],
    ground_truth: Dict[str, Any],
    name_threshold: float = 0.8,
    amount_tolerance: float = 0.01
) -> Tuple[bool, float]:
    """
    Check if predicted item matches ground truth.
    
    Returns:
        (is_match, confidence_score)
    """
    pred_name = normalize_item_name(str(predicted.get("item_name", "")))
    gt_name = normalize_item_name(str(ground_truth.get("item_name", "")))
    
    if not pred_name or not gt_name:
        return False, 0.0
    
    # Fuzzy name matching
    if RAPIDFUZZ_AVAILABLE:
        name_similarity = fuzz.ratio(pred_name, gt_name) / 100.0
    else:
        name_similarity = 1.0 if pred_name == gt_name else 0.0
    
    if name_similarity < name_threshold:
        return False, name_similarity
    
    # Amount matching
    pred_amount = float(predicted.get("item_amount", 0.0))
    gt_amount = float(ground_truth.get("item_amount", 0.0))
    
    if abs(pred_amount - gt_amount) / max(abs(pred_amount), abs(gt_amount), 0.01) <= amount_tolerance:
        return True, name_similarity * 0.7 + 0.3  # Weighted confidence
    
    # Absolute tolerance for small amounts
    if abs(pred_amount - gt_amount) <= 0.1:
        return True, name_similarity * 0.7 + 0.3
    
    return False, name_similarity * 0.5


def calculate_metrics(
    predicted_items: List[Dict[str, Any]],
    ground_truth_items: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 for item extraction.
    
    Returns:
        {
            "precision": float,
            "recall": float,
            "f1": float,
            "name_precision": float,
            "name_recall": float,
            "amount_precision": float,
            "amount_recall": float
        }
    """
    if not ground_truth_items:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "name_precision": 0.0,
            "name_recall": 0.0,
            "amount_precision": 0.0,
            "amount_recall": 0.0
        }
    
    # Match predicted items to ground truth
    matched_gt = set()
    matched_pred = set()
    matches = []
    
    for i, pred_item in enumerate(predicted_items):
        best_match_idx = None
        best_score = 0.0
        
        for j, gt_item in enumerate(ground_truth_items):
            if j in matched_gt:
                continue
            
            is_match, score = match_item(pred_item, gt_item)
            if is_match and score > best_score:
                best_score = score
                best_match_idx = j
        
        if best_match_idx is not None:
            matches.append((i, best_match_idx, best_score))
            matched_pred.add(i)
            matched_gt.add(best_match_idx)
    
    # Calculate metrics
    tp = len(matches)  # True positives
    fp = len(predicted_items) - tp  # False positives
    fn = len(ground_truth_items) - tp  # False negatives
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Name-only metrics (exact match)
    name_matches = 0
    for i, pred_item in enumerate(predicted_items):
        pred_name = normalize_item_name(str(pred_item.get("item_name", "")))
        for gt_item in ground_truth_items:
            gt_name = normalize_item_name(str(gt_item.get("item_name", "")))
            if pred_name == gt_name or (RAPIDFUZZ_AVAILABLE and fuzz.ratio(pred_name, gt_name) > 90):
                name_matches += 1
                break
    
    name_precision = name_matches / len(predicted_items) if predicted_items else 0.0
    name_recall = name_matches / len(ground_truth_items) if ground_truth_items else 0.0
    
    # Amount-only metrics (within tolerance)
    amount_matches = 0
    for i, pred_item in enumerate(predicted_items):
        pred_amount = float(pred_item.get("item_amount", 0.0))
        for gt_item in ground_truth_items:
            gt_amount = float(gt_item.get("item_amount", 0.0))
            if abs(pred_amount - gt_amount) / max(abs(pred_amount), abs(gt_amount), 0.01) <= 0.01:
                amount_matches += 1
                break
    
    amount_precision = amount_matches / len(predicted_items) if predicted_items else 0.0
    amount_recall = amount_matches / len(ground_truth_items) if ground_truth_items else 0.0
    
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "name_precision": round(name_precision, 3),
        "name_recall": round(name_recall, 3),
        "amount_precision": round(amount_precision, 3),
        "amount_recall": round(amount_recall, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


def test_pdf_extraction(pdf_path: Path) -> Dict[str, Any]:
    """
    Test extraction on a single PDF and compare with ground truth.
    
    Returns:
        Dictionary with metrics and results
    """
    print(f"\n{'='*60}")
    print(f"Testing: {pdf_path.name}")
    print(f"{'='*60}")
    
    # Load ground truth
    ground_truth = load_ground_truth(pdf_path)
    if not ground_truth:
        print(f"  WARNING: No ground truth found for {pdf_path.name}")
        print(f"  Expected: {pdf_path.parent / f'{pdf_path.stem}_ground_truth.json'}")
        return None
    
    # Load PDF and extract
    try:
        images = load_pdf_to_images(pdf_path, dpi=300)
        extraction_result = extract_bill_data(images, enable_profiling=False)
    except Exception as e:
        print(f"  ERROR: Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Compare results page by page
    all_metrics = []
    page_results = []
    
    for gt_page in ground_truth.get("pages", []):
        page_no = str(gt_page.get("page_no", "1"))
        gt_items = gt_page.get("items", [])
        
        # Find corresponding page in extraction result
        pred_page = None
        for p in extraction_result.get("pagewise_line_items", []):
            if str(p.get("page_no", "")) == page_no:
                pred_page = p
                break
        
        if not pred_page:
            print(f"  WARNING: Page {page_no} not found in extraction result")
            continue
        
        pred_items = pred_page.get("bill_items", [])
        
        # Calculate metrics
        metrics = calculate_metrics(pred_items, gt_items)
        all_metrics.append(metrics)
        page_results.append({
            "page_no": page_no,
            "predicted_count": len(pred_items),
            "ground_truth_count": len(gt_items),
            "metrics": metrics
        })
        
        print(f"\n  Page {page_no}:")
        print(f"    Predicted: {len(pred_items)} items")
        print(f"    Ground Truth: {len(gt_items)} items")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall: {metrics['recall']:.3f}")
        print(f"    F1: {metrics['f1']:.3f}")
        print(f"    Name Precision: {metrics['name_precision']:.3f}")
        print(f"    Name Recall: {metrics['name_recall']:.3f}")
        print(f"    Amount Precision: {metrics['amount_precision']:.3f}")
        print(f"    Amount Recall: {metrics['amount_recall']:.3f}")
    
    # Aggregate metrics
    if all_metrics:
        avg_metrics = {
            "precision": sum(m["precision"] for m in all_metrics) / len(all_metrics),
            "recall": sum(m["recall"] for m in all_metrics) / len(all_metrics),
            "f1": sum(m["f1"] for m in all_metrics) / len(all_metrics),
            "name_precision": sum(m["name_precision"] for m in all_metrics) / len(all_metrics),
            "name_recall": sum(m["name_recall"] for m in all_metrics) / len(all_metrics),
            "amount_precision": sum(m["amount_precision"] for m in all_metrics) / len(all_metrics),
            "amount_recall": sum(m["amount_recall"] for m in all_metrics) / len(all_metrics),
        }
        
        print(f"\n  Overall Metrics:")
        print(f"    Precision: {avg_metrics['precision']:.3f}")
        print(f"    Recall: {avg_metrics['recall']:.3f}")
        print(f"    F1: {avg_metrics['f1']:.3f}")
        print(f"    Name Precision: {avg_metrics['name_precision']:.3f}")
        print(f"    Name Recall: {avg_metrics['name_recall']:.3f}")
        print(f"    Amount Precision: {avg_metrics['amount_precision']:.3f}")
        print(f"    Amount Recall: {avg_metrics['amount_recall']:.3f}")
        
        return {
            "pdf_name": pdf_path.name,
            "page_results": page_results,
            "overall_metrics": avg_metrics,
            "extraction_result": extraction_result
        }
    
    return None


def main():
    """Main test driver."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test extraction accuracy")
    parser.add_argument("--pdf-dir", type=str, default="data/raw/training_samples",
                       help="Directory containing test PDFs")
    parser.add_argument("--pdf", type=str, default=None,
                       help="Single PDF file to test")
    args = parser.parse_args()
    
    # Find PDFs to test
    if args.pdf:
        pdf_paths = [Path(args.pdf)]
    else:
        pdf_dir = Path(args.pdf_dir)
        pdf_paths = list(pdf_dir.rglob("*.pdf"))
    
    if not pdf_paths:
        print(f"No PDFs found in {args.pdf_dir}")
        return
    
    print(f"Found {len(pdf_paths)} PDF(s) to test")
    
    # Test each PDF
    all_results = []
    for pdf_path in pdf_paths:
        result = test_pdf_extraction(pdf_path)
        if result:
            all_results.append(result)
    
    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        overall_precision = sum(r["overall_metrics"]["precision"] for r in all_results) / len(all_results)
        overall_recall = sum(r["overall_metrics"]["recall"] for r in all_results) / len(all_results)
        overall_f1 = sum(r["overall_metrics"]["f1"] for r in all_results) / len(all_results)
        
        print(f"\nAverage across {len(all_results)} PDF(s):")
        print(f"  Precision: {overall_precision:.3f}")
        print(f"  Recall: {overall_recall:.3f}")
        print(f"  F1: {overall_f1:.3f}")
        
        # Save results
        output_path = Path("test_results.json")
        with open(output_path, "w", encoding="utf8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()


