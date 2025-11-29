"""
Create a comprehensive test report for the 15 training samples.
Shows what was extracted, what failed, and why.
"""
import sys
from pathlib import Path
import json
import csv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def analyze_batch_results(results_dir: Path):
    """Analyze batch test results and create a detailed report."""
    results_dir = Path(results_dir)
    summary_csv = results_dir / "summary.csv"
    
    if not summary_csv.exists():
        print(f"ERROR: Summary CSV not found: {summary_csv}")
        return
    
    # Read summary
    results = []
    with open(summary_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    
    # Analyze each PDF
    report = []
    for result in results:
        filename = result['filename']
        pdf_dir = results_dir / filename.replace('.pdf', '')
        last_response = pdf_dir / "last_response.json"
        
        analysis = {
            "filename": filename,
            "pages": int(result.get('n_pages', 0)),
            "items_extracted": int(result.get('item_count', 0)),
            "reconciled_total": float(result.get('reconciled_total', 0)),
            "success": result.get('extraction_success', 'False') == 'True',
            "fraud_flags": result.get('fraud_flags', 'None'),
            "issues": []
        }
        
        # Check for OCR output
        ocr_files = list(pdf_dir.rglob('*_ocr.json'))
        if ocr_files:
            try:
                ocr_data = json.loads(ocr_files[0].read_text(encoding='utf-8'))
                text_tokens = [t for t in ocr_data.get('text', []) if t.strip()]
                analysis['ocr_tokens'] = len(text_tokens)
                analysis['has_ocr'] = True
                
                # Check if OCR found any amounts
                amounts_found = []
                for token in text_tokens:
                    # Look for numeric patterns
                    import re
                    if re.search(r'\d+[.,]\d+', token) or (token.replace(',', '').replace('.', '').isdigit() and len(token) > 1):
                        try:
                            clean = token.replace('₹', '').replace('Rs', '').replace(',', '').replace('$', '')
                            val = float(''.join(c for c in clean if c.isdigit() or c in '.-'))
                            if 1 <= val <= 1000000:
                                amounts_found.append(val)
                        except:
                            pass
                
                analysis['amounts_detected'] = len(amounts_found)
                if amounts_found:
                    analysis['sample_amounts'] = amounts_found[:5]
            except Exception as e:
                analysis['has_ocr'] = False
                analysis['ocr_error'] = str(e)
        else:
            analysis['has_ocr'] = False
            analysis['issues'].append("No OCR output found")
        
        # Check last_response for details
        if last_response.exists():
            try:
                response_data = json.loads(last_response.read_text(encoding='utf-8'))
                if response_data.get('error'):
                    analysis['issues'].append(f"Error: {response_data['error']}")
                
                # Check page results
                page_results = response_data.get('page_results', [])
                for page_res in page_results:
                    items = page_res.get('items', [])
                    if items:
                        analysis['issues'].append(f"Page {page_res.get('page_no')}: {len(items)} items parsed but filtered out")
            except Exception as e:
                analysis['issues'].append(f"Failed to read last_response: {e}")
        
        # Determine root cause
        if analysis['items_extracted'] == 0:
            if not analysis.get('has_ocr', False):
                analysis['root_cause'] = "OCR failed or no text detected"
            elif analysis.get('amounts_detected', 0) == 0:
                analysis['root_cause'] = "No amounts detected in OCR (might be handwritten/whitener issue)"
            elif analysis.get('amounts_detected', 0) > 0:
                analysis['root_cause'] = "Amounts detected but items not parsed (parser/column detection issue)"
            else:
                analysis['root_cause'] = "Unknown - needs manual inspection"
        else:
            analysis['root_cause'] = "Extraction successful"
        
        report.append(analysis)
    
    # Print report
    print("\n" + "="*100)
    print("COMPREHENSIVE TEST REPORT - 15 TRAINING SAMPLES")
    print("="*100)
    print(f"\n{'Filename':<30} {'Pages':<7} {'Items':<7} {'OCR':<6} {'Amounts':<8} {'Root Cause':<40}")
    print("-"*100)
    
    for r in report:
        ocr_str = "Yes" if r.get('has_ocr') else "No"
        amounts_str = str(r.get('amounts_detected', 0))
        print(f"{r['filename']:<30} {r['pages']:<7} {r['items_extracted']:<7} {ocr_str:<6} {amounts_str:<8} {r.get('root_cause', 'Unknown'):<40}")
    
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    print(f"Total PDFs: {len(report)}")
    print(f"Successful extractions: {sum(1 for r in report if r['items_extracted'] > 0)}")
    print(f"Failed extractions: {sum(1 for r in report if r['items_extracted'] == 0)}")
    print(f"PDFs with OCR output: {sum(1 for r in report if r.get('has_ocr', False))}")
    print(f"PDFs with amounts detected: {sum(1 for r in report if r.get('amounts_detected', 0) > 0)}")
    print(f"Total items extracted: {sum(r['items_extracted'] for r in report)}")
    print(f"Total reconciled amount: {sum(r['reconciled_total'] for r in report):.2f}")
    
    # Save detailed report
    report_json = results_dir / "detailed_report.json"
    with open(report_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed report saved to: {report_json}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Directory containing batch test results")
    args = parser.parse_args()
    
    analyze_batch_results(Path(args.results_dir))

