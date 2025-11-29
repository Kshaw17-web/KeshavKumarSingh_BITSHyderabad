"""
Test script to verify API file upload works.
Run this after starting uvicorn: uvicorn src.api:app --reload --port 8000
"""

import requests
import sys
from pathlib import Path

def test_file_upload(pdf_path: str, api_url: str = "http://127.0.0.1:8000/api/v1/hackrx/run"):
    """Test file upload to API."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}")
        return False
    
    print(f"Testing upload of: {pdf_path.name}")
    
    try:
        with open(pdf_path, 'rb') as f:
            files = {'document': (pdf_path.name, f, 'application/pdf')}
            response = requests.post(api_url, files=files, timeout=60)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response JSON keys: {list(data.keys())}")
            print(f"is_success: {data.get('is_success')}")
            
            if data.get('is_success'):
                pagewise = data.get('data', {}).get('pagewise_line_items', [])
                print(f"Pages extracted: {len(pagewise)}")
                total_items = sum(len(p.get('bill_items', [])) for p in pagewise)
                print(f"Total items: {total_items}")
                return True
            else:
                print(f"Error: {data.get('message', 'Unknown error')}")
                if 'traceback' in data:
                    print(f"Traceback (last 10 lines):")
                    print("\n".join(data['traceback'].split('\n')[-10:]))
                return False
        else:
            print(f"ERROR: Non-200 status: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python TEST_API_UPLOAD.py <path_to_pdf>")
        print("Example: python TEST_API_UPLOAD.py C:\\temp\\train_sample_2.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    success = test_file_upload(pdf_path)
    sys.exit(0 if success else 1)

