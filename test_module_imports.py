"""
Test script to verify all module imports work correctly.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# Add src to path if needed
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

errors = []
successes = []

# Test 1: preprocessing_helpers
try:
    from src.preprocessing_helpers import preprocess_image_local, detect_whiteout_and_lowconf
    successes.append("✓ preprocessing_helpers loaded")
    print("✓ preprocessing_helpers loaded")
except Exception as e:
    errors.append(f"✗ preprocessing_helpers failed: {e}")
    print(f"✗ preprocessing_helpers failed: {e}")

# Test 2: extract_bill_data
try:
    from src.extractor.bill_extractor import extract_bill_data
    successes.append("✓ extract_bill_data available")
    print("✓ extract_bill_data available")
except Exception:
    try:
        from extractor.bill_extractor import extract_bill_data
        successes.append("✓ extract_bill_data available (via extractor)")
        print("✓ extract_bill_data available (via extractor)")
    except Exception as e:
        errors.append(f"✗ extract_bill_data failed: {e}")
        print(f"✗ extract_bill_data failed: {e}")

# Test 3: schemas
try:
    from src.schemas import DocumentRequest, BillItem, PageItems, DataResponse, FullResponse
    successes.append("✓ schemas loaded")
    print("✓ schemas loaded")
except Exception as e:
    errors.append(f"✗ schemas failed: {e}")
    print(f"✗ schemas failed: {e}")

# Test 4: api
try:
    from src.api import app
    successes.append("✓ api module loaded")
    print("✓ api module loaded")
except Exception as e:
    errors.append(f"✗ api module failed: {e}")
    print(f"✗ api module failed: {e}")

# Test 5: Dependencies
deps = ["pytesseract", "pdf2image", "PIL", "numpy", "cv2", "fastapi"]
for dep in deps:
    try:
        if dep == "PIL":
            import PIL
        elif dep == "cv2":
            import cv2
        else:
            __import__(dep)
        successes.append(f"✓ {dep} available")
        print(f"✓ {dep} available")
    except ImportError:
        errors.append(f"✗ {dep} not available")
        print(f"✗ {dep} not available")

# Summary
print("\n" + "="*50)
print("SUMMARY:")
print(f"Successes: {len(successes)}")
print(f"Errors: {len(errors)}")
print("="*50)

if errors:
    print("\nERRORS FOUND:")
    for err in errors:
        print(f"  {err}")
    sys.exit(1)
else:
    print("\n✓ all dependency checks ok")
    print("✓ All imports successful!")

