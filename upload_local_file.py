# upload_local_file.py
import sys
import requests
from pathlib import Path
import json

if len(sys.argv) < 2:
    print("Usage: python upload_local_file.py <path-to-file>")
    sys.exit(1)

file_path = Path(sys.argv[1])
if not file_path.exists():
    print("File not found:", file_path)
    sys.exit(2)

url = "http://127.0.0.1:8000/extract-bill-file"

with file_path.open("rb") as f:
    files = {"file": (file_path.name, f, "application/pdf")}
    try:
        r = requests.post(url, files=files, timeout=120)
    except Exception as e:
        print("Request failed:", e)
        sys.exit(3)

print("HTTP", r.status_code)
try:
    # pretty-print JSON if possible
    print(json.dumps(r.json(), indent=2))
except Exception:
    print("Response text:", r.text)
