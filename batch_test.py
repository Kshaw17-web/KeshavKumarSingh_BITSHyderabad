# batch_test.py
import requests, json, sys
from pathlib import Path

ROOT = Path("data/raw/training_samples/TRAINING_SAMPLES")
OUT = Path("local_test_outputs")
OUT.mkdir(exist_ok=True)

url = "http://127.0.0.1:8000/extract-bill-file"

for f in sorted(ROOT.glob("train_sample_*.pdf")):
    print("Testing", f.name)
    with f.open("rb") as fh:
        r = requests.post(url, files={"file": (f.name, fh, "application/pdf")}, timeout=120)
    out_file = OUT / (f.stem + ".json")
    try:
        out_file.write_text(json.dumps(r.json(), indent=2))
    except Exception:
        out_file.write_text(r.text)
    print(" -> saved", out_file)
print("Done")
