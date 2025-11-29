# inspect_outputs.py
import json
from pathlib import Path

OUT_DIR = Path("local_test_outputs")
for f in sorted(OUT_DIR.glob("*.json")):
    j = json.loads(f.read_text())
    print("===", f.name, "===")
    for i, p in enumerate(j.get("data", {}).get("pagewise_line_items", []), 1):
        print(f"Page {i} type={p.get('page_type')} items={len(p.get('bill_items', []))}")
        # show first 200 chars of item names & amounts
        for bi in p.get("bill_items", [])[:10]:
            print("  ", bi.get("item_name")[:80], "|", bi.get("item_amount"))
    print()
