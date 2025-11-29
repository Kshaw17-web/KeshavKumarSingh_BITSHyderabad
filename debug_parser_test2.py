# debug_parser_test2.py
import json, sys, os
from pathlib import Path

# import your parser module (make sure python path is repo root)
from src.extractor.parsers import (
    group_words_to_lines, detect_column_centers,
    map_tokens_to_columns, parse_row_from_columns, is_probable_item
)

ocr_json = Path(".") / "debug_logs" / "page1_ocr.json"
if not ocr_json.exists():
    print("ERROR: debug_logs/page1_ocr.json not found. Run debug_ocr_test.py first.")
    sys.exit(1)

ocr = json.loads(ocr_json.read_text(encoding="utf-8"))

lines = group_words_to_lines(ocr)
print(f"Num lines detected: {len(lines)}")

centers = detect_column_centers(lines)
print("Detected column centers:", centers)

probable_count = 0
all_parsed = []
for ln in lines:
    cols = map_tokens_to_columns(ln, centers)
    parsed = parse_row_from_columns(cols)
    probable = is_probable_item(parsed)
    if probable:
        probable_count += 1
        print("LINE:", " | ".join([t['text'] for t in ln]))
        print("COLUMNS:", cols)
        print("PARSED:", parsed)
        print("-----")
    all_parsed.append({"columns": cols, "parsed": parsed, "probable": probable})

print("Total probable item rows found:", probable_count)

# save everything for inspection
out = {"num_lines": len(lines), "centers": centers, "probable_count": probable_count, "rows": all_parsed}
Path("debug_logs/parser_diagnostic.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("Wrote debug_logs/parser_diagnostic.json")
