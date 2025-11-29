from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
from PIL import Image
import sys, os, json

pdf_path = sys.argv[1]  # full path to PDF
page_no = int(sys.argv[2]) if len(sys.argv) > 2 else 1
out_dir = sys.argv[3] if len(sys.argv) > 3 else "debug_logs"

os.makedirs(out_dir, exist_ok=True)

# convert page at 300 DPI
pages = convert_from_path(pdf_path, dpi=300, first_page=page_no, last_page=page_no)
pil_img = pages[0]
pil_img.save(os.path.join(out_dir, f"page{page_no}_pil.png"))

# convert to grayscale cv2
cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)

# basic adaptive threshold to see how it looks
th = cv2.adaptiveThreshold(cv2_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                           cv2.THRESH_BINARY, 31, 10)
cv2.imwrite(os.path.join(out_dir, f"page{page_no}_th.png"), th)

# run pytesseract with image_to_data
data = pytesseract.image_to_data(th, output_type=Output.DICT,
                                 config='--oem 1 --psm 6', lang='eng')

with open(os.path.join(out_dir, f"page{page_no}_ocr.json"), "w",
          encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

texts = [t for t in data.get('text', []) if t and t.strip()]
confs = [c for c in data.get('conf', []) if str(c).strip() not in ("-1","")]

print("Total tokens:", len(data.get('text', [])))
print("Non-empty tokens:", len(texts))
print("Sample tokens (first 50):", texts[:50])
print("Sample confs (first 50):", confs[:50])

# write small TSV of tokens, confidences, bounding boxes
with open(os.path.join(out_dir, f"page{page_no}_tokens.tsv"), "w",
          encoding="utf-8") as f:
    f.write("text\tconf\tleft\ttop\twidth\theight\n")
    for i,t in enumerate(data.get('text', [])):
        f.write(f"{t}\t{data.get('conf',[None])[i]}\t"
                f"{data.get('left',[None])[i]}\t"
                f"{data.get('top',[None])[i]}\t"
                f"{data.get('width',[None])[i]}\t"
                f"{data.get('height',[None])[i]}\n")

print("Wrote debug images & OCR json to", out_dir)
