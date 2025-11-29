from pdf2image import convert_from_path
from PIL import Image
from src.preprocessing_helpers import preprocess_image_local, detect_whiteout_and_lowconf
import os, sys

# Set explicit poppler path here (edit to match your machine)
POPPLER_PATH = r"C:\poppler-25.11.0\Library\bin"

pdf_path = r"data\raw\training_samples\TRAINING_SAMPLES\train_sample_13.pdf"
if not os.path.exists(pdf_path):
    print("ERROR: expected file at", pdf_path)
    sys.exit(1)

print("Using Poppler at:", POPPLER_PATH)

imgs = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)

page0 = imgs[0]
print("Loaded page size:", page0.size)

pre = preprocess_image_local(page0)

flags = detect_whiteout_and_lowconf([pre], [""])
print("Flags returned:", flags)

os.makedirs("local_test_outputs/debug_masks", exist_ok=True)
out = "local_test_outputs/debug_masks/train_sample_13_preprocessed.png"
pre.save(out)
print("Saved preprocessed image:", out)
print("Done.")
