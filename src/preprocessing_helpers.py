from typing import Tuple, Dict

import math

import io

import numpy as np

import cv2

from PIL import Image, ImageChops

import pytesseract



def auto_rotate_and_deskew(pil_img: Image.Image) -> Image.Image:

    try:

        osd = pytesseract.image_to_osd(pil_img)

        rot = 0

        for line in osd.splitlines():

            if line.startswith("Rotate:"):

                rot = int(line.split(":")[1].strip())

                break

        if rot != 0:

            return pil_img.rotate(-rot, expand=True)

    except Exception:

        pass

    img = np.array(pil_img.convert("RGB"))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:

        return pil_img

    angles = []

    for x1, y1, x2, y2 in lines[:, 0]:

        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))

        angles.append(ang)

    if not angles:

        return pil_img

    median_angle = float(np.median(angles))

    if abs(median_angle) > 0.5:

        return pil_img.rotate(-median_angle, expand=True)

    return pil_img



def enhance_contrast_clahe(pil_img: Image.Image) -> Image.Image:

    arr = np.array(pil_img.convert("RGB"))

    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    l2 = clahe.apply(l)

    lab2 = cv2.merge((l2, a, b))

    img2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

    return Image.fromarray(img2)



def remove_small_artifacts(gray_np: np.ndarray) -> np.ndarray:

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    opened = cv2.morphologyEx(gray_np, cv2.MORPH_OPEN, kernel)

    return opened



def fix_whitener(pil_img: Image.Image) -> Tuple[Image.Image, np.ndarray]:

    arr = np.array(pil_img.convert("RGB"))

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    v = hsv[:, :, 2]

    sat = hsv[:, :, 1]

    mask = ((v > 220) & (sat < 20)).astype("uint8") * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    mask = cv2.dilate(mask, kernel, iterations=1)

    try:

        inpainted = cv2.inpaint(arr, mask, 5, cv2.INPAINT_TELEA)

        return Image.fromarray(inpainted), mask

    except Exception:

        return pil_img, mask



def preprocess_full_pipeline(pil_img: Image.Image) -> Tuple[Image.Image, Dict]:

    img = auto_rotate_and_deskew(pil_img)

    img = enhance_contrast_clahe(img)

    arr = np.array(img.convert("RGB"))

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape

    if max(h, w) < 1200:

        gray = cv2.resize(gray, (int(w * 2.0), int(h * 2.0)), interpolation=cv2.INTER_CUBIC)

    gray = remove_small_artifacts(gray)

    pil_for_whitener = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))

    repaired_img, white_mask = fix_whitener(pil_for_whitener)

    arr2 = np.array(repaired_img.convert("RGB"))

    gray2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2GRAY)

    th = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)

    prepped = Image.fromarray(th)

    white_coverage = float((white_mask > 0).sum()) / float(white_mask.size) if white_mask is not None else 0.0

    diagnostics = {"white_coverage": white_coverage, "width": w, "height": h}

    return prepped, diagnostics



def ela_map(pil_img: Image.Image, quality: int = 90, scale: int = 10) -> Image.Image:

    buf = io.BytesIO()

    pil_img.convert("RGB").save(buf, "JPEG", quality=quality)

    buf.seek(0)

    reloaded = Image.open(buf).convert("RGB")

    ela = ImageChops.difference(pil_img.convert("RGB"), reloaded)

    def amplify(x):

        v = x * scale

        return 255 if v > 255 else v

    ela = ela.point(amplify)

    return ela



def detect_whiteout_and_lowconf(prepped_img: Image.Image) -> Tuple[list, dict]:

    tsv = pytesseract.image_to_data(prepped_img, output_type=pytesseract.Output.DICT)

    confs = [int(c) for c in tsv.get("conf", []) if c not in ("-1", "")]

    low_conf_pct = float(sum(1 for c in confs if c < 60)) / float(len(confs)) if confs else 0.0

    ela = ela_map(prepped_img)

    ela_gray = np.array(ela.convert("L"))

    ela_high_pct = float((ela_gray > 20).sum()) / float(ela_gray.size)

    mask = None

    try:

        _, mask = fix_whitener(prepped_img)

    except Exception:

        mask = np.zeros((prepped_img.height, prepped_img.width), dtype="uint8")

    white_coverage = float((mask > 0).sum()) / float(mask.size) if mask is not None else 0.0

    flags = []

    if white_coverage > 0.005:

        flags.append({"type": "whiteout", "score": white_coverage})

    if low_conf_pct > 0.30:

        flags.append({"type": "low_ocr_confidence", "score": low_conf_pct})

    if ela_high_pct > 0.02:

        flags.append({"type": "ela_suspect", "score": ela_high_pct})

    metadata = {"white_coverage": white_coverage, "low_conf_pct": low_conf_pct, "ela_high_pct": ela_high_pct}

    return flags, metadata

