"""
Primitive OCR + regex helper.
IMPORTANT: rename functions/variables and expand logic before competition use.
"""

from pathlib import Path
from typing import Iterable, List, Dict
import re

import pytesseract
from PIL import Image


def _ocr_image(image_path: Path) -> str:
    with Image.open(image_path) as img:
        return pytesseract.image_to_string(img)


def extract_bill_lines(image_paths: Iterable[Path]) -> List[Dict[str, str]]:
    """
    Super-naive parser that looks for line items like 'Item - Qty - Amount'.
    TODO: swap regexes, add currency normalization, add unit tests.
    """
    entries: List[Dict[str, str]] = []
    pattern = re.compile(r"(?P<name>[A-Za-z ]+)\s+(?P<qty>\d+)\s+(?P<price>\\d+\\.\\d{2})")

    for img_path in image_paths:
        text = _ocr_image(img_path)
        for match in pattern.finditer(text):
            entries.append(
                {
                    "name": match.group("name").strip(),
                    "quantity": match.group("qty"),
                    "amount": match.group("price"),
                    "source": str(img_path),
                }
            )

    if not entries:
        entries.append(
            {"name": "TODO_ITEM", "quantity": "1", "amount": "0.00", "source": "heuristic-fallback"}
        )
    return entries

