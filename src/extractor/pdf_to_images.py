from pathlib import Path
from typing import List

from pdf2image import convert_from_path


def pdf_to_images(pdf_path: Path) -> List[Path]:
    """
    Convert a PDF into temporary PNG files for downstream OCR.
    NOTE: customize DPI/output_dir before production use.
    """
    if pdf_path.suffix.lower() != ".pdf":
        return [pdf_path]

    output_dir = pdf_path.parent / "generated_pages"
    output_dir.mkdir(exist_ok=True)

    images = convert_from_path(pdf_path, dpi=200, fmt="png", output_folder=output_dir)
    return [Path(img.filename) for img in images]

