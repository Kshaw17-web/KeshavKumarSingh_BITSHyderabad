"""
Root-level extractor package for global imports.
Re-exports from src.extractor to enable: from extractor.bill_extractor import ...
"""

# Import from src.extractor modules
from src.extractor.bill_extractor import extract_bill_data
from src.extractor.pipeline import extract_bill_data_from_file

__all__ = [
    "extract_bill_data",
    "extract_bill_data_from_file",
]
