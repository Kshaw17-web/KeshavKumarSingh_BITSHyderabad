"""
Root-level bill_extractor module.
Re-exports from src.extractor.bill_extractor for global imports.
"""

from src.extractor.bill_extractor import extract_bill_data

__all__ = ["extract_bill_data"]
