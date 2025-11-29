"""
Root-level pipeline module.
Re-exports from src.extractor.pipeline for global imports.
"""

from src.extractor.pipeline import extract_bill_data_from_file

__all__ = ["extract_bill_data_from_file"]



