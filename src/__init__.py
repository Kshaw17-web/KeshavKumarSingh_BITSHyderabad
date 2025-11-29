"""
Bajaj Finserv Datathon - Main package.
Makes extractor available for global imports.
"""

# Re-export extractor functions for convenience
from .extractor import extract_bill_data, extract_bill_data_from_file

__all__ = [
    "extract_bill_data",
    "extract_bill_data_from_file",
]



