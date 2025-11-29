"""
Setup script to make extractor package importable globally.
Install with: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="bajaj-datathon",
    version="1.0.0",
    description="Bajaj Finserv Datathon Bill Extraction API",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "requests",
        "pdf2image",
        "pytesseract",
        "Pillow",
    ],
)



