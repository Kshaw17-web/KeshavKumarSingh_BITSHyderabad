"""
Test script for PaddleOCR-based extraction with forensic regex cleaning.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.text_utils import clean_amount, clean_item_name

def test_clean_amount():
    """Test clean_amount function with various inputs."""
    print("Testing clean_amount():")
    test_cases = [
        ("₹ 1,200.00", 1200.0),
        ("Rs 1.200,00", 1200.0),  # European format
        ("12.500.00", 12500.0),  # Double dot error
        ("1,234.56", 1234.56),
        ("₹1,50,000.00", 150000.0),
        ("O123.45", 123.45),  # OCR typo: O -> 0
        ("S500", 500.0),  # OCR typo: S -> 5
        ("B88", 88.0),  # OCR typo: B -> 8
        ("invalid", 0.0),  # Invalid input
    ]
    
    for input_text, expected in test_cases:
        result = clean_amount(input_text)
        status = "✓" if abs(result - expected) < 0.01 else "✗"
        print(f"  {status} '{input_text}' -> {result} (expected: {expected})")


def test_clean_item_name():
    """Test clean_item_name function with various inputs."""
    print("\nTesting clean_item_name():")
    test_cases = [
        ("1. ITEM NAME", "Item Name"),
        ("2) PRODUCT NAME", "Product Name"),
        ("• BULLET ITEM", "Bullet Item"),
        ("ALL CAPS TEXT", "All Caps Text"),
        ("  multiple   spaces  ", "multiple spaces"),
        ("Normal Text", "Normal Text"),
    ]
    
    for input_text, expected_pattern in test_cases:
        result = clean_item_name(input_text)
        # Check if result contains expected pattern (case-insensitive)
        status = "✓" if expected_pattern.lower() in result.lower() or result.lower() in expected_pattern.lower() else "✗"
        print(f"  {status} '{input_text}' -> '{result}'")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Text Utilities")
    print("=" * 60)
    test_clean_amount()
    test_clean_item_name()
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)

