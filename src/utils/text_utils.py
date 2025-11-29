"""
Text normalization and cleaning utilities for OCR output.
Handles common OCR errors and formatting inconsistencies in Indian invoices.
"""

import re
from typing import Optional


def clean_amount(text: str) -> float:
    """
    Clean and parse amount text from OCR output.
    
    Fixes common OCR typos and handles various number formats:
    - Replaces common OCR character confusions (O->0, S->5, etc.)
    - Removes currency symbols and whitespace
    - Handles both Indian (1,200.00) and European (1.200,00) formats
    - Fixes double dot errors (12.500.00 -> 12500.00)
    
    Args:
        text: Raw text string from OCR (e.g., "₹ 1,200.00" or "Rs 1.200,00")
        
    Returns:
        Parsed float value, or 0.0 if parsing fails
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    # Step 1: Remove currency symbols and common prefixes FIRST
    # This prevents OCR typo fixes from affecting currency symbols
    currency_symbols = ['₹', 'Rs', 'rs', 'RS', '$', 'USD', 'EUR', '€', '£', 'INR']
    for symbol in currency_symbols:
        text = text.replace(symbol, '')
    
    # Step 2: Fix common OCR typos (only after removing currency symbols)
    # O, D, Q -> 0 (common OCR confusion)
    text = text.replace('O', '0').replace('o', '0')
    text = text.replace('D', '0').replace('d', '0')
    text = text.replace('Q', '0').replace('q', '0')
    
    # S -> 5 (common OCR confusion)
    text = text.replace('S', '5').replace('s', '5')
    
    # B -> 8 (common OCR confusion)
    text = text.replace('B', '8').replace('b', '8')
    
    # l, |, I -> 1 (common OCR confusion in numbers)
    # But be careful - only replace in numeric contexts
    # Replace standalone l/|/I that appear in numeric sequences
    text = re.sub(r'(?<=\d)[l|I](?=\d)', '1', text)
    text = re.sub(r'^[l|I](?=\d)', '1', text)
    text = re.sub(r'(?<=\d)[l|I]$', '1', text)
    
    # Step 3: Remove all whitespace
    text = text.replace(' ', '').replace('\t', '').replace('\n', '')
    
    # Step 4: Handle different number formats
    # Check for European format (1.200,00) vs Indian format (1,200.00)
    has_comma_as_decimal = ',' in text and '.' in text
    has_only_comma = ',' in text and '.' not in text
    has_only_dot = '.' in text and ',' not in text
    
    # European format: 1.200,00 (dot as thousand separator, comma as decimal)
    if has_comma_as_decimal:
        # Check if comma comes after dot (European) or before dot (Indian)
        comma_pos = text.find(',')
        dot_pos = text.find('.')
        
        if comma_pos > dot_pos:
            # European format: 1.200,00
            # Remove dots (thousand separators), replace comma with dot
            text = text.replace('.', '').replace(',', '.')
        else:
            # Indian format: 1,200.00
            # Remove commas (thousand separators), keep dot
            text = text.replace(',', '')
    
    # If only comma (could be European decimal or Indian thousand separator)
    elif has_only_comma:
        # Check if comma is likely decimal (appears near end with 2 digits after)
        # or thousand separator (appears earlier)
        parts = text.split(',')
        if len(parts) == 2 and len(parts[1]) <= 2:
            # Likely European decimal: 1200,50
            text = text.replace(',', '.')
        else:
            # Likely thousand separator: 1,200
            text = text.replace(',', '')
    
    # If only dot, it's already in correct format (keep as is)
    # But check for double dot errors
    
    # Step 5: Fix double dot errors (e.g., "12.500.00" -> "12500.00")
    # Count dots - if more than one, likely a formatting error
    dot_count = text.count('.')
    if dot_count > 1:
        # Find the rightmost dot (likely the decimal point)
        last_dot_pos = text.rfind('.')
        # Remove all other dots
        text = text[:last_dot_pos].replace('.', '') + text[last_dot_pos:]
    
    # Step 6: Clean any remaining non-numeric characters (except decimal point and minus)
    text = re.sub(r'[^\d.\-]', '', text)
    
    # Step 7: Handle negative numbers
    is_negative = text.startswith('-')
    if is_negative:
        text = text[1:]
    
    # Step 8: Validate and parse
    if not text or text in ('.', '-', '--', ''):
        return 0.0
    
    try:
        value = float(text)
        if is_negative:
            value = -value
        return value
    except (ValueError, TypeError):
        return 0.0


def clean_item_name(text: str) -> str:
    """
    Clean and normalize item name text from OCR output.
    
    Performs the following normalizations:
    - Removes leading bullet points, numbers (1., 2.), or special characters
    - Converts ALL CAPS text to Title Case for readability
    - Normalizes whitespace (collapses multiple spaces)
    
    Args:
        text: Raw text string from OCR (e.g., "1. ITEM NAME" or "ITEM NAME")
        
    Returns:
        Cleaned and normalized item name string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Step 1: Remove leading bullet points and numbering
    # Patterns: "1.", "1)", "•", "-", "*", etc.
    text = re.sub(r'^[\d]+[\.\)\-\s]*', '', text)  # Remove "1.", "1)", "1-", etc.
    text = re.sub(r'^[•\-\*\#\s]+', '', text)  # Remove leading bullets and dashes
    
    # Step 2: Remove trailing special characters
    text = re.sub(r'[•\-\*\#\s]+$', '', text)
    
    # Step 3: Normalize whitespace (collapse multiple spaces/tabs/newlines)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Step 4: Convert ALL CAPS to Title Case
    # Check if text is all uppercase (excluding numbers and special chars)
    if text and text.isupper() and any(c.isalpha() for c in text):
        # Convert to title case, but preserve acronyms (all caps words)
        # Simple approach: convert entire string to title case
        # More sophisticated: detect acronyms and preserve them
        words = text.split()
        title_words = []
        for word in words:
            # If word is all caps and has more than 1 letter, might be acronym
            # But for item names, usually better to title case everything
            if word.isupper() and len(word) > 1 and word.isalpha():
                # Check if it's likely an acronym (short, all caps)
                # For now, convert to title case
                title_words.append(word.title())
            else:
                title_words.append(word.title())
        text = ' '.join(title_words)
    
    # Step 5: Final whitespace normalization
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

