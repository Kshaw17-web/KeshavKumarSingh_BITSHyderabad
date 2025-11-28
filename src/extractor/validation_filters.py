# src/extractor/validation_filters.py
import re
from datetime import datetime
from typing import List, Tuple

# Regex patterns
CURRENCY_RE = re.compile(
    r'(?P<cur>₹|Rs\.?|INR|Rs|Rs\.)?\s*(?P<amt>\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)'
)
DATE_RE = re.compile(
    r'(?:(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})|(?:\d{4}[-/]\d{1,2}[-/]\d{1,2})|(?:\d{1,2}\s+[A-Za-z]{3,}\s+\d{4}))'
)
ID_RE = re.compile(r'\b(?:INV|Invoice|Bill|BILL|Ref|Receipt|RCPT|MRN)[\s:\-#]*[A-Za-z0-9\-\/]{2,}\b', re.I)
STRICT_AMOUNT_RE = re.compile(r'(?P<cur>₹|Rs\.?|INR)?\s*(?P<amt>\d{1,6}(?:[,\s]\d{3})*(?:\.\d{1,2})?)')

QxR_RE = re.compile(r'(?P<qty>\d+(?:\.\d+)?)\s*(?:x|X|\*)\s*(?P<rate>\d+(?:\.\d+)?)')

NEGATIVE_CONTEXT_RE = re.compile(r'\b(Date|Invoice No|Invoice|Bill No|Ref|Receipt|Time|DOB|MRN|Patient)\b', re.I)


def looks_like_date(token: str) -> bool:
    return bool(DATE_RE.search(token))


def looks_like_id(token: str) -> bool:
    return bool(ID_RE.search(token))


def parse_number_str(num_str: str) -> float | None:
    if num_str is None:
        return None
    s = str(num_str).replace(',', '').replace(' ', '')
    try:
        return float(s)
    except Exception:
        return None


def find_currency_candidates(line: str) -> List[Tuple[float, str, Tuple[int, int]]]:
    """Return list of (amt_float, matched_text, span) for currency-like candidates."""
    out = []
    for m in CURRENCY_RE.finditer(line):
        amt = parse_number_str(m.group('amt'))
        if amt is None:
            continue
        out.append((amt, m.group(0).strip(), m.span()))
    return out


def is_likely_money_token(line: str, surrounding_text: str | None = None) -> bool:
    """
    Heuristic check if a number token in `line` is likely monetary:
      - not a date or invoice id
      - matches strict amount pattern
      - not ridiculously large (tunable)
      - not found near negative context words like 'Invoice' or 'Date'
    """
    if looks_like_date(line):
        return False
    if looks_like_id(line):
        return False
    m = STRICT_AMOUNT_RE.search(line)
    if not m:
        return False
    amt = parse_number_str(m.group('amt'))
    if amt is None:
        return False
    # tune this threshold as needed for your dataset
    if amt > 5_000_000:
        return False
    if surrounding_text:
        if NEGATIVE_CONTEXT_RE.search(surrounding_text):
            return False
    return True


def extract_qty_rate(line: str):
    """If a 'qty x rate' pattern exists, return (qty, rate, computed_amount) else None."""
    m = QxR_RE.search(line)
    if not m:
        return None
    try:
        qty = float(m.group('qty'))
        rate = float(m.group('rate'))
        return qty, rate, round(qty * rate, 2)
    except Exception:
        return None


def find_reported_total(lines: List[str]) -> float | None:
    """
    Scan all lines for 'Total' keywords and return the first strict amount found next to them.
    """
    for line in lines[::-1]:  # scan from bottom up (totals often near end)
        if re.search(r'\b(Total|Grand Total|Net Payable|Amount Payable|Balance Due)\b', line, re.I):
            m = STRICT_AMOUNT_RE.search(line)
            if m:
                amt = parse_number_str(m.group('amt'))
                if amt is not None:
                    return amt
    return None


def reconcile(items: list, reported_total: float | None, tol: float = 0.05):
    """
    Compare sum(items amounts) with reported_total. Returns (ok_bool, sum, rel_error)
    """
    s = sum((item.get('item_amount') or 0) for item in items)
    if reported_total is None:
        return True, s, None
    if reported_total == 0:
        return False, s, 1.0
    diff = abs(s - reported_total)
    rel = diff / reported_total
    return (rel <= tol), s, rel
