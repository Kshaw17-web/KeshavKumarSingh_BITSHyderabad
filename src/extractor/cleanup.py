"""
Deduplication and totals reconciliation helpers.
"""

from typing import List, Dict, Any, Tuple, Optional

from rapidfuzz import fuzz


def dedupe_items(items: List[Dict[str, Any]], name_threshold: int = 90) -> List[Dict[str, Any]]:
    """
    Merge items with similar names and similar amounts.
    """
    kept = []
    for item in items:
        matched = False
        name_a = (item.get('item_name') or "").strip()
        for k in kept:
            name_b = (k.get('item_name') or "").strip()
            score = fuzz.token_set_ratio(name_a, name_b)
            amt_a = item.get('item_amount')
            amt_b = k.get('item_amount')
            amt_ok = (amt_a is None and amt_b is None) or (amt_a is not None and amt_b is not None and abs(amt_a - amt_b) <= 0.01 * max(1, amt_b))
            if score >= name_threshold and amt_ok:
                # merge quantity if available
                if k.get('item_quantity') and item.get('item_quantity'):
                    try:
                        k['item_quantity'] = (k.get('item_quantity') or 0) + (item.get('item_quantity') or 0)
                    except Exception:
                        pass
                matched = True
                break
        if not matched:
            kept.append(item.copy())
    return kept


def reconcile_totals(bill_items: List[Dict[str, Any]], extracted_total: Optional[float]) -> Tuple[float, str]:
    """
    Compare sum of item amounts with extracted_total and decide final_total and method.
    
    Returns (final_total, method)
    """
    sum_items = sum([it.get('item_amount') or 0 for it in bill_items])
    if extracted_total is None:
        return (sum_items, "derived_from_items")
    if extracted_total > 0:
        diff = abs(sum_items - extracted_total)
        if diff / extracted_total <= 0.02:
            return (sum_items, "items_sum_preferred")
        if sum_items == 0:
            return (extracted_total, "extracted_total")
        return (extracted_total, "mismatch_flagged")
    return (sum_items, "fallback")
