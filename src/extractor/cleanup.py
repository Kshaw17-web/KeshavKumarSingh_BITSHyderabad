"""
Cleanup and reconciliation utilities for bill extraction.

Functions:
- dedupe_items(items, name_threshold=0.85, amount_tolerance=0.01)
- reconcile_totals(items, reported_total=None, tolerance=0.05)

These helpers are used to clean up extracted bill items by removing duplicates
and reconciling totals against reported values.
"""

from typing import List, Dict, Any, Optional, Tuple
import re

# Optional dependency for fuzzy matching
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    fuzz = None


def _normalize_item_name(name: str) -> str:
    """
    Normalize item name for comparison.
    
    Args:
        name: Item name string
    
    Returns:
        Normalized string (lowercase, whitespace normalized, common prefixes removed)
    """
    if not name:
        return ""
    
    # Convert to lowercase and normalize whitespace
    normalized = re.sub(r'\s+', ' ', name.lower().strip())
    
    # Remove common prefixes that don't affect matching
    prefixes = ['item', 'product', 'medicine', 'drug', 'test']
    for prefix in prefixes:
        if normalized.startswith(prefix + ' '):
            normalized = normalized[len(prefix) + 1:]
    
    return normalized


def _items_similar(item1: Dict[str, Any], item2: Dict[str, Any], 
                   name_threshold: float = 0.85, 
                   amount_tolerance: float = 0.01) -> bool:
    """
    Check if two items are likely duplicates using fuzzy matching.
    
    Args:
        item1: First item dictionary with keys: item_name, item_amount, etc.
        item2: Second item dictionary
        name_threshold: Similarity threshold for item names (0-1, default 0.85)
        amount_tolerance: Relative tolerance for amount matching (default 0.01 = 1%)
    
    Returns:
        True if items are likely duplicates, False otherwise
    """
    name1 = str(item1.get("item_name", "")).strip()
    name2 = str(item2.get("item_name", "")).strip()
    
    if not name1 or not name2:
        return False
    
    # Normalize names
    norm_name1 = _normalize_item_name(name1)
    norm_name2 = _normalize_item_name(name2)
    
    # Exact match after normalization
    if norm_name1 == norm_name2:
        name_similarity = 1.0
    elif RAPIDFUZZ_AVAILABLE:
        # Use fuzzy matching
        name_similarity = fuzz.ratio(norm_name1, norm_name2) / 100.0
    else:
        # Fallback: simple substring check
        if norm_name1 in norm_name2 or norm_name2 in norm_name1:
            name_similarity = 0.9
        else:
            name_similarity = 0.0
    
    # Check name similarity threshold
    if name_similarity < name_threshold:
        return False
    
    # Check amount similarity (if both have amounts)
    amount1 = item1.get("item_amount")
    amount2 = item2.get("item_amount")
    
    if amount1 is not None and amount2 is not None:
        # Calculate relative difference
        max_amount = max(abs(amount1), abs(amount2))
        if max_amount > 0:
            relative_diff = abs(amount1 - amount2) / max_amount
            if relative_diff > amount_tolerance:
                # Amounts differ significantly - likely different items
                return False
    
    return True


def dedupe_items(items: List[Dict[str, Any]], 
                  name_threshold: float = 0.85,
                  amount_tolerance: float = 0.01,
                  merge_strategy: str = "keep_first") -> List[Dict[str, Any]]:
    """
    Remove duplicate items from a list using fuzzy matching.
    
    This function identifies duplicate items based on:
    - Name similarity (using fuzzy matching if rapidfuzz available)
    - Amount similarity (within tolerance)
    
    When duplicates are found, they are merged according to the merge_strategy:
    - "keep_first": Keep the first occurrence, discard others
    - "keep_best": Keep item with highest confidence or amount
    - "merge": Merge amounts and take best name/confidence
    
    Args:
        items: List of item dictionaries with keys: item_name, item_amount, etc.
        name_threshold: Similarity threshold for item names (0-1, default 0.85)
                       Higher = stricter matching (fewer duplicates detected)
        amount_tolerance: Relative tolerance for amount matching (default 0.01 = 1%)
        merge_strategy: How to handle duplicates - "keep_first", "keep_best", or "merge"
    
    Returns:
        Deduplicated list of items
    
    Example:
        >>> items = [
        ...     {"item_name": "Paracetamol 500mg", "item_amount": 100.0},
        ...     {"item_name": "Paracetamol 500 mg", "item_amount": 100.0},
        ...     {"item_name": "Aspirin", "item_amount": 50.0}
        ... ]
        >>> deduped = dedupe_items(items)
        >>> len(deduped)  # Returns 2 (duplicates merged)
    """
    if not items:
        return []
    
    if len(items) == 1:
        return items
    
    # Track which items have been processed
    seen_indices = set()
    deduplicated = []
    
    for i, item in enumerate(items):
        if i in seen_indices:
            continue
        
        # Find all duplicates of this item
        duplicates = [i]
        for j in range(i + 1, len(items)):
            if j in seen_indices:
                continue
            
            if _items_similar(item, items[j], name_threshold, amount_tolerance):
                duplicates.append(j)
        
        # Merge duplicates according to strategy
        if len(duplicates) == 1:
            # No duplicates, keep as-is
            deduplicated.append(item)
        else:
            # Duplicates found - merge them
            duplicate_items = [items[idx] for idx in duplicates]
            
            if merge_strategy == "keep_first":
                merged = duplicate_items[0].copy()
            elif merge_strategy == "keep_best":
                # Find item with highest confidence or amount
                best_item = max(duplicate_items, key=lambda x: (
                    float(x.get("confidence", 0.5)) if isinstance(x.get("confidence"), (int, float)) else 0.5,
                    x.get("item_amount", 0.0)
                ))
                merged = best_item.copy()
            else:  # merge
                # Merge: take best name, sum amounts, take best rate/qty
                merged = duplicate_items[0].copy()
                
                # Sum amounts
                total_amount = sum(d.get("item_amount", 0.0) for d in duplicate_items if d.get("item_amount") is not None)
                if total_amount > 0:
                    merged["item_amount"] = total_amount
                
                # Take longest/most complete name
                names = [d.get("item_name", "") for d in duplicate_items]
                merged["item_name"] = max(names, key=len)
                
                # Take best rate/quantity (non-None preferred)
                for field in ["item_rate", "item_quantity"]:
                    values = [d.get(field) for d in duplicate_items if d.get(field) is not None]
                    if values:
                        merged[field] = values[0]  # Take first non-None
            
            deduplicated.append(merged)
        
        # Mark all duplicates as processed
        seen_indices.update(duplicates)
    
    return deduplicated


def reconcile_totals(items: List[Dict[str, Any]], 
                     reported_total: Optional[float] = None,
                     tolerance: float = 0.05,
                     use_weighted: bool = False) -> Tuple[bool, float, float]:
    """
    Reconcile calculated total against reported total.
    
    This function calculates the sum of item amounts and compares it against
    a reported total (if provided). It can use either simple summation or
    weighted summation based on confidence scores.
    
    Args:
        items: List of item dictionaries with keys: item_amount, confidence (optional)
        reported_total: Reported total from the bill (if available)
        tolerance: Relative error tolerance for reconciliation (default 0.05 = 5%)
        use_weighted: If True, use confidence-weighted sum; else use simple sum
    
    Returns:
        Tuple of (reconciliation_ok, calculated_sum, relative_error):
        - reconciliation_ok: True if calculated sum matches reported total within tolerance,
                            None if reported_total is None, False otherwise
        - calculated_sum: Sum of item amounts (weighted or simple)
        - relative_error: Relative error between calculated and reported (0.0 if no reported_total)
    
    Example:
        >>> items = [
        ...     {"item_name": "Item 1", "item_amount": 100.0, "confidence": 0.9},
        ...     {"item_name": "Item 2", "item_amount": 50.0, "confidence": 0.8}
        ... ]
        >>> ok, calc_sum, error = reconcile_totals(items, reported_total=150.0)
        >>> print(f"Reconciliation: {ok}, Sum: {calc_sum}, Error: {error}")
    """
    if not items:
        return False, 0.0, 0.0
    
    if use_weighted:
        # Weighted sum based on confidence
        weighted_sum = 0.0
        total_weight = 0.0
        
        for item in items:
            amount = item.get("item_amount", 0.0)
            if amount is None:
                continue
            
            # Get confidence (default to 0.7 if not available)
            confidence = item.get("confidence", 0.7)
            if isinstance(confidence, str):
                # Convert string confidence to float
                confidence_map = {"low": 0.5, "medium": 0.7, "high": 0.9}
                confidence = confidence_map.get(confidence.lower(), 0.7)
            elif not isinstance(confidence, (int, float)):
                confidence = 0.7
            
            weight = float(confidence)
            weighted_sum += amount * weight
            total_weight += weight
        
        # Use weighted average if weights available, else fall back to simple sum
        if total_weight > 0:
            calculated_sum = weighted_sum / total_weight
        else:
            calculated_sum = sum(item.get("item_amount", 0.0) for item in items if item.get("item_amount") is not None)
    else:
        # Simple sum
        calculated_sum = sum(
            item.get("item_amount", 0.0) 
            for item in items 
            if item.get("item_amount") is not None
        )
    
    # Round to 2 decimal places
    calculated_sum = round(calculated_sum, 2)
    
    # If no reported total, return None for reconciliation_ok
    if reported_total is None:
        return None, calculated_sum, 0.0
    
    # Calculate relative error
    reported_total = float(reported_total)
    
    if reported_total > 0:
        relative_error = abs(calculated_sum - reported_total) / reported_total
        reconciliation_ok = relative_error <= tolerance
    else:
        relative_error = 0.0
        reconciliation_ok = False
    
    return reconciliation_ok, calculated_sum, relative_error


if __name__ == "__main__":
    """
    CLI usage example for testing the cleanup module.
    
    Usage:
        python -m src.extractor.cleanup
    """
    # Example usage
    sample_items = [
        {"item_name": "Paracetamol 500mg", "item_amount": 100.0, "confidence": 0.9},
        {"item_name": "Paracetamol 500 mg", "item_amount": 100.0, "confidence": 0.8},
        {"item_name": "Aspirin 100mg", "item_amount": 50.0, "confidence": 0.95},
        {"item_name": "Aspirin 100 mg", "item_amount": 50.0, "confidence": 0.85},
    ]
    
    print("Original items:", len(sample_items))
    for item in sample_items:
        print(f"  - {item['item_name']}: ₹{item['item_amount']}")
    
    print("\n" + "="*60)
    print("Deduplication Test:")
    print("="*60)
    
    deduped = dedupe_items(sample_items, name_threshold=0.85)
    print(f"\nDeduplicated items: {len(deduped)}")
    for item in deduped:
        print(f"  - {item['item_name']}: ₹{item['item_amount']}")
    
    print("\n" + "="*60)
    print("Reconciliation Test:")
    print("="*60)
    
    # Test with reported total
    reported = 200.0
    ok, calc_sum, error = reconcile_totals(deduped, reported_total=reported)
    print(f"\nReported total: ₹{reported}")
    print(f"Calculated sum: ₹{calc_sum}")
    print(f"Relative error: {error:.2%}")
    print(f"Reconciliation OK: {ok}")
    
    # Test without reported total
    ok2, calc_sum2, error2 = reconcile_totals(deduped, reported_total=None)
    print(f"\nWithout reported total:")
    print(f"Calculated sum: ₹{calc_sum2}")
    print(f"Reconciliation OK: {ok2}")

