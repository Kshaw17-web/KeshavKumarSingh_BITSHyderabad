"""
Ensemble reconciler that merges heuristic and LayoutLMv3 model results.
Uses fuzzy matching to deduplicate and produces final JSON schema.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    fuzz = None
    process = None


def fuzzy_match_items(
    item1: Dict[str, Any],
    item2: Dict[str, Any],
    name_threshold: float = 0.8,
    amount_tolerance: float = 0.01
) -> bool:
    """
    Check if two items are likely the same using fuzzy matching.
    
    Args:
        item1: First item dict
        item2: Second item dict
        name_threshold: Similarity threshold for item names (0-1)
        amount_tolerance: Relative tolerance for amounts
        
    Returns:
        True if items match
    """
    name1 = str(item1.get("item_name", "")).strip().lower()
    name2 = str(item2.get("item_name", "")).strip().lower()
    
    if not name1 or not name2:
        return False
    
    # Fuzzy match names
    if RAPIDFUZZ_AVAILABLE:
        name_similarity = fuzz.ratio(name1, name2) / 100.0
    else:
        # Simple fallback
        name_similarity = 1.0 if name1 == name2 else 0.0
    
    if name_similarity < name_threshold:
        return False
    
    # Check amount similarity
    amount1 = item1.get("item_amount")
    amount2 = item2.get("item_amount")
    
    if amount1 is None or amount2 is None:
        return name_similarity > 0.9  # High name similarity if amounts missing
    
    # Relative tolerance check
    if abs(amount1 - amount2) / max(abs(amount1), abs(amount2), 0.01) <= amount_tolerance:
        return True
    
    # Absolute tolerance for small amounts
    if abs(amount1 - amount2) <= 0.1:
        return True
    
    return False


def merge_item(
    item1: Dict[str, Any],
    item2: Dict[str, Any],
    prefer_model: bool = True
) -> Dict[str, Any]:
    """
    Merge two matching items, preferring model results if prefer_model=True.
    
    Args:
        item1: First item
        item2: Second item
        prefer_model: Prefer model results over heuristic
        
    Returns:
        Merged item
    """
    # Determine which is model result (has confidence > 0.7)
    model_item = item1 if item1.get("confidence", 0.0) > 0.7 else item2
    heuristic_item = item2 if model_item == item1 else item1
    
    if prefer_model and model_item.get("confidence", 0.0) > 0.7:
        base_item = model_item.copy()
        # Fill missing fields from heuristic
        if not base_item.get("item_rate") and heuristic_item.get("item_rate"):
            base_item["item_rate"] = heuristic_item["item_rate"]
        if not base_item.get("item_quantity") and heuristic_item.get("item_quantity"):
            base_item["item_quantity"] = heuristic_item["item_quantity"]
    else:
        base_item = heuristic_item.copy()
        # Fill missing fields from model
        if not base_item.get("item_rate") and model_item.get("item_rate"):
            base_item["item_rate"] = model_item["item_rate"]
        if not base_item.get("item_quantity") and model_item.get("item_quantity"):
            base_item["item_quantity"] = model_item["item_quantity"]
    
    # Use better item_name (longer or from model)
    if len(str(model_item.get("item_name", ""))) > len(str(base_item.get("item_name", ""))):
        base_item["item_name"] = model_item["item_name"]
    
    # Use more confident amount
    if model_item.get("confidence", 0.0) > 0.7:
        base_item["item_amount"] = model_item.get("item_amount", base_item.get("item_amount", 0.0))
    
    # Update confidence (average or max)
    conf1 = item1.get("confidence", 0.5)
    conf2 = item2.get("confidence", 0.5)
    base_item["confidence"] = max(conf1, conf2)
    
    return base_item


def reconcile_ensemble(
    heuristic_items: List[Dict[str, Any]],
    model_items: List[Dict[str, Any]],
    prefer_model: bool = True,
    name_threshold: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Reconcile heuristic and model results into final item list.
    
    Steps:
    1. Fuzzy match items between heuristic and model results
    2. Merge matching items (prefer model if prefer_model=True)
    3. Add unmatched items
    4. Deduplicate remaining items
    
    Args:
        heuristic_items: Items from heuristic extraction
        model_items: Items from LayoutLMv3 model
        prefer_model: Prefer model results when merging
        name_threshold: Fuzzy matching threshold for names
        
    Returns:
        Final reconciled list of items
    """
    if not model_items:
        return heuristic_items
    
    if not heuristic_items:
        return model_items
    
    # Step 1: Match items between heuristic and model
    matched_heuristic = set()
    matched_model = set()
    merged_items = []
    
    for i, h_item in enumerate(heuristic_items):
        best_match_idx = None
        best_match_score = 0.0
        
        for j, m_item in enumerate(model_items):
            if j in matched_model:
                continue
            
            if fuzzy_match_items(h_item, m_item, name_threshold=name_threshold):
                # Calculate match score
                name1 = str(h_item.get("item_name", "")).lower()
                name2 = str(m_item.get("item_name", "")).lower()
                
                if RAPIDFUZZ_AVAILABLE:
                    score = fuzz.ratio(name1, name2) / 100.0
                else:
                    score = 1.0 if name1 == name2 else 0.0
                
                if score > best_match_score:
                    best_match_score = score
                    best_match_idx = j
        
        if best_match_idx is not None:
            # Merge items
            merged = merge_item(h_item, model_items[best_match_idx], prefer_model=prefer_model)
            merged_items.append(merged)
            matched_heuristic.add(i)
            matched_model.add(best_match_idx)
        else:
            # No match found, add heuristic item
            merged_items.append(h_item)
            matched_heuristic.add(i)
    
    # Step 2: Add unmatched model items
    for j, m_item in enumerate(model_items):
        if j not in matched_model:
            merged_items.append(m_item)
    
    # Step 3: Deduplicate within merged list
    final_items = []
    seen_indices = set()
    
    for i, item in enumerate(merged_items):
        if i in seen_indices:
            continue
        
        # Check for duplicates with remaining items
        duplicates = [i]
        for j in range(i + 1, len(merged_items)):
            if j in seen_indices:
                continue
            if fuzzy_match_items(item, merged_items[j], name_threshold=name_threshold):
                duplicates.append(j)
        
        if len(duplicates) > 1:
            # Merge all duplicates
            merged = item.copy()
            for dup_idx in duplicates[1:]:
                merged = merge_item(merged, merged_items[dup_idx], prefer_model=prefer_model)
            final_items.append(merged)
        else:
            final_items.append(item)
        
        seen_indices.update(duplicates)
    
    return final_items


def format_items_for_schema(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format items to match HackRx schema.
    
    Args:
        items: List of item dictionaries
        
    Returns:
        Formatted items with required fields
    """
    formatted = []
    
    for item in items:
        formatted_item = {
            "item_name": str(item.get("item_name", "UNKNOWN")).strip(),
            "item_amount": float(item.get("item_amount", 0.0)),
            "item_rate": float(item.get("item_rate")) if item.get("item_rate") is not None else None,
            "item_quantity": float(item.get("item_quantity")) if item.get("item_quantity") is not None else None,
            "confidence": round(float(item.get("confidence", 0.5)), 2)
        }
        
        # Validate and clean
        if formatted_item["item_amount"] < 0.01:
            continue  # Skip invalid amounts
        
        if not formatted_item["item_name"] or len(formatted_item["item_name"]) < 2:
            formatted_item["item_name"] = "UNKNOWN_ITEM"
        
        formatted.append(formatted_item)
    
    return formatted


