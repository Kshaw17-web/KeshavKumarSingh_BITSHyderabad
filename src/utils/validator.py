from typing import List, Dict, Any


def build_stub_payload(items: List[Dict[str, str]]) -> Dict[str, Any]:
    """Build minimal HackRX-style payload from parsed line items."""
    total_amount = sum(float(item.get("amount", 0)) for item in items)
    return {
        "document_id": items[0].get("source", "unknown") if items else "unknown",
        "currency": "INR",
        "line_items": items,
        "totals": {"sub_total": total_amount, "tax": 0.0, "grand_total": total_amount},
    }


def validate_payload(payload: Dict[str, Any]) -> None:
    """Raise ValueError when required keys are missing. Expand for schema checks."""
    required_keys = {"document_id", "currency", "line_items", "totals"}
    missing = required_keys - payload.keys()
    if missing:
        raise ValueError(f"payload missing keys: {missing}")

    reconcile_totals(payload["totals"])


def reconcile_totals(totals: Dict[str, float]) -> float:
    """Compute grand_total = sub_total + tax; returns computed grand total."""
    sub_total = float(totals.get("sub_total", 0))
    tax = float(totals.get("tax", 0))
    grand_total = sub_total + tax
    totals["grand_total"] = grand_total
    return grand_total

