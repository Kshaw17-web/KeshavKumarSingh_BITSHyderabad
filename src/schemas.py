# src/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# -------------------------
# Request models
# -------------------------
class DocumentRequest(BaseModel):
    document: str = Field(..., description="URL to a PDF/image of the bill")


# -------------------------
# Core response models
# -------------------------
class BillItem(BaseModel):
    item_name: str = Field(..., description="Item name exactly as on bill")
    item_amount: float = Field(..., description="Net amount for the item (post discount)")
    item_rate: Optional[float] = Field(None, description="Rate per unit (if available)")
    item_quantity: Optional[float] = Field(None, description="Quantity (if available)")
    confidence: Optional[str] = Field(None, description="Optional confidence label (low/medium/high)")

class PageItems(BaseModel):
    page_no: str = Field(..., description="Page number as string")
    page_type: str = Field(..., description='One of "Bill Detail" | "Final Bill" | "Pharmacy"')
    bill_items: List[BillItem] = Field(..., description="List of line items found on this page")
    # Optional reconciliation metadata
    reported_total: Optional[float] = Field(None, description="Total amount reported on the page (if any)")
    reconciliation_ok: Optional[bool] = Field(None, description="Whether local reconciliation matched reported_total")
    reconciliation_relative_error: Optional[float] = Field(None, description="Relative error if reconciliation performed")
    preprocessing: Optional[Dict[str, Any]] = Field(None, description="Preprocessing diagnostics (white_coverage, dimensions, etc.)")
    fraud_flags: Optional[List[Dict[str, Any]]] = Field(None, description="Fraud detection flags (whiteout, low OCR confidence, ELA suspect)")

# Backwards-compatible alias: some code expects name PagewiseLineItem
PagewiseLineItem = PageItems


class DataResponse(BaseModel):
    pagewise_line_items: List[PageItems] = Field(..., description="All pages with extracted items")
    total_item_count: int = Field(..., description="Total count of extracted items across pages")
    reconciled_amount: Optional[float] = Field(None, description="Sum of item amounts after reconciliation")

# Compatibility alias: api.py uses ExtractionData
ExtractionData = DataResponse


class FullResponse(BaseModel):
    is_success: bool = Field(..., description="True if the response follows the expected schema")
    token_usage: Dict[str, Any] = Field(..., description="Token usage object (total,input,output) - keep zeros if not used")
    data: DataResponse = Field(..., description="Extraction result payload")


# Additional compatibility aliases (in case api.py used different names)
# If api.py expects PagewiseLineItem (singular) or PagewiseLineItems, map them
PagewiseLineItems = List[PageItems]  # type alias convenience (not a Pydantic model)
Pagewise_Line_Item = PageItems  # alternative name (rare)
