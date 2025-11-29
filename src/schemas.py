from typing import List, Optional, Any
from pydantic import BaseModel

class DocumentRequest(BaseModel):
    document: str

class BillItem(BaseModel):
    item_name: str
    item_amount: float
    item_rate: float
    item_quantity: float

class PagewiseItem(BaseModel):
    page_no: str
    bill_items: List[BillItem]

class ExtractionData(BaseModel):
    pagewise_line_items: List[PagewiseItem]
    total_item_count: int
    reconciled_amount: float

class FullResponse(BaseModel):
    is_success: bool
    data: ExtractionData
    # We removed 'token_usage' and 'fraud_flags' to match the strict sample provided