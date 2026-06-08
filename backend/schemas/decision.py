# backend/schemas/decision.py
from __future__ import annotations
from datetime import date
from typing import Literal, Optional
from pydantic import BaseModel

DecisionStatus = Literal["accepted", "dismissed"]

class DecisionKey(BaseModel):
    check_in: date
    nights: int
    adults: int
    boarding_canonical: str
    recommended_price_tnd: Optional[float] = None

class DecisionIn(DecisionKey):
    status: DecisionStatus

class DecisionBulkIn(BaseModel):
    status: DecisionStatus
    items: list[DecisionKey]

class DecisionRow(BaseModel):
    check_in: date
    nights: int
    adults: int
    boarding_canonical: str
    recommended_price_tnd: float | None
    status: DecisionStatus

class DecisionBulkOut(BaseModel):
    count: int
