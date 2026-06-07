from __future__ import annotations
from pydantic import BaseModel
from schemas.common import DataResponse


class Alert(BaseModel):
    type: str
    severity: str
    message: str
    run_ts: str
    log_filename: str


AlertListResponse = DataResponse[Alert]
