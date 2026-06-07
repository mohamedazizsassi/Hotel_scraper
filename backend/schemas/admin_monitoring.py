from __future__ import annotations
from pydantic import BaseModel
from schemas.common import DataResponse


class MonitoringSummary(BaseModel):
    total_rows: int | None
    logged_window_items: int
    runs_count: int
    finished_runs: int
    failed_runs: int
    latest_scrape_at: str | None
    last_run_status: str | None
    last_run_items: int | None
    hotels_scraped_distinct: int


class ScrapeRunRow(BaseModel):
    run_ts: str
    log_filename: str
    source: str | None
    items_total: int
    errors_total: int
    duration_s: float | None
    status: str


class DailyRow(BaseModel):
    day: str
    items_total: int
    runs: int


ScrapeRunListResponse = DataResponse[ScrapeRunRow]
DailyResponse = DataResponse[DailyRow]
