from __future__ import annotations
from pydantic import BaseModel


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
