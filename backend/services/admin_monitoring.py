from __future__ import annotations
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from schemas.admin_monitoring import MonitoringSummary

_SUMMARY_SQL = text("""
    SELECT
        COALESCE(SUM(items_total), 0)                              AS logged_window_items,
        COUNT(*)                                                   AS runs_count,
        COUNT(*) FILTER (WHERE status = 'finished')                AS finished_runs,
        COUNT(*) FILTER (WHERE status <> 'finished')               AS failed_runs,
        MAX(run_ts)::text                                          AS latest_scrape_at
    FROM scrape_runs
""")

_LAST_RUN_SQL = text("""
    SELECT status, items_total
    FROM scrape_runs ORDER BY run_ts DESC LIMIT 1
""")


async def build_summary(db: AsyncSession, total_rows: int | None) -> MonitoringSummary:
    agg = (await db.execute(_SUMMARY_SQL)).mappings().one()
    last = (await db.execute(_LAST_RUN_SQL)).mappings().first()
    hotels = await db.scalar(
        text("SELECT COUNT(DISTINCT hotel_name_normalized) FROM hotel_features"))
    return MonitoringSummary(
        total_rows=total_rows,
        logged_window_items=int(agg["logged_window_items"]),
        runs_count=int(agg["runs_count"]),
        finished_runs=int(agg["finished_runs"]),
        failed_runs=int(agg["failed_runs"]),
        latest_scrape_at=agg["latest_scrape_at"],
        last_run_status=last["status"] if last else None,
        last_run_items=last["items_total"] if last else None,
        hotels_scraped_distinct=int(hotels or 0),
    )
