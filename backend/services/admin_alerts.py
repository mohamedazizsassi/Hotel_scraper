from __future__ import annotations
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from schemas.admin_alert import Alert

# One row per run with: the run fields + the median items_total over FINISHED runs.
_RUNS_SQL = text("""
    SELECT run_ts::text AS run_ts, log_filename, items_total, errors_total, status,
           (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY items_total)
            FROM scrape_runs WHERE status = 'finished') AS median_items
    FROM scrape_runs
    ORDER BY run_ts DESC
""")


async def list_alerts(db: AsyncSession) -> list[Alert]:
    rows = (await db.execute(_RUNS_SQL)).mappings().fetchall()
    alerts: list[Alert] = []
    for r in rows:
        if r["status"] != "finished":
            alerts.append(Alert(
                type="failed_run", severity="error",
                message=f"Run {r['log_filename']} did not finish (status={r['status']}).",
                run_ts=r["run_ts"], log_filename=r["log_filename"]))
            continue
        median = r["median_items"]
        if median and r["items_total"] < 0.5 * float(median):
            alerts.append(Alert(
                type="low_volume", severity="warning",
                message=(f"Run {r['log_filename']} collected {r['items_total']} rows, "
                         f"well below the median ({int(median)})."),
                run_ts=r["run_ts"], log_filename=r["log_filename"]))
    return alerts
