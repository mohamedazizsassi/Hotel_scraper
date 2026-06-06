"""Backfill/refresh scrape_runs from scraper/logs/*.log. Idempotent (UPSERT on
log_filename). Read-only on the scraper.

Run with the default logs dir:   python -m scripts.load_scrape_runs
Or point at a specific dir:       python -m scripts.load_scrape_runs "C:/path/to/scraper/logs"
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import ScrapeRun
from db.session import AsyncSessionLocal
from scripts.scrape_log_parser import ScrapeRunRecord, parse_log_file

# backend/scripts/load_scrape_runs.py -> parents[2] == repo root
DEFAULT_LOGS_DIR = Path(__file__).resolve().parents[2] / "scraper" / "logs"


def _record_to_values(rec: ScrapeRunRecord) -> dict:
    return {
        "run_ts": rec.run_ts,
        "log_filename": rec.log_filename,
        "source": rec.source,
        "spiders_count": rec.spiders_count,
        "items_total": rec.items_total,
        "errors_total": rec.errors_total,
        "duration_s": rec.duration_s,
        "status": rec.status,
    }


async def upsert_run(session: AsyncSession, rec: ScrapeRunRecord) -> None:
    values = _record_to_values(rec)
    stmt = pg_insert(ScrapeRun).values(**values)
    update_cols = {k: v for k, v in values.items() if k != "log_filename"}
    stmt = stmt.on_conflict_do_update(index_elements=["log_filename"], set_=update_cols)
    await session.execute(stmt)


async def load_scrape_runs(logs_dir: Path, session: AsyncSession) -> int:
    files = sorted(Path(logs_dir).glob("run_*.log"))
    for path in files:
        await upsert_run(session, parse_log_file(path))
    await session.commit()
    return len(files)


async def main(logs_dir: Path = DEFAULT_LOGS_DIR) -> None:
    async with AsyncSessionLocal() as session:
        n = await load_scrape_runs(logs_dir, session)
        print(f"Loaded/updated {n} scrape run logs from {logs_dir}")


if __name__ == "__main__":
    import sys
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOGS_DIR
    asyncio.run(main(target))
