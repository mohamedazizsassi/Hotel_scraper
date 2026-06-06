import datetime
from sqlalchemy import select
from db.models import ScrapeRun


async def test_scrape_run_round_trip(db_session):
    run = ScrapeRun(
        run_ts=datetime.datetime(2026, 6, 5, 10, 0, tzinfo=datetime.timezone.utc),
        log_filename="run_2026-06-05_10-00.log",
        source="mixed",
        spiders_count=3,
        items_total=100,
        errors_total=5,
        duration_s=12.5,
        status="finished",
    )
    db_session.add(run)
    await db_session.commit()

    res = await db_session.execute(
        select(ScrapeRun).where(ScrapeRun.log_filename == "run_2026-06-05_10-00.log")
    )
    row = res.scalar_one()
    assert row.items_total == 100
    assert row.errors_total == 5
    assert row.status == "finished"
    assert float(row.duration_s) == 12.5
    assert row.id is not None
