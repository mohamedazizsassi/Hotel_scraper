from sqlalchemy import select, func
from db.models import ScrapeRun
from scripts.load_scrape_runs import load_scrape_runs

LOG_A = """promohotel starting | city=a
Dumping Scrapy stats:
{'elapsed_time_seconds': 100.0, 'finish_reason': 'finished', 'item_scraped_count': 10, 'log_count/ERROR': 2}
"""
LOG_B = """tunisiepromo starting | city=b
Dumping Scrapy stats:
{'elapsed_time_seconds': 50.0, 'finish_reason': 'finished', 'item_scraped_count': 7, 'log_count/ERROR': 1}
"""


async def test_loader_inserts_and_is_idempotent(db_session, tmp_path):
    (tmp_path / "run_2026-01-01_10-00.log").write_text(LOG_A, encoding="utf-8")
    (tmp_path / "run_2026-01-01_15-00.log").write_text(LOG_B, encoding="utf-8")

    n1 = await load_scrape_runs(tmp_path, db_session)
    assert n1 == 2

    names = ["run_2026-01-01_10-00.log", "run_2026-01-01_15-00.log"]
    res = await db_session.execute(
        select(func.count()).select_from(ScrapeRun).where(ScrapeRun.log_filename.in_(names))
    )
    assert res.scalar_one() == 2

    res_a = await db_session.execute(
        select(ScrapeRun).where(ScrapeRun.log_filename == "run_2026-01-01_10-00.log")
    )
    row_a = res_a.scalar_one()
    assert row_a.items_total == 10
    assert row_a.source == "promohotel"
    assert row_a.status == "finished"

    # Second run over the same dir must UPDATE, not duplicate.
    n2 = await load_scrape_runs(tmp_path, db_session)
    assert n2 == 2
    res2 = await db_session.execute(
        select(func.count()).select_from(ScrapeRun).where(ScrapeRun.log_filename.in_(names))
    )
    assert res2.scalar_one() == 2
