import datetime
from scripts.scrape_log_parser import parse_log_text, parse_run_ts, detect_source

SAMPLE = """2026-06-05 10:00:06 [hotel_scraper.spiders.base] INFO: promohotel starting | city=ain-draham (18) | days=60 | nights=1
2026-06-05 10:02:40 [scrapy.statscollectors] INFO: Dumping Scrapy stats:
{'elapsed_time_seconds': 153.13,
 'finish_reason': 'finished',
 'item_scraped_count': 6,
 'log_count/ERROR': 183}
2026-06-05 10:02:41 [scrapy.statscollectors] INFO: Dumping Scrapy stats:
{'elapsed_time_seconds': 160.50,
 'finish_reason': 'finished',
 'log_count/ERROR': 12}
"""


def test_parse_run_ts_from_filename():
    ts = parse_run_ts("run_2026-06-05_10-00.log")
    assert (ts.year, ts.month, ts.day, ts.hour, ts.minute) == (2026, 6, 5, 10, 0)
    assert ts.tzinfo is not None


def test_parse_aggregates_blocks():
    rec = parse_log_text(SAMPLE, "run_2026-06-05_10-00.log")
    assert rec.spiders_count == 2
    assert rec.items_total == 6           # block 2 has no item_scraped_count -> 0
    assert rec.errors_total == 195        # 183 + 12
    assert rec.duration_s == 160.50       # max elapsed
    assert rec.status == "finished"
    assert rec.source == "promohotel"
    assert rec.log_filename == "run_2026-06-05_10-00.log"


def test_status_partial_when_some_not_finished():
    text = (
        "tunisiepromo starting | city=x\n"
        "Dumping Scrapy stats:\n{'finish_reason': 'finished', 'item_scraped_count': 4, 'log_count/ERROR': 1}\n"
        "Dumping Scrapy stats:\n{'finish_reason': 'shutdown', 'log_count/ERROR': 0}\n"
    )
    rec = parse_log_text(text, "run_2026-05-01_15-00.log")
    assert rec.status == "partial"
    assert rec.source == "tunisiepromo"


def test_status_failed_when_no_blocks():
    rec = parse_log_text("nothing useful here\n", "run_2026-05-02_10-00.log")
    assert rec.spiders_count == 0
    assert rec.status == "failed"
    assert rec.items_total == 0


def test_detect_source_mixed():
    text = "promohotel starting | city=a\n...\ntunisiepromo starting | city=b\n"
    assert detect_source(text) == "mixed"
