"""Read-only parser for Scrapy run logs in scraper/logs/.

Each run log holds many 'Dumping Scrapy stats:' blocks (one per spider). We
aggregate them into a single ScrapeRunRecord per file. No scraper code is
touched; this only reads the logs already written to disk.
"""
from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from pathlib import Path

_STATS_MARKER = "Dumping Scrapy stats:"
_ITEMS_RE = re.compile(r"'item_scraped_count':\s*(\d+)")
_ERRORS_RE = re.compile(r"'log_count/ERROR':\s*(\d+)")
_ELAPSED_RE = re.compile(r"'elapsed_time_seconds':\s*([\d.]+)")
_FINISH_RE = re.compile(r"'finish_reason':\s*'([^']+)'")
_FILENAME_RE = re.compile(r"run_(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})")


@dataclass
class ScrapeRunRecord:
    run_ts: datetime.datetime
    log_filename: str
    source: str | None
    spiders_count: int
    items_total: int
    errors_total: int
    duration_s: float
    status: str


def parse_run_ts(filename: str) -> datetime.datetime:
    """Parse the scheduled run time from 'run_YYYY-MM-DD_HH-MM(.log)'.

    The filename carries the scheduler's local clock; we stamp it as UTC for
    storage/display consistency (it is a label, not an instant to convert)."""
    m = _FILENAME_RE.search(filename)
    if not m:
        raise ValueError(f"Cannot parse run timestamp from filename: {filename!r}")
    y, mo, d, h, mi = (int(g) for g in m.groups())
    return datetime.datetime(y, mo, d, h, mi, tzinfo=datetime.timezone.utc)


def detect_source(text: str) -> str | None:
    """Detect which spider family produced the run from '<spider> starting' lines."""
    has_promo = "promohotel starting" in text
    has_tunisie = "tunisiepromo starting" in text
    if has_promo and has_tunisie:
        return "mixed"
    if has_promo:
        return "promohotel"
    if has_tunisie:
        return "tunisiepromo"
    return None


def _status_from_finishes(finishes: list[str | None]) -> str:
    if not finishes:
        return "failed"
    finished = [f for f in finishes if f == "finished"]
    if len(finished) == len(finishes):
        return "finished"
    if finished:
        return "partial"
    return "failed"


def parse_log_text(text: str, filename: str) -> ScrapeRunRecord:
    blocks = text.split(_STATS_MARKER)[1:]  # element 0 is the preamble
    items_total = 0
    errors_total = 0
    max_elapsed = 0.0
    finishes: list[str | None] = []
    for block in blocks:
        m_items = _ITEMS_RE.search(block)
        items_total += int(m_items.group(1)) if m_items else 0
        m_err = _ERRORS_RE.search(block)
        errors_total += int(m_err.group(1)) if m_err else 0
        m_el = _ELAPSED_RE.search(block)
        if m_el:
            max_elapsed = max(max_elapsed, float(m_el.group(1)))
        m_fin = _FINISH_RE.search(block)
        finishes.append(m_fin.group(1) if m_fin else None)

    return ScrapeRunRecord(
        run_ts=parse_run_ts(filename),
        log_filename=filename,
        source=detect_source(text),
        spiders_count=len(blocks),
        items_total=items_total,
        errors_total=errors_total,
        duration_s=max_elapsed,
        status=_status_from_finishes(finishes),
    )


def parse_log_file(path: Path) -> ScrapeRunRecord:
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_log_text(text, path.name)
