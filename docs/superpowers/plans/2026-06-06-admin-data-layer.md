# Admin Data Layer & Monitoring Infrastructure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the schema + read-only plumbing the Admin monitoring features need — a `scrape_runs` table populated from the Scrapy logs, hotel contact columns, and a Mongo client for the live `hotel_prices` row count — with zero changes to the scraper.

**Architecture:** New ORM models drive the test DB (`Base.metadata.create_all`); matching raw-SQL migrations drive the real dev DB. A pure log parser turns each `scraper/logs/run_*.log` into one aggregated record; an idempotent async loader upserts those into `scrape_runs`. A thin `motor` wrapper exposes only `estimated_document_count()` (no scans). This is Plan 1 of 3 (data → backend API → frontend).

**Tech Stack:** FastAPI/SQLAlchemy 2.0 async, asyncpg, PostgreSQL, `motor` (async MongoDB), pytest + pytest-asyncio (`asyncio_mode=auto`).

---

## Conventions for every backend command

- Run from the `backend/` directory with its venv active:
  `cd backend; .\.venv\Scripts\Activate.ps1` (PowerShell) — the project is on Windows.
- Tests: `pytest tests/<file>::<test> -v`. The test DB is `settings.test_db_url`
  (`postgresql+asyncpg://revway:REDACTED@localhost:5432/revway_test`), already
  configured in `backend/.env`. The autouse `setup_test_db` fixture drops+creates
  **all ORM tables** at session start — so new ORM models appear automatically.
- Commit messages: Conventional Commits, **no `Co-Authored-By: Claude` trailer**
  (project rule).

## Prerequisites

- [ ] **Branch:** create/switch to `feat/admin-platform` (off `main`) for all three
  admin plans. Untracked dirs (`backend/`, `frontend/`) carry across the switch;
  if a tracked-file checkout conflict appears, stash or commit the unrelated
  in-flight ML changes first. (If using a worktree per `superpowers:using-git-worktrees`, create it now.)

## Deferred to later plans (do NOT do here)

- Mongo client lifespan open/close + `/health` reporting → **Plan 2** (when an
  endpoint first uses it).
- `get_current_admin`, admin routers/services/schemas → **Plan 2**.
- Any frontend work → **Plan 3**.

---

### Task 1: `scrape_runs` table (ORM model + migration 003)

**Files:**
- Modify: `backend/db/models.py`
- Create: `database/postgres/migrations/003_create_scrape_runs.sql`
- Test: `backend/tests/test_scrape_runs_model.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_scrape_runs_model.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scrape_runs_model.py -v`
Expected: FAIL — `ImportError: cannot import name 'ScrapeRun' from 'db.models'`.

- [ ] **Step 3: Add the ORM model**

In `backend/db/models.py`, extend the top-level import line:

```python
from sqlalchemy import String, Boolean, Integer, SmallInteger, DateTime, ForeignKey, Numeric, func
```

Then append at the end of the file:

```python
class ScrapeRun(Base):
    __tablename__ = "scrape_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_ts: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    log_filename: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    source: Mapped[Optional[str]] = mapped_column(String)
    spiders_count: Mapped[int] = mapped_column(Integer, default=0)
    items_total: Mapped[int] = mapped_column(Integer, default=0)
    errors_total: Mapped[int] = mapped_column(Integer, default=0)
    duration_s: Mapped[Optional[float]] = mapped_column(Numeric)
    status: Mapped[str] = mapped_column(String, nullable=False)
    ingested_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scrape_runs_model.py -v`
Expected: PASS.

- [ ] **Step 5: Create migration 003 (for the real dev DB)**

Create `database/postgres/migrations/003_create_scrape_runs.sql`:

```sql
-- =========================================================
-- 003_create_scrape_runs.sql
-- =========================================================
-- Per-run scrape statistics parsed read-only from scraper/logs/*.log.
-- One row per log file (= one scheduled run). Populated by
-- backend/scripts/load_scrape_runs.py. The scraper is NOT modified.

CREATE TABLE scrape_runs (
    id            INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_ts        TIMESTAMPTZ NOT NULL,        -- parsed from the filename
    log_filename  TEXT NOT NULL UNIQUE,        -- idempotency key
    source        TEXT,                         -- promohotel / tunisiepromo / mixed
    spiders_count INTEGER NOT NULL DEFAULT 0,
    items_total   INTEGER NOT NULL DEFAULT 0,   -- = rows inserted into hotel_prices
    errors_total  INTEGER NOT NULL DEFAULT 0,
    duration_s    NUMERIC,
    status        TEXT NOT NULL,                -- finished / partial / failed
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_scrape_runs_ts ON scrape_runs (run_ts);

INSERT INTO schema_migrations (version, description)
VALUES ('003', 'scrape_runs: per-run scrape stats parsed from logs');
```

- [ ] **Step 6: Commit**

```bash
git add backend/db/models.py database/postgres/migrations/003_create_scrape_runs.sql backend/tests/test_scrape_runs_model.py
git commit -m "feat(db): scrape_runs table (ORM model + migration 003)"
```

---

### Task 2: Hotel contact columns (ORM + migration 004)

**Files:**
- Modify: `backend/db/models.py` (`PlatformHotel`)
- Create: `database/postgres/migrations/004_add_hotel_contact_columns.sql`
- Test: `backend/tests/test_hotel_contact_fields.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_hotel_contact_fields.py`:

```python
from sqlalchemy import select
from db.models import PlatformHotel


async def test_hotel_contact_fields_persist(db_session):
    res = await db_session.execute(
        select(PlatformHotel).where(PlatformHotel.hotel_name_normalized == "hotel_comp_1")
    )
    hotel = res.scalar_one()
    hotel.contact_email = "contact@hotelcomp1.tn"
    hotel.contact_phone = "+216 71 000 000"
    await db_session.commit()

    # Force the re-read to hit the DB instead of the session identity map. Without
    # this, SQLAlchemy returns the same in-memory object and the assertions pass
    # even if the columns were never mapped/persisted (false positive).
    db_session.expunge_all()

    res2 = await db_session.execute(
        select(PlatformHotel).where(PlatformHotel.hotel_name_normalized == "hotel_comp_1")
    )
    reread = res2.scalar_one()
    assert reread.contact_email == "contact@hotelcomp1.tn"
    assert reread.contact_phone == "+216 71 000 000"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hotel_contact_fields.py -v`
Expected: FAIL — `AttributeError: 'PlatformHotel' object has no attribute 'contact_email'`.

- [ ] **Step 3: Add the columns to the ORM model**

In `backend/db/models.py`, inside `class PlatformHotel`, after the `is_active` line, add:

```python
    contact_email: Mapped[Optional[str]] = mapped_column(String)
    contact_phone: Mapped[Optional[str]] = mapped_column(String)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hotel_contact_fields.py -v`
Expected: PASS.

- [ ] **Step 5: Create migration 004**

Create `database/postgres/migrations/004_add_hotel_contact_columns.sql`:

```sql
-- =========================================================
-- 004_add_hotel_contact_columns.sql
-- =========================================================
-- Admin-editable contact info on platform_hotels (not scraped).

ALTER TABLE platform_hotels
    ADD COLUMN contact_email TEXT,
    ADD COLUMN contact_phone TEXT;

INSERT INTO schema_migrations (version, description)
VALUES ('004', 'platform_hotels: contact_email, contact_phone');
```

- [ ] **Step 6: Commit**

```bash
git add backend/db/models.py database/postgres/migrations/004_add_hotel_contact_columns.sql backend/tests/test_hotel_contact_fields.py
git commit -m "feat(db): platform_hotels contact columns (ORM + migration 004)"
```

---

### Task 3: Scrapy-log parser (pure functions)

**Files:**
- Create: `backend/scripts/__init__.py`
- Create: `backend/scripts/scrape_log_parser.py`
- Test: `backend/tests/test_scrape_log_parser.py`

The parser turns one log file into one `ScrapeRunRecord`. Each file contains many
`Dumping Scrapy stats:` blocks (one per spider). We aggregate: `items_total` =
Σ `item_scraped_count` (= rows inserted into Mongo, because the pipeline is
insert-only), `errors_total` = Σ `log_count/ERROR`, `duration_s` = max
`elapsed_time_seconds`, `spiders_count` = number of blocks, `status` from the
`finish_reason` values, `source` from the `"<spider> starting"` log lines.

- [ ] **Step 1: Write the failing tests**

Create `backend/scripts/__init__.py` (empty file, makes `scripts` importable):

```python
```

Create `backend/tests/test_scrape_log_parser.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scrape_log_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.scrape_log_parser'`.

- [ ] **Step 3: Implement the parser**

Create `backend/scripts/scrape_log_parser.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scrape_log_parser.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/__init__.py backend/scripts/scrape_log_parser.py backend/tests/test_scrape_log_parser.py
git commit -m "feat(monitoring): read-only Scrapy log parser"
```

---

### Task 4: Idempotent loader (logs → `scrape_runs`)

**Files:**
- Create: `backend/scripts/load_scrape_runs.py`
- Test: `backend/tests/test_load_scrape_runs.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_load_scrape_runs.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_load_scrape_runs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.load_scrape_runs'`.

- [ ] **Step 3: Implement the loader**

Create `backend/scripts/load_scrape_runs.py`:

```python
"""Backfill/refresh scrape_runs from scraper/logs/*.log. Idempotent (UPSERT on
log_filename). Read-only on the scraper. Run: python -m scripts.load_scrape_runs
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import ScrapeRun
from db.session import AsyncSessionLocal
from scripts.scrape_log_parser import ScrapeRunRecord, parse_log_file

# backend/scripts/load_scrape_runs.py -> parents[2] == repo root (revway/)
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
```

> Note: `main()` takes an optional logs-dir CLI argument because the real
> `scraper/logs` lives outside this worktree (logs are not committed). Task 6
> passes that path explicitly.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_load_scrape_runs.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/load_scrape_runs.py backend/tests/test_load_scrape_runs.py
git commit -m "feat(monitoring): idempotent scrape_runs loader"
```

---

### Task 5: Mongo client for the live row count

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/core/config.py`
- Modify: `backend/.env.example`
- Create: `backend/db/mongo.py`
- Test: `backend/tests/test_mongo_client.py`

- [ ] **Step 1: Add `motor` and install it**

Append to `backend/requirements.txt`:

```
motor==3.6.0
```

Run: `pip install motor==3.6.0`
Expected: installs `motor` and its `pymongo` dependency. (If `3.6.0` is
unavailable in your index, use the latest `motor` 3.x and pin that.)

- [ ] **Step 2: Add Mongo settings to config**

In `backend/core/config.py`, inside `class Settings`, after the `test_db_url`
line, add:

```python
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "hotel_scraper"
```

- [ ] **Step 3: Write the failing test**

Create `backend/tests/test_mongo_client.py`:

```python
from unittest.mock import AsyncMock, MagicMock
from db.mongo import count_hotel_prices


async def test_count_hotel_prices_returns_count():
    coll = MagicMock()
    coll.estimated_document_count = AsyncMock(return_value=24_400_000)
    fake_db = {"hotel_prices": coll}
    assert await count_hotel_prices(fake_db) == 24_400_000


async def test_count_hotel_prices_returns_none_on_error():
    coll = MagicMock()
    coll.estimated_document_count = AsyncMock(side_effect=RuntimeError("mongo down"))
    fake_db = {"hotel_prices": coll}
    assert await count_hotel_prices(fake_db) is None
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_mongo_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'db.mongo'`.

- [ ] **Step 5: Implement the Mongo client**

Create `backend/db/mongo.py`:

```python
"""Thin async MongoDB client. Used ONLY for the live hotel_prices total via
estimated_document_count() (reads collection metadata — no scan). Degrades to
None if Mongo is unreachable so the rest of the (PG-sourced) API still works.
"""
from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient

from core.config import settings

_client: AsyncIOMotorClient | None = None


def get_mongo_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongo_uri, serverSelectionTimeoutMS=3000)
    return _client


def get_mongo_db():
    return get_mongo_client()[settings.mongo_db]


async def count_hotel_prices(db) -> int | None:
    """Estimated row count of hotel_prices. Returns None if Mongo errors out."""
    try:
        return await db["hotel_prices"].estimated_document_count()
    except Exception:
        return None


def close_mongo_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_mongo_client.py -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Update the env template**

Append to `backend/.env.example`:

```
# MongoDB (read-only: live hotel_prices total for the admin monitoring page)
MONGO_URI=mongodb://localhost:27017
MONGO_DB=hotel_scraper
```

Also add the same two keys to your local (gitignored) `backend/.env` so the
running app/loader pick up the real values.

- [ ] **Step 8: Commit**

```bash
git add backend/requirements.txt backend/core/config.py backend/db/mongo.py backend/tests/test_mongo_client.py backend/.env.example
git commit -m "feat(monitoring): async Mongo client for live hotel_prices count"
```

---

### Task 6: Apply migrations + backfill against the real dev DB (verification)

No new code — this proves the data layer works end-to-end on the real database.
Uses the documented dev creds (`revway` / `root`). Adjust the connection string
if your `backend/.env` differs.

- [ ] **Step 1: Run the full backend test suite (no regressions)**

Run: `pytest -q`
Expected: all tests pass, including the new `test_scrape_runs_model`,
`test_hotel_contact_fields`, `test_scrape_log_parser`, `test_load_scrape_runs`,
`test_mongo_client`, plus the pre-existing auth/recommendation/ml_store tests.

- [ ] **Step 2: Apply migrations 003 and 004 to the dev DB**

Run (from repo root):

```bash
psql "postgresql://revway:REDACTED@localhost:5432/revway" -f database/postgres/migrations/003_create_scrape_runs.sql
psql "postgresql://revway:REDACTED@localhost:5432/revway" -f database/postgres/migrations/004_add_hotel_contact_columns.sql
```

Expected: `CREATE TABLE`, `CREATE INDEX`, `INSERT 0 1` for 003; `ALTER TABLE`,
`INSERT 0 1` for 004. (If `schema_migrations` already has '003'/'004', the
INSERT errors on the PK — safe to ignore on re-apply.)

- [ ] **Step 3: Backfill scrape_runs from the real logs**

The logs live in the **original tree** (not committed, so absent from the
worktree). Pass that path explicitly:

Run (from `backend/`, venv active):
`python -m scripts.load_scrape_runs "C:\Users\ASUS\Desktop\PFE\revway\scraper\logs"`
Expected: `Loaded/updated 88 scrape run logs from C:\...\scraper\logs`
(the count equals the number of `run_*.log` files; ~88 at time of writing).

- [ ] **Step 4: Spot-check the loaded data**

Run:

```bash
psql "postgresql://revway:REDACTED@localhost:5432/revway" -c "SELECT count(*), min(run_ts)::date, max(run_ts)::date, sum(items_total) FROM scrape_runs;"
```

Expected: row count ≈ 88, dates spanning 2026-04-16 → 2026-06-05, and a
`sum(items_total)` in the low millions. **NOTE (measured 2026-06-06):** this is
the yield of the *logged* runs only, NOT the full collection. One sampled run
(promohotel, `2026-06-05_10-00`) scraped 26,085 items against 43,004 errors, and
the 88 logs do not span the collection's whole history — so `sum(items_total)`
will be well below the Mongo `hotel_prices` total (~24M). Treat the Mongo count
as the authoritative total and per-run `items_total` as per-run yield; do **not**
expect them to reconcile. The last log (`2026-06-05_15-00`) is an aborted run
with no stats blocks → it correctly loads as `status='failed'`.

---

## Self-review

**Spec coverage (vs §3, §5.3, §6 of the design spec):**
- D3 `scrape_runs` table + loader → Tasks 1, 3, 4. ✓
- D6 Mongo client (`motor`, count-only) → Task 5. ✓
- §6.1 migration 003 → Task 1. ✓
- §6.2 migration 004 (contact columns) → Task 2. ✓
- §6.3 idempotent loader, run_ts from filename, status derivation, source detection → Tasks 3, 4. ✓
- D5 "items_total = rows added" (insert-only justification) → encoded in parser docstring + tests. ✓
- Backfill of all 88 logs → Task 6. ✓
- Deferred items (lifespan/health, `get_current_admin`, endpoints, frontend) explicitly listed as out of scope. ✓

**Placeholder scan:** none — every code/SQL/command step is complete.

**Type consistency:** `ScrapeRunRecord` fields (`run_ts, log_filename, source, spiders_count, items_total, errors_total, duration_s, status`) match the `ScrapeRun` ORM columns and the migration columns; `_record_to_values` maps exactly those keys; the loader's `index_elements=["log_filename"]` matches the `UNIQUE` constraint on `log_filename`. `parse_log_text`/`parse_run_ts`/`detect_source` signatures match their test imports. `count_hotel_prices(db)` indexes `db["hotel_prices"]`, matching the test's `fake_db` dict.
