"""
Feature pipeline CLI with two paths: single-load (incremental) and chunked (full reprocess).

**Single-load path** (default, for incremental daily runs):
    MongoDB raw read (filtered)
      → assemble_features (clean / supplements / taxonomy / calendar / competitive / demand)
      → validate_features (optional; failure exits non-zero BEFORE any write)
      → write_parquet_snapshot (immutable training input)
      → write_postgres (atomic TRUNCATE+INSERT into hotel_features)

**Chunked path** (--full-reprocess, for safe reprocessing of large collections):
    For each chunk from MongoDB:
      → assemble_features
      → write_postgres (TRUNCATE on first, INSERT only after)
    Accumulate all chunks:
      → validate_features (optional)
      → write_parquet_snapshot

Daily incremental (loads only new rows since last run)::

    python -m feature_engineering.build_features \\
        --mongo-uri    $MONGO_URI \\
        --postgres-uri $POSTGRES_URI \\
        --artifacts-dir ./artifacts \\
        --incremental

Full reprocess (chunked to avoid RAM pressure, handles all 15M+ rows)::

    python -m feature_engineering.build_features \\
        --mongo-uri    $MONGO_URI \\
        --postgres-uri $POSTGRES_URI \\
        --artifacts-dir ./artifacts \\
        --full-reprocess [--chunk-size 100000]

Optional flags (both paths)::

    [--since 2026-04-01T00:00:00Z]  # Explicit cutoff (single-load only)
    [--parquet-only]                 # Skip Postgres write
    [--postgres-only]                # Skip Parquet write
    [--validate]                     # Run validators before writes
    [--overwrite]                    # Allow same-day parquet overwrite
    [--limit N]                      # Cap total rows (smoke tests only)
    [--log-level {DEBUG,INFO,WARNING,ERROR}]

Defaults read from ``ml/.env`` via ``config.py``. Per-stage timing is
logged at INFO; the final summary is printed to stdout regardless of log
level so an operator running the CLI sees it without parsing logs.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from . import config
from .assemble import assemble_features
from .mongo_loader import (
    enumerate_scrape_dates,
    load_raw_from_mongo,
    load_raw_from_mongo_chunked,
)
from .validators import merge_reports, validate_features
from .writers import (
    PARQUET_FILENAME_TEMPLATE,
    open_parquet_stream,
    swap_table_atomic,
    write_parquet_snapshot,
    write_postgres,
    write_postgres_append,
)

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR: Path = Path(__file__).resolve().parents[1] / "artifacts"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the feature table from MongoDB and persist to Parquet + Postgres.",
    )
    parser.add_argument(
        "--mongo-uri", default=config.MONGO_URI,
        help="MongoDB URI. Default: $MONGO_URI from ml/.env.",
    )
    parser.add_argument(
        "--postgres-uri", default=config.POSTGRES_URI,
        help="PostgreSQL URI. Default: $POSTGRES_URI from ml/.env.",
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR,
        help="Directory for parquet snapshots and validation reports.",
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help=(
            "Load only rows scraped after the latest scraped_at currently in "
            "the Postgres feature table. Falls back to a full load if the "
            "table is missing or empty. Mutually exclusive with --since and --full-reprocess."
        ),
    )
    parser.add_argument(
        "--since", type=_parse_iso8601, default=None,
        help=(
            "Explicit scraped_after cutoff (ISO 8601, e.g. "
            "2026-04-01T00:00:00Z). Overrides --incremental if both passed. "
            "Mutually exclusive with --full-reprocess."
        ),
    )
    parser.add_argument(
        "--full-reprocess", action="store_true",
        help=(
            "Process all rows from MongoDB in chunks (100k default) to avoid RAM pressure. "
            "Use this for reprocessing the full collection. Mutually exclusive with "
            "--incremental and --since."
        ),
    )
    parser.add_argument(
        "--chunk-size", type=int, default=100_000,
        help="Rows per chunk during --full-reprocess (default 100k).",
    )
    parser.add_argument(
        "--parquet-only", action="store_true",
        help="Write the Parquet snapshot but skip the Postgres write.",
    )
    parser.add_argument(
        "--postgres-only", action="store_true",
        help="Write Postgres but skip the Parquet snapshot.",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run validators before writing. On failure, exit non-zero with no writes.",
    )
    parser.add_argument(
        "--leakage-sample", type=int, default=None,
        help=(
            "Max rows to sample for leakage audit (default 1000). "
            "Reduce for faster validation on large datasets (e.g. 100 for ~3 min)."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow the parquet snapshot to overwrite an existing same-day file.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of raw rows read (smoke tests / EDA only).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Root logger level.",
    )
    return parser


def _parse_iso8601(s: str) -> datetime:
    """argparse type for ``--since``. Returns a tz-aware UTC datetime."""
    try:
        # Accept the trailing-Z form Python<3.11 doesn't parse natively.
        normalised = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalised)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ISO 8601 timestamp: {s!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.parquet_only and args.postgres_only:
        logger.error("--parquet-only and --postgres-only are mutually exclusive")
        return 2

    # Validate mode combinations.
    mode_count = sum([args.full_reprocess, args.incremental, args.since is not None])
    if mode_count > 1:
        logger.error("--full-reprocess, --incremental, and --since are mutually exclusive")
        return 2

    # Route: chunked full-reprocess or single-load path.
    if args.full_reprocess:
        return _main_chunked(args)
    else:
        return _main_single_load(args)


def _main_single_load(args: argparse.Namespace) -> int:
    """Original single-load path (for incremental or small datasets)."""
    timings: dict[str, float] = {}

    scraped_after = _resolve_scraped_after(args)

    # ---- 1. Load -----------------------------------------------------------
    t0 = time.perf_counter()
    raw = load_raw_from_mongo(
        mongo_uri=args.mongo_uri,
        scraped_after=scraped_after,
        limit=args.limit,
    )
    timings["load"] = time.perf_counter() - t0

    # ---- 2. Assemble -------------------------------------------------------
    t0 = time.perf_counter()
    features = assemble_features(raw)
    timings["assemble"] = time.perf_counter() - t0

    # ---- 3. Validate (gate writes) ----------------------------------------
    if args.validate:
        t0 = time.perf_counter()
        validate_kwargs = {"reports_dir": args.artifacts_dir / "reports"}
        if args.leakage_sample is not None:
            validate_kwargs["sample_size"] = args.leakage_sample
        report = validate_features(features, **validate_kwargs)
        timings["validate"] = time.perf_counter() - t0
        if not report.passed:
            logger.error(
                "Validation FAILED with %d failure(s). No writes performed.",
                len(report.failures),
            )
            for msg in report.failures:
                logger.error("  - %s", msg)
            return 1

    # ---- 4. Parquet --------------------------------------------------------
    parquet_path: Path | None = None
    if not args.postgres_only:
        t0 = time.perf_counter()
        parquet_path = write_parquet_snapshot(
            features, args.artifacts_dir, overwrite=args.overwrite,
        )
        timings["parquet"] = time.perf_counter() - t0

    # ---- 5. Postgres -------------------------------------------------------
    rows_written: int | None = None
    if not args.parquet_only:
        t0 = time.perf_counter()
        rows_written = write_postgres(features, args.postgres_uri)
        timings["postgres"] = time.perf_counter() - t0

    _print_summary(features, timings, parquet_path, rows_written)
    return 0


def _main_chunked(args: argparse.Namespace) -> int:
    """
    Full-reprocess path (--full-reprocess), looping one scrape-day at a time.

    The outer loop iterates the distinct UTC scrape-days returned by
    ``enumerate_scrape_dates``. For each day we load only that day's rows
    (incremental read, never a full-collection scan) and run the complete
    assemble pipeline on that isolated slice. This is the correctness
    requirement after the 2026-05-14 C1 fix: peer-group and demand
    aggregates are keyed by ``scrape_date``, so a day must be processed
    as one atomic batch -- splitting a day across batches would yield
    truncated peer sets and silently-wrong aggregates.
    """
    timings: dict[str, float] = {}
    rows_written: int = 0
    first_day = True

    if args.chunk_size != 100_000:
        logger.info(
            "--chunk-size=%d ignored in per-scrape-date mode "
            "(each scrape-day is one atomic chunk).",
            args.chunk_size,
        )

    # Stream each day into a staging table; promote with a transactional
    # rename only after every day succeeds. Closes the half-written
    # hotel_features risk that the day-by-day APPEND approach would have
    # left if a day mid-loop crashed (CLAUDE.md Stage 10 requires atomic
    # writes; the swap restores that guarantee for the chunked path).
    staging_table = f"{config.POSTGRES_FEATURES_TABLE}_staging"

    t0_total = time.perf_counter()

    t0 = time.perf_counter()
    scrape_days = enumerate_scrape_dates(mongo_uri=args.mongo_uri)
    timings["enumerate_days"] = time.perf_counter() - t0

    if not scrape_days:
        logger.error("No scrape-days found in MongoDB; nothing to process.")
        return 1

    logger.info("Starting per-scrape-date full-reprocess: %d days to process", len(scrape_days))

    cumulative_rows = 0
    parquet_stream = None
    parquet_path: Path | None = None
    day_reports: list = []  # list[ValidationReport]

    try:
        for day_idx, day_str in enumerate(scrape_days):
            day_start = datetime.fromisoformat(day_str).replace(tzinfo=timezone.utc)
            day_end = day_start + pd.Timedelta(days=1).to_pytimedelta()

            logger.info(
                "Processing scrape-day %d/%d: %s",
                day_idx + 1, len(scrape_days), day_str,
            )

            t0 = time.perf_counter()
            raw_day = load_raw_from_mongo(
                mongo_uri=args.mongo_uri,
                scraped_after=day_start - pd.Timedelta(microseconds=1).to_pytimedelta(),
                scraped_before=day_end,
                limit=None,
            )
            timings[f"load_day_{day_str}"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            features_day = assemble_features(raw_day)
            timings[f"assemble_day_{day_str}"] = time.perf_counter() - t0

            # Validate per-day BEFORE any write -- fail fast on bad data
            # rather than write a full rebuild and discover the leak later.
            if args.validate:
                t0 = time.perf_counter()
                validate_kwargs: dict = {"reports_dir": None}
                if args.leakage_sample is not None:
                    validate_kwargs["sample_size"] = args.leakage_sample
                day_report = validate_features(features_day, **validate_kwargs)
                timings[f"validate_day_{day_str}"] = time.perf_counter() - t0
                day_reports.append(day_report)
                if not day_report.passed:
                    logger.error(
                        "Validation FAILED on scrape-day %s with %d failure(s). "
                        "Aborting before any write for this day.",
                        day_str, len(day_report.failures),
                    )
                    for msg in day_report.failures:
                        logger.error("  - %s", msg)
                    return 1

            if not args.parquet_only:
                t0 = time.perf_counter()
                if first_day:
                    rows_written += write_postgres(
                        features_day, args.postgres_uri, table_name=staging_table,
                    )
                    first_day = False
                else:
                    rows_written += write_postgres_append(
                        features_day, args.postgres_uri, table_name=staging_table,
                    )
                timings[f"postgres_day_{day_str}"] = time.perf_counter() - t0

            if not args.postgres_only:
                t0 = time.perf_counter()
                if parquet_stream is None:
                    parquet_path = args.artifacts_dir / PARQUET_FILENAME_TEMPLATE.format(
                        date=date.today().isoformat(),
                    )
                    parquet_stream = open_parquet_stream(
                        parquet_path, features_day, overwrite=args.overwrite,
                    )
                parquet_stream.append(features_day)
                timings[f"parquet_day_{day_str}"] = time.perf_counter() - t0

            cumulative_rows += len(features_day)
            # Free per-day frames promptly so memory footprint stays at
            # ~one day even on a 4 GB host.
            del features_day, raw_day

            if args.limit is not None and cumulative_rows >= args.limit:
                logger.info("Reached --limit=%d after day %s; stopping.", args.limit, day_str)
                break

    except Exception as e:
        logger.error("Per-scrape-date processing failed: %s", e, exc_info=True)
        if parquet_stream is not None:
            parquet_stream.close()
        return 1

    if parquet_stream is not None:
        parquet_stream.close()

    if cumulative_rows == 0:
        logger.error("No scrape-days produced any rows")
        return 1

    # Aggregate per-day validation reports and persist a merged file.
    if args.validate and day_reports:
        merged = merge_reports(day_reports)
        reports_dir = args.artifacts_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        merged_path = reports_dir / f"validation_{merged.timestamp}.json"
        merged.to_json_file(merged_path)
        logger.info("merged validation report written to %s", merged_path)
        logger.info(
            "validation summary: rows=%d failures=%d warnings=%d leakage_mismatches=%d",
            merged.n_rows, len(merged.failures), len(merged.warnings),
            merged.leakage_mismatches,
        )

    # Promote staging -> live hotel_features atomically. Only reached
    # if every day succeeded -- partial runs leave the live table alone
    # and the orphan staging table will be DROP+CREATE'd by the next
    # run's day-1 write.
    if not args.parquet_only:
        t0 = time.perf_counter()
        swap_table_atomic(
            args.postgres_uri,
            staging_table=staging_table,
            final_table=config.POSTGRES_FEATURES_TABLE,
        )
        timings["postgres_swap"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - t0_total
    _print_summary_streaming(timings, parquet_path, rows_written, cumulative_rows)
    return 0


def _resolve_scraped_after(args: argparse.Namespace) -> datetime | None:
    """``--since`` wins; otherwise ``--incremental`` queries Postgres."""
    if args.since is not None:
        logger.info("Using explicit --since cutoff: %s", args.since.isoformat())
        return args.since
    if args.incremental:
        cutoff = _latest_scraped_at(args.postgres_uri)
        if cutoff is None:
            logger.info("Incremental requested but Postgres table is empty/missing — full load.")
            return None
        logger.info("Incremental cutoff from Postgres max(scraped_at): %s", cutoff.isoformat())
        return cutoff
    return None


def _latest_scraped_at(postgres_uri: str) -> datetime | None:
    """Return the largest ``scraped_at`` already in the feature table, or None."""
    engine = create_engine(postgres_uri)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(f'SELECT MAX(scraped_at) FROM "{config.POSTGRES_FEATURES_TABLE}"')
            ).scalar_one_or_none()
    except SQLAlchemyError as exc:
        logger.warning("Could not read incremental cutoff from Postgres: %s", exc)
        return None
    finally:
        engine.dispose()

    if row is None:
        return None
    if isinstance(row, datetime):
        return row if row.tzinfo else row.replace(tzinfo=timezone.utc)
    return None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary_streaming(
    timings: dict[str, float],
    parquet_path: Path | None,
    rows_written: int,
    rows_seen: int,
) -> None:
    """
    Console summary for the streaming per-scrape-date build path.

    The streaming path never holds the full concatenated frame in
    memory, so the rich per-column dimensions ``_print_summary`` prints
    are not available. We surface row counts and timings instead -- the
    merged validation report carries the full coverage breakdown.
    """
    bar = "=" * 64
    print(bar)
    print(f"rows seen          : {rows_seen:,}")
    print(f"rows -> postgres   : {rows_written:,}")
    print(f"parquet path       : {parquet_path if parquet_path else '(skipped)'}")
    print("-" * 64)
    print("stage timings (s):")
    # Collapse per-day timings to summary stats so the output stays
    # readable across ~15+ scrape-days.
    by_prefix: dict[str, list[float]] = {}
    other: dict[str, float] = {}
    for name, sec in timings.items():
        for prefix in ("load_day_", "assemble_day_", "validate_day_", "postgres_day_", "parquet_day_"):
            if name.startswith(prefix):
                by_prefix.setdefault(prefix.rstrip("_"), []).append(sec)
                break
        else:
            other[name] = sec
    for name, sec in other.items():
        print(f"  {name:<22} {sec:>8.2f}")
    for prefix, secs in by_prefix.items():
        print(
            f"  {prefix:<22} n={len(secs):<3} sum={sum(secs):>7.1f} "
            f"min={min(secs):>5.1f} max={max(secs):>5.1f}"
        )
    print(bar)


def _print_summary(
    df: pd.DataFrame,
    timings: dict[str, float],
    parquet_path: Path | None,
    rows_written: int | None,
) -> None:
    bar = "=" * 64
    print(bar)
    print(f"rows               : {len(df):,}")
    print(f"columns            : {df.shape[1]}")
    if "hotel_name_normalized" in df.columns:
        print(f"unique hotels      : {df['hotel_name_normalized'].nunique():,}")
    if "city_name" in df.columns:
        print(f"unique cities      : {df['city_name'].nunique()}")
    if "source" in df.columns:
        sources = sorted(df["source"].dropna().unique().tolist())
        print(f"sources            : {sources}")
    if "check_in" in df.columns and df["check_in"].notna().any():
        lo = pd.to_datetime(df["check_in"]).min().date()
        hi = pd.to_datetime(df["check_in"]).max().date()
        print(f"check_in range     : {lo} -> {hi}")
    print("-" * 64)
    print("stage timings (s):")
    for name, sec in timings.items():
        print(f"  {name:<10} {sec:>8.2f}")
    print("-" * 64)
    if parquet_path is not None:
        print(f"parquet snapshot   : {parquet_path}")
    if rows_written is not None:
        print(f"postgres rows      : {rows_written:,} -> {config.POSTGRES_FEATURES_TABLE}")
    print(bar)


if __name__ == "__main__":
    sys.exit(main())
