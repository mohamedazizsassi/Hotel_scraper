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
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from . import config
from .assemble import assemble_features
from .mongo_loader import load_raw_from_mongo, load_raw_from_mongo_chunked
from .validators import validate_features
from .writers import write_parquet_snapshot, write_postgres, write_postgres_append

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
    """Chunked full-reprocess path (--full-reprocess)."""
    timings: dict[str, float] = {}
    all_features: list[pd.DataFrame] = []
    rows_written: int = 0
    first_chunk = True

    logger.info("Starting chunked full-reprocess (chunk_size=%d)", args.chunk_size)

    t0_total = time.perf_counter()

    try:
        for chunk_idx, raw_chunk in enumerate(
            load_raw_from_mongo_chunked(
                mongo_uri=args.mongo_uri,
                chunk_size=args.chunk_size,
                limit=args.limit,
            )
        ):
            logger.info("Processing chunk %d (rows=%d)", chunk_idx + 1, len(raw_chunk))

            t0 = time.perf_counter()
            features_chunk = assemble_features(raw_chunk)
            timings[f"assemble_chunk_{chunk_idx}"] = time.perf_counter() - t0

            all_features.append(features_chunk)

            # Write to Postgres per chunk (TRUNCATE on first, INSERT only after).
            if not args.parquet_only:
                t0 = time.perf_counter()
                if first_chunk:
                    rows_written += write_postgres(
                        features_chunk, args.postgres_uri,
                    )
                    first_chunk = False
                else:
                    rows_written += write_postgres_append(
                        features_chunk, args.postgres_uri,
                    )
                timings[f"postgres_chunk_{chunk_idx}"] = time.perf_counter() - t0

    except Exception as e:
        logger.error("Chunked processing failed: %s", e, exc_info=True)
        return 1

    if not all_features:
        logger.error("No chunks were processed")
        return 1

    # Accumulate all chunks for final Parquet snapshot.
    features = pd.concat(all_features, ignore_index=True)
    logger.info("Accumulated %d rows across %d chunks", len(features), len(all_features))

    # Validate final result (optional).
    if args.validate:
        t0 = time.perf_counter()
        validate_kwargs = {"reports_dir": args.artifacts_dir / "reports"}
        if args.leakage_sample is not None:
            validate_kwargs["sample_size"] = args.leakage_sample
        report = validate_features(features, **validate_kwargs)
        timings["validate"] = time.perf_counter() - t0
        if not report.passed:
            logger.error(
                "Validation FAILED with %d failure(s). Postgres already written; "
                "no parquet write.",
                len(report.failures),
            )
            for msg in report.failures:
                logger.error("  - %s", msg)
            return 1

    # Write final Parquet snapshot.
    parquet_path: Path | None = None
    if not args.postgres_only:
        t0 = time.perf_counter()
        parquet_path = write_parquet_snapshot(
            features, args.artifacts_dir, overwrite=args.overwrite,
        )
        timings["parquet"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - t0_total
    _print_summary(features, timings, parquet_path, rows_written)
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
