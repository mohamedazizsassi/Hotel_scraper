"""
Rebuild Postgres serving indexes on hotel_features — standalone CLI.

The normal pipeline (``build_features.py``) already calls
``writers.add_serving_indexes`` at the end of every run, so this script
is an escape hatch:

* Indexes were dropped manually (e.g. while debugging a query plan).
* You want to add or modify the index set without rerunning the 21h
  full reprocess.
* You changed ``writers.HOTEL_FEATURES_SERVING_INDEXES`` and need to
  re-materialise the indexes against the existing table.

Run from ml/ with:

    .\\.venv\\Scripts\\python.exe -m feature_engineering.rebuild_indexes
"""
from __future__ import annotations

import os

from sqlalchemy import create_engine, text

from .writers import HOTEL_FEATURES_SERVING_INDEXES, add_serving_indexes


def main() -> int:
    uri = os.environ.get(
        "POSTGRES_URI", "postgresql://revway:REDACTED@localhost:5432/revway"
    )
    engine = create_engine(uri)

    print(
        f"Building {len(HOTEL_FEATURES_SERVING_INDEXES)} serving indexes "
        f"on hotel_features ..."
    )
    add_serving_indexes(engine, table_name="hotel_features")

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT indexname FROM pg_indexes "
            "WHERE schemaname='public' AND tablename='hotel_features' "
            "ORDER BY indexname"
        )).fetchall()
        print(f"\nIndexes now on hotel_features ({len(rows)}):")
        for (name,) in rows:
            print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
