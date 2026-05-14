#!/usr/bin/env python
"""Clear the hotel_features table from PostgreSQL."""
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv('.env')

postgres_uri = os.environ.get('POSTGRES_URI', 'postgresql://revway:root@localhost:5432/revway')
table_name = 'hotel_features'

try:
    engine = create_engine(postgres_uri)
    with engine.connect() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))
        conn.commit()
    print(f'OK: Cleared table {table_name}')
    exit(0)
except Exception as e:
    print(f'ERROR: {e}')
    exit(1)
finally:
    engine.dispose()
