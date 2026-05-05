#!/usr/bin/env python
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
uri = os.getenv('POSTGRES_URI', 'postgresql://revway:change_me@localhost:5432/revway')

try:
    engine = create_engine(uri)
    with engine.connect() as conn:
        # Check if table exists
        result = conn.execute(text("""
            SELECT EXISTS(
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'hotel_features'
            ) AS table_exists
        """)).scalar()

        if result:
            # Get row count and stats
            count = conn.execute(text('SELECT COUNT(*) FROM hotel_features')).scalar()
            cols = conn.execute(text('SELECT COUNT(*) FROM information_schema.columns WHERE table_name = \'hotel_features\'')).scalar()

            # Get date range
            min_date = conn.execute(text('SELECT MIN(scraped_at) FROM hotel_features')).scalar()
            max_date = conn.execute(text('SELECT MAX(scraped_at) FROM hotel_features')).scalar()

            # Get unique hotels and cities
            hotels = conn.execute(text('SELECT COUNT(DISTINCT hotel_name_normalized) FROM hotel_features')).scalar()
            cities = conn.execute(text('SELECT COUNT(DISTINCT city_name) FROM hotel_features')).scalar()

            print('[OK] hotel_features table EXISTS')
            print(f'  Rows: {count:,}')
            print(f'  Columns: {cols}')
            print(f'  Unique hotels: {hotels:,}')
            print(f'  Unique cities: {cities}')
            print(f'  Date range: {min_date} to {max_date}')
        else:
            print('[MISSING] hotel_features table does NOT exist')
except Exception as e:
    print(f'[ERROR] {e}')
finally:
    engine.dispose()
