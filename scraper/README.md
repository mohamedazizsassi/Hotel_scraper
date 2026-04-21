# Hotel Scraper

Scrapy project for scraping hotel offers and prices from:

- `promohotel.tn`
- `tunisiepromo.tn`

The project stores data in MongoDB and exports training-ready Parquet files.

## Features

- Two spiders:
	- `hotel_scraper/spiders/promohotel_spider.py`
	- `hotel_scraper/spiders/tunisiepromo_spider.py`
- Flexible scrape parameters per request:
	- `city_id`, `days`, `start_day`, `nights` (1/3/5), `adults`, `children`
- Tiered scheduling logic in `run_scrape.py` aligned with API update windows
- Data enrichment fields for ML (`days_until_checkin`, `city_name`, normalized hotel name)
- MongoDB insert pipeline with indexes
- Parquet export pipeline (`.parquet` + snappy compression)

## Project Structure

- `hotel_scraper/items.py`: canonical item schema
- `hotel_scraper/utils.py`: payload builder, date range helper, city mappings
- `hotel_scraper/spiders/promohotel_spider.py`: promohotel spider
- `hotel_scraper/spiders/tunisiepromo_spider.py`: tunisiepromo spider
- `hotel_scraper/pipelines.py`: normalization, duplicate filtering, MongoDB, Parquet export
- `hotel_scraper/settings.py`: Scrapy settings and pipeline order
- `run_scrape.py`: orchestrates all cities/night lengths in tiers
- `run_scrape.bat`: Windows Task Scheduler entrypoint

## Requirements

- Python 3.10+
- MongoDB running locally or remotely (`mongodb://localhost:27017` by default)

Dependencies (from `requirements.txt`):

- `scrapy`
- `pymongo`
- `python-dotenv`

## Setup

1. Create and activate virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Configure environment variables:

```powershell
Copy-Item .envexemple .env
```

Then edit `.env` and set:

- `MONGO_URI`
- `MONGO_DB`

## Run Spiders Manually

List available spiders:

```powershell
python -m scrapy list
```

Run `promohotel`:

```powershell
python -m scrapy crawl promohotel
python -m scrapy crawl promohotel -a city_id=3 -a days=10 -a nights=3 -a adults=2 -a children=0
```

Run `tunisiepromo`:

```powershell
python -m scrapy crawl tunisiepromo
python -m scrapy crawl tunisiepromo -a city_id=34 -a days=10 -a nights=5 -a adults=2 -a children=0
```

Notes:

- `nights` accepts `1`, `3`, `5` (invalid values are coerced to `1` by spiders)
- `start_day` offsets the search window (example: `start_day=60` starts at today+60)

## Scheduled Batch Run

Use one of:

```powershell
python run_scrape.py
```

or

```powershell
run_scrape.bat
```

Scheduling strategy in `run_scrape.py`:

- Morning run (`hour < 12`):
	- Near tier: `start_day=0`, `days=60`
	- Far tier: `start_day=60`, `days=120`
- Afternoon run:
	- Near tier only

For each tier, the script crawls:

- All `PROMOHOTEL_CITIES`
- All `TUNISIEPROMO_CITIES`
- All `nights` options: `(1, 3, 5)`

Each batch run gets a `SCRAPE_RUN_ID` injected in settings and attached to each item.

## Data Model

Key fields in `hotel_scraper/items.py`:

- Metadata: `source`, `scraped_at`, `scrape_run_id`
- Search params: `check_in`, `check_out`, `nights`, `days_until_checkin`, `city_id`, `city_name`, `adults`, `children`
- Hotel: `hotel_name`, `hotel_name_normalized`, `stars`
- Offer: `boarding_name`, `room_name`, `price`, `sur_demande`, `supplements`

## Pipeline Flow

Configured in `hotel_scraper/settings.py`:

1. `NormalizationPipeline` (adds `hotel_name_normalized`)
2. `DuplicateFilterPipeline` (in-memory dedupe within run)
3. `MongoDBPipeline` (insert into `hotel_prices`)
4. `ParquetExportPipeline` (writes batched Parquet file)

## Outputs

- MongoDB collection: `hotel_prices`
- Parquet files in `output/`:
	- `<spider_name>_<timestamp>.parquet`

Parquet schema is explicitly typed in `hotel_scraper/pipelines.py` for ML compatibility.

## MongoDB Indexes

Created automatically by `MongoDBPipeline`:

- `idx_hotel_price_time` on `(source, hotel_name, check_in, boarding_name, room_name, scraped_at)`
- `idx_check_in`
- `idx_source`

## Notes for ML Usage

- Use `days_until_checkin` and `nights` as primary temporal features
- Use `hotel_name_normalized` for cross-source matching and grouping
- Keep `scrape_run_id` to reconstruct one complete market snapshot

## Troubleshooting

- Spider runs but nothing in DB:
	- check logs for `MongoDB pipeline connected`
	- verify `.env` values (`MONGO_URI`, `MONGO_DB`)
- Missing Parquet file:
	- verify `PARQUET_OUTPUT_DIR` in `hotel_scraper/settings.py`
	- ensure `output/` is writable
- Unexpected duplicates filtered:
	- check `DuplicateFilterPipeline` key definition in `hotel_scraper/pipelines.py`
