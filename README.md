# Hotel Scraper

Scrapy project for scraping hotel offers and prices from multiple providers.

## Project Structure

- `hotel_scraper/spiders/promohotel_spider.py`
- `hotel_scraper/spiders/tunisiepromo_spider.py`
- `hotel_scraper/settings.py`
- `hotel_scraper/pipelines.py`
- `hotel_scraper/utils.py`

## Requirements

- Python 3.10+
- MongoDB running locally (default: `mongodb://localhost:27017`)

## Setup

1. Create and activate virtual environment (already done in this workspace):

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

Then edit `.env` with your real MongoDB values.

## Run Spiders

List available spiders:

```powershell
python -m scrapy list
```

Run a spider:

```powershell
python -m scrapy crawl promohotel
python -m scrapy crawl tunisiepromo
```

## Output and Storage

Items are processed through pipelines and stored in MongoDB according to `hotel_scraper/pipelines.py` and the settings in `hotel_scraper/settings.py`.

MongoDB credentials are loaded from environment variables in `.env`.

## Git Workflow

Initialize and commit locally:

```powershell
git add .
git commit -m "Initial project setup"
```

Connect to a remote GitHub repository:

```powershell
git remote add origin <YOUR_REPOSITORY_URL>
git branch -M main
git push -u origin main
```
