# Player Line Predictor V2.6.0

A sports prop prediction web app using FastAPI + scikit-learn, deployable on Render with Docker.

## Supported sports
- **NBA** — via Basketball-Reference search + game logs
- **NHL** — via the official `api-web.nhle.com` API
- **MLB** — via the official `statsapi.mlb.com` API

## Features
1. **Search player** — returns deduplicated results
2. **Sync game logs** — fetches and caches recent seasons in SQLite
3. **Run prediction** — Random Forest regression + Logistic Regression classifier
4. **Results** — predicted stat, edge, probability over/under, hit rate, trend chart

## Local development
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Deploy on Render
Push this repo to GitHub, then create a new **Web Service** on Render:
- **Environment:** Docker
- Render will automatically pick up `render.yaml`

## File structure
```
app/
  __init__.py
  config.py        – paths / SQLite location
  data_sources.py  – NBA / NHL / MLB API clients
  modeling.py      – feature engineering + RF + LR model
  storage.py       – SQLite read/write helpers
  main.py          – FastAPI routes
  static/
    styles.css
  templates/
    index.html
requirements.txt
Dockerfile
render.yaml
```

## Changelog
### V2.6.0
- **NBA**: switched from Basketball Reference scraping to `stats.nba.com` API (numeric player IDs, structured JSON, faster and more reliable)
- **Modeling**: robust handling of 1-class datasets and small sample sizes — no more "solver needs 2 classes" crash
- **Dockerfile**: fixed `$PORT` shell expansion for Render
- **render.yaml**: set `env: docker` to use the Dockerfile properly
- Deduplication of search results across all sports
- Cleaner error messages throughout
