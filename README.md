# Player Line Predictor V2.4

This version keeps NHL and MLB support and switches NBA to BALLDONTLIE.

## Required environment variable for NBA
Set this in Render before deploying:

- `BALLDONTLIE_API_KEY` = your BALLDONTLIE API key

## Run locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`
