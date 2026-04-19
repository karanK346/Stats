# Player Line Predictor V2.4.1

This rebuild keeps NHL and MLB support and switches NBA to an unofficial ESPN-based source so you do not need a paid NBA API key.

## Run locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`

## Deploy on Render
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

## Notes
- NHL and MLB use public endpoints and cached local storage.
- NBA uses unofficial ESPN endpoints, so it can still change over time.
