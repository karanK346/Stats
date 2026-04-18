# Player Line Predictor V2.3

This version fixes the requirements file, keeps the improved NHL and MLB sync logic, and replaces the NBA sync source with Basketball Reference scraping so it can work more reliably on hosted platforms than `nba_api`.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Notes

- NHL uses the public NHL API.
- MLB uses the `MLB-StatsAPI` package and the raw hydrated `person` endpoint for game logs.
- NBA now uses Basketball Reference search + game log scraping instead of `nba_api`.
- The app auto-syncs when you select a player and also auto-syncs during Predict if the player is not cached yet.
