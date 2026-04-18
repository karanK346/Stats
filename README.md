# Player Line Predictor V2.2

A deploy-ready FastAPI web app that works well on desktop and phone.

## What is new in V2.2
- cleaner mobile-friendly UI
- recent 5 / recent 10 season shortcuts
- trend chart for the last 20 games
- stronger feature set: recent averages, medians, volatility, opponent history, home/away, days rest
- summary cards for edge, hit rate, probability, and error
- top model feature importance display
- SQLite cache so synced player logs stay saved

## Sports supported
- NBA: points, rebounds, assists, threes, PRA
- NHL: goals, assists, points, shots
- MLB: hits, home runs, RBI, strikeouts

## Local run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Then open `http://127.0.0.1:8000`

## Render deploy
Build command:
```bash
pip install -r requirements.txt
```
Start command:
```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Practical notes
- This is an experiment tool, not a guaranteed predictor.
- Public sports endpoints can change over time.
- First syncs can take a bit because data is pulled live.
- Pulling all 10 seasons for many players in one session can be slow on free hosting.

## Suggested workflow
1. Search a player.
2. Sync 3 to 5 seasons first.
3. Run a prediction.
4. If needed, sync more seasons.
5. Compare the chart, hit rate, and model edge rather than only one number.


## What's new in V2.2
- safer API error handling in the frontend and backend
- auto-sync when you select a player
- predict can auto-sync if the player is not cached yet
- better error messages when a provider returns non-JSON responses
