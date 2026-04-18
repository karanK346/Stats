from __future__ import annotations
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .data_sources import (
    search_players,
    fetch_nba_player_logs,
    fetch_nhl_player_logs,
    fetch_mlb_player_logs,
    default_recent_seasons,
)
from .storage import upsert_logs, get_player_logs, list_cached_players
from .modeling import train_and_predict, STAT_MAP

app = FastAPI(title="Player Line Predictor V2")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


class SearchPayload(BaseModel):
    sport: str
    query: str


class SyncPayload(BaseModel):
    sport: str
    player_id: str
    player_name: str
    seasons: list[str] = Field(default_factory=list)


class PredictPayload(BaseModel):
    sport: str
    player_id: str
    stat: str
    line: float


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "stat_map": STAT_MAP})


@app.post("/api/search")
async def api_search(payload: SearchPayload):
    try:
        return {"results": search_players(payload.sport.lower(), payload.query)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/default-seasons/{sport}")
async def api_default_seasons(sport: str, count: int = 5):
    return {"seasons": default_recent_seasons(sport.lower(), count)}


@app.post("/api/sync")
async def api_sync(payload: SyncPayload):
    sport = payload.sport.lower()
    seasons = payload.seasons or default_recent_seasons(sport, 5)
    if sport == "nba":
        df = fetch_nba_player_logs(payload.player_id, payload.player_name, seasons)
    elif sport == "nhl":
        df = fetch_nhl_player_logs(payload.player_id, payload.player_name, seasons)
    elif sport == "mlb":
        df = fetch_mlb_player_logs(payload.player_id, payload.player_name, seasons)
    else:
        raise HTTPException(status_code=400, detail="Unsupported sport.")
    if df.empty:
        raise HTTPException(status_code=404, detail="No logs found for that player/seasons.")
    count = upsert_logs(df)
    return {"inserted_games": count, "seasons": seasons}


@app.get("/api/cached")
async def api_cached():
    df = list_cached_players()
    return {"players": df.to_dict(orient="records")}


@app.post("/api/predict")
async def api_predict(payload: PredictPayload):
    sport = payload.sport.lower()
    if payload.stat not in STAT_MAP.get(sport, []):
        raise HTTPException(status_code=400, detail="Stat not supported for this sport.")
    df = get_player_logs(sport, payload.player_id)
    if df.empty:
        raise HTTPException(status_code=404, detail="No cached data for this player. Sync first.")
    try:
        return train_and_predict(df, payload.stat, payload.line)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    return {"ok": True, "app": "v2"}
