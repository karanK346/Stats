from __future__ import annotations
import time
import json
import requests
import pandas as pd

# NBA
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog

# MLB
import statsapi

NBA_SEASON_FORMAT_HINT = "YYYY-YY"
NHL_TIMEOUT = 30
UA = {"User-Agent": "Mozilla/5.0"}

def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None

def search_players(sport: str, query: str):
    q = (query or "").strip().lower()
    if not q:
        return []

    if sport == "nba":
        matches = nba_players.find_players_by_full_name(q)
        return [{"id": str(m["id"]), "name": m["full_name"], "sport": "nba"} for m in matches[:15]]

    if sport == "mlb":
        res = statsapi.lookup_player(q)
        return [{"id": str(m["id"]), "name": m["fullName"], "sport": "mlb"} for m in res[:15]]

    if sport == "nhl":
        # NHL public search is not always stable, so use suggest endpoint.
        url = f"https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit=20&q={requests.utils.quote(q)}"
        r = requests.get(url, headers=UA, timeout=NHL_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        out = []
        for item in data:
            pid = item.get("playerId") or item.get("player_id")
            name = item.get("name") or f"{item.get('firstName', {}).get('default','')} {item.get('lastName', {}).get('default','')}".strip()
            if pid and name:
                out.append({"id": str(pid), "name": name, "sport": "nhl"})
        return out[:15]

    return []

def fetch_nba_player_logs(player_id: str, player_name: str, seasons: list[str]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        try:
            gl = playergamelog.PlayerGameLog(player_id=player_id, season=season, timeout=30)
            df = gl.get_data_frames()[0]
            if df.empty:
                continue
            df = df.copy()
            df["sport"] = "nba"
            df["player_id"] = str(player_id)
            df["player_name"] = player_name
            df["season"] = season
            df["game_date"] = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d")
            df["opponent"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s+([A-Z]{2,3})")
            df["team"] = df["MATCHUP"].str.extract(r"^([A-Z]{2,3})")
            df["home_away"] = df["MATCHUP"].apply(lambda x: "AWAY" if "@" in str(x) else "HOME")
            df["stat_points"] = pd.to_numeric(df["PTS"], errors="coerce")
            df["stat_rebounds"] = pd.to_numeric(df["REB"], errors="coerce")
            df["stat_assists"] = pd.to_numeric(df["AST"], errors="coerce")
            df["stat_threes"] = pd.to_numeric(df.get("FG3M"), errors="coerce")
            df["stat_pra"] = df["stat_points"].fillna(0) + df["stat_rebounds"].fillna(0) + df["stat_assists"].fillna(0)
            df["raw_json"] = df.to_dict(orient="records")
            frames.append(df)
            time.sleep(0.7)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _nhl_season_id_from_label(season_label: str) -> str:
    # "2024-25" -> "20242025"
    start = season_label[:4]
    end_two = season_label[-2:]
    end = str(int(start) + 1) if len(end_two) == 2 else season_label[-4:]
    if len(end) == 2:
        end = start[:2] + end
    return f"{start}{end}"

def fetch_nhl_player_logs(player_id: str, player_name: str, seasons: list[str]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        season_id = _nhl_season_id_from_label(season)
        url = f"https://api-web.nhle.com/v1/player/{player_id}/game-log/{season_id}/2"
        try:
            r = requests.get(url, headers=UA, timeout=NHL_TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            games = payload.get("gameLog", []) or payload.get("games", [])
            if not games:
                continue
            rows = []
            for g in games:
                game_date = g.get("gameDate")
                rows.append({
                    "sport": "nhl",
                    "player_id": str(player_id),
                    "player_name": player_name,
                    "season": season,
                    "game_date": game_date,
                    "opponent": (g.get("opponentAbbrev") or g.get("opponentCommonName", {}) or {}).get("default") if isinstance(g.get("opponentCommonName"), dict) else g.get("opponentAbbrev"),
                    "team": g.get("teamAbbrev"),
                    "home_away": "HOME" if g.get("homeRoadFlag") == "H" else "AWAY",
                    "stat_goals": safe_float(g.get("goals")),
                    "stat_assists": safe_float(g.get("assists")),
                    "stat_shots": safe_float(g.get("shots")),
                    "stat_points": safe_float(g.get("points")),
                    "raw_json": json.dumps(g),
                })
            df = pd.DataFrame(rows)
            frames.append(df)
            time.sleep(0.5)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _mlb_years_from_labels(seasons: list[str]) -> list[int]:
    years = []
    for s in seasons:
        try:
            years.append(int(str(s)[:4]))
        except Exception:
            pass
    return years

def fetch_mlb_player_logs(player_id: str, player_name: str, seasons: list[str]) -> pd.DataFrame:
    years = _mlb_years_from_labels(seasons)
    frames = []
    for year in years:
        try:
            logs = statsapi.player_stat_data(player_id, group="[hitting]", type="gameLog", sportId=1, season=year)
            stats = logs.get("stats", [{}])[0].get("splits", [])
            rows = []
            for s in stats:
                stat = s.get("stat", {})
                game = s.get("game", {})
                team = s.get("team", {})
                opp = s.get("opponent", {})
                rows.append({
                    "sport": "mlb",
                    "player_id": str(player_id),
                    "player_name": player_name,
                    "season": str(year),
                    "game_date": game.get("gameDate", "")[:10],
                    "opponent": opp.get("abbreviation"),
                    "team": team.get("abbreviation"),
                    "home_away": "HOME" if s.get("isHome") else "AWAY",
                    "stat_hits": safe_float(stat.get("hits")),
                    "stat_home_runs": safe_float(stat.get("homeRuns")),
                    "stat_rbi": safe_float(stat.get("rbi")),
                    "stat_strikeouts": safe_float(stat.get("strikeOuts")),
                    "raw_json": json.dumps(s),
                })
            if rows:
                frames.append(pd.DataFrame(rows))
            time.sleep(0.3)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def default_recent_seasons(sport: str, count: int = 5) -> list[str]:
    # A simple static recent window is enough for this starter app.
    if sport in {"nba", "nhl"}:
        starts = list(range(2025, 2025 - count, -1))
        return [f"{y}-{str((y+1)%100).zfill(2)}" for y in starts]
    if sport == "mlb":
        return [str(y) for y in range(2025, 2025 - count, -1)]
    return []
