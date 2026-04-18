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

NHL_TIMEOUT = 30
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _get_json(url: str, timeout: int = 30):
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    ctype = (r.headers.get("content-type") or "").lower()
    if "json" not in ctype:
        preview = r.text[:160].replace("\n", " ")
        
        raise RuntimeError(f"Expected JSON but got {ctype or 'unknown content type'} from {url}. Preview: {preview}")
    return r.json()


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
        url = f"https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit=20&q={requests.utils.quote(q)}"
        data = _get_json(url, timeout=NHL_TIMEOUT)
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
    last_error = None
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
            df["raw_json"] = [json.dumps(r) for r in df.to_dict(orient="records")]
            frames.append(df)
            time.sleep(0.7)
        except Exception as e:
            last_error = e
    if frames:
        return pd.concat(frames, ignore_index=True)
    if last_error:
        raise RuntimeError(f"NBA sync failed: {last_error}")
    return pd.DataFrame()


def _nhl_season_id_from_label(season_label: str) -> str:
    season_label = str(season_label).strip()
    if len(season_label) == 8 and season_label.isdigit():
        return season_label
    if "-" in season_label:
        start, end = season_label.split("-", 1)
        end = (start[:2] + end) if len(end) == 2 else end
        return f"{start}{end}"
    if len(season_label) == 4 and season_label.isdigit():
        return f"{season_label}{int(season_label)+1}"
    return season_label


def fetch_nhl_player_logs(player_id: str, player_name: str, seasons: list[str]) -> pd.DataFrame:
    frames = []
    last_error = None
    for season in seasons:
        season_id = _nhl_season_id_from_label(season)
        url = f"https://api-web.nhle.com/v1/player/{player_id}/game-log/{season_id}/2"
        try:
            payload = _get_json(url, timeout=NHL_TIMEOUT)
            games = payload.get("gameLog") or payload.get("games") or payload.get("log") or []
            if not games:
                continue
            rows = []
            for g in games:
                opp_name = g.get("opponentAbbrev")
                if not opp_name and isinstance(g.get("opponentCommonName"), dict):
                    opp_name = g["opponentCommonName"].get("default")
                rows.append({
                    "sport": "nhl",
                    "player_id": str(player_id),
                    "player_name": player_name,
                    "season": str(season),
                    "game_date": g.get("gameDate") or g.get("date"),
                    "opponent": opp_name,
                    "team": g.get("teamAbbrev"),
                    "home_away": "HOME" if g.get("homeRoadFlag") == "H" else "AWAY",
                    "stat_goals": safe_float(g.get("goals")),
                    "stat_assists": safe_float(g.get("assists")),
                    "stat_shots": safe_float(g.get("shots")),
                    "stat_points": safe_float(g.get("points")),
                    "raw_json": json.dumps(g),
                })
            if rows:
                frames.append(pd.DataFrame(rows))
            time.sleep(0.5)
        except Exception as e:
            last_error = e
    if frames:
        return pd.concat(frames, ignore_index=True)
    if last_error:
        raise RuntimeError(f"NHL sync failed: {last_error}")
    return pd.DataFrame()


def _mlb_years_from_labels(seasons: list[str]) -> list[int]:
    years = []
    for s in seasons:
        s = str(s).strip()
        try:
            years.append(int(s[:4]))
        except Exception:
            pass
    return years


def _fetch_mlb_group(player_id: str, player_name: str, year: int, group: str) -> pd.DataFrame:
    logs = statsapi.player_stat_data(player_id, group=f"[{group}]", type="gameLog", sportId=1, season=year)
    stats = logs.get("stats", [])
    splits = stats[0].get("splits", []) if stats else []
    rows = []
    for s in splits:
        stat = s.get("stat", {})
        game = s.get("game", {})
        team = s.get("team", {})
        opp = s.get("opponent", {})
        row = {
            "sport": "mlb",
            "player_id": str(player_id),
            "player_name": player_name,
            "season": str(year),
            "game_date": (game.get("gameDate") or "")[:10],
            "opponent": opp.get("abbreviation"),
            "team": team.get("abbreviation"),
            "home_away": "HOME" if s.get("isHome") else "AWAY",
            "raw_json": json.dumps(s),
        }
        if group == "hitting":
            row.update({
                "stat_hits": safe_float(stat.get("hits")),
                "stat_home_runs": safe_float(stat.get("homeRuns")),
                "stat_rbi": safe_float(stat.get("rbi")),
                "stat_strikeouts": safe_float(stat.get("strikeOuts")),
            })
        elif group == "pitching":
            row.update({
                "stat_hits": safe_float(stat.get("hits")),
                "stat_home_runs": safe_float(stat.get("homeRuns")),
                "stat_rbi": safe_float(stat.get("rbi")),
                "stat_strikeouts": safe_float(stat.get("strikeOuts")),
            })
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_mlb_player_logs(player_id: str, player_name: str, seasons: list[str]) -> pd.DataFrame:
    years = _mlb_years_from_labels(seasons)
    frames = []
    last_error = None
    for year in years:
        try:
            hit_df = _fetch_mlb_group(player_id, player_name, year, "hitting")
            pitch_df = _fetch_mlb_group(player_id, player_name, year, "pitching")
            if not hit_df.empty and not pitch_df.empty:
                key_cols = ["sport", "player_id", "player_name", "season", "game_date", "opponent", "team", "home_away"]
                merged = pd.merge(hit_df, pitch_df, on=key_cols, how="outer", suffixes=("_hit", "_pit"))
                # collapse duplicated stat columns with pitching taking precedence for K props if present
                out = merged[key_cols].copy()
                for col in ["stat_hits", "stat_home_runs", "stat_rbi", "stat_strikeouts"]:
                    a = merged.get(f"{col}_pit")
                    b = merged.get(f"{col}_hit")
                    out[col] = a.combine_first(b) if a is not None and b is not None else (a if a is not None else b)
                out["raw_json"] = merged.get("raw_json_pit").combine_first(merged.get("raw_json_hit"))
                frames.append(out)
            elif not hit_df.empty:
                frames.append(hit_df)
            elif not pitch_df.empty:
                frames.append(pitch_df)
            time.sleep(0.3)
        except Exception as e:
            last_error = e
    if frames:
        return pd.concat(frames, ignore_index=True)
    if last_error:
        raise RuntimeError(f"MLB sync failed: {last_error}")
    return pd.DataFrame()


def default_recent_seasons(sport: str, count: int = 5) -> list[str]:
    if sport in {"nba", "nhl"}:
        starts = list(range(2025, 2025 - count, -1))
        return [f"{y}-{str((y+1)%100).zfill(2)}" for y in starts]
    if sport == "mlb":
        return [str(y) for y in range(2025, 2025 - count, -1)]
    return []
