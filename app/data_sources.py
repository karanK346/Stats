from __future__ import annotations
import os
import time
import json
import requests
import pandas as pd

# MLB
import statsapi

NHL_TIMEOUT = 30
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}
BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY", "").strip()
BALLDONTLIE_BASE = "https://api.balldontlie.io/v1"


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _get_json(url: str, timeout: int = 30, headers: dict | None = None, params: dict | None = None):
    merged_headers = dict(UA)
    if headers:
        merged_headers.update(headers)
    r = requests.get(url, headers=merged_headers, timeout=timeout, params=params)
    ctype = (r.headers.get("content-type") or "").lower()
    if r.status_code >= 400:
        preview = r.text[:220].replace("\n", " ")
        raise RuntimeError(f"HTTP {r.status_code} from {url}. {preview}")
    if "json" not in ctype:
        preview = r.text[:160].replace("\n", " ")
        raise RuntimeError(f"Expected JSON but got {ctype or 'unknown content type'} from {url}. Preview: {preview}")
    return r.json()


def _balldontlie_headers() -> dict:
    if not BALLDONTLIE_API_KEY:
        raise RuntimeError("NBA sync requires BALLDONTLIE_API_KEY in Render environment variables.")
    return {"Authorization": BALLDONTLIE_API_KEY}


def _balldontlie_get(path: str, params: dict | None = None):
    try:
        return _get_json(f"{BALLDONTLIE_BASE}{path}", headers=_balldontlie_headers(), params=params)
    except RuntimeError as e:
        msg = str(e)
        if "HTTP 401" in msg:
            raise RuntimeError("BALLDONTLIE rejected the NBA request. Check that your API key is set correctly and that your account tier includes NBA game stats.")
        if "HTTP 429" in msg:
            raise RuntimeError("BALLDONTLIE rate limit reached. Try again in a minute.")
        raise


def _normalize_nba_season_start(season_label: str) -> int:
    s = str(season_label).strip()
    if "-" in s:
        return int(s.split("-", 1)[0])
    if len(s) == 8 and s.isdigit():
        return int(s[:4])
    return int(s[:4])


def search_players(sport: str, query: str):
    q = (query or "").strip().lower()
    if not q:
        return []

    if sport == "nba":
        payload = _balldontlie_get("/players", params={"search": q, "per_page": 15})
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        out = []
        for row in rows:
            first = (row.get("first_name") or "").strip()
            last = (row.get("last_name") or "").strip()
            name = f"{first} {last}".strip()
            if not name:
                continue
            out.append({"id": str(row.get("id")), "name": name, "sport": "nba"})
        # prefer exact startswith matches, then contains
        out.sort(key=lambda x: (not x["name"].lower().startswith(q), q not in x["name"].lower(), x["name"]))
        return out[:15]

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
        cursor = None
        season_start = _normalize_nba_season_start(season)
        while True:
            params = {
                "player_ids[]": str(player_id),
                "seasons[]": season_start,
                "postseason": "false",
                "per_page": 100,
            }
            if cursor is not None:
                params["cursor"] = cursor
            try:
                payload = _balldontlie_get("/stats", params=params)
                rows = payload.get("data", []) if isinstance(payload, dict) else []
                if rows:
                    normalized = []
                    for r in rows:
                        game = r.get("game") or {}
                        team = r.get("team") or {}
                        player_team_id = team.get("id")
                        home_team_id = game.get("home_team_id")
                        visitor_team_id = game.get("visitor_team_id")
                        if player_team_id == home_team_id:
                            opp = game.get("visitor_team", {}) if isinstance(game.get("visitor_team"), dict) else {}
                            home_away = "HOME"
                        elif player_team_id == visitor_team_id:
                            opp = game.get("home_team", {}) if isinstance(game.get("home_team"), dict) else {}
                            home_away = "AWAY"
                        else:
                            opp = {}
                            home_away = None
                        normalized.append({
                            "sport": "nba",
                            "player_id": str(player_id),
                            "player_name": player_name,
                            "season": str(season),
                            "game_date": str(game.get("date") or "")[:10],
                            "opponent": opp.get("abbreviation") or opp.get("full_name") or opp.get("name"),
                            "team": team.get("abbreviation") or team.get("full_name") or team.get("name"),
                            "home_away": home_away,
                            "stat_points": safe_float(r.get("pts")),
                            "stat_rebounds": safe_float(r.get("reb")),
                            "stat_assists": safe_float(r.get("ast")),
                            "stat_threes": safe_float(r.get("fg3m")),
                            "stat_pra": (safe_float(r.get("pts")) or 0) + (safe_float(r.get("reb")) or 0) + (safe_float(r.get("ast")) or 0),
                            "raw_json": json.dumps(r),
                        })
                    frames.append(pd.DataFrame(normalized))
                meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
                cursor = meta.get("next_cursor")
                if not cursor:
                    break
                time.sleep(0.25)
            except Exception as e:
                last_error = e
                break
        time.sleep(0.15)
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = out.dropna(subset=["game_date"]).drop_duplicates(subset=["player_id", "season", "game_date"], keep="last")
        return out
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
    payload = statsapi.get(
        "person",
        {
            "personId": str(player_id),
            "hydrate": f"stats(group=[{group}],type=[gameLog],season={year},sportId=1),currentTeam",
        },
    )
    people = payload.get("people", []) if isinstance(payload, dict) else []
    if not people:
        return pd.DataFrame()

    stats_blocks = people[0].get("stats", []) or []
    splits = []
    for block in stats_blocks:
        block_splits = block.get("splits", []) or []
        if block_splits:
            splits.extend(block_splits)

    rows = []
    for s in splits:
        stat = s.get("stat", {}) or {}
        game = s.get("game", {}) or {}
        team = s.get("team", {}) or {}
        opp = s.get("opponent", {}) or {}
        date_val = game.get("gameDate") or s.get("date") or s.get("gameDate") or ""
        row = {
            "sport": "mlb",
            "player_id": str(player_id),
            "player_name": player_name,
            "season": str(year),
            "game_date": str(date_val)[:10] if date_val else None,
            "opponent": opp.get("abbreviation") or opp.get("name"),
            "team": team.get("abbreviation") or team.get("name"),
            "home_away": "HOME" if s.get("isHome") else ("AWAY" if s.get("isHome") is not None else None),
            "raw_json": json.dumps(s),
        }
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
