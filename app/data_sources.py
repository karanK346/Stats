from __future__ import annotations
import json
import time
from typing import Any, Iterable

import pandas as pd
import requests
import statsapi

NHL_TIMEOUT = 30
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def safe_float(v):
    try:
        if v is None or v == "":
            return None
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


# ---------------- ESPN / NBA ---------------- #


def _walk_json(obj: Any) -> Iterable[dict]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_json(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_json(item)


def _collect_espn_search_results(payload: dict, query: str) -> list[dict]:
    query_l = query.lower().strip()
    seen: set[str] = set()
    out: list[dict] = []
    for node in _walk_json(payload):
        cand_id = node.get("id") or node.get("athleteId") or node.get("uid")
        name = node.get("displayName") or node.get("fullName") or node.get("name")
        node_type = str(node.get("type") or node.get("entityType") or node.get("description") or "").lower()
        sport_hint = json.dumps(node).lower()
        if not cand_id or not name:
            continue
        if not any(k in sport_hint for k in ["basketball", "nba"]):
            continue
        if not any(k in node_type for k in ["athlete", "player", "person"]):
            if not any(k in sport_hint for k in ["athlete", "player"]):
                continue
        key = str(cand_id)
        clean_name = str(name).replace(" Overview", "").strip()
        if key in seen:
            continue
        if query_l and query_l not in clean_name.lower() and not clean_name.lower().startswith(query_l):
            # still allow a few broad matches if response is sparse, but prefer closer ones later
            pass
        seen.add(key)
        out.append({"id": key, "name": clean_name, "sport": "nba"})
    out.sort(key=lambda x: (not x["name"].lower().startswith(query_l), query_l not in x["name"].lower(), x["name"]))
    return out[:15]


def _espn_search_nba(query: str) -> list[dict]:
    endpoints = [
        ("https://site.web.api.espn.com/apis/search/v2", {"query": query, "limit": 15, "sport": "basketball"}),
        ("https://site.web.api.espn.com/apis/common/v3/search", {"query": query, "limit": 15, "sport": "basketball"}),
    ]
    last_error = None
    for url, params in endpoints:
        try:
            payload = _get_json(url, params=params)
            results = _collect_espn_search_results(payload, query)
            if results:
                return results
        except Exception as e:
            last_error = e
    if last_error:
        raise RuntimeError(f"NBA search failed: {last_error}")
    return []


STAT_KEY_MAP = {
    "pts": "stat_points",
    "points": "stat_points",
    "reb": "stat_rebounds",
    "rebounds": "stat_rebounds",
    "ast": "stat_assists",
    "assists": "stat_assists",
    "fg3m": "stat_threes",
    "threepointfieldgoalsmade": "stat_threes",
    "threesmade": "stat_threes",
    "3pm": "stat_threes",
}


def _parse_espn_stats_list(stats: Any) -> dict[str, float | None]:
    out = {
        "stat_points": None,
        "stat_rebounds": None,
        "stat_assists": None,
        "stat_threes": None,
    }
    if isinstance(stats, dict):
        for key, value in stats.items():
            norm = STAT_KEY_MAP.get(str(key).replace("_", "").replace("-", "").lower())
            if norm:
                out[norm] = safe_float(value)
        return out
    if not isinstance(stats, list):
        return out

    # list of named stats
    named_found = False
    for item in stats:
        if isinstance(item, dict):
            label = (
                item.get("name")
                or item.get("abbreviation")
                or item.get("shortDisplayName")
                or item.get("displayName")
                or item.get("label")
            )
            value = item.get("value")
            if value is None:
                value = item.get("displayValue")
            norm = STAT_KEY_MAP.get(str(label).replace("_", "").replace("-", "").replace(" ", "").lower()) if label else None
            if norm:
                out[norm] = safe_float(value)
                named_found = True
    if named_found:
        return out

    # fallback: ESPN common gamelog often uses [PTS, REB, AST, ...]
    numeric = [safe_float(x) for x in stats]
    if len(numeric) >= 3:
        out["stat_points"] = numeric[0]
        out["stat_rebounds"] = numeric[1]
        out["stat_assists"] = numeric[2]
    if len(numeric) >= 9:
        out["stat_threes"] = numeric[8]
    return out


def _extract_espn_events(payload: dict) -> list[dict]:
    events: list[dict] = []
    direct = payload.get("events")
    if isinstance(direct, list):
        events.extend([e for e in direct if isinstance(e, dict)])

    for season_type in payload.get("seasonTypes", []) or []:
        for category in season_type.get("categories", []) or []:
            cat_events = category.get("events") or category.get("items") or []
            for event in cat_events:
                if isinstance(event, dict):
                    merged = dict(event)
                    if "category" not in merged:
                        merged["category"] = category.get("name") or category.get("displayName")
                    events.append(merged)

    return events


def _normalize_espn_nba_logs(payload: dict, player_id: str, player_name: str, season_label: str) -> pd.DataFrame:
    rows: list[dict] = []
    events = _extract_espn_events(payload)
    for event in events:
        date_val = event.get("gameDate") or event.get("date") or (event.get("event") or {}).get("date")
        opp = (
            event.get("opponentDisplayName")
            or event.get("opponentShortName")
            or ((event.get("opponent") or {}).get("displayName") if isinstance(event.get("opponent"), dict) else None)
            or ((event.get("competitor") or {}).get("displayName") if isinstance(event.get("competitor"), dict) else None)
        )
        home_away = event.get("homeAway") or event.get("home_away")
        if isinstance(home_away, str):
            home_away = home_away.upper()
            if home_away in {"HOME", "AWAY"}:
                pass
            elif home_away in {"VS", "H"}:
                home_away = "HOME"
            elif home_away in {"@", "A"}:
                home_away = "AWAY"
            else:
                home_away = None
        stats = _parse_espn_stats_list(event.get("stats") or event.get("statistics"))
        if all(v is None for v in stats.values()):
            continue
        row = {
            "sport": "nba",
            "player_id": str(player_id),
            "player_name": player_name,
            "season": str(season_label),
            "game_date": str(date_val)[:10] if date_val else None,
            "opponent": opp,
            "team": event.get("teamAbbrev") or event.get("teamShortName") or None,
            "home_away": home_away,
            "raw_json": json.dumps(event),
        }
        row.update(stats)
        pts = row.get("stat_points") or 0
        reb = row.get("stat_rebounds") or 0
        ast = row.get("stat_assists") or 0
        row["stat_pra"] = pts + reb + ast
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _normalize_nba_season_start(season_label: str) -> int:
    s = str(season_label).strip()
    if "-" in s:
        return int(s.split("-", 1)[0])
    if len(s) == 8 and s.isdigit():
        return int(s[:4])
    return int(s[:4])


def fetch_nba_player_logs(player_id: str, player_name: str, seasons: list[str]) -> pd.DataFrame:
    frames = []
    last_error = None
    base_url = f"https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{player_id}/gamelog"
    for season in seasons:
        season_start = _normalize_nba_season_start(season)
        tried_any = False
        for params in (
            {"season": season_start},
            {"year": season_start},
            {},
        ):
            try:
                tried_any = True
                payload = _get_json(base_url, params=params)
                df = _normalize_espn_nba_logs(payload, player_id, player_name, str(season))
                if not df.empty:
                    if "season" not in params and len(seasons) > 1:
                        # avoid duplicating current-season data into every season when the endpoint ignores params
                        if season != seasons[0]:
                            continue
                    frames.append(df)
                    break
            except Exception as e:
                last_error = e
                continue
        if tried_any:
            time.sleep(0.2)
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = out.dropna(subset=["game_date"]).drop_duplicates(subset=["player_id", "game_date"], keep="last")
        return out
    if last_error:
        raise RuntimeError(f"NBA sync failed: {last_error}")
    return pd.DataFrame()


# ---------------- NHL ---------------- #


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
            time.sleep(0.3)
        except Exception as e:
            last_error = e
    if frames:
        return pd.concat(frames, ignore_index=True)
    if last_error:
        raise RuntimeError(f"NHL sync failed: {last_error}")
    return pd.DataFrame()


# ---------------- MLB ---------------- #


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
            time.sleep(0.2)
        except Exception as e:
            last_error = e
    if frames:
        return pd.concat(frames, ignore_index=True)
    if last_error:
        raise RuntimeError(f"MLB sync failed: {last_error}")
    return pd.DataFrame()


# ---------------- Unified search + helpers ---------------- #


def search_players(sport: str, query: str):
    q = (query or "").strip()
    if not q:
        return []

    if sport == "nba":
        return _espn_search_nba(q)

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


def default_recent_seasons(sport: str, count: int = 5) -> list[str]:
    if sport in {"nba", "nhl"}:
        starts = list(range(2025, 2025 - count, -1))
        return [f"{y}-{str((y + 1) % 100).zfill(2)}" for y in starts]
    if sport == "mlb":
        return [str(y) for y in range(2025, 2025 - count, -1)]
    return []
