from __future__ import annotations

import json
import time
from datetime import datetime
from urllib.parse import quote
import re
from typing import Any, Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ─────────────────────────── Timeouts ────────────────────────────── #
NHL_TIMEOUT = 30
NBA_TIMEOUT = 30
MLB_TIMEOUT = 30

# ─────────────────────────── Headers ─────────────────────────────── #
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

# stats.nba.com requires these specific headers or it returns 400
NBA_HEADERS = {
    **UA,
    "Host": "stats.nba.com",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Sec-Fetch-Site": "same-site",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "Connection": "keep-alive",
}


# ─────────────────────────── Helpers ─────────────────────────────── #

def safe_float(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _get_json(
    url: str,
    timeout: int = 30,
    headers: dict | None = None,
    params: dict | None = None,
):
    merged = dict(UA)
    if headers:
        merged.update(headers)
    r = requests.get(url, headers=merged, timeout=timeout, params=params)
    ctype = (r.headers.get("content-type") or "").lower()
    if r.status_code >= 400:
        preview = r.text[:300].replace("\n", " ")
        raise RuntimeError(f"HTTP {r.status_code} from {url}. {preview}")
    if "json" not in ctype and "javascript" not in ctype:
        preview = r.text[:160].replace("\n", " ")
        raise RuntimeError(
            f"Expected JSON but got '{ctype}' from {url}. Preview: {preview}"
        )
    return r.json()


# ══════════════════════════════════════════════════════════════════════
#  NBA  –  ESPN search/gamelog (primary) / stats.nba.com / Basketball‑Reference
# ══════════════════════════════════════════════════════════════════════

def _nba_season_label(season: str) -> str:
    season = str(season).strip()
    if "-" in season:
        start, end = season.split("-", 1)
        if len(end) == 4:
            end = end[2:]
        return f"{start}-{end.zfill(2)}"
    try:
        y = int(season[:4])
        return f"{y}-{str(y + 1)[2:]}"
    except Exception:
        return season


def _nba_parse_date(raw: str) -> str:
    for fmt in ("%b %d, %Y", "%Y-%m-%d", "%a %m/%d"):
        try:
            parsed = datetime.strptime(raw.strip(), fmt)
            if fmt == "%a %m/%d":
                parsed = parsed.replace(year=datetime.utcnow().year)
            return parsed.strftime("%Y-%m-%d")
        except Exception:
            continue
    return raw[:10] if raw else raw


def _extract_numeric_espn_id(node: dict) -> str | None:
    for key in ("athleteId", "playerId", "id"):
        val = node.get(key)
        if isinstance(val, int):
            return str(val)
        if isinstance(val, str) and val.isdigit():
            return val
    for key in ("uid", "guid", "$ref", "href", "link"):
        val = node.get(key)
        if not isinstance(val, str):
            continue
        m = re.search(r"a:(\d+)", val)
        if m:
            return m.group(1)
        m = re.search(r"/athletes/(\d+)", val)
        if m:
            return m.group(1)
    return None


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
        cand_id = _extract_numeric_espn_id(node)
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
        seen.add(key)
        out.append({"id": key, "name": clean_name, "sport": "nba"})
    out.sort(key=lambda x: (not x["name"].lower().startswith(query_l), query_l not in x["name"].lower(), x["name"]))
    return out[:15]


def _espn_search_nba(query: str) -> list[dict]:
    endpoints = [
        ("https://site.web.api.espn.com/apis/search/v2", {"query": query, "limit": 15, "sport": "basketball"}),
        ("https://site.web.api.espn.com/apis/common/v3/search", {"query": query, "limit": 15, "sport": "basketball"}),
    ]
    for url, params in endpoints:
        try:
            payload = _get_json(url, params=params)
            results = _collect_espn_search_results(payload, query)
            if results:
                return results
        except Exception:
            continue
    return []


_nba_all_players: list[dict] = []
_nba_all_players_ts: float = 0.0


def _fetch_all_nba_players() -> list[dict]:
    global _nba_all_players, _nba_all_players_ts
    now = time.time()
    if _nba_all_players and (now - _nba_all_players_ts) < 3600:
        return _nba_all_players

    url = "https://stats.nba.com/stats/commonallplayers"
    params = {
        "LeagueID": "00",
        "Season": _nba_season_label(default_recent_seasons("nba", 1)[0]),
        "IsOnlyCurrentSeason": "0",
    }
    data = _get_json(url, timeout=NBA_TIMEOUT, headers=NBA_HEADERS, params=params)
    rs = data["resultSets"][0]
    hdrs = rs["headers"]
    pid_idx = hdrs.index("PERSON_ID")
    name_idx = hdrs.index("DISPLAY_FIRST_LAST")
    seen: set[str] = set()
    players: list[dict] = []
    for row in rs["rowSet"]:
        pid = str(row[pid_idx])
        name = str(row[name_idx] or "").strip()
        if pid and name and pid not in seen:
            seen.add(pid)
            players.append({"id": pid, "name": name, "sport": "nba"})
    _nba_all_players = players
    _nba_all_players_ts = now
    return players


def _bref_search_nba(query: str) -> list[dict]:
    results: list[dict] = []
    seen: set[str] = set()
    for letter in "abcdefghijklmnopqrstuvwxyz":
        url = f"https://www.basketball-reference.com/players/{letter}/"
        try:
            r = requests.get(url, headers=UA, timeout=15)
            if r.status_code >= 400:
                continue
            soup = BeautifulSoup(r.text, "lxml")
            table = soup.find("table", {"id": "players"})
            if not table:
                continue
            body = table.find("tbody")
            if not body:
                continue
            for row in body.find_all("tr"):
                if "thead" in (row.get("class") or []):
                    continue
                th = row.find("th", {"data-stat": "player"}) or row.find("th")
                if not th:
                    continue
                a = th.find("a")
                if not a:
                    continue
                name = a.get_text(" ", strip=True)
                slug = a.get("href", "").rstrip("/").split("/")[-1].replace(".html", "")
                if not slug or slug in seen:
                    continue
                if query not in f"{name} {slug}".lower():
                    continue
                seen.add(slug)
                results.append({"id": slug, "name": name, "sport": "nba"})
                if len(results) >= 15:
                    return results
            time.sleep(0.05)
        except Exception:
            continue
    return results


def search_nba_players(query: str) -> list[dict]:
    q = (query or "").strip().lower()
    if not q:
        return []
    espn = _espn_search_nba(q)
    if espn:
        return espn
    try:
        all_players = _fetch_all_nba_players()
        results = [p for p in all_players if q in p["name"].lower()][:15]
        if results:
            return results
    except Exception:
        pass
    return _bref_search_nba(q)


def _find_bref_slug_by_name(player_name: str) -> str | None:
    q = (player_name or "").strip().lower()
    if not q:
        return None
    candidates = _bref_search_nba(q)
    for c in candidates:
        if c["name"].strip().lower() == q:
            return c["id"]
    return candidates[0]["id"] if candidates else None


def _parse_espn_stats_list(stats: Any) -> dict[str, float | None]:
    out = {"stat_points": None, "stat_rebounds": None, "stat_assists": None, "stat_threes": None}
    if isinstance(stats, dict):
        aliases = {
            "points": "stat_points", "pts": "stat_points",
            "rebounds": "stat_rebounds", "reb": "stat_rebounds",
            "assists": "stat_assists", "ast": "stat_assists",
            "threepointfieldgoalsmade": "stat_threes", "3pm": "stat_threes", "fg3m": "stat_threes",
        }
        for key, value in stats.items():
            norm = aliases.get(str(key).replace("_", "").replace("-", "").lower())
            if norm:
                out[norm] = safe_float(value)
        return out
    if not isinstance(stats, list):
        return out
    named_found = False
    aliases = {
        "points": "stat_points", "pts": "stat_points",
        "rebounds": "stat_rebounds", "reb": "stat_rebounds",
        "assists": "stat_assists", "ast": "stat_assists",
        "3pm": "stat_threes", "fg3m": "stat_threes", "threesmade": "stat_threes",
    }
    for item in stats:
        if isinstance(item, dict):
            label = item.get("name") or item.get("abbreviation") or item.get("shortDisplayName") or item.get("displayName") or item.get("label")
            value = item.get("value") if item.get("value") is not None else item.get("displayValue")
            norm = aliases.get(str(label).replace("_", "").replace("-", "").replace(" ", "").lower()) if label else None
            if norm:
                out[norm] = safe_float(value)
                named_found = True
    if named_found:
        return out
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
                    events.append(event)
    return events


def _nba_logs_espn(player_id: str, player_name: str, seasons: list[str]) -> pd.DataFrame:
    frames = []
    last_error = None
    base_url = f"https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{player_id}/gamelog"
    for season in seasons:
        season_start = int(str(season)[:4])
        for params in ({"season": season_start}, {"year": season_start}, {}):
            try:
                payload = _get_json(base_url, params=params)
                rows = []
                for event in _extract_espn_events(payload):
                    date_val = event.get("gameDate") or event.get("date") or (event.get("event") or {}).get("date")
                    opp = event.get("opponentDisplayName") or event.get("opponentShortName") or ((event.get("opponent") or {}).get("displayName") if isinstance(event.get("opponent"), dict) else None)
                    home_away = event.get("homeAway") or event.get("home_away")
                    if isinstance(home_away, str):
                        home_away = home_away.upper()
                        if home_away in {"VS", "H"}:
                            home_away = "HOME"
                        elif home_away in {"@", "A"}:
                            home_away = "AWAY"
                    stats = _parse_espn_stats_list(event.get("stats") or event.get("statistics"))
                    if all(v is None for v in stats.values()):
                        continue
                    pts = stats.get("stat_points") or 0
                    reb = stats.get("stat_rebounds") or 0
                    ast = stats.get("stat_assists") or 0
                    rows.append({
                        "sport": "nba",
                        "player_id": str(player_id),
                        "player_name": player_name,
                        "season": str(season),
                        "game_date": str(date_val)[:10] if date_val else None,
                        "opponent": opp,
                        "team": event.get("teamAbbrev") or event.get("teamShortName") or None,
                        "home_away": home_away,
                        "stat_points": stats.get("stat_points"),
                        "stat_rebounds": stats.get("stat_rebounds"),
                        "stat_assists": stats.get("stat_assists"),
                        "stat_threes": stats.get("stat_threes"),
                        "stat_pra": pts + reb + ast,
                        "raw_json": json.dumps(event),
                    })
                if rows:
                    frames.append(pd.DataFrame(rows))
                    break
            except Exception as e:
                last_error = e
                continue
        time.sleep(0.2)
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = out.dropna(subset=["game_date"]).drop_duplicates(subset=["player_id", "game_date"], keep="last")
        return out
    if last_error:
        raise RuntimeError(f"NBA sync failed: {last_error}")
    return pd.DataFrame()


def fetch_nba_player_logs(player_id: str, player_name: str, seasons: list[str]) -> pd.DataFrame:
    pid = str(player_id)
    if pid.isdigit():
        try:
            df = _nba_logs_espn(pid, player_name, seasons)
            if not df.empty:
                return df
        except Exception:
            pass
        try:
            df = _nba_logs_statscom(pid, player_name, seasons)
            if not df.empty:
                return df
        except Exception:
            pass
        slug = _find_bref_slug_by_name(player_name)
        if slug:
            return _nba_logs_bref(slug, player_name, seasons)
        return pd.DataFrame()
    return _nba_logs_bref(pid, player_name, seasons)


def _nba_logs_statscom(player_id: str, player_name: str, seasons: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for season in seasons:
        season_label = _nba_season_label(season)
        url = "https://stats.nba.com/stats/playergamelog"
        params = {
            "PlayerID": player_id,
            "Season": season_label,
            "SeasonType": "Regular Season",
            "LeagueID": "00",
        }
        try:
            data = _get_json(url, timeout=NBA_TIMEOUT, headers=NBA_HEADERS, params=params)
            rs = data["resultSets"][0]
            h = {col: i for i, col in enumerate(rs["headers"])}
            for row in rs["rowSet"]:
                def g(key: str):
                    idx = h.get(key)
                    return row[idx] if idx is not None else None
                matchup = g("MATCHUP") or ""
                is_home = "@" not in matchup
                parts = matchup.split()
                team = parts[0] if parts else None
                opp = parts[-1] if len(parts) >= 3 else None
                raw_date = g("GAME_DATE") or ""
                game_date = _nba_parse_date(raw_date) if raw_date else None
                pts = safe_float(g("PTS"))
                reb = safe_float(g("REB"))
                ast = safe_float(g("AST"))
                threes = safe_float(g("FG3M"))
                rows.append({
                    "sport": "nba", "player_id": str(player_id), "player_name": player_name,
                    "season": str(season), "game_date": game_date, "opponent": opp, "team": team,
                    "home_away": "HOME" if is_home else "AWAY", "stat_points": pts,
                    "stat_rebounds": reb, "stat_assists": ast, "stat_threes": threes,
                    "stat_pra": (pts or 0) + (reb or 0) + (ast or 0),
                    "raw_json": json.dumps({"source": "stats.nba.com", "season": season_label, "raw_date": raw_date}),
                })
            time.sleep(0.4)
        except Exception:
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _nba_logs_bref(player_id: str, player_name: str, seasons: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for season in seasons:
        try:
            label = _nba_season_label(season)
            end_year = int(label.split("-")[0]) + 1
        except Exception:
            continue
        url = f"https://www.basketball-reference.com/players/{player_id[0]}/{player_id}/gamelog/{end_year}"
        try:
            r = requests.get(url, headers=UA, timeout=20)
            if r.status_code >= 400:
                continue
            soup = BeautifulSoup(r.text, "lxml")
            table = soup.find("table", {"id": "pgl_basic"})
            if not table:
                continue
            body = table.find("tbody")
            if not body:
                continue
            for tr in body.find_all("tr"):
                if "thead" in (tr.get("class") or []):
                    continue
                reason = tr.find("td", {"data-stat": "reason"})
                if reason and reason.get_text(strip=True):
                    continue
                date_td = tr.find("td", {"data-stat": "date_game"})
                if not date_td:
                    continue
                date = date_td.get_text(strip=True)
                if not date:
                    continue
                def gs(key: str):
                    td = tr.find("td", {"data-stat": key})
                    return safe_float(td.get_text(strip=True) if td else None)
                pts = gs("pts")
                reb = gs("trb")
                ast = gs("ast")
                threes = gs("fg3")
                ha_td = tr.find("td", {"data-stat": "game_location"})
                opp_td = tr.find("td", {"data-stat": "opp_id"})
                team_td = tr.find("td", {"data-stat": "team_id"})
                rows.append({
                    "sport": "nba", "player_id": str(player_id), "player_name": player_name,
                    "season": str(season), "game_date": date,
                    "opponent": opp_td.get_text(strip=True) if opp_td else None,
                    "team": team_td.get_text(strip=True) if team_td else None,
                    "home_away": "AWAY" if ha_td and ha_td.get_text(strip=True) == "@" else "HOME",
                    "stat_points": pts, "stat_rebounds": reb, "stat_assists": ast,
                    "stat_threes": threes, "stat_pra": (pts or 0) + (reb or 0) + (ast or 0),
                    "raw_json": json.dumps({"source": "basketball-reference", "season": end_year}),
                })
            time.sleep(0.15)
        except Exception:
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════
#  NHL  –  api‑web.nhle.com
# ══════════════════════════════════════════════════════════════════════

def _nhl_season_id(season_label: str) -> str:
    s = str(season_label).strip()
    if len(s) == 8 and s.isdigit():
        return s
    if "-" in s:
        start, end = s.split("-", 1)
        end = (start[:2] + end) if len(end) == 2 else end
        return f"{start}{end}"
    if len(s) == 4 and s.isdigit():
        return f"{s}{int(s) + 1}"
    return s


def search_nhl_players(query: str) -> list[dict]:
    q = (query or "").strip()
    if not q:
        return []
    url = (
        f"https://search.d3.nhle.com/api/v1/search/player"
        f"?culture=en-us&limit=20&q={quote(q)}"
    )
    data = _get_json(url, timeout=NHL_TIMEOUT)
    out: list[dict] = []
    seen: set[str] = set()
    for item in data:
        pid = item.get("playerId") or item.get("player_id")
        fn = item.get("firstName") or {}
        ln = item.get("lastName") or {}
        name = item.get("name") or (
            f"{fn.get('default','') if isinstance(fn, dict) else fn} "
            f"{ln.get('default','') if isinstance(ln, dict) else ln}".strip()
        )
        if pid and name and str(pid) not in seen:
            seen.add(str(pid))
            out.append({"id": str(pid), "name": name, "sport": "nhl"})
    return out[:15]


def fetch_nhl_player_logs(
    player_id: str, player_name: str, seasons: list[str]
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    last_error = None
    for season in seasons:
        sid = _nhl_season_id(season)
        url = f"https://api-web.nhle.com/v1/player/{player_id}/game-log/{sid}/2"
        try:
            payload = _get_json(url, timeout=NHL_TIMEOUT)
            games = (
                payload.get("gameLog")
                or payload.get("games")
                or payload.get("log")
                or []
            )
            if not games:
                continue
            rows: list[dict] = []
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


# ══════════════════════════════════════════════════════════════════════
#  MLB  –  statsapi.mlb.com
# ══════════════════════════════════════════════════════════════════════

def _mlb_years(seasons: list[str]) -> list[int]:
    years: list[int] = []
    for s in seasons:
        try:
            years.append(int(str(s).strip()[:4]))
        except Exception:
            pass
    return years


def search_mlb_players(query: str) -> list[dict]:
    q = (query or "").strip()
    if not q:
        return []
    url = f"https://statsapi.mlb.com/api/v1/people/search?names={quote(q)}"
    data = _get_json(url, timeout=MLB_TIMEOUT)
    out: list[dict] = []
    seen: set[str] = set()
    for p in (data.get("people") or [])[:15]:
        pid = p.get("id")
        name = p.get("fullName")
        if pid and name and str(pid) not in seen:
            seen.add(str(pid))
            out.append({"id": str(pid), "name": name, "sport": "mlb"})
    return out


def fetch_mlb_player_logs(
    player_id: str, player_name: str, seasons: list[str]
) -> pd.DataFrame:
    games: list[dict] = []
    for year in _mlb_years(seasons):
        url = (
            f"https://statsapi.mlb.com/api/v1/people/{player_id}"
            f"?hydrate=stats(group=[hitting,pitching],type=[gameLog],season={year})"
        )
        try:
            data = _get_json(url, timeout=MLB_TIMEOUT)
            people = data.get("people") or []
            if not people:
                continue
            for block in people[0].get("stats") or []:
                group = (
                    (block.get("group") or {}).get("displayName") or ""
                ).lower()
                for split in block.get("splits") or []:
                    stat = split.get("stat") or {}
                    opp = split.get("opponent") or {}
                    team = split.get("team") or {}
                    date = ((split.get("date") or split.get("gameDate") or "")[:10]) or None
                    row: dict = {
                        "sport": "mlb",
                        "player_id": str(player_id),
                        "player_name": player_name,
                        "season": str(year),
                        "game_date": date,
                        "opponent": (
                            opp.get("abbreviation") or opp.get("name")
                        ),
                        "team": team.get("abbreviation") or team.get("name"),
                        "home_away": None,
                        "stat_hits": 0.0,
                        "stat_home_runs": 0.0,
                        "stat_rbi": 0.0,
                        "stat_strikeouts": 0.0,
                        "raw_json": json.dumps(split),
                    }
                    if group == "hitting":
                        row["stat_hits"] = float(stat.get("hits", 0) or 0)
                        row["stat_home_runs"] = float(stat.get("homeRuns", 0) or 0)
                        row["stat_rbi"] = float(stat.get("rbi", 0) or 0)
                        row["stat_strikeouts"] = float(
                            stat.get("strikeOuts", 0) or 0
                        )
                    elif group == "pitching":
                        row["stat_strikeouts"] = float(
                            stat.get("strikeOuts", 0) or 0
                        )
                    games.append(row)
            time.sleep(0.2)
        except Exception:
            continue
    return pd.DataFrame(games) if games else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════
#  Unified dispatch
# ══════════════════════════════════════════════════════════════════════

def search_players(sport: str, query: str) -> list[dict]:
    q = (query or "").strip()
    if not q:
        return []
    sport = sport.lower()
    if sport == "nba":
        return search_nba_players(q)
    if sport == "nhl":
        return search_nhl_players(q)
    if sport == "mlb":
        return search_mlb_players(q)
    return []


def default_recent_seasons(sport: str, count: int = 5) -> list[str]:
    sport = sport.lower()
    now = datetime.utcnow()
    if sport in {"nba", "nhl"}:
        current_start = now.year if now.month >= 10 else now.year - 1
        starts = list(range(current_start, current_start - count, -1))
        return [f"{y}-{str(y + 1)[2:]}" for y in starts]
    if sport == "mlb":
        current_year = now.year
        return [str(y) for y in range(current_year, current_year - count, -1)]
    return []
