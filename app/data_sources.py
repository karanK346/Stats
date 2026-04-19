from __future__ import annotations
import json
import time
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup

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


# ---------------- NBA (Basketball Reference) ---------------- #

def search_nba_players(query: str) -> list[dict]:
    q = (query or "").strip().lower()
    if not q:
        return []
    results: list[dict] = []
    seen: set[str] = set()
    for letter in "abcdefghijklmnopqrstuvwxyz":
        url = f"https://www.basketball-reference.com/players/{letter}/"
        try:
            r = requests.get(url, headers=UA, timeout=20)
            if r.status_code >= 400:
                continue
            soup = BeautifulSoup(r.text, "lxml")
            table = soup.find("table", {"id": "players"})
            if table is None:
                continue
            body = table.find("tbody")
            if body is None:
                continue
            for row in body.find_all("tr"):
                if "thead" in (row.get("class") or []):
                    continue
                th = row.find("th", {"data-stat": "player"}) or row.find("th")
                if th is None:
                    continue
                a = th.find("a")
                if not a:
                    continue
                name = a.get_text(" ", strip=True)
                href = a.get("href", "")
                slug = href.rstrip("/").split("/")[-1].replace(".html", "")
                if not slug or slug in seen:
                    continue
                hay = f"{name} {slug}".lower()
                if q not in hay:
                    continue
                seen.add(slug)
                results.append({"id": slug, "name": name, "sport": "nba"})
                if len(results) >= 15:
                    return results
            time.sleep(0.05)
        except Exception:
            continue
    return results


def fetch_nba_player_logs(player_id: str, player_name: str, seasons: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for season in seasons:
        try:
            end_year = int(str(season)[:4]) + 1 if "-" in str(season) else int(str(season)[:4])
        except Exception:
            continue
        url = f"https://www.basketball-reference.com/players/{player_id[0]}/{player_id}/gamelog/{end_year}"
        try:
            r = requests.get(url, headers=UA, timeout=20)
            if r.status_code >= 400:
                continue
            soup = BeautifulSoup(r.text, "lxml")
            table = soup.find("table", {"id": "pgl_basic"})
            if table is None:
                continue
            body = table.find("tbody")
            if body is None:
                continue
            for tr in body.find_all("tr"):
                if "thead" in (tr.get("class") or []):
                    continue
                reason_td = tr.find("td", {"data-stat": "reason"})
                if reason_td and reason_td.get_text(strip=True):
                    continue
                date_td = tr.find("td", {"data-stat": "date_game"})
                if date_td is None:
                    continue
                date = date_td.get_text(strip=True)
                homeaway = tr.find("td", {"data-stat": "game_location"})
                opp = tr.find("td", {"data-stat": "opp_id"})
                team = tr.find("td", {"data-stat": "team_id"})
                def get_stat(key: str):
                    td = tr.find("td", {"data-stat": key})
                    return safe_float(td.get_text(strip=True) if td else None)
                pts = get_stat("pts")
                reb = get_stat("trb")
                ast = get_stat("ast")
                threes = get_stat("fg3")
                rows.append({
                    "sport": "nba",
                    "player_id": str(player_id),
                    "player_name": player_name,
                    "season": str(season),
                    "game_date": date,
                    "opponent": opp.get_text(strip=True) if opp else None,
                    "team": team.get_text(strip=True) if team else None,
                    "home_away": "AWAY" if homeaway and homeaway.get_text(strip=True) == "@" else "HOME",
                    "stat_points": pts,
                    "stat_rebounds": reb,
                    "stat_assists": ast,
                    "stat_threes": threes,
                    "stat_pra": (pts or 0) + (reb or 0) + (ast or 0),
                    "raw_json": json.dumps({"source": "basketball_reference", "season": season, "date": date}),
                })
            time.sleep(0.2)
        except Exception:
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


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


def fetch_mlb_player_logs(player_id: str, player_name: str, seasons: list[str]) -> pd.DataFrame:
    games: list[dict] = []
    for year in _mlb_years_from_labels(seasons):
        url = f"https://statsapi.mlb.com/api/v1/people/{player_id}?hydrate=stats(group=[hitting,pitching],type=[gameLog],season={year})"
        try:
            data = _get_json(url)
            people = data.get("people", [])
            if not people:
                continue
            for block in people[0].get("stats", []) or []:
                group = ((block.get("group") or {}).get("displayName") or "").lower()
                for g in block.get("splits", []) or []:
                    stat = g.get("stat", {}) or {}
                    opp = g.get("opponent") or {}
                    team = g.get("team") or {}
                    date = (g.get("date") or g.get("gameDate") or "")[:10] or None
                    row = {
                        "sport": "mlb",
                        "player_id": str(player_id),
                        "player_name": player_name,
                        "season": str(year),
                        "game_date": date,
                        "opponent": opp.get("abbreviation") or opp.get("name"),
                        "team": team.get("abbreviation") or team.get("name"),
                        "home_away": None,
                        "stat_hits": 0.0,
                        "stat_home_runs": 0.0,
                        "stat_rbi": 0.0,
                        "stat_strikeouts": 0.0,
                        "raw_json": json.dumps(g),
                    }
                    if group == "hitting":
                        row["stat_hits"] = float(stat.get("hits", 0) or 0)
                        row["stat_home_runs"] = float(stat.get("homeRuns", 0) or 0)
                        row["stat_rbi"] = float(stat.get("rbi", 0) or 0)
                        row["stat_strikeouts"] = float(stat.get("strikeOuts", 0) or 0)
                    elif group == "pitching":
                        row["stat_strikeouts"] = float(stat.get("strikeOuts", 0) or 0)
                    games.append(row)
            time.sleep(0.2)
        except Exception:
            continue
    return pd.DataFrame(games) if games else pd.DataFrame()


# ---------------- Unified search + helpers ---------------- #

def search_players(sport: str, query: str):
    q = (query or "").strip()
    if not q:
        return []

    if sport == "nba":
        return search_nba_players(q)

    if sport == "mlb":
        url = f"https://statsapi.mlb.com/api/v1/people/search?names={quote(q)}"
        data = _get_json(url)
        out = []
        for p in data.get("people", [])[:15]:
            pid = p.get("id")
            name = p.get("fullName")
            if pid and name:
                out.append({"id": str(pid), "name": name, "sport": "mlb"})
        return out

    if sport == "nhl":
        url = f"https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit=20&q={quote(q)}"
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
