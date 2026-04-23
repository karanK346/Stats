from __future__ import annotations

import json
import time
from datetime import datetime
from urllib.parse import quote

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
#  NBA  –  Basketball-Reference (search + game logs)
# ══════════════════════════════════════════════════════════════════════

def _nba_season_label(season: str) -> str:
    """Normalise any season string to the 'YYYY-YY' format."""
    season = str(season).strip()
    if "-" in season:
        start, end = season.split("-", 1)
        start = start.strip()
        end = end.strip()
        if len(end) == 4:
            end = end[2:]
        return f"{start}-{end.zfill(2)}"
    try:
        y = int(season[:4])
        return f"{y}-{str(y + 1)[2:]}"
    except Exception:
        return season


_nba_all_players: list[dict] = []
_nba_all_players_ts: float = 0.0


def _fetch_all_nba_players() -> list[dict]:
    global _nba_all_players, _nba_all_players_ts
    now = time.time()
    if _nba_all_players and (now - _nba_all_players_ts) < 24 * 3600:
        return _nba_all_players

    players: list[dict] = []
    seen: set[str] = set()
    for letter in "abcdefghijklmnopqrstuvwxyz":
        url = f"https://www.basketball-reference.com/players/{letter}/"
        try:
            r = requests.get(url, headers=UA, timeout=20)
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
                link = th.find("a")
                if not link:
                    continue
                name = link.get_text(" ", strip=True)
                slug = link.get("href", "").rstrip("/").split("/")[-1].replace(".html", "")
                if not slug or slug in seen:
                    continue
                seen.add(slug)
                players.append({"id": slug, "name": name, "sport": "nba"})
            time.sleep(0.03)
        except Exception:
            continue
    _nba_all_players = players
    _nba_all_players_ts = now
    return players


def search_nba_players(query: str) -> list[dict]:
    q = (query or "").strip().lower()
    if not q:
        return []
    players = _fetch_all_nba_players()
    ranked = []
    for p in players:
        name = p["name"].lower()
        score = 0
        if name.startswith(q):
            score = 0
        elif q in name:
            score = 1
        else:
            continue
        ranked.append((score, len(name), p["name"], p))
    ranked.sort(key=lambda x: (x[0], x[1], x[2]))
    return [x[3] for x in ranked[:15]]


def fetch_nba_player_logs(
    player_id: str, player_name: str, seasons: list[str]
) -> pd.DataFrame:
    rows: list[dict] = []
    for season in seasons:
        try:
            label = _nba_season_label(season)
            end_year = int(label.split("-")[0]) + 1
        except Exception:
            continue
        url = (
            f"https://www.basketball-reference.com/players/"
            f"{player_id[0]}/{player_id}/gamelog/{end_year}"
        )
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
                    "sport": "nba",
                    "player_id": str(player_id),
                    "player_name": player_name,
                    "season": str(season),
                    "game_date": date,
                    "opponent": opp_td.get_text(strip=True) if opp_td else None,
                    "team": team_td.get_text(strip=True) if team_td else None,
                    "home_away": (
                        "AWAY"
                        if ha_td and ha_td.get_text(strip=True) == "@"
                        else "HOME"
                    ),
                    "stat_points": pts,
                    "stat_rebounds": reb,
                    "stat_assists": ast,
                    "stat_threes": threes,
                    "stat_pra": (pts or 0) + (reb or 0) + (ast or 0),
                    "raw_json": json.dumps({
                        "source": "basketball_reference",
                        "season": season,
                        "date": date,
                    }),
                })
            time.sleep(0.18)
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
        current_start = now.year if now.month >= 7 else now.year - 1
        starts = list(range(current_start, current_start - count, -1))
        return [f"{y}-{str(y + 1)[2:]}" for y in starts]
    if sport == "mlb":
        current_year = now.year
        return [str(y) for y in range(current_year, current_year - count, -1)]
    return []
