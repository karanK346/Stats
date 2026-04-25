"""
data_sources.py  –  Player Line Predictor V3.0
=================================================
NBA  search  → nba_api static JSON  (bundled in the package, NO HTTP call)
NBA  logs    → Basketball Reference  (scrape; more permissive than NBA.com)
NHL          → api-web.nhle.com  (official, no-auth public API)
MLB          → statsapi.mlb.com   (official, no-auth public API)
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── timeouts ────────────────────────────────────────────────────────
NHL_TIMEOUT = 30
MLB_TIMEOUT = 30
BREF_TIMEOUT = 20

# ── generic browser-like headers ────────────────────────────────────
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

UA_JSON = {**UA, "Accept": "application/json, text/plain, */*"}


def safe_float(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _get_json(url: str, timeout: int = 30,
              headers: dict | None = None,
              params: dict | None = None) -> dict:
    h = dict(UA_JSON)
    if headers:
        h.update(headers)
    r = requests.get(url, headers=h, timeout=timeout, params=params)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} from {url}")
    ct = (r.headers.get("content-type") or "").lower()
    if "json" not in ct and "javascript" not in ct:
        raise RuntimeError(f"Non-JSON response ({ct}) from {url}")
    return r.json()


# ══════════════════════════════════════════════════════════════════
#  NBA  – search via nba_api static list  /  logs via BBRef scrape
# ══════════════════════════════════════════════════════════════════

# ---------- season helpers ----------

def _nba_season_label(season: str) -> str:
    """'2024-25', '2024-2025', '2024' → '2024-25'"""
    s = str(season).strip()
    if "-" in s:
        start, end = s.split("-", 1)
        if len(end) == 4:
            end = end[2:]
        return f"{start.strip()}-{end.strip().zfill(2)}"
    try:
        y = int(s[:4])
        return f"{y}-{str(y + 1)[2:]}"
    except Exception:
        return s


def _nba_end_year(season: str) -> int:
    """'2024-25' → 2025"""
    label = _nba_season_label(season)
    return int(label.split("-")[0]) + 1


def _nba_parse_date(raw: str) -> str:
    """'OCT 23, 2024' → '2024-10-23'"""
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw


# ---------- player search (LOCAL – no HTTP) ----------

_nba_static_cache: list[dict] = []


def _nba_static_players() -> list[dict]:
    """
    nba_api bundles the full player list as a JSON file inside the package.
    get_players() just reads that file – no network call at all.
    """
    global _nba_static_cache
    if _nba_static_cache:
        return _nba_static_cache
    try:
        from nba_api.stats.static import players as _p
        out = []
        seen: set[str] = set()
        for p in _p.get_players():
            pid = str(p.get("id") or "")
            name = (p.get("full_name") or "").strip()
            if pid and name and pid not in seen:
                seen.add(pid)
                out.append({"id": pid, "name": name, "sport": "nba"})
        _nba_static_cache = out
        return out
    except Exception:
        return []


def search_nba_players(query: str) -> list[dict]:
    q = (query or "").strip().lower()
    if not q:
        return []

    # Primary: local nba_api static JSON  (zero HTTP, instant)
    players = _nba_static_players()
    if players:
        hits = [p for p in players if q in p["name"].lower()]
        hits.sort(key=lambda p: (
            not p["name"].lower().startswith(q),
            len(p["name"]),
            p["name"],
        ))
        return hits[:15]

    # Fallback: Basketball Reference search page
    return _bref_player_search(q)


# ---------- BBRef player search (used as fallback) ----------

def _bref_player_search(query: str) -> list[dict]:
    """
    Use BBRef's /search endpoint – single HTTP request, fast.
    Handles both direct-redirect (exact match) and search results page.
    """
    url = f"https://www.basketball-reference.com/search/search.fcgi?search={quote(query)}"
    try:
        r = requests.get(url, headers=UA, timeout=BREF_TIMEOUT, allow_redirects=True)
        if r.status_code >= 400:
            return []

        soup = BeautifulSoup(r.text, "lxml")
        results: list[dict] = []
        seen: set[str] = set()

        # BBRef may redirect straight to a player page on exact match
        if "/players/" in r.url and "search" not in r.url:
            slug = r.url.rstrip("/").split("/")[-1].replace(".html", "")
            h1 = soup.find("h1", {"itemprop": "name"}) or soup.find("h1")
            name = h1.get_text(strip=True) if h1 else slug
            return [{"id": slug, "name": name, "sport": "nba"}]

        # Search results page
        for div in soup.select("div.search-item-name"):
            a = div.find("a", href=True)
            if not a or "/players/" not in a["href"]:
                continue
            slug = a["href"].rstrip("/").split("/")[-1].replace(".html", "")
            name = a.get_text(strip=True)
            if slug and slug not in seen:
                seen.add(slug)
                results.append({"id": slug, "name": name, "sport": "nba"})
            if len(results) >= 15:
                break

        return results
    except Exception:
        return []


# ---------- game log fetching ----------

def _nba_slug_for_name(player_name: str) -> str | None:
    """Find a BBRef slug for a player name, using the search page."""
    hits = _bref_player_search(player_name)
    if not hits:
        return None
    # prefer exact name match
    nl = player_name.lower()
    for h in hits:
        if h["name"].lower() == nl:
            return h["id"]
    return hits[0]["id"]


def _nba_logs_bref(slug: str, player_name: str, seasons: list[str]) -> pd.DataFrame:
    """Scrape game logs from Basketball-Reference using a slug."""
    rows: list[dict] = []
    first = slug[0] if slug else "a"
    for season in seasons:
        end_year = _nba_end_year(season)
        url = (
            f"https://www.basketball-reference.com/players/"
            f"{first}/{slug}/gamelog/{end_year}"
        )
        try:
            r = requests.get(url, headers=UA, timeout=BREF_TIMEOUT)
            if r.status_code >= 400:
                continue
            soup = BeautifulSoup(r.text, "lxml")
            table = soup.find("table", {"id": "pgl_basic"})
            if not table:
                continue
            tbody = table.find("tbody")
            if not tbody:
                continue

            for tr in tbody.find_all("tr"):
                cls = tr.get("class") or []
                # skip header rows and partial-season separators
                if "thead" in cls or "partial_table" in cls:
                    continue
                # skip rows where player didn't play / was inactive
                reason_td = tr.find("td", {"data-stat": "reason"})
                if reason_td and reason_td.get_text(strip=True):
                    continue

                date_td = tr.find("td", {"data-stat": "date_game"})
                if not date_td:
                    continue
                date_str = date_td.get_text(strip=True)
                if not date_str or date_str == "Date":
                    continue

                # helper – capture tr correctly inside loop
                def gs(key: str, _tr=tr) -> float | None:
                    td = _tr.find("td", {"data-stat": key})
                    if not td:
                        return None
                    return safe_float(td.get_text(strip=True))

                pts    = gs("pts")
                reb    = gs("trb")
                ast    = gs("ast")
                threes = gs("fg3")

                ha_td   = tr.find("td", {"data-stat": "game_location"})
                opp_td  = tr.find("td", {"data-stat": "opp_id"})
                team_td = tr.find("td", {"data-stat": "team_id"})

                is_home = not (ha_td and ha_td.get_text(strip=True) == "@")

                rows.append({
                    "sport":          "nba",
                    "player_id":      slug,
                    "player_name":    player_name,
                    "season":         str(season),
                    "game_date":      date_str,
                    "opponent":       opp_td.get_text(strip=True)  if opp_td  else None,
                    "team":           team_td.get_text(strip=True) if team_td else None,
                    "home_away":      "HOME" if is_home else "AWAY",
                    "stat_points":    pts,
                    "stat_rebounds":  reb,
                    "stat_assists":   ast,
                    "stat_threes":    threes,
                    "stat_pra":       (pts or 0) + (reb or 0) + (ast or 0),
                    "raw_json":       json.dumps({
                        "source": "basketball_reference",
                        "season": season,
                        "date":   date_str,
                    }),
                })
            time.sleep(0.4)           # polite crawl delay
        except Exception:
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _nba_logs_statscom(player_id: str, player_name: str, seasons: list[str]) -> pd.DataFrame:
    """
    Direct stats.nba.com request.
    Works from local/residential IPs; often blocked on cloud servers.
    Kept as first-try for numeric IDs in case Render's IP works.
    """
    NBA_HEADERS = {
        **UA_JSON,
        "Host":                  "stats.nba.com",
        "Referer":               "https://www.nba.com/",
        "Origin":                "https://www.nba.com",
        "x-nba-stats-origin":    "stats",
        "x-nba-stats-token":     "true",
        "Sec-Fetch-Site":        "same-site",
        "Sec-Fetch-Mode":        "cors",
        "Sec-Fetch-Dest":        "empty",
    }
    rows: list[dict] = []
    for season in seasons:
        season_label = _nba_season_label(season)
        try:
            data = _get_json(
                "https://stats.nba.com/stats/playergamelog",
                timeout=20,
                headers=NBA_HEADERS,
                params={
                    "PlayerID":   player_id,
                    "Season":     season_label,
                    "SeasonType": "Regular Season",
                    "LeagueID":   "00",
                },
            )
            rs = data["resultSets"][0]
            h  = {col: i for i, col in enumerate(rs["headers"])}
            for row in rs["rowSet"]:
                def g(key: str, _h=h, _row=row):
                    idx = _h.get(key)
                    return _row[idx] if idx is not None else None
                matchup  = g("MATCHUP") or ""
                is_home  = "@" not in matchup
                parts    = matchup.split()
                raw_date = g("GAME_DATE") or ""
                pts      = safe_float(g("PTS"))
                reb      = safe_float(g("REB"))
                ast      = safe_float(g("AST"))
                threes   = safe_float(g("FG3M"))
                rows.append({
                    "sport":         "nba",
                    "player_id":     str(player_id),
                    "player_name":   player_name,
                    "season":        str(season),
                    "game_date":     _nba_parse_date(raw_date) if raw_date else None,
                    "opponent":      parts[-1] if len(parts) >= 3 else None,
                    "team":          parts[0] if parts else None,
                    "home_away":     "HOME" if is_home else "AWAY",
                    "stat_points":   pts,
                    "stat_rebounds": reb,
                    "stat_assists":  ast,
                    "stat_threes":   threes,
                    "stat_pra":      (pts or 0) + (reb or 0) + (ast or 0),
                    "raw_json":      json.dumps({
                        "source": "stats.nba.com",
                        "season": season_label,
                    }),
                })
            time.sleep(0.4)
        except Exception:
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_nba_player_logs(player_id: str, player_name: str, seasons: list[str]) -> pd.DataFrame:
    """
    Strategy:
      1. If numeric ID → try stats.nba.com (works on some server IPs)
      2. Find BBRef slug from player name → scrape Basketball Reference
      3. If slug-based ID provided directly → go straight to BBRef
    """
    slug: str | None = None

    # numeric ID from nba_api static list
    if str(player_id).isdigit():
        # try the official API first (fast if it works)
        try:
            df = _nba_logs_statscom(player_id, player_name, seasons)
            if not df.empty:
                return df
        except Exception:
            pass
        # resolve to a BBRef slug via search
        slug = _nba_slug_for_name(player_name)
    else:
        slug = player_id   # already a BBRef slug

    if not slug:
        return pd.DataFrame()

    df = _nba_logs_bref(slug, player_name, seasons)
    # rewrite player_id so storage key stays consistent
    if not df.empty:
        df["player_id"] = str(player_id)
    return df


# ══════════════════════════════════════════════════════════════════
#  NHL  –  api-web.nhle.com
# ══════════════════════════════════════════════════════════════════

def _nhl_season_id(label: str) -> str:
    s = str(label).strip()
    if len(s) == 8 and s.isdigit():
        return s
    if "-" in s:
        start, end = s.split("-", 1)
        if len(end) == 2:
            end = start[:2] + end
        return f"{start}{end}"
    if len(s) == 4 and s.isdigit():
        return f"{s}{int(s) + 1}"
    return s


def search_nhl_players(query: str) -> list[dict]:
    q = (query or "").strip()
    if not q:
        return []
    data = _get_json(
        f"https://search.d3.nhle.com/api/v1/search/player"
        f"?culture=en-us&limit=20&q={quote(q)}",
        timeout=NHL_TIMEOUT,
    )
    out: list[dict] = []
    seen: set[str] = set()
    for item in data:
        pid = item.get("playerId") or item.get("player_id")
        fn  = item.get("firstName") or {}
        ln  = item.get("lastName")  or {}
        name = item.get("name") or (
            f"{fn.get('default','') if isinstance(fn, dict) else fn} "
            f"{ln.get('default','') if isinstance(ln, dict) else ln}".strip()
        )
        if pid and name and str(pid) not in seen:
            seen.add(str(pid))
            out.append({"id": str(pid), "name": name, "sport": "nhl"})
    return out[:15]


def fetch_nhl_player_logs(player_id: str, player_name: str, seasons: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    last_err = None
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
            rows = []
            for g in games:
                opp = g.get("opponentAbbrev")
                if not opp and isinstance(g.get("opponentCommonName"), dict):
                    opp = g["opponentCommonName"].get("default")
                rows.append({
                    "sport":         "nhl",
                    "player_id":     str(player_id),
                    "player_name":   player_name,
                    "season":        str(season),
                    "game_date":     g.get("gameDate") or g.get("date"),
                    "opponent":      opp,
                    "team":          g.get("teamAbbrev"),
                    "home_away":     "HOME" if g.get("homeRoadFlag") == "H" else "AWAY",
                    "stat_goals":    safe_float(g.get("goals")),
                    "stat_assists":  safe_float(g.get("assists")),
                    "stat_shots":    safe_float(g.get("shots")),
                    "stat_points":   safe_float(g.get("points")),
                    "raw_json":      json.dumps(g),
                })
            if rows:
                frames.append(pd.DataFrame(rows))
            time.sleep(0.3)
        except Exception as e:
            last_err = e
    if frames:
        return pd.concat(frames, ignore_index=True)
    if last_err:
        raise RuntimeError(f"NHL sync failed: {last_err}")
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
#  MLB  –  statsapi.mlb.com
# ══════════════════════════════════════════════════════════════════

def _mlb_years(seasons: list[str]) -> list[int]:
    out: list[int] = []
    for s in seasons:
        try:
            out.append(int(str(s).strip()[:4]))
        except Exception:
            pass
    return out


def search_mlb_players(query: str) -> list[dict]:
    q = (query or "").strip()
    if not q:
        return []
    data = _get_json(
        f"https://statsapi.mlb.com/api/v1/people/search?names={quote(q)}",
        timeout=MLB_TIMEOUT,
    )
    out: list[dict] = []
    seen: set[str] = set()
    for p in (data.get("people") or [])[:15]:
        pid  = p.get("id")
        name = p.get("fullName")
        if pid and name and str(pid) not in seen:
            seen.add(str(pid))
            out.append({"id": str(pid), "name": name, "sport": "mlb"})
    return out


def fetch_mlb_player_logs(player_id: str, player_name: str, seasons: list[str]) -> pd.DataFrame:
    games: list[dict] = []
    for year in _mlb_years(seasons):
        url = (
            f"https://statsapi.mlb.com/api/v1/people/{player_id}"
            f"?hydrate=stats(group=[hitting,pitching],type=[gameLog],season={year})"
        )
        try:
            data   = _get_json(url, timeout=MLB_TIMEOUT)
            people = data.get("people") or []
            if not people:
                continue
            for block in people[0].get("stats") or []:
                group = ((block.get("group") or {}).get("displayName") or "").lower()
                for split in block.get("splits") or []:
                    stat = split.get("stat") or {}
                    opp  = split.get("opponent") or {}
                    team = split.get("team") or {}
                    date = ((split.get("date") or split.get("gameDate") or "")[:10]) or None
                    row: dict = {
                        "sport":           "mlb",
                        "player_id":       str(player_id),
                        "player_name":     player_name,
                        "season":          str(year),
                        "game_date":       date,
                        "opponent":        opp.get("abbreviation") or opp.get("name"),
                        "team":            team.get("abbreviation") or team.get("name"),
                        "home_away":       None,
                        "stat_hits":       0.0,
                        "stat_home_runs":  0.0,
                        "stat_rbi":        0.0,
                        "stat_strikeouts": 0.0,
                        "raw_json":        json.dumps(split),
                    }
                    if group == "hitting":
                        row["stat_hits"]       = float(stat.get("hits",       0) or 0)
                        row["stat_home_runs"]  = float(stat.get("homeRuns",   0) or 0)
                        row["stat_rbi"]        = float(stat.get("rbi",        0) or 0)
                        row["stat_strikeouts"] = float(stat.get("strikeOuts", 0) or 0)
                    elif group == "pitching":
                        row["stat_strikeouts"] = float(stat.get("strikeOuts", 0) or 0)
                    games.append(row)
            time.sleep(0.2)
        except Exception:
            continue
    return pd.DataFrame(games) if games else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
#  Unified dispatch
# ══════════════════════════════════════════════════════════════════

def search_players(sport: str, query: str) -> list[dict]:
    sport = sport.lower()
    if sport == "nba":
        return search_nba_players((query or "").strip())
    if sport == "nhl":
        return search_nhl_players((query or "").strip())
    if sport == "mlb":
        return search_mlb_players((query or "").strip())
    return []


def default_recent_seasons(sport: str, count: int = 5) -> list[str]:
    """
    Current date: April 2026
      NBA/NHL → 2025-26 season is the current one (playoffs)
      MLB     → 2026 season is live (started March 2026)
    """
    sport = sport.lower()
    if sport in {"nba", "nhl"}:
        starts = list(range(2025, 2025 - count, -1))
        return [f"{y}-{str(y + 1)[2:]}" for y in starts]
    if sport == "mlb":
        return [str(y) for y in range(2026, 2026 - count, -1)]
    return []
