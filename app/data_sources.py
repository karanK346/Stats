from __future__ import annotations
import json
import time
from typing import Any, Iterable

from bs4 import BeautifulSoup
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

# ---------------- NBA (FIXED VERSION) ---------------- #

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

def search_nba_players(query: str) -> list[dict]:
    """
    Clean search using Basketball Reference player index
    """
    query = query.lower()
    results = []

    alphabet = list("abcdefghijklmnopqrstuvwxyz")

    for letter in alphabet:
        url = f"https://www.basketball-reference.com/players/{letter}/"
        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")

            rows = soup.select("table#players tbody tr")
            for row in rows:
                name = row.find("th").text.strip()
                link = row.find("a")
                if not link:
                    continue

                player_id = link["href"].split("/")[-1].replace(".html", "")

                if query in name.lower():
                    results.append({
                        "id": player_id,
                        "name": name,
                        "sport": "nba"
                    })

                if len(results) >= 15:
                    return results

        except:
            continue

    return results


def fetch_nba_player_logs(player_id: str, player_name: str, seasons: list[str]) -> pd.DataFrame:
    """
    Scrapes Basketball Reference game logs
    """
    all_games = []

    for season in seasons:
        try:
            year = int(season.split("-")[0]) + 1  # BR uses end year
        except:
            continue

        url = f"https://www.basketball-reference.com/players/{player_id[0]}/{player_id}/gamelog/{year}"

        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")

            table = soup.find("table", {"id": "pgl_basic"})
            if table is None:
                continue

            rows = table.find("tbody").find_all("tr")

            for row in rows:
                if "thead" in row.get("class", []):
                    continue

                cols = row.find_all("td")
                if len(cols) < 20:
                    continue

                try:
                    game_date = cols[1].text.strip()
                    pts = float(cols[27].text.strip() or 0)
                    reb = float(cols[18].text.strip() or 0)
                    ast = float(cols[19].text.strip() or 0)
                    threes = float(cols[12].text.strip() or 0)

                    all_games.append({
                        "sport": "nba",
                        "player_id": player_id,
                        "player_name": player_name,
                        "season": season,
                        "game_date": game_date,
                        "stat_points": pts,
                        "stat_rebounds": reb,
                        "stat_assists": ast,
                        "stat_threes": threes,
                        "stat_pra": pts + reb + ast
                    })
                except:
                    continue

            time.sleep(0.3)

        except:
            continue

    if not all_games:
        return pd.DataFrame()

    df = pd.DataFrame(all_games)

    df = df.drop_duplicates(subset=["player_id", "game_date"])
    df = df.sort_values("game_date")

    return df


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
