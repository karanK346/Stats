from __future__ import annotations
import time
import json
import re
from io import StringIO
from bs4 import BeautifulSoup, Comment
import requests
import pandas as pd

# MLB
import statsapi

NHL_TIMEOUT = 30
UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://www.basketball-reference.com/",
    "Origin": "https://www.basketball-reference.com",
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


def _get_html(url: str, timeout: int = 30) -> str:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.text


def _extract_table_df_from_bref_html(html: str, table_id: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": table_id})
    if table is None:
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            if table_id in comment:
                comment_soup = BeautifulSoup(comment, "html.parser")
                table = comment_soup.find("table", {"id": table_id})
                if table is not None:
                    break
    if table is None:
        raise RuntimeError(f"Could not find table {table_id} on Basketball Reference page.")
    df = pd.read_html(StringIO(str(table)))[0]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
    return df


def _nba_search_bref(query: str):
    url = f"https://www.basketball-reference.com/search/search.fcgi?search={requests.utils.quote(query)}"
    html = _get_html(url)
    soup = BeautifulSoup(html, "html.parser")
    out = []
    seen = set()

    for a in soup.select('a[href^="/players/"]'):
        href = a.get("href", "")
        m = re.match(r"/players/[a-z]/([a-z0-9]+)\.html", href)
        if not m:
            continue
        slug = m.group(1)
        name = a.get_text(" ", strip=True)
        if not name or slug in seen:
            continue
        seen.add(slug)
        out.append({"id": slug, "name": name, "sport": "nba"})

    if not out:
        # fallback heuristic from page body
        for m in re.finditer(r'/players/[a-z]/([a-z0-9]+)\.html', html):
            slug = m.group(1)
            if slug in seen:
                continue
            seen.add(slug)
            out.append({"id": slug, "name": slug, "sport": "nba"})

    return out[:15]


def search_players(sport: str, query: str):
    q = (query or "").strip().lower()
    if not q:
        return []

    if sport == "nba":
        return _nba_search_bref(q)

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


def _nba_bref_year_from_season_label(season_label: str) -> int:
    s = str(season_label).strip()
    if '-' in s:
        start, end = s.split('-', 1)
        if len(end) == 2:
            end = start[:2] + end
        return int(end)
    if len(s) == 4 and s.isdigit():
        return int(s) + 1
    if len(s) == 8 and s.isdigit():
        return int(s[4:])
    return int(s)


def fetch_nba_player_logs(player_id: str, player_name: str, seasons: list[str]) -> pd.DataFrame:
    frames = []
    last_error = None
    for season in seasons:
        try:
            year_id = _nba_bref_year_from_season_label(season)
            slug = str(player_id)
            url = f"https://www.basketball-reference.com/players/{slug[0]}/{slug}/gamelog/{year_id}"
            html = _get_html(url, timeout=30)
            df = _extract_table_df_from_bref_html(html, "pgl_basic")
            if df.empty:
                continue
            df = df[df.iloc[:, 0].astype(str) != "Rk"].copy()
            # normalize common column names
            col_map = {str(c): str(c).strip() for c in df.columns}
            df = df.rename(columns=col_map)
            away_col = next((c for c in df.columns if str(c).startswith("Unnamed:") or str(c) == ""), None)
            date_col = "Date" if "Date" in df.columns else next((c for c in df.columns if str(c).lower() == "date"), None)
            team_col = "Tm" if "Tm" in df.columns else next((c for c in df.columns if str(c).lower() in {"tm","team"}), None)
            opp_col = "Opp" if "Opp" in df.columns else next((c for c in df.columns if str(c).lower() in {"opp","opponent"}), None)
            pts_col = "PTS" if "PTS" in df.columns else None
            reb_col = "TRB" if "TRB" in df.columns else ("REB" if "REB" in df.columns else None)
            ast_col = "AST" if "AST" in df.columns else None
            threes_col = "3P" if "3P" in df.columns else ("FG3" if "FG3" in df.columns else None)

            if not date_col or not opp_col or not pts_col:
                raise RuntimeError("Basketball Reference game log columns were not in the expected format.")

            # filter to played games only
            played = pd.to_numeric(df.get("GmSc"), errors="coerce").notna() | pd.to_numeric(df.get(pts_col), errors="coerce").notna()
            df = df[played].copy()
            if df.empty:
                continue

            out = pd.DataFrame()
            out["sport"] = "nba"
            out["player_id"] = slug
            out["player_name"] = player_name
            out["season"] = str(season)
            out["game_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
            out["opponent"] = df[opp_col].astype(str)
            out["team"] = df[team_col].astype(str) if team_col else None
            if away_col:
                out["home_away"] = df[away_col].astype(str).apply(lambda x: "AWAY" if x.strip() == "@" else "HOME")
            else:
                out["home_away"] = None
            out["stat_points"] = pd.to_numeric(df[pts_col], errors="coerce")
            out["stat_rebounds"] = pd.to_numeric(df[reb_col], errors="coerce") if reb_col else None
            out["stat_assists"] = pd.to_numeric(df[ast_col], errors="coerce") if ast_col else None
            out["stat_threes"] = pd.to_numeric(df[threes_col], errors="coerce") if threes_col else None
            out["stat_pra"] = out["stat_points"].fillna(0) + out["stat_rebounds"].fillna(0) + out["stat_assists"].fillna(0)
            out["raw_json"] = [json.dumps(r) for r in df.to_dict(orient="records")]
            frames.append(out.dropna(subset=["game_date"]))
            time.sleep(0.7)
        except Exception as e:
            last_error = e
            time.sleep(0.7)
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
