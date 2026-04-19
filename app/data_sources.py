import requests
import pandas as pd

# ---------------- HELPERS ---------------- #

UA = {
    "User-Agent": "Mozilla/5.0"
}

def _get_json(url):
    r = requests.get(url, headers=UA)
    r.raise_for_status()
    return r.json()

# ---------------- NBA (ESPN) ---------------- #

def search_nba_player(name: str):
    url = f"https://site.web.api.espn.com/apis/common/v3/search?query={name}&limit=5"
    data = _get_json(url)

    results = []
    for item in data.get("items", []):
        athlete = item.get("resource", {})
        if athlete.get("type") == "athlete":
            results.append({
                "id": athlete.get("id"),
                "name": athlete.get("displayName")
            })
    return results


def fetch_nba_player_logs(player_id: str, player_name: str, seasons: list):
    url = f"https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{player_id}/gamelog"
    data = _get_json(url)

    games = []
    for game in data.get("events", []):
        stats = game.get("stats", [])
        if not stats:
            continue

        games.append({
            "player_name": player_name,
            "points": float(stats[0]) if len(stats) > 0 else 0,
            "rebounds": float(stats[1]) if len(stats) > 1 else 0,
            "assists": float(stats[2]) if len(stats) > 2 else 0
        })

    return pd.DataFrame(games)

# ---------------- NHL ---------------- #

def search_nhl_player(name: str):
    url = f"https://suggest.svc.nhl.com/svc/suggest/v1/minactiveplayers/{name}/5"
    r = requests.get(url)
    data = r.json()

    results = []
    for item in data.get("suggestions", []):
        parts = item.split("|")
        results.append({
            "id": parts[0],
            "name": parts[2] + " " + parts[1]
        })
    return results


def fetch_nhl_player_logs(player_id: str, player_name: str, seasons: list):
    games = []

    for season in seasons:
        season = season.replace("-", "")
        url = f"https://api-web.nhle.com/v1/player/{player_id}/game-log/{season}/2"
        data = _get_json(url)

        for g in data.get("gameLog", []):
            games.append({
                "player_name": player_name,
                "goals": g.get("goals", 0),
                "assists": g.get("assists", 0),
                "shots": g.get("shots", 0)
            })

    return pd.DataFrame(games)

# ---------------- MLB ---------------- #

def search_mlb_player(name: str):
    url = f"https://statsapi.mlb.com/api/v1/people/search?names={name}"
    data = _get_json(url)

    results = []
    for p in data.get("people", []):
        results.append({
            "id": p.get("id"),
            "name": p.get("fullName")
        })
    return results


def fetch_mlb_player_logs(player_id: str, player_name: str, seasons: list):
    games = []

    for season in seasons:
        url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=gameLog&season={season}"
        data = _get_json(url)

        stats = data.get("stats", [])
        if not stats:
            continue

        splits = stats[0].get("splits", [])
        for g in splits:
            stat = g.get("stat", {})

            games.append({
                "player_name": player_name,
                "hits": stat.get("hits", 0),
                "home_runs": stat.get("homeRuns", 0),
                "strikeouts": stat.get("strikeOuts", 0)
            })

    return pd.DataFrame(games)
