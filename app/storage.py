import sqlite3
import pandas as pd
from .config import DB_PATH


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_logs (
            sport       TEXT NOT NULL,
            player_id   TEXT NOT NULL,
            player_name TEXT NOT NULL,
            season      TEXT NOT NULL,
            game_date   TEXT NOT NULL,
            opponent    TEXT,
            team        TEXT,
            home_away   TEXT,
            stat_points    REAL,
            stat_rebounds  REAL,
            stat_assists   REAL,
            stat_threes    REAL,
            stat_pra       REAL,
            stat_goals     REAL,
            stat_shots     REAL,
            stat_hits      REAL,
            stat_home_runs REAL,
            stat_rbi       REAL,
            stat_strikeouts REAL,
            raw_json    TEXT,
            PRIMARY KEY (sport, player_id, season, game_date)
        )
        """
    )
    conn.commit()
    return conn


def upsert_logs(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    conn = get_conn()
    cols = [
        "sport", "player_id", "player_name", "season", "game_date",
        "opponent", "team", "home_away",
        "stat_points", "stat_rebounds", "stat_assists", "stat_threes", "stat_pra",
        "stat_goals", "stat_shots", "stat_hits", "stat_home_runs", "stat_rbi",
        "stat_strikeouts", "raw_json",
    ]
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    df.to_sql("player_logs_tmp", conn, if_exists="replace", index=False)
    conn.execute(
        "INSERT OR REPLACE INTO player_logs SELECT * FROM player_logs_tmp"
    )
    conn.execute("DROP TABLE player_logs_tmp")
    conn.commit()
    count = len(df)
    conn.close()
    return count


def get_player_logs(sport: str, player_id: str) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM player_logs WHERE sport = ? AND player_id = ? ORDER BY game_date",
        conn,
        params=[sport, str(player_id)],
    )
    conn.close()
    return df


def list_cached_players() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT sport, player_id, player_name,
               COUNT(*) AS games,
               MIN(game_date) AS first_game,
               MAX(game_date) AS last_game
        FROM player_logs
        GROUP BY sport, player_id, player_name
        ORDER BY last_game DESC
        """,
        conn,
    )
    conn.close()
    return df
