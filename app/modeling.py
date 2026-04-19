from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error

STAT_MAP = {
    "nba": ["points", "rebounds", "assists", "threes", "pra"],
    "nhl": ["goals", "assists", "points", "shots"],
    "mlb": ["hits", "home_runs", "rbi", "strikeouts"],
}

COLUMN_MAP = {
    "points": "stat_points",
    "rebounds": "stat_rebounds",
    "assists": "stat_assists",
    "threes": "stat_threes",
    "pra": "stat_pra",
    "goals": "stat_goals",
    "shots": "stat_shots",
    "hits": "stat_hits",
    "home_runs": "stat_home_runs",
    "rbi": "stat_rbi",
    "strikeouts": "stat_strikeouts",
}


def _derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "stat_points" not in df.columns:
        df["stat_points"] = np.nan
    if "stat_goals" in df.columns and "stat_assists" in df.columns:
        mask = df["sport"].eq("nhl") & df["stat_points"].isna()
        df.loc[mask, "stat_points"] = df.loc[mask, "stat_goals"].fillna(0) + df.loc[mask, "stat_assists"].fillna(0)
    if "stat_pra" not in df.columns:
        df["stat_pra"] = np.nan
    if {"stat_points", "stat_rebounds", "stat_assists"}.issubset(df.columns):
        mask = df["sport"].eq("nba") & df["stat_pra"].isna()
        df.loc[mask, "stat_pra"] = (
            df.loc[mask, "stat_points"].fillna(0)
            + df.loc[mask, "stat_rebounds"].fillna(0)
            + df.loc[mask, "stat_assists"].fillna(0)
        )
    return df


def prepare_features(df: pd.DataFrame, target_stat: str):
    df = _derive_columns(df)
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"]).sort_values("game_date").reset_index(drop=True)

    target_col = COLUMN_MAP[target_stat]
    if target_col not in df.columns:
        raise ValueError(f"Target column not available for stat '{target_stat}'.")

    df["target"] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=["target"]).copy()
    if len(df) < 15:
        raise ValueError("Not enough games. Sync more seasons first.")

    # Time-series features
    df["lag1"] = df["target"].shift(1)
    df["lag2"] = df["target"].shift(2)
    df["lag3"] = df["target"].shift(3)
    for window in (3, 5, 10):
        df[f"avg_{window}"] = df["target"].rolling(window).mean().shift(1)
        df[f"median_{window}"] = df["target"].rolling(window).median().shift(1)
    df["std_5"] = df["target"].rolling(5).std().shift(1)
    df["ceil_10"] = df["target"].rolling(10).max().shift(1)
    df["floor_10"] = df["target"].rolling(10).min().shift(1)
    df["trend_5_vs_10"] = df["avg_5"] - df["avg_10"]
    df["days_rest"] = df["game_date"].diff().dt.days.clip(lower=0, upper=14)
    df["is_home"] = (df["home_away"].fillna("").str.upper() == "HOME").astype(int)

    # Opponent history features from the player's own past games vs that opponent
    opp_group = df.groupby("opponent", dropna=False)["target"]
    df["opp_avg_prior"] = opp_group.transform(lambda s: s.shift(1).expanding().mean())
    df["opp_games_prior"] = opp_group.cumcount()

    feature_cols = [
        "lag1", "lag2", "lag3", "avg_3", "avg_5", "avg_10", "median_3", "median_5", "median_10",
        "std_5", "ceil_10", "floor_10", "trend_5_vs_10", "days_rest", "is_home", "opp_avg_prior", "opp_games_prior"
    ]

    X = (
        df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .bfill()
        .ffill()
        .fillna(0)
    )
    y = df["target"].astype(float)
    return df, X, y, feature_cols


def train_and_predict(df: pd.DataFrame, target_stat: str, line: float):
    df, X, y, feature_cols = prepare_features(df, target_stat)
    split = max(int(len(df) * 0.8), len(df) - 8)
    split = min(split, len(df) - 2)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    reg = RandomForestRegressor(
        n_estimators=350,
        random_state=42,
        min_samples_leaf=2,
        max_depth=8,
    )
    reg.fit(X_train, y_train)

    preds = reg.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds)) if len(y_test) else None

    latest_features = X.iloc[[-1]]
    next_prediction = float(reg.predict(latest_features)[0])

    y_over = (y > float(line)).astype(int)
    clf_split = min(split, len(y_over) - 1)
    if clf_split < 10:
        raise ValueError("Not enough games to estimate probabilities yet.")
    clf = LogisticRegression(max_iter=500)
    clf.fit(X.iloc[:clf_split], y_over.iloc[:clf_split])
    prob_over = float(clf.predict_proba(latest_features)[0][1])

    test_over_preds = clf.predict(X.iloc[clf_split:])
    accuracy = float(accuracy_score(y_over.iloc[clf_split:], test_over_preds)) if len(y_over.iloc[clf_split:]) else None

    latest_10 = df.tail(10).copy()
    hit_rate_10 = float((latest_10["target"] > float(line)).mean()) if len(latest_10) else None
    season_avg = float(df["target"].tail(82 if df["sport"].iloc[-1] == "nba" else 162).mean())
    volatility = float(df["target"].tail(10).std(ddof=0) or 0)

    feature_importance = sorted(
        [
            {"feature": f, "importance": round(float(i), 4)}
            for f, i in zip(feature_cols, reg.feature_importances_)
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )[:8]

    chart_points = [
        {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
        for d, v in zip(df["game_date"].tail(20), df["target"].tail(20))
    ]

    return {
        "games_used": int(len(df)),
        "predicted_stat": round(next_prediction, 2),
        "line": float(line),
        "lean": "OVER" if next_prediction > float(line) else "UNDER",
        "edge": round(abs(next_prediction - float(line)), 2),
        "prob_over": round(prob_over, 3),
        "prob_under": round(1 - prob_over, 3),
        "mae": round(mae, 3) if mae is not None else None,
        "classification_accuracy": round(accuracy, 3) if accuracy is not None else None,
        "hit_rate_last_10": round(hit_rate_10, 3) if hit_rate_10 is not None else None,
        "recent_avg_5": round(float(df["target"].tail(5).mean()), 2),
        "recent_avg_10": round(float(df["target"].tail(10).mean()), 2),
        "season_avg": round(season_avg, 2),
        "volatility_last_10": round(volatility, 2),
        "recent_games": (
            latest_10[["game_date", "opponent", "team", "home_away", "target"]]
            .assign(game_date=lambda x: x["game_date"].dt.strftime("%Y-%m-%d"))
            .to_dict(orient="records")
        ),
        "chart_points": chart_points,
        "feature_importance": feature_importance,
    }
