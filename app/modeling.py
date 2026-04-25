from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

STAT_MAP = {
    "nba": ["points", "rebounds", "assists", "threes", "pra"],
    "nhl": ["goals", "assists", "points", "shots"],
    "mlb": ["hits", "home_runs", "rbi", "strikeouts"],
}

COLUMN_MAP = {
    "points":     "stat_points",
    "rebounds":   "stat_rebounds",
    "assists":    "stat_assists",
    "threes":     "stat_threes",
    "pra":        "stat_pra",
    "goals":      "stat_goals",
    "shots":      "stat_shots",
    "hits":       "stat_hits",
    "home_runs":  "stat_home_runs",
    "rbi":        "stat_rbi",
    "strikeouts": "stat_strikeouts",
}


def _derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure float columns exist
    for col in ("stat_points", "stat_pra", "stat_goals", "stat_assists",
                "stat_rebounds", "stat_shots", "stat_threes",
                "stat_hits", "stat_home_runs", "stat_rbi", "stat_strikeouts"):
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # NHL: derive points = goals + assists where points is missing
    mask = df["sport"].eq("nhl") & df["stat_points"].isna()
    if mask.any():
        vals = df.loc[mask, "stat_goals"].fillna(0) + df.loc[mask, "stat_assists"].fillna(0)
        df["stat_points"] = df["stat_points"].astype("float64")
        df.loc[mask, "stat_points"] = vals.astype("float64")

    # NBA: derive PRA = points + rebounds + assists where pra is missing
    mask = df["sport"].eq("nba") & df["stat_pra"].isna()
    if mask.any():
        vals = (df.loc[mask, "stat_points"].fillna(0)
                + df.loc[mask, "stat_rebounds"].fillna(0)
                + df.loc[mask, "stat_assists"].fillna(0))
        df["stat_pra"] = df["stat_pra"].astype("float64")
        df.loc[mask, "stat_pra"] = vals.astype("float64")

    return df


def prepare_features(df: pd.DataFrame, target_stat: str):
    df = _derive_columns(df)
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"]).sort_values("game_date").reset_index(drop=True)

    target_col = COLUMN_MAP.get(target_stat)
    if not target_col:
        raise ValueError(f"Unknown stat '{target_stat}'.")
    if target_col not in df.columns:
        raise ValueError(
            f"Column '{target_col}' not present for stat '{target_stat}'. "
            "Sync more data first."
        )

    df["target"] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=["target"]).copy()

    if len(df) < 10:
        raise ValueError(
            f"Only {len(df)} valid game(s) found — need at least 10. "
            "Sync more seasons first."
        )

    # ── Time-series features ──────────────────────────────────────── #
    df["lag1"] = df["target"].shift(1)
    df["lag2"] = df["target"].shift(2)
    df["lag3"] = df["target"].shift(3)

    for window in (3, 5, 10):
        df[f"avg_{window}"]    = df["target"].rolling(window).mean().shift(1)
        df[f"median_{window}"] = df["target"].rolling(window).median().shift(1)

    df["std_5"]         = df["target"].rolling(5).std().shift(1)
    df["ceil_10"]       = df["target"].rolling(10).max().shift(1)
    df["floor_10"]      = df["target"].rolling(10).min().shift(1)
    df["trend_5_vs_10"] = df["avg_5"] - df["avg_10"]
    df["days_rest"]     = df["game_date"].diff().dt.days.clip(lower=0, upper=14)
    df["is_home"]       = (df["home_away"].fillna("").str.upper() == "HOME").astype(int)

    # Opponent history
    opp_group = df.groupby("opponent", dropna=False)["target"]
    df["opp_avg_prior"]   = opp_group.transform(lambda s: s.shift(1).expanding().mean())
    df["opp_games_prior"] = opp_group.cumcount()

    feature_cols = [
        "lag1", "lag2", "lag3",
        "avg_3", "avg_5", "avg_10",
        "median_3", "median_5", "median_10",
        "std_5", "ceil_10", "floor_10", "trend_5_vs_10",
        "days_rest", "is_home",
        "opp_avg_prior", "opp_games_prior",
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


def train_and_predict(df: pd.DataFrame, target_stat: str, line: float) -> dict:
    df, X, y, feature_cols = prepare_features(df, target_stat)

    n = len(df)
    split = max(int(n * 0.8), n - 8)
    split = min(split, n - 2)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ── Regression (Random Forest) ────────────────────────────────── #
    reg = RandomForestRegressor(
        n_estimators=350,
        random_state=42,
        min_samples_leaf=2,
        max_depth=8,
    )
    reg.fit(X_train, y_train)

    test_preds = reg.predict(X_test)
    mae = float(mean_absolute_error(y_test, test_preds)) if len(y_test) else None

    latest_features = X.iloc[[-1]]
    next_prediction = float(reg.predict(latest_features)[0])

    # ── Classification (Logistic Regression) ─────────────────────── #
    y_over = (y > float(line)).astype(int)
    clf_split = min(split, n - 1)

    prob_over: float
    accuracy: float | None

    if clf_split < 8:
        # Too few samples – fall back to simple rate
        prob_over = float(y_over.mean())
        accuracy = None
    else:
        train_labels = y_over.iloc[:clf_split]
        unique_classes = sorted(set(int(v) for v in train_labels))

        if len(unique_classes) < 2:
            # All games went the same way vs this line
            prob_over = 1.0 if unique_classes and unique_classes[0] == 1 else 0.0
            accuracy = 1.0
        else:
            try:
                scaler = StandardScaler()
                X_clf_train = scaler.fit_transform(X.iloc[:clf_split])
                X_clf_latest = scaler.transform(latest_features)
                clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
                clf.fit(X_clf_train, train_labels)
                prob_over = float(clf.predict_proba(X_clf_latest)[0][1])
                test_over = y_over.iloc[clf_split:]
                if len(test_over) > 0:
                    X_clf_test = scaler.transform(X.iloc[clf_split:])
                    accuracy = float(accuracy_score(test_over, clf.predict(X_clf_test)))
                else:
                    accuracy = None
            except Exception:
                prob_over = float(train_labels.mean())
                accuracy = None

    # ── Summary stats ─────────────────────────────────────────────── #
    latest_10 = df.tail(10).copy()
    hit_rate_10 = float((latest_10["target"] > float(line)).mean()) if len(latest_10) else None

    tail_n = 82 if df["sport"].iloc[-1] == "nba" else (162 if df["sport"].iloc[-1] == "mlb" else 82)
    season_avg = float(df["target"].tail(tail_n).mean())
    volatility = float(df["target"].tail(10).std(ddof=0) or 0)

    feature_importance = sorted(
        [
            {"feature": f, "importance": round(float(imp), 4)}
            for f, imp in zip(feature_cols, reg.feature_importances_)
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )[:8]

    chart_points = [
        {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
        for d, v in zip(df["game_date"].tail(20), df["target"].tail(20))
    ]

    return {
        "games_used": int(n),
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
