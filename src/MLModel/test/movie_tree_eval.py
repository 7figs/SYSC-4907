#!/usr/bin/env python3
"""
Decision Tree evaluation on movie features + one user's preferences.

Assumptions:
- Both CSV files are in the SAME DIRECTORY as this script.
- Preferences CSV format:
    movie_id, user_label (0 = dislike, 1 = like)
- Features CSV:
    Rows ordered by movie_id 1..500
    Contains numeric features (year, binary genres, binary actors)
    May also contain text columns (title, overview) → automatically ignored
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


# =========================
# CONSTANTS (EDIT THESE)
# =========================

PREFERENCES_CSV_NAME = "preferences.csv"
FEATURES_CSV_NAME = "top500_movies_features.csv"

N_TRAIN = 100                # number of movies used for training
RANDOM_SEED = None           # None = random split every run

MAX_DEPTH = None             # e.g. 6 or None
MIN_SAMPLES_LEAF = 1


# =========================
# IMPLEMENTATION
# =========================

def resolve_path(csv_name: str) -> Path:
    return Path(__file__).resolve().parent / csv_name


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "movie_id" not in df.columns:
        df.insert(0, "movie_id", np.arange(1, len(df) + 1, dtype=int))

    df["movie_id"] = df["movie_id"].astype(int)
    return df


def load_preferences(path: Path) -> tuple[pd.DataFrame, str]:
    prefs = pd.read_csv(path)
    prefs.columns = [c.strip() for c in prefs.columns]

    if prefs.shape[1] < 2:
        raise ValueError("Preferences CSV must have movie_id + one user column")

    if prefs.columns[0] != "movie_id":
        prefs = prefs.rename(columns={prefs.columns[0]: "movie_id"})

    label_col = prefs.columns[1]

    prefs["movie_id"] = prefs["movie_id"].astype(int)
    prefs[label_col] = prefs[label_col].astype(int)

    return prefs[["movie_id", label_col]], label_col


def main() -> None:
    if not (1 <= N_TRAIN <= 499):
        raise ValueError("N_TRAIN must be between 1 and 499")

    prefs_path = resolve_path(PREFERENCES_CSV_NAME)
    feats_path = resolve_path(FEATURES_CSV_NAME)

    if not prefs_path.exists():
        raise FileNotFoundError(prefs_path)
    if not feats_path.exists():
        raise FileNotFoundError(feats_path)

    # Load data
    features = load_features(feats_path)
    prefs, label_col = load_preferences(prefs_path)

    # Merge
    df = features.merge(prefs, on="movie_id", how="inner")

    # Split X / y
    y = df[label_col].to_numpy(dtype=int)
    feature_cols = [c for c in df.columns if c not in ("movie_id", label_col)]
    X = df[feature_cols]

    # 🔹 IGNORE NON-NUMERIC FEATURES 🔹
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    dropped_cols = sorted(set(X.columns) - set(numeric_cols))

    if dropped_cols:
        print(f"Ignoring non-numeric feature columns: {dropped_cols}")

    X = X[numeric_cols]

    # Random train/test split (new every run)
    rng = np.random.default_rng(RANDOM_SEED)
    all_idx = np.arange(len(df))
    train_idx = rng.choice(all_idx, size=N_TRAIN, replace=False)
    test_idx = np.setdiff1d(all_idx, train_idx)

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    # Train decision tree
    clf = DecisionTreeClassifier(
        random_state=None,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
    )
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Stats
    correct = int((y_pred == y_test).sum())
    total = len(y_test)
    accuracy = correct / total

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Output
    print("\n=== Decision Tree Evaluation ===")
    print(f"Training size:     {N_TRAIN}")
    print(f"Test size:         {total}")
    print(f"Features used:     {X.shape[1]}")
    print(f"Tree max_depth:    {MAX_DEPTH}")
    print()
    print("=== Results ===")
    print(f"Correct:           {correct}/{total}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"False Positives:   {fp}")
    print(f"False Negatives:   {fn}")
    print()
    print("Confusion Matrix (rows=true [0,1], cols=pred [0,1]):")
    print(cm)


if __name__ == "__main__":
    main()
