#!/usr/bin/env python3
"""
Decision Tree evaluation on movie features + one user's preferences.

Handles preferences.csv even if values are quoted, e.g.:
  "movie_id","user_1"
  "1","0"
  "2","1"

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
# HELPERS
# =========================

def resolve_path(csv_name: str) -> Path:
    return Path(__file__).resolve().parent / csv_name


def clean_int_series(s: pd.Series, colname: str) -> pd.Series:
    """
    Convert a column that might be like: 1, "1", ' "1" ', etc. into int.
    """
    # Convert to string, strip whitespace, strip surrounding quotes
    s2 = (
        s.astype(str)
         .str.strip()
         .str.strip('"')
         .str.strip("'")
         .str.strip()
    )

    # Convert to numeric safely
    num = pd.to_numeric(s2, errors="coerce")

    if num.isna().any():
        bad = s[num.isna()].head(10).tolist()
        raise ValueError(f"Column '{colname}' contains non-numeric values. Examples: {bad}")

    return num.astype(int)


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip().strip('"').strip("'").strip() for c in df.columns]

    if "movie_id" not in df.columns:
        df.insert(0, "movie_id", np.arange(1, len(df) + 1, dtype=int))

    df["movie_id"] = clean_int_series(df["movie_id"], "movie_id")
    return df


def load_preferences(path: Path) -> tuple[pd.DataFrame, str]:
    # Robust CSV read (handles quotes, extra spaces, BOM)
    prefs = pd.read_csv(
        path,
        encoding="utf-8-sig",
        engine="python",
        skipinitialspace=True,
        quotechar='"',
    )
    prefs.columns = [str(c).strip().strip('"').strip("'").strip() for c in prefs.columns]

    if prefs.shape[1] < 2:
        raise ValueError("Preferences CSV must have movie_id + one user column")

    # First column treated as movie_id, rename if needed
    if prefs.columns[0] != "movie_id":
        prefs = prefs.rename(columns={prefs.columns[0]: "movie_id"})

    label_col = prefs.columns[1]

    prefs["movie_id"] = clean_int_series(prefs["movie_id"], "movie_id")
    prefs[label_col] = clean_int_series(prefs[label_col], label_col)

    # Validate labels are only 0/1
    invalid = ~prefs[label_col].isin([0, 1])
    if invalid.any():
        bad_rows = prefs.loc[invalid, ["movie_id", label_col]].head(10)
        raise ValueError(
            f"Labels must be 0/1 only. Bad rows (first 10):\n{bad_rows.to_string(index=False)}"
        )

    return prefs[["movie_id", label_col]], label_col


# =========================
# MAIN
# =========================

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
    if df.shape[0] < 2:
        raise ValueError("After merge, too few rows. Check that movie_id values match between files.")

    # Split X / y
    y = df[label_col].to_numpy(dtype=int)
    feature_cols = [c for c in df.columns if c not in ("movie_id", label_col)]
    X = df[feature_cols]

    # Ignore non-numeric feature columns (e.g., title, overview)
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
    total = int(len(y_test))
    accuracy = correct / total if total else 0.0

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    # Output
    print("\n=== Decision Tree Evaluation ===")
    print(f"Training size:     {N_TRAIN}")
    print(f"Test size:         {total}")
    print(f"Features used:     {X.shape[1]}")
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
