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

Change vs. prior version:
- Training set selection is now STRATIFIED / PROPORTIONAL:
  the number of 1s and 0s in the N_TRAIN sample matches the overall label ratio
  (as closely as possible, subject to available counts).
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
    s2 = (
        s.astype(str)
         .str.strip()
         .str.strip('"')
         .str.strip("'")
         .str.strip()
    )

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

    if prefs.columns[0] != "movie_id":
        prefs = prefs.rename(columns={prefs.columns[0]: "movie_id"})

    label_col = prefs.columns[1]

    prefs["movie_id"] = clean_int_series(prefs["movie_id"], "movie_id")
    prefs[label_col] = clean_int_series(prefs[label_col], label_col)

    invalid = ~prefs[label_col].isin([0, 1])
    if invalid.any():
        bad_rows = prefs.loc[invalid, ["movie_id", label_col]].head(10)
        raise ValueError(
            f"Labels must be 0/1 only. Bad rows (first 10):\n{bad_rows.to_string(index=False)}"
        )

    return prefs[["movie_id", label_col]], label_col


def stratified_train_indices(
    y: np.ndarray,
    n_train: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Pick n_train indices such that the number of 0/1 labels in the sample
    is proportional to their counts in the full y (as closely as possible).

    If exact proportional counts aren't possible (due to limited samples in a class),
    this function clamps to what's available and fills the remainder from the other class.
    """
    if n_train < 1 or n_train >= len(y):
        raise ValueError("n_train must be between 1 and len(y)-1")

    idx_ones = np.where(y == 1)[0]
    idx_zeros = np.where(y == 0)[0]

    n_ones = int(len(idx_ones))
    n_zeros = int(len(idx_zeros))
    n_total = n_ones + n_zeros

    if n_total != len(y):
        raise ValueError("Unexpected label indexing issue.")

    if n_ones == 0 or n_zeros == 0:
        # Degenerate case: only one class exists; fall back to uniform sampling.
        return rng.choice(np.arange(len(y)), size=n_train, replace=False)

    # Target proportional counts
    p_one = n_ones / n_total
    n_train_ones = int(round(n_train * p_one))
    n_train_zeros = n_train - n_train_ones

    # Clamp to availability and re-balance
    # Clamp ones first
    if n_train_ones > n_ones:
        n_train_ones = n_ones
        n_train_zeros = n_train - n_train_ones
    # Clamp zeros
    if n_train_zeros > n_zeros:
        n_train_zeros = n_zeros
        n_train_ones = n_train - n_train_zeros

    # Final safety check (should always pass now)
    if n_train_ones < 0 or n_train_zeros < 0:
        raise ValueError("Could not compute a valid stratified split.")
    if n_train_ones + n_train_zeros != n_train:
        raise ValueError("Stratified allocation failed to sum to n_train.")

    train_ones = rng.choice(idx_ones, size=n_train_ones, replace=False) if n_train_ones else np.array([], dtype=int)
    train_zeros = rng.choice(idx_zeros, size=n_train_zeros, replace=False) if n_train_zeros else np.array([], dtype=int)

    train_idx = np.concatenate([train_ones, train_zeros])
    rng.shuffle(train_idx)
    return train_idx


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

    if not (1 <= N_TRAIN <= (len(df) - 1)):
        raise ValueError(f"N_TRAIN must be between 1 and {len(df)-1} for this merged dataset.")

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

    # Proportional / stratified train/test split
    rng = np.random.default_rng(RANDOM_SEED)
    all_idx = np.arange(len(df))

    train_idx = stratified_train_indices(y=y, n_train=N_TRAIN, rng=rng)
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

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    # Output
    print("\n=== Decision Tree Evaluation ===")
    print(f"Training size:       {N_TRAIN}")
    print(f"Test size:           {total}")
    print(f"Features used:       {X.shape[1]}")
    print()

    # Optional: show class balance (useful sanity check)
    print("=== Label Balance ===")
    print(f"Full dataset:        ones={int((y==1).sum())}, zeros={int((y==0).sum())}")
    print(f"Training subset:     ones={int((y_train==1).sum())}, zeros={int((y_train==0).sum())}")
    print()

    print("=== Results ===")
    print(f"Correct:             {correct}/{total}")
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    print(f"False Positives:     {fp}")
    print(f"False Negatives:     {fn}")
    print(f"True Positives:      {tp}")
    print(f"True Negatives:      {tn}")
    print()
    print("Confusion Matrix (rows=true [0,1], cols=pred [0,1]):")
    print(cm)


if __name__ == "__main__":
    main()
