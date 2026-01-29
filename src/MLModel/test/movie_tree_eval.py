#!/usr/bin/env python3
"""
Decision Tree evaluation on movie features + one or more users' preferences.

Handles preferences.csv even if values are quoted, e.g.:
  "movie_id","user_1","user_2"
  "1","0","1"
  "2","1","0"

Assumptions:
- Both CSV files are in the SAME DIRECTORY as this script.
- Preferences CSV format:
    movie_id, user_1, user_2, ... (each user column is 0/1)
- Features CSV:
    Rows ordered by movie_id 1..500
    Contains numeric features (year, binary genres, binary actors)
    May also contain text columns (title, overview) → automatically ignored

Split behavior:
- Each user is evaluated independently with its own stratified/proportional split,
  because label balance can differ across users.

Reporting:
- Per-user stats
- Aggregate stats: pooled by summing confusion matrices across users
- Average stats: mean of per-user metrics (accuracy/recall/precision/FPR)
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

N_TRAIN = 100                # number of movies used for training (per user)
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


def load_preferences(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns:
      prefs_df: movie_id + all user columns
      label_cols: list of user column names
    """
    prefs = pd.read_csv(
        path,
        encoding="utf-8-sig",
        engine="python",
        skipinitialspace=True,
        quotechar='"',
    )
    prefs.columns = [str(c).strip().strip('"').strip("'").strip() for c in prefs.columns]

    if prefs.shape[1] < 2:
        raise ValueError("Preferences CSV must have movie_id + at least one user column")

    if prefs.columns[0] != "movie_id":
        prefs = prefs.rename(columns={prefs.columns[0]: "movie_id"})

    label_cols = list(prefs.columns[1:])

    prefs["movie_id"] = clean_int_series(prefs["movie_id"], "movie_id")

    # Clean and validate each label column
    for c in label_cols:
        prefs[c] = clean_int_series(prefs[c], c)
        invalid = ~prefs[c].isin([0, 1])
        if invalid.any():
            bad_rows = prefs.loc[invalid, ["movie_id", c]].head(10)
            raise ValueError(
                f"Labels in column '{c}' must be 0/1 only. Bad rows (first 10):\n{bad_rows.to_string(index=False)}"
            )

    return prefs[["movie_id"] + label_cols], label_cols


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
    if n_train_ones > n_ones:
        n_train_ones = n_ones
        n_train_zeros = n_train - n_train_ones
    if n_train_zeros > n_zeros:
        n_train_zeros = n_zeros
        n_train_ones = n_train - n_train_zeros

    if n_train_ones < 0 or n_train_zeros < 0:
        raise ValueError("Could not compute a valid stratified split.")
    if n_train_ones + n_train_zeros != n_train:
        raise ValueError("Stratified allocation failed to sum to n_train.")

    train_ones = (
        rng.choice(idx_ones, size=n_train_ones, replace=False)
        if n_train_ones else np.array([], dtype=int)
    )
    train_zeros = (
        rng.choice(idx_zeros, size=n_train_zeros, replace=False)
        if n_train_zeros else np.array([], dtype=int)
    )

    train_idx = np.concatenate([train_ones, train_zeros])
    rng.shuffle(train_idx)
    return train_idx


def metrics_from_confusion(tn: int, fp: int, fn: int, tp: int) -> dict:
    total = tn + fp + fn + tp
    correct = tn + tp
    accuracy = correct / total if total else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "fpr": fpr,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


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
    prefs, label_cols = load_preferences(prefs_path)

    # Merge once (keep all labels)
    df = features.merge(prefs, on="movie_id", how="inner")
    if df.shape[0] < 2:
        raise ValueError("After merge, too few rows. Check that movie_id values match between files.")

    if not (1 <= N_TRAIN <= (len(df) - 1)):
        raise ValueError(f"N_TRAIN must be between 1 and {len(df)-1} for this merged dataset.")

    # Build X once
    feature_cols = [c for c in df.columns if c not in (["movie_id"] + label_cols)]
    X_all = df[feature_cols]

    # Ignore non-numeric feature columns (e.g., title, overview)
    numeric_cols = [c for c in X_all.columns if pd.api.types.is_numeric_dtype(X_all[c])]
    dropped_cols = sorted(set(X_all.columns) - set(numeric_cols))
    if dropped_cols:
        print(f"Ignoring non-numeric feature columns: {dropped_cols}")

    X_all = X_all[numeric_cols]

    # Output header
    print("\n=== Decision Tree Evaluation (Multi-User) ===")
    print(f"Users found:          {len(label_cols)} -> {label_cols}")
    print(f"Training size/user:   {N_TRAIN}")
    print(f"Rows (after merge):   {len(df)}")
    print(f"Features used:        {X_all.shape[1]}")
    print()

    rng = np.random.default_rng(RANDOM_SEED)
    all_idx = np.arange(len(df))

    # Accumulators for aggregate + averages
    agg_tn = agg_fp = agg_fn = agg_tp = 0
    per_user_metrics: list[dict] = []

    for user_col in label_cols:
        y = df[user_col].to_numpy(dtype=int)

        # Split indices per user (label distribution may differ)
        train_idx = stratified_train_indices(y=y, n_train=N_TRAIN, rng=rng)
        test_idx = np.setdiff1d(all_idx, train_idx)

        X_train, y_train = X_all.iloc[train_idx], y[train_idx]
        X_test, y_test = X_all.iloc[test_idx], y[test_idx]

        # Train
        clf = DecisionTreeClassifier(
            random_state=None,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
        )
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)

        # Confusion + metrics
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
        m = metrics_from_confusion(tn, fp, fn, tp)

        # Label balance info
        full_ones = int((y == 1).sum())
        full_zeros = int((y == 0).sum())
        train_ones = int((y_train == 1).sum())
        train_zeros = int((y_train == 0).sum())

        # Print per-user block
        print(f"--- User: {user_col} ---")
        print("Label Balance")
        print(f"  Full dataset:        ones={full_ones}, zeros={full_zeros}")
        print(f"  Training subset:     ones={train_ones}, zeros={train_zeros}")
        print("Results")
        print(f"  Test size:           {m['total']}")
        print(f"  Correct:             {m['correct']}/{m['total']}")
        print(f"  Accuracy:            {m['accuracy']:.4f}")
        print(f"  Recall:              {m['recall']:.4f}")
        print(f"  Precision:           {m['precision']:.4f}")
        print(f"  False Positive Rate: {m['fpr']:.4f}")
        print(f"  False Positives:     {m['fp']}")
        print(f"  False Negatives:     {m['fn']}")
        print(f"  True Positives:      {m['tp']}")
        print(f"  True Negatives:      {m['tn']}")
        print("  Confusion Matrix (rows=true [0,1], cols=pred [0,1]):")
        print(f"  {cm[0,0]}  {cm[0,1]}")
        print(f"  {cm[1,0]}  {cm[1,1]}")
        print()

        # Accumulate aggregate
        agg_tn += tn
        agg_fp += fp
        agg_fn += fn
        agg_tp += tp

        per_user_metrics.append({
            "user": user_col,
            "accuracy": m["accuracy"],
            "recall": m["recall"],
            "precision": m["precision"],
            "fpr": m["fpr"],
            "test_size": m["total"],
        })

    # Aggregate (pooled confusion)
    agg = metrics_from_confusion(agg_tn, agg_fp, agg_fn, agg_tp)

    # Average (mean of per-user metrics)
    avg_accuracy = float(np.mean([d["accuracy"] for d in per_user_metrics])) if per_user_metrics else 0.0
    avg_recall = float(np.mean([d["recall"] for d in per_user_metrics])) if per_user_metrics else 0.0
    avg_precision = float(np.mean([d["precision"] for d in per_user_metrics])) if per_user_metrics else 0.0
    avg_fpr = float(np.mean([d["fpr"] for d in per_user_metrics])) if per_user_metrics else 0.0

    print("=== Aggregate (Pooled Across All Users) ===")
    print(f"Total test examples:  {agg['total']}")
    print(f"Correct:              {agg['correct']}/{agg['total']}")
    print(f"Accuracy:             {agg['accuracy']:.4f}")
    print(f"Recall:               {agg['recall']:.4f}")
    print(f"Precision:            {agg['precision']:.4f}")
    print(f"False Positive Rate:  {agg['fpr']:.4f}")
    print(f"False Positives:      {agg['fp']}")
    print(f"False Negatives:      {agg['fn']}")
    print(f"True Positives:       {agg['tp']}")
    print(f"True Negatives:       {agg['tn']}")
    print("Confusion Matrix (rows=true [0,1], cols=pred [0,1]):")
    print(np.array([[agg_tn, agg_fp], [agg_fn, agg_tp]], dtype=int))
    print()

    print("=== Average (Mean of Per-User Metrics) ===")
    print(f"Users averaged:       {len(per_user_metrics)}")
    print(f"Avg Accuracy:         {avg_accuracy:.4f}")
    print(f"Avg Recall:           {avg_recall:.4f}")
    print(f"Avg Precision:        {avg_precision:.4f}")
    print(f"Avg False Pos Rate:   {avg_fpr:.4f}")


if __name__ == "__main__":
    main()
