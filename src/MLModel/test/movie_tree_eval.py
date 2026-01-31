#!/usr/bin/env python3
"""
Decision Tree evaluation on movie features + one or more users' preferences.

Now supports MULTI-RUN evaluation per user:
- Keep N_TRAIN fixed (e.g., 100)
- Run each user evaluation M_RUNS times with fresh random splits
- Aggregate results for that user by summing confusion matrices across runs
- Report per-user metrics from that aggregated confusion matrix
- Then report aggregate stats across users, and average stats across users.

Adds: p@k (precision@k) "user-vector recommender" evaluation per run:
- For each run, compute user vector = sum(training vectors), where dislikes are subtracted
- Score all non-training movies by dot(user_vector, movie_vector)
- Take top k (k=10), count how many are actually liked (p)
- Aggregate p and k across runs, report p@k per user, plus pooled + average across users

Assumptions:
- Both CSV files are in the SAME DIRECTORY as this script.
- Preferences CSV format:
    movie_id, user_1, user_2, ... (each user column is 0/1)
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

# NEW: toggle balanced (proportional/stratified) sampling for the N_TRAIN set
# True  -> keep label proportions similar to full user dataset
# False -> choose N_TRAIN uniformly at random (no balancing)
BALANCED_TRAIN_SAMPLING = True

N_TRAIN = 100                # number of movies used for training (per run, per user)
M_RUNS = 100                 # number of runs per user (aggregate over these runs)
RANDOM_SEED = None           # None = random each run; set int for reproducibility

MAX_DEPTH = None             # e.g. 6 or None
MIN_SAMPLES_LEAF = 0.2

# p@k settings
P_AT_K = 10                  # k is always 10 as requested


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
        # Degenerate: only one class exists; fall back to uniform sampling.
        return rng.choice(np.arange(len(y)), size=n_train, replace=False)

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


def choose_train_indices(
    y: np.ndarray,
    n_train: int,
    rng: np.random.Generator,
    balanced: bool,
) -> np.ndarray:
    """
    Wrapper to toggle between:
      - balanced=True: stratified proportional sampling
      - balanced=False: uniform random sampling
    """
    if balanced:
        return stratified_train_indices(y=y, n_train=n_train, rng=rng)
    # Uniform sampling across all items
    if n_train < 1 or n_train >= len(y):
        raise ValueError("n_train must be between 1 and len(y)-1")
    return rng.choice(np.arange(len(y)), size=n_train, replace=False)


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


def precision_at_k_user_vector(
    X_all_np: np.ndarray,
    y_all: np.ndarray,
    train_idx: np.ndarray,
    k: int,
) -> tuple[int, int, float]:
    """
    Build user vector from training examples:
      user_vec = sum( +x_i for likes ) + sum( -x_i for dislikes )
    Score every non-training movie by dot(user_vec, x_j), take top k.
    Return (p, k, p_at_k).
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    # weights: like (1) -> +1, dislike (0) -> -1
    w = np.where(y_all[train_idx] == 1, 1.0, -1.0)  # shape (N_TRAIN,)
    user_vec = (X_all_np[train_idx] * w[:, None]).sum(axis=0)  # shape (D,)

    all_idx = np.arange(X_all_np.shape[0])
    cand_idx = np.setdiff1d(all_idx, train_idx, assume_unique=False)

    if cand_idx.size == 0:
        return 0, k, 0.0

    scores = X_all_np[cand_idx] @ user_vec  # dot product for each candidate

    k_eff = min(k, cand_idx.size)
    # argpartition for efficiency, then sort those top candidates
    top_part = np.argpartition(scores, -k_eff)[-k_eff:]
    top_sorted_local = top_part[np.argsort(scores[top_part])[::-1]]
    top_idx = cand_idx[top_sorted_local]

    p = int((y_all[top_idx] == 1).sum())
    return p, k_eff, (p / k_eff) if k_eff else 0.0


# =========================
# MAIN
# =========================

def main() -> None:
    if not (1 <= N_TRAIN <= 499):
        raise ValueError("N_TRAIN must be between 1 and 499")
    if M_RUNS < 1:
        raise ValueError("M_RUNS must be >= 1")

    prefs_path = resolve_path(PREFERENCES_CSV_NAME)
    feats_path = resolve_path(FEATURES_CSV_NAME)

    if not prefs_path.exists():
        raise FileNotFoundError(prefs_path)
    if not feats_path.exists():
        raise FileNotFoundError(feats_path)

    features = load_features(feats_path)
    prefs, label_cols = load_preferences(prefs_path)

    df = features.merge(prefs, on="movie_id", how="inner")
    if df.shape[0] < 2:
        raise ValueError("After merge, too few rows. Check that movie_id values match between files.")

    if not (1 <= N_TRAIN <= (len(df) - 1)):
        raise ValueError(f"N_TRAIN must be between 1 and {len(df)-1} for this merged dataset.")

    feature_cols = [c for c in df.columns if c not in (["movie_id"] + label_cols)]
    X_all = df[feature_cols]

    numeric_cols = [c for c in X_all.columns if pd.api.types.is_numeric_dtype(X_all[c])]
    dropped_cols = sorted(set(X_all.columns) - set(numeric_cols))
    if dropped_cols:
        print(f"Ignoring non-numeric feature columns: {dropped_cols}")

    X_all = X_all[numeric_cols]

    # For fast dot products in p@k
    X_all_np = X_all.to_numpy(dtype=float)

    print("\n=== Decision Tree Evaluation (Multi-User, Multi-Run) ===")
    print(f"Users found:            {len(label_cols)} -> {label_cols}")
    print(f"Training size/run/user: {N_TRAIN}")
    print(f"Runs per user (M):      {M_RUNS}")
    print(f"Rows (after merge):     {len(df)}")
    print(f"Features used:          {X_all.shape[1]}")
    print(f"p@k enabled:            k={P_AT_K}")
    print(f"Balanced sampling:      {BALANCED_TRAIN_SAMPLING}")
    print()

    rng = np.random.default_rng(RANDOM_SEED)
    all_idx = np.arange(len(df))

    # Aggregate across users (pooled confusion, after each user has already been pooled across runs)
    agg_tn = agg_fp = agg_fn = agg_tp = 0

    # Aggregate p@k across users (pooled)
    agg_p_sum = 0
    agg_k_sum = 0

    per_user_metrics: list[dict] = []

    for user_col in label_cols:
        y = df[user_col].to_numpy(dtype=int)

        # Label balance info (same every run)
        full_ones = int((y == 1).sum())
        full_zeros = int((y == 0).sum())

        # Pool confusion across M runs for this user
        user_tn = user_fp = user_fn = user_tp = 0

        # Pool p@k across M runs for this user
        user_p_sum = 0
        user_k_sum = 0

        for _ in range(M_RUNS):
            train_idx = choose_train_indices(
                y=y,
                n_train=N_TRAIN,
                rng=rng,
                balanced=BALANCED_TRAIN_SAMPLING,
            )
            test_idx = np.setdiff1d(all_idx, train_idx)

            X_train, y_train = X_all.iloc[train_idx], y[train_idx]
            X_test, y_test = X_all.iloc[test_idx], y[test_idx]

            clf = DecisionTreeClassifier(
                random_state=42,
                max_depth=MAX_DEPTH,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                class_weight={0.0: 1, 1.0: 3}
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

            user_tn += tn
            user_fp += fp
            user_fn += fn
            user_tp += tp

            # ---- p@k computation (k=10) using the special "user vector" method ----
            p, k_eff, _p_at_k = precision_at_k_user_vector(
                X_all_np=X_all_np,
                y_all=y,
                train_idx=train_idx,
                k=P_AT_K,
            )
            user_p_sum += p
            user_k_sum += k_eff

        # Metrics from aggregated confusion for this user
        m = metrics_from_confusion(user_tn, user_fp, user_fn, user_tp)

        # p@k from aggregated p and k
        user_p_at_k = (user_p_sum / user_k_sum) if user_k_sum else 0.0
        user_avg_p_per_run = (user_p_sum / M_RUNS) if M_RUNS else 0.0

        print(f"--- User: {user_col} ---")
        print("Label Balance")
        print(f"  Full dataset:        ones={full_ones}, zeros={full_zeros}")
        print("Multi-Run Results (Aggregated over runs)")
        print(f"  Runs:                {M_RUNS}")
        print(f"  Total test examples: {m['total']}  (should equal {M_RUNS} * (N_total - N_TRAIN) = {M_RUNS} * ({len(df)} - {N_TRAIN}) = {M_RUNS*(len(df)-N_TRAIN)})")
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
        print(f"  {user_tn}  {user_fp}")
        print(f"  {user_fn}  {user_tp}")
        print("p@k (User-Vector Recommender, Aggregated over runs)")
        print(f"  k (fixed):           {P_AT_K}")
        print(f"  Total p (liked in top-k across runs): {user_p_sum}")
        print(f"  Total k (top-k counted across runs):  {user_k_sum}  (should equal {M_RUNS}*{min(P_AT_K, len(df)-N_TRAIN)})")
        print(f"  p@{P_AT_K}:            {user_p_at_k:.4f}")
        print(f"  Avg p per run:       {user_avg_p_per_run:.2f} / {P_AT_K}")
        print()

        # Add to overall aggregate (pooled across users)
        agg_tn += user_tn
        agg_fp += user_fp
        agg_fn += user_fn
        agg_tp += user_tp

        agg_p_sum += user_p_sum
        agg_k_sum += user_k_sum

        per_user_metrics.append({
            "user": user_col,
            "accuracy": m["accuracy"],
            "recall": m["recall"],
            "precision": m["precision"],
            "fpr": m["fpr"],
            "test_size": m["total"],
            "p_at_10": user_p_at_k,
            "p_sum": user_p_sum,
            "k_sum": user_k_sum,
        })

    # Overall aggregate (pooled across users)
    agg = metrics_from_confusion(agg_tn, agg_fp, agg_fn, agg_tp)

    # Overall pooled p@k across users
    agg_p_at_k = (agg_p_sum / agg_k_sum) if agg_k_sum else 0.0

    # Average across users (mean of per-user metrics)
    avg_accuracy = float(np.mean([d["accuracy"] for d in per_user_metrics])) if per_user_metrics else 0.0
    avg_recall = float(np.mean([d["recall"] for d in per_user_metrics])) if per_user_metrics else 0.0
    avg_precision = float(np.mean([d["precision"] for d in per_user_metrics])) if per_user_metrics else 0.0
    avg_fpr = float(np.mean([d["fpr"] for d in per_user_metrics])) if per_user_metrics else 0.0
    avg_p_at_k = float(np.mean([d["p_at_10"] for d in per_user_metrics])) if per_user_metrics else 0.0

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
    print("p@k (User-Vector Recommender, Pooled Across All Users)")
    print(f"  k (fixed):          {P_AT_K}")
    print(f"  Total p:            {agg_p_sum}")
    print(f"  Total k:            {agg_k_sum}")
    print(f"  p@{P_AT_K}:           {agg_p_at_k:.4f}")
    print()

    print("=== Average (Mean of Per-User Metrics) ===")
    print(f"Users averaged:       {len(per_user_metrics)}")
    print(f"Avg Accuracy:         {avg_accuracy:.4f}")
    print(f"Avg Recall:           {avg_recall:.4f}")
    print(f"Avg Precision:        {avg_precision:.4f}")
    print(f"Avg False Pos Rate:   {avg_fpr:.4f}")
    print(f"Avg p@{P_AT_K}:         {avg_p_at_k:.4f}")


if __name__ == "__main__":
    main()
