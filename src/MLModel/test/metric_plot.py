#!/usr/bin/env python3
"""
Sweep N_TRAIN from N_MIN..N_MAX and collect:
- Aggregate pooled across users: accuracy/recall/precision/FPR/FNR + p@k
- Average across users: avg accuracy/recall/precision/FPR/FNR + avg p@k

Then generate scatterplots of ONLY the RATES (no totals/confusion counts plots).

Adds:
- False Negative Rate (FNR) = FN / (FN + TP)

Sampling:
- BALANCED_TRAIN_SAMPLING True => proportional stratified sampling
- False => uniform random sampling
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


# =========================
# CONSTANTS (EDIT THESE)
# =========================

PREFERENCES_CSV_NAME = "preferences.csv"
FEATURES_CSV_NAME = "top500_movies_features.csv"

# Toggle balanced (proportional/stratified) sampling for the N_TRAIN set
BALANCED_TRAIN_SAMPLING = True

N_MIN = 1
N_MAX = 100

M_RUNS = 100
RANDOM_SEED = None  # None = random each run; set int for reproducibility

MAX_DEPTH = None
MIN_SAMPLES_LEAF = 0.2

P_AT_K = 10
PLOTS_SUBFOLDER = "n_sweep_plots"


# =========================
# HELPERS
# =========================

def resolve_path(csv_name: str) -> Path:
    return Path(__file__).resolve().parent / csv_name


def clean_int_series(s: pd.Series, colname: str) -> pd.Series:
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


def stratified_train_indices(y: np.ndarray, n_train: int, rng: np.random.Generator) -> np.ndarray:
    if n_train < 1 or n_train >= len(y):
        raise ValueError("n_train must be between 1 and len(y)-1")

    idx_ones = np.where(y == 1)[0]
    idx_zeros = np.where(y == 0)[0]
    n_ones = int(idx_ones.size)
    n_zeros = int(idx_zeros.size)

    if n_ones == 0 or n_zeros == 0:
        return rng.choice(np.arange(len(y)), size=n_train, replace=False)

    p_one = n_ones / (n_ones + n_zeros)
    n_train_ones = int(round(n_train * p_one))
    n_train_zeros = n_train - n_train_ones

    if n_train_ones > n_ones:
        n_train_ones = n_ones
        n_train_zeros = n_train - n_train_ones
    if n_train_zeros > n_zeros:
        n_train_zeros = n_zeros
        n_train_ones = n_train - n_train_zeros

    train_ones = rng.choice(idx_ones, size=n_train_ones, replace=False) if n_train_ones else np.array([], dtype=int)
    train_zeros = rng.choice(idx_zeros, size=n_train_zeros, replace=False) if n_train_zeros else np.array([], dtype=int)

    train_idx = np.concatenate([train_ones, train_zeros])
    rng.shuffle(train_idx)
    return train_idx


def choose_train_indices(y: np.ndarray, n_train: int, rng: np.random.Generator, balanced: bool) -> np.ndarray:
    if balanced:
        return stratified_train_indices(y=y, n_train=n_train, rng=rng)
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
    fnr = fn / (fn + tp) if (fn + tp) else 0.0  # NEW

    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "fpr": fpr,
        "fnr": fnr,  # NEW
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def precision_at_k_user_vector(X_all_np: np.ndarray, y_all: np.ndarray, train_idx: np.ndarray, k: int) -> tuple[int, int, float]:
    if k < 1:
        raise ValueError("k must be >= 1")

    w = np.where(y_all[train_idx] == 1, 1.0, -1.0)
    user_vec = (X_all_np[train_idx] * w[:, None]).sum(axis=0)

    all_idx = np.arange(X_all_np.shape[0])
    cand_idx = np.setdiff1d(all_idx, train_idx, assume_unique=False)
    if cand_idx.size == 0:
        return 0, k, 0.0

    scores = X_all_np[cand_idx] @ user_vec
    k_eff = min(k, cand_idx.size)

    top_part = np.argpartition(scores, -k_eff)[-k_eff:]
    top_sorted_local = top_part[np.argsort(scores[top_part])[::-1]]
    top_idx = cand_idx[top_sorted_local]

    p = int((y_all[top_idx] == 1).sum())
    return p, k_eff, (p / k_eff) if k_eff else 0.0


def seed_for_n(base_seed: int | None, n_train: int) -> int | None:
    if base_seed is None:
        return None
    return int((base_seed * 1_000_003 + n_train * 10_007) % (2**32 - 1))


def evaluate_for_n(
    df: pd.DataFrame,
    label_cols: list[str],
    X_all: pd.DataFrame,
    X_all_np: np.ndarray,
    n_train: int,
    m_runs: int,
    base_seed: int | None,
) -> dict:
    rng = np.random.default_rng(seed_for_n(base_seed, n_train))
    all_idx = np.arange(len(df))

    # pooled across users (but already pooled across runs per user)
    agg_tn = agg_fp = agg_fn = agg_tp = 0
    agg_p_sum = 0
    agg_k_sum = 0

    per_user_metrics: list[dict] = []

    for user_col in label_cols:
        y = df[user_col].to_numpy(dtype=int)

        user_tn = user_fp = user_fn = user_tp = 0
        user_p_sum = 0
        user_k_sum = 0

        for _ in range(m_runs):
            train_idx = choose_train_indices(y=y, n_train=n_train, rng=rng, balanced=BALANCED_TRAIN_SAMPLING)
            test_idx = np.setdiff1d(all_idx, train_idx)

            X_train, y_train = X_all.iloc[train_idx], y[train_idx]
            X_test, y_test = X_all.iloc[test_idx], y[test_idx]

            clf = DecisionTreeClassifier(
                random_state=42,
                max_depth=MAX_DEPTH,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                class_weight={0.0: 1, 1.0: 3},
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

            user_tn += tn
            user_fp += fp
            user_fn += fn
            user_tp += tp

            p, k_eff, _ = precision_at_k_user_vector(X_all_np=X_all_np, y_all=y, train_idx=train_idx, k=P_AT_K)
            user_p_sum += p
            user_k_sum += k_eff

        m = metrics_from_confusion(user_tn, user_fp, user_fn, user_tp)
        user_p_at_k = (user_p_sum / user_k_sum) if user_k_sum else 0.0

        agg_tn += user_tn
        agg_fp += user_fp
        agg_fn += user_fn
        agg_tp += user_tp
        agg_p_sum += user_p_sum
        agg_k_sum += user_k_sum

        per_user_metrics.append({
            "accuracy": m["accuracy"],
            "recall": m["recall"],
            "precision": m["precision"],
            "fpr": m["fpr"],
            "fnr": m["fnr"],     # NEW
            "p_at_k": user_p_at_k,
        })

    agg = metrics_from_confusion(agg_tn, agg_fp, agg_fn, agg_tp)
    agg_p_at_k = (agg_p_sum / agg_k_sum) if agg_k_sum else 0.0

    avg_accuracy = float(np.mean([d["accuracy"] for d in per_user_metrics])) if per_user_metrics else 0.0
    avg_recall = float(np.mean([d["recall"] for d in per_user_metrics])) if per_user_metrics else 0.0
    avg_precision = float(np.mean([d["precision"] for d in per_user_metrics])) if per_user_metrics else 0.0
    avg_fpr = float(np.mean([d["fpr"] for d in per_user_metrics])) if per_user_metrics else 0.0
    avg_fnr = float(np.mean([d["fnr"] for d in per_user_metrics])) if per_user_metrics else 0.0  # NEW
    avg_p_at_k = float(np.mean([d["p_at_k"] for d in per_user_metrics])) if per_user_metrics else 0.0

    return {
        "n_train": n_train,

        # pooled aggregate rates
        "agg_accuracy": agg["accuracy"],
        "agg_recall": agg["recall"],
        "agg_precision": agg["precision"],
        "agg_fpr": agg["fpr"],
        "agg_fnr": agg["fnr"],      # NEW
        "agg_p_at_k": agg_p_at_k,

        # mean across users rates
        "avg_accuracy": avg_accuracy,
        "avg_recall": avg_recall,
        "avg_precision": avg_precision,
        "avg_fpr": avg_fpr,
        "avg_fnr": avg_fnr,         # NEW
        "avg_p_at_k": avg_p_at_k,
    }


def save_scatter(xs: list[int], ys: list[float], title: str, xlabel: str, ylabel: str, outpath: Path) -> None:
    plt.figure()
    plt.scatter(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main() -> None:
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

    feature_cols = [c for c in df.columns if c not in (["movie_id"] + label_cols)]
    X_all = df[feature_cols]
    numeric_cols = [c for c in X_all.columns if pd.api.types.is_numeric_dtype(X_all[c])]
    X_all = X_all[numeric_cols]
    X_all_np = X_all.to_numpy(dtype=float)

    n_min = max(N_MIN, 1)
    n_max = min(N_MAX, len(df) - 1)
    if n_min > n_max:
        raise ValueError(f"Invalid sweep bounds after clamping: {n_min}..{n_max} (rows={len(df)})")

    outdir = Path(__file__).resolve().parent / PLOTS_SUBFOLDER
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Balanced sampling: {BALANCED_TRAIN_SAMPLING}")
    print("Plotting rates only (no totals).")
    print()

    results: list[dict] = []
    for n_train in range(n_min, n_max + 1):
        r = evaluate_for_n(
            df=df,
            label_cols=label_cols,
            X_all=X_all,
            X_all_np=X_all_np,
            n_train=n_train,
            m_runs=M_RUNS,
            base_seed=RANDOM_SEED,
        )
        results.append(r)
        print(f"Done N_TRAIN={n_train} ({n_train - n_min + 1}/{n_max - n_min + 1})")

    xs = [r["n_train"] for r in results]

    # RATES ONLY (plus p@k, which is also a rate)
    float_metrics = [
        ("agg_accuracy", "Aggregate Accuracy (pooled across users)"),
        ("agg_recall", "Aggregate Recall/TPR (pooled across users)"),
        ("agg_precision", "Aggregate Precision (pooled across users)"),
        ("agg_fpr", "Aggregate FPR (pooled across users)"),
        ("agg_fnr", "Aggregate FNR (pooled across users)"),  # NEW
        ("agg_p_at_k", f"Aggregate p@{P_AT_K} (pooled across users)"),

        ("avg_accuracy", "Average Accuracy (mean across users)"),
        ("avg_recall", "Average Recall/TPR (mean across users)"),
        ("avg_precision", "Average Precision (mean across users)"),
        ("avg_fpr", "Average FPR (mean across users)"),
        ("avg_fnr", "Average FNR (mean across users)"),      # NEW
        ("avg_p_at_k", f"Average p@{P_AT_K} (mean across users)"),
    ]

    for key, nice in float_metrics:
        ys = [float(r[key]) for r in results]
        save_scatter(
            xs=xs,
            ys=ys,
            title=f"{nice} vs N_TRAIN",
            xlabel="N_TRAIN",
            ylabel=key,
            outpath=outdir / f"{key}_vs_n_train.png",
        )

    print(f"\nSaved plots to: {outdir}")


if __name__ == "__main__":
    main()
