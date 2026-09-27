"""Validation-based model selection using the official per-S1 macro F0.5."""
import json
import os
import sys
import time
import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

sys.path.insert(0, "/home/mahidhar14/amazonml2026/code/business_entity_resolution")
from src.features import FEATURE_NAMES
from src.metrics import compute_entity_er_metrics
from src.model import ProbabilityEnsemble

EXP_DIR = "/home/mahidhar14/amazonml2026/experiments"


def _load_validation():
    X_train = np.nan_to_num(np.load(os.path.join(EXP_DIR, "X_train.npy")), nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.load(os.path.join(EXP_DIR, "y_train.npy")).astype(np.uint8)
    X_val = np.nan_to_num(np.load(os.path.join(EXP_DIR, "X_val.npy")), nan=0.0, posinf=0.0, neginf=0.0)
    pairs = pl.read_parquet(os.path.join(EXP_DIR, "val_pairs.parquet"))
    gt_df = pl.read_parquet(os.path.join(EXP_DIR, "val_gt.parquet"))
    gt = {}
    for row in gt_df.iter_rows(named=True):
        mids = row.get("matched_entity_ids")
        gt[row["source1_entity_id"]] = (set(x.strip() for x in str(mids).split(",") if x.strip())
            if mids and str(mids).strip() and str(mids).strip() != "None" else set())
    if len(X_train) != len(y_train) or len(X_val) != pairs.height:
        raise ValueError("feature/label/pair row counts do not agree")
    if X_train.ndim != 2 or X_val.ndim != 2 or X_train.shape[1] != len(FEATURE_NAMES) or X_val.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Saved feature matrices use a stale schema; rebuild train and validation features")
    return X_train, y_train, X_val, pairs, gt


def _threshold_search(probs, pairs, gt):
    by_s1 = {}
    for sid, tid, prob in zip(pairs["source1_entity_id"].to_list(), pairs["target_entity_id"].to_list(), probs):
        by_s1.setdefault(sid, []).append((tid, float(prob)))
    # Compact deterministic search. Include the all-empty operating point.
    thresholds = list(np.arange(0.05, 1.0, 0.01)) + [1.000001]
    best = (-1.0, 1.000001, {})
    for threshold in thresholds:
        pred = {sid: {tid for tid, p in by_s1.get(sid, ()) if p >= threshold} for sid in gt}
        metrics = compute_entity_er_metrics(gt, pred)
        if metrics["macro_f05"] > best[0]:
            best = (metrics["macro_f05"], float(threshold), metrics)
    return best[1], best[2]


def train_and_compare():
    X_train, y_train, X_val, val_pairs, gt = _load_validation()
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=160, max_depth=14, min_samples_leaf=4, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "lightgbm": LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    fitted, results = {}, {}
    for name, model in candidates.items():
        start = time.time()
        model.fit(X_train, y_train)
        p = model.predict_proba(X_val)[:, 1]
        tau, metrics = _threshold_search(p, val_pairs, gt)
        fitted[name] = model
        results[name] = {"model": model, "threshold": tau, "metrics": metrics}
        print(f"{name}: macro_F0.5={metrics['macro_f05']:.5f}, threshold={tau:.3f}, elapsed={time.time()-start:.1f}s")

    # Ensemble is considered only by validation. A small fixed weight grid avoids
    # an unbounded search; each blend gets its own official-metric threshold.
    rf, lgb = fitted["random_forest"], fitted["lightgbm"]
    best_blend = None
    for w_rf in (0.2, 0.4, 0.6, 0.8):
        blend = ProbabilityEnsemble([rf, lgb], [w_rf, 1.0 - w_rf])
        tau, metrics = _threshold_search(blend.predict_proba(X_val)[:, 1], val_pairs, gt)
        if best_blend is None or metrics["macro_f05"] > best_blend["metrics"]["macro_f05"]:
            best_blend = {"model": blend, "threshold": tau, "metrics": metrics,
                          "name": "rf_lightgbm_ensemble", "weights": [w_rf, 1.0-w_rf]}
    results["rf_lightgbm_ensemble"] = best_blend
    best_name = max(results, key=lambda k: results[k]["metrics"]["macro_f05"])
    selected = results[best_name]
    config_path = os.path.join(EXP_DIR, "blocking_config.json")
    blocking_config = json.load(open(config_path, encoding="utf-8")) if os.path.exists(config_path) else {
        "candidate_limit": 100, "same_country_only": True
    }
    artifact = {
        "model": selected["model"], "model_name": best_name,
        "threshold": selected["threshold"], "metrics": selected["metrics"],
        "features": list(FEATURE_NAMES), "config": blocking_config,
        "ensemble_weights": selected.get("weights"),
    }
    out = os.path.join(EXP_DIR, "best_model.joblib")
    joblib.dump(artifact, out)
    print(f"Selected {best_name}: macro_F0.5={selected['metrics']['macro_f05']:.5f}, threshold={selected['threshold']:.3f}")
    print(f"Saved selected validation model to {out}")


if __name__ == "__main__":
    train_and_compare()
