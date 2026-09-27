import os
import sys
import numpy as np
import polars as pl
from collections import defaultdict
import joblib

sys.path.insert(0, "/home/mahidhar14/amazonml2026/code/business_entity_resolution")
from src.metrics import compute_entity_er_metrics
from src.features import FEATURE_NAMES

EXP_DIR = "/home/mahidhar14/amazonml2026/experiments"

def test_improvements():
    print("Loading validation predictions and features...")
    val_pairs = pl.read_parquet(os.path.join(EXP_DIR, "val_pairs.parquet"))
    val_gt = pl.read_parquet(os.path.join(EXP_DIR, "val_gt.parquet"))
    X_val = np.load(os.path.join(EXP_DIR, "X_val.npy"))
    
    gt_map = {}
    for row in val_gt.iter_rows(named=True):
        sid = row["source1_entity_id"]
        m = row["matched_entity_ids"]
        if m and str(m).strip() and str(m).strip() != "None":
            gt_map[sid] = set(x.strip() for x in str(m).split(",") if x.strip())
        else:
            gt_map[sid] = set()
            
    best_info = joblib.load(os.path.join(EXP_DIR, "best_model.joblib"))
    model = best_info["model"]
    probs = model.predict_proba(X_val)[:, 1]
    
    s1_ids = val_pairs["source1_entity_id"].to_list()
    t_ids = val_pairs["target_entity_id"].to_list()
    
    hn_mis_idx = FEATURE_NAMES.index("house_num_mismatch")
    hn_mismatches = X_val[:, hn_mis_idx]
    
    addr_mis_idx = FEATURE_NAMES.index("is_addr_missing")
    addr_missings = X_val[:, addr_mis_idx]
    
    jw_idx = FEATURE_NAMES.index("name_jaro_winkler")
    jws = X_val[:, jw_idx]
    
    entity_cands = defaultdict(list)
    for sid, tid, p, h_mis, a_mis, jw in zip(s1_ids, t_ids, probs, hn_mismatches, addr_missings, jws):
        entity_cands[sid].append({
            "tid": tid, 
            "prob": p, 
            "hn_mismatch": h_mis,
            "addr_missing": a_mis,
            "jw": jw
        })
        
    print(f"Total entities: {len(gt_map):,}")
    
    # Baseline with no house number filtering (tau=0.97)
    preds_base = defaultdict(set)
    for sid, cands in entity_cands.items():
        for c in cands:
            if c["prob"] >= 0.97:
                preds_base[sid].add(c["tid"])
    m_base = compute_entity_er_metrics(gt_map, preds_base)
    print("\n--- BASELINE (tau=0.97, no rules) ---")
    print(f"  Macro F_0.5: {m_base['macro_f05']:.4f}")
    print(f"  Precision:   {m_base['macro_precision']:.4f}")
    print(f"  Recall:      {m_base['macro_recall']:.4f}")
    print(f"  Singleton:   {m_base['singleton_accuracy']:.4f}")
    
    # Test Rule 1: Reject house_num_mismatch
    print("\n--- EXPERIMENT 1: House Number Mismatch Filter ---")
    preds_r1 = defaultdict(set)
    for sid, cands in entity_cands.items():
        for c in cands:
            if c["hn_mismatch"] == 1.0:
                continue  # REJECT mismatching house numbers!
            if c["prob"] >= 0.97:
                preds_r1[sid].add(c["tid"])
    m_r1 = compute_entity_er_metrics(gt_map, preds_r1)
    print(f"  Macro F_0.5: {m_r1['macro_f05']:.4f} (diff: {m_r1['macro_f05'] - m_base['macro_f05']:+.4f})")
    print(f"  Precision:   {m_r1['macro_precision']:.4f}")
    print(f"  Recall:      {m_r1['macro_recall']:.4f}")
    print(f"  Singleton:   {m_r1['singleton_accuracy']:.4f}")
    
    # Test Rule 2: House Number Mismatch Filter + Missing Address Guard
    print("\n--- EXPERIMENT 2: House Number Filter + Missing Address Guard ---")
    preds_r2 = defaultdict(set)
    for sid, cands in entity_cands.items():
        for c in cands:
            if c["hn_mismatch"] == 1.0:
                continue
            if c["addr_missing"] == 1.0 and c["prob"] < 0.99:
                continue
            if c["prob"] >= 0.97:
                preds_r2[sid].add(c["tid"])
    m_r2 = compute_entity_er_metrics(gt_map, preds_r2)
    print(f"  Macro F_0.5: {m_r2['macro_f05']:.4f} (diff: {m_r2['macro_f05'] - m_base['macro_f05']:+.4f})")
    print(f"  Precision:   {m_r2['macro_precision']:.4f}")
    print(f"  Recall:      {m_r2['macro_recall']:.4f}")
    print(f"  Singleton:   {m_r2['singleton_accuracy']:.4f}")
    
    # Test Rule 3: Grid search tau with Rules 1 + 2
    print("\n--- EXPERIMENT 3: Threshold Tuning with Rules 1 & 2 ---")
    best_tau = 0.97
    best_f05 = m_r2["macro_f05"]
    best_m = m_r2
    
    for tau in np.arange(0.90, 0.995, 0.01):
        preds = defaultdict(set)
        for sid, cands in entity_cands.items():
            for c in cands:
                if c["hn_mismatch"] == 1.0:
                    continue
                if c["addr_missing"] == 1.0 and c["prob"] < 0.99:
                    continue
                if c["prob"] >= tau:
                    preds[sid].add(c["tid"])
        m = compute_entity_er_metrics(gt_map, preds)
        if m["macro_f05"] > best_f05:
            best_f05 = m["macro_f05"]
            best_tau = tau
            best_m = m
            
    print(f"Best Tuned Threshold: {best_tau:.3f}")
    print(f"  Macro F_0.5: {best_m['macro_f05']:.4f} (total gain: {best_m['macro_f05'] - m_base['macro_f05']:+.4f})")
    print(f"  Precision:   {best_m['macro_precision']:.4f}")
    print(f"  Recall:      {best_m['macro_recall']:.4f}")
    print(f"  Singleton:   {best_m['singleton_accuracy']:.4f}")

if __name__ == "__main__":
    test_improvements()
