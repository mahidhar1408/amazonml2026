import os
import sys
import numpy as np
import polars as pl
from collections import defaultdict
import joblib

sys.path.insert(0, "/home/mahidhar14/amazonml2026/code/business_entity_resolution")
from src.metrics import compute_entity_er_metrics

EXP_DIR = "/home/mahidhar14/amazonml2026/experiments"

def test_rules():
    print("Loading validation predictions and features...")
    val_pairs = pl.read_parquet(os.path.join(EXP_DIR, "val_pairs.parquet"))
    val_gt = pl.read_parquet(os.path.join(EXP_DIR, "val_gt.parquet"))
    X_val = np.load(os.path.join(EXP_DIR, "X_val.npy"))
    
    # Ground truth mapping
    gt_map = {}
    for row in val_gt.iter_rows(named=True):
        sid = row["source1_entity_id"]
        m = row["matched_entity_ids"]
        if m and str(m).strip() and str(m).strip() != "None":
            gt_map[sid] = set(x.strip() for x in str(m).split(",") if x.strip())
        else:
            gt_map[sid] = set()
            
    # Load models
    X_tr_lr = np.nan_to_num(X_val, nan=0.0)
    best_info = joblib.load(os.path.join(EXP_DIR, "best_model.joblib"))
    lgb = best_info["model"]
    probs = lgb.predict_proba(X_val)[:, 1]
    
    s1_ids = val_pairs["source1_entity_id"].to_list()
    t_ids = val_pairs["target_entity_id"].to_list()
    
    # Feature 26 is house_num_mismatch in FEATURE_NAMES (index 26)
    # Let's check feature indices
    from src.features import FEATURE_NAMES
    hn_mismatch_idx = FEATURE_NAMES.index("house_num_mismatch")
    hn_mismatches = X_val[:, hn_mismatch_idx]
    
    name_jw_idx = FEATURE_NAMES.index("name_jaro_winkler")
    name_jws = X_val[:, name_jw_idx]
    
    # Group by s1_id
    entity_data = defaultdict(list)
    for sid, tid, p, h_mis, jw in zip(s1_ids, t_ids, probs, hn_mismatches, name_jws):
        entity_data[sid].append({
            "tid": tid, 
            "prob": p, 
            "hn_mismatch": h_mis,
            "name_jw": jw
        })
        
    print(f"Total validation entities: {len(gt_map):,}")
    
    # Baseline: Simple threshold
    print("\n--- Testing Decision Rules on LightGBM Probabilities ---")
    
    best_overall_f05 = 0.0
    best_rule_params = None
    
    # Grid search:
    # tau_match: probability required to accept a candidate
    # tau_top: top candidate must have at least this prob, otherwise entity is singleton!
    # house_penalty: reject or penalize if house_num_mismatch == 1
    
    for tau_match in [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]:
        for tau_top in [tau_match, min(0.98, tau_match + 0.05), min(0.98, tau_match + 0.10)]:
            for filter_hn in [False, True]:
                preds = {}
                for sid in gt_map.keys():
                    cands = entity_data.get(sid, [])
                    if not cands:
                        preds[sid] = set()
                        continue
                        
                    # Filter candidates
                    valid_cands = []
                    for c in cands:
                        if filter_hn and c["hn_mismatch"] == 1.0 and c["prob"] < 0.98:
                            continue
                        if c["name_jw"] < 0.40 and c["prob"] < 0.98:
                            continue
                        valid_cands.append(c)
                        
                    if not valid_cands:
                        preds[sid] = set()
                        continue
                        
                    # Sort by prob
                    valid_cands.sort(key=lambda x: x["prob"], reverse=True)
                    top_p = valid_cands[0]["prob"]
                    
                    # Singleton check
                    if top_p < tau_top:
                        preds[sid] = set()
                    else:
                        # Accept any candidate meeting tau_match
                        matched = {c["tid"] for c in valid_cands if c["prob"] >= tau_match}
                        preds[sid] = matched
                        
                m = compute_entity_er_metrics(gt_map, preds)
                if m["macro_f05"] > best_overall_f05:
                    best_overall_f05 = m["macro_f05"]
                    best_rule_params = (tau_match, tau_top, filter_hn)
                    print(f"New Best: tau_match={tau_match:.2f}, tau_top={tau_top:.2f}, filter_hn={filter_hn}")
                    print(f"  Macro F_0.5: {m['macro_f05']:.4f}, Prec: {m['macro_precision']:.4f}, Rec: {m['macro_recall']:.4f}, SingletonAcc: {m['singleton_accuracy']:.4f}")

if __name__ == "__main__":
    test_rules()
