import os
import sys
import polars as pl
import numpy as np
from collections import defaultdict
import joblib

sys.path.insert(0, "/home/mahidhar14/amazonml2026/code/business_entity_resolution")
from src.metrics import compute_entity_er_metrics

EXP_DIR = "/home/mahidhar14/amazonml2026/experiments"
DATA_DIR = "/home/mahidhar14/amazonml2026/student_resource/dataset/train"

def analyze_errors():
    print("Loading data for error analysis...")
    val_s1 = pl.read_parquet(os.path.join(EXP_DIR, "val_s1.parquet"))
    val_gt = pl.read_parquet(os.path.join(EXP_DIR, "val_gt.parquet"))
    val_pairs = pl.read_parquet(os.path.join(EXP_DIR, "val_pairs.parquet"))
    X_val = np.load(os.path.join(EXP_DIR, "X_val.npy"))
    
    s1_dict = {r["entity_id"]: r for r in val_s1.to_dicts()}
    gt_map = {}
    for row in val_gt.iter_rows(named=True):
        sid = row["source1_entity_id"]
        m = row["matched_entity_ids"]
        if m and str(m).strip() and str(m).strip() != "None":
            gt_map[sid] = set(x.strip() for x in str(m).split(",") if x.strip())
        else:
            gt_map[sid] = set()
            
    # Load model
    best_info = joblib.load(os.path.join(EXP_DIR, "best_model.joblib"))
    model = best_info["model"]
    tau_s2 = best_info["tau_s2"]
    tau_s3 = best_info["tau_s3"]
    probs = model.predict_proba(X_val)[:, 1]
    
    s1_ids = val_pairs["source1_entity_id"].to_list()
    t_ids = val_pairs["target_entity_id"].to_list()
    
    # Generate predictions
    preds = defaultdict(set)
    for sid, tid, p in zip(s1_ids, t_ids, probs):
        tau = tau_s2 if tid.startswith("S2-") else tau_s3
        if p >= tau:
            preds[sid].add(tid)
            
    # Load sample targets for inspection
    needed_tids = set()
    fp_cases = []
    fn_cases = []
    singleton_fp_cases = []
    
    for sid, true_targets in gt_map.items():
        pred_targets = preds.get(sid, set())
        
        # Singleton false positives
        if len(true_targets) == 0 and len(pred_targets) > 0:
            singleton_fp_cases.append((sid, pred_targets))
            needed_tids.update(pred_targets)
            
        # Standard false positives
        fps = pred_targets - true_targets
        if fps:
            fp_cases.append((sid, fps, true_targets))
            needed_tids.update(fps)
            needed_tids.update(true_targets)
            
        # False negatives
        fns = true_targets - pred_targets
        if fns:
            fn_cases.append((sid, fns, pred_targets))
            needed_tids.update(fns)
            
    print(f"Total validation entities: {len(gt_map):,}")
    print(f"Total entities with False Positives: {len(fp_cases):,}")
    print(f"Total entities with False Negatives: {len(fn_cases):,}")
    print(f"Total Singleton False Positives: {len(singleton_fp_cases):,}")
    
    # Load needed target rows from S2 and S3
    from src.dataset_builder import load_target_records
    sample_needed = set()
    for sid, fps, _ in fp_cases[:20]: sample_needed.update(fps)
    for sid, fns, _ in fn_cases[:20]: sample_needed.update(fns)
    for sid, fps in singleton_fp_cases[:20]: sample_needed.update(fps)
    target_dict = load_target_records(sample_needed, data_split="train")
    
    print("\n==================== SAMPLE FALSE POSITIVES (Wrong Merges) ====================")
    for idx, (sid, fps, trues) in enumerate(fp_cases[:5]):
        s1 = s1_dict[sid]
        print(f"\n--- FP Case {idx+1} ---")
        print(f"  S1:  ID={sid}, Name='{s1['business_name']}', Addr='{s1['business_address']}', Country={s1['country']}")
        for fpid in list(fps)[:2]:
            t = target_dict.get(fpid, {})
            print(f"  FP:  ID={fpid}, Name='{t.get('business_name')}', Addr='{t.get('business_address')}'")
        for tid in list(trues)[:2]:
            t = target_dict.get(tid, {})
            print(f"  True ID={tid}, Name='{t.get('business_name')}', Addr='{t.get('business_address')}'")
            
    print("\n==================== SAMPLE SINGLETON FALSE POSITIVES ====================")
    for idx, (sid, fps) in enumerate(singleton_fp_cases[:5]):
        s1 = s1_dict[sid]
        print(f"\n--- Singleton FP Case {idx+1} ---")
        print(f"  S1:  ID={sid}, Name='{s1['business_name']}', Addr='{s1['business_address']}', Country={s1['country']}")
        for fpid in list(fps)[:2]:
            t = target_dict.get(fpid, {})
            print(f"  FP:  ID={fpid}, Name='{t.get('business_name')}', Addr='{t.get('business_address')}'")
            
    print("\n==================== SAMPLE FALSE NEGATIVES (Missed Matches) ====================")
    for idx, (sid, fns, preds_set) in enumerate(fn_cases[:5]):
        s1 = s1_dict[sid]
        print(f"\n--- FN Case {idx+1} ---")
        print(f"  S1:  ID={sid}, Name='{s1['business_name']}', Addr='{s1['business_address']}', Country={s1['country']}")
        for fnid in list(fns)[:2]:
            t = target_dict.get(fnid, {})
            print(f"  Missed Target: ID={fnid}, Name='{t.get('business_name')}', Addr='{t.get('business_address')}'")

if __name__ == "__main__":
    analyze_errors()
