import os
import sys
import time
import polars as pl
import numpy as np

sys.path.insert(0, "/home/mahidhar14/amazonml2026/code/business_entity_resolution")
from src.dataset_builder import load_target_records, build_feature_matrix_parallel

EXP_DIR = "/home/mahidhar14/amazonml2026/experiments"

def main():
    print("Loading validation candidate pairs and S1 entities...")
    val_cands_df = pl.read_parquet(os.path.join(EXP_DIR, "val_candidate_pairs.parquet"))
    val_s1_df = pl.read_parquet(os.path.join(EXP_DIR, "val_s1.parquet"))
    
    s1_dict = {r["entity_id"]: r for r in val_s1_df.to_dicts()}
    
    # Flatten candidate pairs
    pairs = []
    needed_target_ids = set()
    for row in val_cands_df.iter_rows(named=True):
        sid = row["source1_entity_id"]
        cands = row["candidate_entity_ids"]
        if cands and str(cands).strip():
            for tid in str(cands).split(","):
                tid = tid.strip()
                if tid:
                    pairs.append((sid, tid))
                    needed_target_ids.add(tid)
                    
    print(f"Total candidate pairs to compute: {len(pairs):,}")
    print(f"Unique target IDs: {len(needed_target_ids):,}")
    
    # Load targets
    target_dict = load_target_records(needed_target_ids, data_split="train")
    
    # Build feature matrix
    X_val = build_feature_matrix_parallel(pairs, s1_dict, target_dict, n_jobs=12, batch_size=5000)
    
    # Save feature matrix and pair metadata
    print("Saving X_val.npy and val_pairs.parquet...")
    np.save(os.path.join(EXP_DIR, "X_val.npy"), X_val)
    
    pairs_df = pl.DataFrame({
        "source1_entity_id": [p[0] for p in pairs],
        "target_entity_id": [p[1] for p in pairs],
    })
    pairs_df.write_parquet(os.path.join(EXP_DIR, "val_pairs.parquet"))
    print("Validation features successfully saved!")

if __name__ == "__main__":
    main()
