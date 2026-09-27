import os
import sys
import time
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm

from src.features import extract_pair_features, FEATURE_NAMES, extract_postal_codes
from src.normalize import (
    normalize_text,
    get_name_tokens_sorted,
    get_unspaced_name,
    extract_house_numbers
)

DATA_DIR = "/home/mahidhar14/amazonml2026/student_resource/dataset"

def load_target_records(needed_ids: Set[str], data_split: str = "train") -> Dict[str, dict]:
    """Efficiently load only the required target records from Source 2 and Source 3."""
    split_dir = os.path.join(DATA_DIR, data_split)
    target_dict = {}
    
    needed_s2 = {x for x in needed_ids if x.startswith("S2-")}
    needed_s3 = {x for x in needed_ids if x.startswith("S3-")}
    
    # Load Source 2 records
    if needed_s2:
        s2_path = os.path.join(split_dir, f"{data_split}_source2.tsv")
        print(f"Filtering {data_split}_source2.tsv for {len(needed_s2):,} needed IDs...")
        t0 = time.time()
        s2_df = pl.read_csv(
            s2_path, separator="\t", 
            columns=["entity_id", "business_name", "business_address", "country"]
        ).filter(pl.col("entity_id").is_in(needed_s2))
        print(f"Retrieved {s2_df.shape[0]:,} S2 records in {time.time()-t0:.2f}s")
        for row in s2_df.iter_rows(named=True):
            target_dict[row["entity_id"]] = row
            
    # Load Source 3 records
    if needed_s3:
        s3_path = os.path.join(split_dir, f"{data_split}_source3.tsv")
        print(f"Filtering {data_split}_source3.tsv for {len(needed_s3):,} needed IDs...")
        t0 = time.time()
        s3_df = pl.read_csv(
            s3_path, separator="\t", 
            columns=["entity_id", "business_name", "business_address", "country"]
        ).filter(pl.col("entity_id").is_in(needed_s3))
        print(f"Retrieved {s3_df.shape[0]:,} S3 records in {time.time()-t0:.2f}s")
        for row in s3_df.iter_rows(named=True):
            target_dict[row["entity_id"]] = row
            
    print(f"Total target records loaded: {len(target_dict):,}")
    return target_dict

def _batch_feature_extractor(batch_pairs, s1_dict, target_dict):
    """Worker function to process a batch of (s1_id, target_id) pairs."""
    batch_features = []
    for s1_id, tid in batch_pairs:
        s1 = s1_dict.get(s1_id)
        t = target_dict.get(tid)
        if s1 is None or t is None:
            batch_features.append([0.0] * len(FEATURE_NAMES))
            continue
            
        feat = extract_pair_features(
            s1_name_raw=s1["business_name"],
            s1_addr_raw=s1.get("business_address"),
            t_name_raw=t["business_name"],
            t_addr_raw=t.get("business_address"),
            country=s1["country"],
            target_id=tid,
            s1_norm_name=s1.get("norm_name"),
            s1_sorted_name=s1.get("sorted_name"),
            s1_unspaced_name=s1.get("unspaced_name"),
            s1_norm_addr=s1.get("norm_addr"),
            s1_house_nums=s1.get("house_nums"),
            s1_addr_tokens=s1.get("addr_tokens"),
            s1_postal_codes=s1.get("postal_codes"),
            target_country=t.get("country"),
        )
        batch_features.append(feat)
    return batch_features

def build_feature_matrix_parallel(
    pairs: List[Tuple[str, str]],
    s1_dict: Dict[str, dict],
    target_dict: Dict[str, dict],
    n_jobs: int = 12,
    batch_size: int = 5000
) -> np.ndarray:
    """Precompute S1 normalized fields and extract pair features in parallel."""
    print("Precomputing normalized representations for Source 1 records...")
    t0 = time.time()
    for sid, rec in s1_dict.items():
        if "norm_name" not in rec:
            n = rec["business_name"]
            a = rec.get("business_address")
            norm_a = normalize_text(a) if a else ""
            rec["norm_name"] = normalize_text(n)
            rec["sorted_name"] = get_name_tokens_sorted(n)
            rec["unspaced_name"] = get_unspaced_name(n)
            rec["norm_addr"] = norm_a
            rec["house_nums"] = extract_house_numbers(a)
            rec["addr_tokens"] = set(norm_a.split())
            rec["postal_codes"] = extract_postal_codes(a)
    print(f"Precomputed {len(s1_dict):,} S1 records in {time.time()-t0:.2f}s")
    
    print(f"Extracting features for {len(pairs):,} pairs across {n_jobs} cores...")
    t1 = time.time()
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_batch_feature_extractor)(b, s1_dict, target_dict) for b in batches
    )
    
    all_features = [f for batch_res in results for f in batch_res]
    X = np.array(all_features, dtype=np.float32)
    print(f"Feature matrix built: shape={X.shape} in {time.time()-t1:.2f}s")
    return X
