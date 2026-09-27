"""Build sampled training pairs using the same blocking as validation/inference.

This script scans the complete training S2/S3 target tables when explicitly run.
It is intentionally not run by the code-maintenance workflow.
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import polars as pl

sys.path.insert(0, "/home/mahidhar14/amazonml2026/code/business_entity_resolution")
from src.blocking import generate_candidates, candidate_diagnostics
from src.dataset_builder import build_feature_matrix_parallel

DATA_DIR = "/home/mahidhar14/amazonml2026/student_resource/dataset/train"
EXP_DIR = "/home/mahidhar14/amazonml2026/experiments"


def main(n_s1=35000, candidate_limit=100, hard_negatives_per_s1=8, seed=42):
    rng = random.Random(seed)
    config_path = os.path.join(EXP_DIR, "blocking_config.json")
    blocking_config = json.load(open(config_path, encoding="utf-8")) if os.path.exists(config_path) else {
        "candidate_limit": candidate_limit, "same_country_only": True
    }
    candidate_limit = int(blocking_config["candidate_limit"])
    s1 = pl.read_parquet(os.path.join(EXP_DIR, "train_s1.parquet")).sample(n=n_s1, seed=seed)
    gt = pl.read_parquet(os.path.join(EXP_DIR, "train_gt.parquet"))
    targets = pl.concat([
        pl.read_csv(os.path.join(DATA_DIR, f"train_source{src}.tsv"), separator="\t",
                    columns=["entity_id", "business_name", "business_address", "country"])
        for src in (2, 3)
    ])

    gt_map = defaultdict(set)
    for row in gt.iter_rows(named=True):
        mids = row.get("matched_entity_ids")
        if mids and str(mids).strip() and str(mids).strip() != "None":
            gt_map[row["source1_entity_id"]].update(x.strip() for x in str(mids).split(",") if x.strip())

    s1_rows = s1.to_dicts()
    candidates = generate_candidates(s1_rows, targets.iter_rows(named=True),
                                    max_candidates=candidate_limit,
                                    same_country_only=blocking_config["same_country_only"])
    s1_lookup = {row["entity_id"]: row for row in s1_rows}
    sampled_gt = {sid: gt_map.get(sid, set()) for sid in s1_lookup}
    print("Candidate diagnostics:", candidate_diagnostics(candidates, sampled_gt))

    # Retain every positive even if blocking missed it so the classifier sees
    # positive variation; validation candidate recall separately exposes that miss.
    pairs, labels = [], []
    needed_targets = set()
    for sid, row in s1_lookup.items():
        positives = gt_map.get(sid, set())
        for tid in sorted(positives):
            pairs.append((sid, tid)); labels.append(1); needed_targets.add(tid)
        negatives = [tid for tid in candidates.get(sid, []) if tid not in positives]
        if len(negatives) > hard_negatives_per_s1:
            negatives = rng.sample(negatives, hard_negatives_per_s1)
        for tid in negatives:
            pairs.append((sid, tid)); labels.append(0); needed_targets.add(tid)

    targets_needed = targets.filter(pl.col("entity_id").is_in(list(needed_targets)))
    target_lookup = {r["entity_id"]: r for r in targets_needed.iter_rows(named=True)}
    print(f"Training pairs: {len(pairs):,}; positives: {sum(labels):,}; negatives: {len(labels)-sum(labels):,}")
    X = build_feature_matrix_parallel(pairs, s1_lookup, target_lookup, n_jobs=12, batch_size=5000)
    os.makedirs(EXP_DIR, exist_ok=True)
    np.save(os.path.join(EXP_DIR, "X_train.npy"), X)
    np.save(os.path.join(EXP_DIR, "y_train.npy"), np.asarray(labels, dtype=np.uint8))


if __name__ == "__main__":
    main()
