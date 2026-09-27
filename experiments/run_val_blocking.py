"""Generate validation candidates with the production blocking implementation.

This scans the complete training S2/S3 tables when explicitly run.
"""
import argparse
import json
import os
import sys
import polars as pl

sys.path.insert(0, "/home/mahidhar14/amazonml2026/code/business_entity_resolution")
from src.blocking import generate_candidates, candidate_diagnostics

DATA_DIR = "/home/mahidhar14/amazonml2026/student_resource/dataset/train"
EXP_DIR = "/home/mahidhar14/amazonml2026/experiments"


def main(candidate_limit=100):
    s1 = pl.read_parquet(os.path.join(EXP_DIR, "val_s1.parquet"))
    gt_df = pl.read_parquet(os.path.join(EXP_DIR, "val_gt.parquet"))
    targets = pl.concat([
        pl.read_csv(os.path.join(DATA_DIR, f"train_source{src}.tsv"), separator="\t",
                    columns=["entity_id", "business_name", "business_address", "country"])
        for src in (2, 3)
    ])
    ground_truth = {}
    for row in gt_df.iter_rows(named=True):
        mids = row.get("matched_entity_ids")
        ground_truth[row["source1_entity_id"]] = (
            set(x.strip() for x in str(mids).split(",") if x.strip())
            if mids and str(mids).strip() and str(mids).strip() != "None" else set()
        )
    s1_country = dict(zip(s1["entity_id"].to_list(), s1["country"].to_list()))
    true_ids = set().union(*ground_truth.values()) if ground_truth else set()
    true_target_rows = targets.filter(pl.col("entity_id").is_in(list(true_ids)))
    target_country = dict(zip(true_target_rows["entity_id"].to_list(), true_target_rows["country"].to_list()))
    cross_country = sum(
        1 for sid, mids in ground_truth.items() for tid in mids
        if tid in target_country and s1_country.get(sid) != target_country[tid]
    )
    print(f"Cross-country validation ground-truth links: {cross_country}")
    same_country_only = cross_country == 0
    candidates = generate_candidates(
        s1.to_dicts(), targets.iter_rows(named=True), max_candidates=candidate_limit,
        same_country_only=same_country_only,
    )
    with open(os.path.join(EXP_DIR, "blocking_config.json"), "w", encoding="utf-8") as f:
        json.dump({"candidate_limit": candidate_limit, "same_country_only": same_country_only}, f, indent=2)
    print("Candidate diagnostics:", candidate_diagnostics(candidates, ground_truth))
    out = pl.DataFrame({"source1_entity_id": list(candidates),
                        "candidate_entity_ids": [",".join(ids) for ids in candidates.values()]})
    out.write_parquet(os.path.join(EXP_DIR, "val_candidate_pairs.parquet"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-limit", type=int, default=100)
    main(parser.parse_args().candidate_limit)
