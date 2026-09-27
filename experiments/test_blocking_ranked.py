import os
import sys
import time
from collections import defaultdict
import numpy as np
import polars as pl

sys.path.insert(0, "/home/mahidhar14/amazonml2026/code/business_entity_resolution")
from src.normalize import (
    normalize_text,
    extract_core_name,
    extract_address_keys,
)

DATA_DIR = "/home/mahidhar14/amazonml2026/student_resource/dataset/train"
EXP_DIR = "/home/mahidhar14/amazonml2026/experiments"

class RankedBlockingIndex:
    """Multi-stage inverted index with prioritized candidate ranking."""
    def __init__(self, max_bucket_size: int = 100):
        self.max_bucket_size = max_bucket_size
        self.idx_exact_name = defaultdict(list)
        self.idx_sorted_name = defaultdict(list)
        self.idx_unspaced_name = defaultdict(list)
        self.idx_prefix_name = defaultdict(list)
        self.idx_address = defaultdict(list)
        
    def add_target(self, tid: str, name: str, addr: str):
        norm_name = normalize_text(name)
        if norm_name:
            self.idx_exact_name[norm_name].append(tid)
            
        core = extract_core_name(name)
        if core:
            words = core.split()
            sorted_core = " ".join(sorted(set(words)))
            self.idx_sorted_name[sorted_core].append(tid)
            
            unspaced = core.replace(" ", "")
            if len(unspaced) >= 4:
                self.idx_unspaced_name[unspaced].append(tid)
                
            pfx = " ".join(words[:2]) if len(words) >= 2 else words[0]
            if len(pfx) >= 4:
                self.idx_prefix_name[pfx].append(tid)
                
        if addr:
            for akey in extract_address_keys(addr):
                self.idx_address[akey].append(tid)
                
    def query_ranked(self, name: str, addr: str, max_candidates: int = 150):
        scores = defaultdict(float)
        
        # 1. Exact normalized name (+100)
        norm_name = normalize_text(name)
        if norm_name and norm_name in self.idx_exact_name:
            for tid in self.idx_exact_name[norm_name]:
                scores[tid] += 100.0
                
        core = extract_core_name(name)
        if core:
            words = core.split()
            # 2. Sorted core name tokens (+80)
            sorted_core = " ".join(sorted(set(words)))
            if sorted_core in self.idx_sorted_name:
                for tid in self.idx_sorted_name[sorted_core]:
                    scores[tid] += 80.0
                    
            # 3. Unspaced core name (+70)
            unspaced = core.replace(" ", "")
            if len(unspaced) >= 4 and unspaced in self.idx_unspaced_name:
                for tid in self.idx_unspaced_name[unspaced]:
                    scores[tid] += 70.0
                    
            # 4. Core name prefix (+30, capped)
            pfx = " ".join(words[:2]) if len(words) >= 2 else words[0]
            if len(pfx) >= 4 and pfx in self.idx_prefix_name:
                bucket = self.idx_prefix_name[pfx]
                limit_bucket = bucket if len(bucket) <= self.max_bucket_size else bucket[:self.max_bucket_size]
                for tid in limit_bucket:
                    scores[tid] += 30.0
                    
        # 5. Address keys (+50, capped)
        if addr:
            for akey in extract_address_keys(addr):
                if akey in self.idx_address:
                    bucket = self.idx_address[akey]
                    limit_bucket = bucket if len(bucket) <= self.max_bucket_size else bucket[:self.max_bucket_size]
                    for tid in limit_bucket:
                        scores[tid] += 50.0
                        
        if not scores:
            return []
            
        # Sort candidates by score descending
        sorted_candidates = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_candidates[:max_candidates]

def evaluate_limits():
    print("Loading validation dataset...")
    val_s1 = pl.read_parquet(os.path.join(EXP_DIR, "val_s1.parquet"))
    val_gt = pl.read_parquet(os.path.join(EXP_DIR, "val_gt.parquet"))
    
    # Ground truth mapping
    gt_map = {}
    gt_s2_map = {}
    gt_s3_map = {}
    
    for row in val_gt.iter_rows(named=True):
        sid = row["source1_entity_id"]
        m = row["matched_entity_ids"]
        if m and str(m).strip() and str(m).strip() != "None":
            all_m = set(x.strip() for x in str(m).split(",") if x.strip())
            gt_map[sid] = all_m
            gt_s2_map[sid] = {x for x in all_m if x.startswith("S2-")}
            gt_s3_map[sid] = {x for x in all_m if x.startswith("S3-")}
        else:
            gt_map[sid] = set()
            gt_s2_map[sid] = set()
            gt_s3_map[sid] = set()
            
    limits = [35, 50, 75, 100, 150]
    
    # Track stats per limit
    # We will test on 10,000 validation entities for rapid exact benchmarking
    test_sub = val_s1.sample(n=10000, seed=42)
    print(f"Benchmarking on {test_sub.shape[0]:,} validation entities...")
    
    # Group by country
    ranked_results = {}
    
    for country in test_sub["country"].unique().to_list():
        c_sub = test_sub.filter(pl.col("country") == country)
        print(f"\nLoading targets for {country}...")
        t0 = time.time()
        s2 = pl.read_csv(
            os.path.join(DATA_DIR, "train_source2.tsv"), 
            separator="\t",
            columns=["entity_id", "business_name", "business_address", "country"]
        ).filter(pl.col("country") == country)
        
        s3 = pl.read_csv(
            os.path.join(DATA_DIR, "train_source3.tsv"), 
            separator="\t",
            columns=["entity_id", "business_name", "business_address", "country"]
        ).filter(pl.col("country") == country)
        
        targets = pl.concat([s2, s3])
        print(f"Loaded {targets.shape[0]:,} targets in {time.time()-t0:.2f}s")
        
        index = RankedBlockingIndex(max_bucket_size=100)
        t1 = time.time()
        t_ids = targets["entity_id"].to_list()
        t_names = targets["business_name"].to_list()
        t_addrs = targets["business_address"].to_list()
        for tid, n, a in zip(t_ids, t_names, t_addrs):
            index.add_target(tid, n, a)
        print(f"Indexed {len(t_ids):,} targets in {time.time()-t1:.2f}s")
        
        t2 = time.time()
        s1_ids = c_sub["entity_id"].to_list()
        s1_names = c_sub["business_name"].to_list()
        s1_addrs = c_sub["business_address"].to_list()
        for sid, n, a in zip(s1_ids, s1_names, s1_addrs):
            ranked_results[sid] = index.query_ranked(n, a, max_candidates=150)
        print(f"Queried {len(s1_ids):,} entities in {time.time()-t2:.2f}s")
        
        del s2, s3, targets, index, t_ids, t_names, t_addrs
        
    print("\n==================== EVALUATING CANDIDATE LIMITS ====================")
    for lim in limits:
        total_true = 0
        hits = 0
        s2_true = 0
        s2_hits = 0
        s3_true = 0
        s3_hits = 0
        
        country_stats = defaultdict(lambda: {"true": 0, "hits": 0})
        cand_counts = []
        
        for row in test_sub.iter_rows(named=True):
            sid = row["entity_id"]
            cntry = row["country"]
            true_all = gt_map[sid]
            true_s2 = gt_s2_map[sid]
            true_s3 = gt_s3_map[sid]
            
            cands = set(ranked_results[sid][:lim])
            cand_counts.append(len(cands))
            
            total_true += len(true_all)
            hits += len(true_all & cands)
            
            s2_true += len(true_s2)
            s2_hits += len(true_s2 & cands)
            
            s3_true += len(true_s3)
            s3_hits += len(true_s3 & cands)
            
            country_stats[cntry]["true"] += len(true_all)
            country_stats[cntry]["hits"] += len(true_all & cands)
            
        recall = 100.0 * hits / total_true if total_true > 0 else 0.0
        s2_recall = 100.0 * s2_hits / s2_true if s2_true > 0 else 0.0
        s3_recall = 100.0 * s3_hits / s3_true if s3_true > 0 else 0.0
        avg_c = np.mean(cand_counts)
        max_c = np.max(cand_counts)
        p50_c = np.median(cand_counts)
        p95_c = np.percentile(cand_counts, 95)
        
        print(f"\n--- Candidate Limit: {lim} ---")
        print(f"  Overall Candidate Recall: {recall:.2f}% ({hits:,} / {total_true:,})")
        print(f"  S1 -> S2 Recall:          {s2_recall:.2f}% ({s2_hits:,} / {s2_true:,})")
        print(f"  S1 -> S3 Recall:          {s3_recall:.2f}% ({s3_hits:,} / {s3_true:,})")
        for cntry, cdata in country_stats.items():
            crecall = 100.0 * cdata["hits"] / cdata["true"] if cdata["true"] > 0 else 0.0
            print(f"  Country {cntry} Recall:       {crecall:.2f}% ({cdata['hits']:,} / {cdata['true']:,})")
        print(f"  Candidate count: avg={avg_c:.1f}, median={p50_c:.0f}, p95={p95_c:.0f}, max={max_c}")

if __name__ == "__main__":
    evaluate_limits()
