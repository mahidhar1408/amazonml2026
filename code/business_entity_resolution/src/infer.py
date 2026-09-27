import os
import sys
import time
import argparse
import gc
import multiprocessing as mp
from collections import defaultdict
import joblib
import numpy as np
import polars as pl
from tqdm import tqdm

sys.path.insert(0, "/home/mahidhar14/amazonml2026/code/business_entity_resolution")
from src.blocking import RankedBlockingIndex
from src.features import extract_pair_features, FEATURE_NAMES, extract_postal_codes
from src.normalize import (
    normalize_text,
    extract_core_name,
    get_name_tokens_sorted,
    get_unspaced_name,
    extract_house_numbers,
    extract_address_keys,
    NOISE_PREFIX_REGEX,
    LEGAL_TERMS
)
from src.model import EntityResolutionModel


# Global variables for forked worker processes (zero IPC serialization overhead)
G_S1 = None
G_TARGET = None
G_COUNTRY = None

def init_worker(s1_dict, target_lookup, country):
    global G_S1, G_TARGET, G_COUNTRY
    G_S1 = s1_dict
    G_TARGET = target_lookup
    G_COUNTRY = country

def worker_feature_batch(batch_pairs):
    batch_features = []
    for s1_id, tid in batch_pairs:
        s1 = G_S1.get(s1_id)
        t_data = G_TARGET.get(tid)
        if s1 is None or t_data is None:
            batch_features.append([0.0] * len(FEATURE_NAMES))
            continue
            
        raw_n, raw_a, norm_n, sorted_n, unspaced_n, norm_a, target_country = t_data
        feat = extract_pair_features(
            s1_name_raw=s1["name"],
            s1_addr_raw=s1["addr"],
            t_name_raw=raw_n,
            t_addr_raw=raw_a,
            country=s1["country"],
            target_id=tid,
            s1_norm_name=s1["norm_name"],
            s1_sorted_name=s1["sorted_name"],
            s1_unspaced_name=s1["unspaced_name"],
            s1_norm_addr=s1["norm_addr"],
            s1_house_nums=s1["house_nums"],
            s1_addr_tokens=s1["addr_tokens"],
            s1_postal_codes=s1["postal_codes"],
            t_norm_name=norm_n,
            t_sorted_name=sorted_n,
            t_unspaced_name=unspaced_n,
            t_norm_addr=norm_a,
            target_country=target_country,
        )
        batch_features.append(feat)
    return batch_features

def run_inference(
    test_dir: str,
    output_dir: str,
    model_path: str,
    max_candidates: int = None,
    chunk_size: int = 50000,
    n_jobs: int = 8,
    dry_run: bool = False
):
    os.makedirs(output_dir, exist_ok=True)
    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")
    
    print(f"Loading trained production model from {model_path}...")
    model_data = joblib.load(model_path)
    er_model = model_data["model"]
    threshold = float(model_data["threshold"])
    saved_features = model_data.get("features")
    if saved_features is not None and list(saved_features) != list(FEATURE_NAMES):
        raise ValueError("Saved model feature schema does not match current FEATURE_NAMES; retrain/select model first")
    model_config = model_data.get("config", {})
    if max_candidates is None:
        max_candidates = int(model_config.get("candidate_limit", 100))
    same_country_only = bool(model_config.get("same_country_only", True))
    print(f"Model loaded ({model_data.get('model_name', 'unknown')}, threshold={threshold:.4f}, candidates={max_candidates})")
    
    # Load test_source1
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"Reading test Source 1 from {s1_path}...")
    test_s1_df = pl.read_csv(
        s1_path, separator="\t", 
        columns=["entity_id", "business_name", "business_address", "country"]
    )
    all_s1_ids = test_s1_df["entity_id"].to_list()
    total_s1 = len(all_s1_ids)
    if len(set(all_s1_ids)) != total_s1:
        raise ValueError("test_source1.tsv contains duplicate entity IDs")
    print(f"Total test Source 1 entities: {total_s1:,}")
    
    # Result mappings: entity_id -> comma-separated string
    res_matches = {sid: "" for sid in all_s1_ids}
    res_candidates = {sid: "" for sid in all_s1_ids}
    
    total_pairs_evaluated = 0
    total_matches_found = 0
    total_t0 = time.time()
    
    countries = test_s1_df["country"].unique().to_list() if same_country_only else [None]
    for country in countries:
        if not same_country_only:
            c_s1 = test_s1_df
        elif country is None:
            c_s1 = test_s1_df.filter(pl.col("country").is_null())
        else:
            c_s1 = test_s1_df.filter(pl.col("country") == country)
        if dry_run:
            c_s1 = c_s1.head(2000)
            print(f"\n[DRY RUN] Processing {c_s1.shape[0]} entities for {country}...")
        else:
            print(f"\n==================== PROCESSING COUNTRY: {country} ====================")
            print(f"Test S1 entities in {country}: {c_s1.shape[0]:,}")
            
        t0 = time.time()
        print(f"[{country}] Loading targets from test_source2.tsv and test_source3.tsv...")
        s2_c = pl.read_csv(
            os.path.join(test_dir, "test_source2.tsv"), separator="\t",
            columns=["entity_id", "business_name", "business_address", "country"]
        )
        s3_c = pl.read_csv(
            os.path.join(test_dir, "test_source3.tsv"), separator="\t",
            columns=["entity_id", "business_name", "business_address", "country"]
        )
        if same_country_only:
            if country is None:
                s2_c = s2_c.filter(pl.col("country").is_null())
                s3_c = s3_c.filter(pl.col("country").is_null())
            else:
                s2_c = s2_c.filter(pl.col("country") == country)
                s3_c = s3_c.filter(pl.col("country") == country)
        
        targets_c = pl.concat([s2_c, s3_c])
        del s2_c, s3_c
        n_targets = targets_c.shape[0]
        print(f"[{country}] Loaded {n_targets:,} targets in {time.time()-t0:.2f}s")
        
        t1 = time.time()
        print(f"[{country}] Building RankedBlockingIndex and target lookup...")
        index = RankedBlockingIndex(max_bucket_size=500)
        target_lookup = {}
        
        t_ids = targets_c["entity_id"].to_list()
        t_names = targets_c["business_name"].to_list()
        t_addrs = targets_c["business_address"].to_list()
        t_countries = targets_c["country"].to_list() if "country" in targets_c.columns else [country] * len(t_ids)
        del targets_c
        for tid, name, addr, t_country in zip(t_ids, t_names, t_addrs, t_countries):
            # Index name and address signals independently: missing names still
            # permit address-only blocking through RankedBlockingIndex.
            index.add_target(tid, name, addr, t_country)
            norm_name = normalize_text(name)
            core = extract_core_name(name)
            sorted_core = " ".join(sorted(set(core.split()))) if core else ""
            unspaced = core.replace(" ", "") if core else ""
            norm_a = normalize_text(addr) if addr else ""
            target_lookup[tid] = (name, addr, norm_name, sorted_core, unspaced, norm_a, t_country)
            
        print(f"[{country}] Indexed {len(target_lookup):,} targets in {time.time()-t1:.2f}s")
        del t_ids, t_names, t_addrs
        
        # Process S1 entities in chunks
        s1_rows = c_s1.to_dicts()
        del c_s1
        n_chunks = (len(s1_rows) + chunk_size - 1) // chunk_size
        
        for ch_idx in range(n_chunks):
            ch_start = ch_idx * chunk_size
            ch_end = min(ch_start + chunk_size, len(s1_rows))
            chunk_s1 = s1_rows[ch_start:ch_end]
            
            t_ch0 = time.time()
            s1_dict = {}
            candidate_map = {}
            pairs = []
            
            for s1_rec in chunk_s1:
                sid = s1_rec["entity_id"]
                n = s1_rec["business_name"]
                a = s1_rec["business_address"]
                norm_a = normalize_text(a) if a else ""
                
                s1_dict[sid] = {
                    "name": n,
                    "addr": a,
                    "country": s1_rec["country"],
                    "norm_name": normalize_text(n),
                    "sorted_name": get_name_tokens_sorted(n),
                    "unspaced_name": get_unspaced_name(n),
                    "norm_addr": norm_a,
                    "house_nums": extract_house_numbers(None, norm_text=norm_a) if norm_a else [],
                    "addr_tokens": set(norm_a.split()) if norm_a else set(),
                    "postal_codes": extract_postal_codes(a),
                }
                
                cands = index.query_ranked(n, a, max_candidates=max_candidates)
                candidate_map[sid] = cands
                for tid in cands:
                    pairs.append((sid, tid))
                    
            n_pairs = len(pairs)
            total_pairs_evaluated += n_pairs
            
            # Parallel feature extraction via Fork Pool
            if n_pairs > 0:
                pair_batch_size = 10000
                batches = [pairs[i:i + pair_batch_size] for i in range(0, n_pairs, pair_batch_size)]
                
                ctx = mp.get_context("fork")
                with ctx.Pool(processes=n_jobs, initializer=init_worker, initargs=(s1_dict, target_lookup, country)) as pool:
                    feat_batches = pool.map(worker_feature_batch, batches)
                    
                X_chunk = np.array([f for b_res in feat_batches for f in b_res], dtype=np.float32)
                del feat_batches
                
                probs = er_model.predict_proba(X_chunk)
                keep = probs[:, 1] >= threshold
                del X_chunk, probs
                
                match_map = defaultdict(list)
                for (sid, tid), k in zip(pairs, keep):
                    if k:
                        match_map[sid].append(tid)
                del keep
            else:
                match_map = defaultdict(list)
                
            # Save results to memory map
            ch_matches = 0
            for s1_rec in chunk_s1:
                sid = s1_rec["entity_id"]
                cands = candidate_map.get(sid, [])
                matches = match_map.get(sid, [])
                
                uniq_matches = list(dict.fromkeys(matches))
                uniq_cands = list(dict.fromkeys(cands))
                
                if uniq_matches:
                    ch_matches += 1
                    res_matches[sid] = ",".join(uniq_matches)
                else:
                    res_matches[sid] = ""
                    
                res_candidates[sid] = ",".join(uniq_cands) if uniq_cands else ""
                
            total_matches_found += ch_matches
            ch_time = time.time() - t_ch0
            print(f"  [{country}] Chunk {ch_idx+1}/{n_chunks} ({len(chunk_s1):,} S1, {n_pairs:,} pairs, {ch_matches:,} matches) in {ch_time:.1f}s ({len(chunk_s1)/ch_time:.0f} S1/s, {n_pairs/ch_time:.0f} pairs/s)", flush=True)
            
        del index, target_lookup, s1_rows
        gc.collect()
        if dry_run:
            break
            
    if dry_run:
        print("[DRY RUN] Sample processing finished; submission files were not written.")
        return

    if set(res_matches) != set(all_s1_ids) or set(res_candidates) != set(all_s1_ids):
        raise ValueError("Output mappings do not cover exactly the test S1 IDs")
    for sid, match_text in res_matches.items():
        mids = [x for x in match_text.split(",") if x]
        candidate_list = [x for x in res_candidates[sid].split(",") if x]
        cids = set(candidate_list)
        if len(candidate_list) != len(cids):
            raise ValueError("Duplicate candidate IDs for " + sid)
        if any(not x.startswith(("S2-", "S3-")) for x in candidate_list):
            raise ValueError("Invalid candidate ID for " + sid)
        if len(mids) != len(set(mids)) or any(not x.startswith(("S2-", "S3-")) for x in mids):
            raise ValueError("Invalid or duplicate match IDs for " + sid)
        if not set(mids).issubset(cids):
            raise ValueError("Predicted matches are not a subset of candidates for " + sid)

    print(f"\n==================== WRITING SUBMISSION FILES ====================")
    print(f"Writing final submission TSVs in exact test_source1.tsv order ({len(all_s1_ids):,} entities)...")
    t_w0 = time.time()
    
    # Write matching_results.tsv and candidate_pairs.tsv
    with open(matching_file, "w", encoding="utf-8") as f_m, \
         open(candidate_file, "w", encoding="utf-8") as f_c:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for sid in all_s1_ids:
            m_str = res_matches.get(sid, "")
            c_str = res_candidates.get(sid, "")
            f_m.write(f"{sid}\t{m_str}\n")
            f_c.write(f"{sid}\t{c_str}\n")
            
    print(f"Successfully written both TSVs in {time.time()-t_w0:.2f}s")
    
    total_singletons = total_s1 - total_matches_found if not dry_run else len(res_matches) - total_matches_found
    print(f"\n==================== INFERENCE COMPLETE ====================")
    print(f"Total Source 1 entities: {total_s1:,}")
    print(f"Total evaluated candidate pairs: {total_pairs_evaluated:,}")
    print(f"Total entities with matches: {total_matches_found:,}")
    print(f"Total predicted singletons: {total_singletons:,}")
    if total_s1 > 0:
        print(f"Match rate: {total_matches_found/total_s1*100:.2f}% | Singleton rate: {total_singletons/total_s1*100:.2f}%")
    print(f"Total execution runtime: {time.time()-total_t0:.1f}s ({(time.time()-total_t0)/60:.2f} minutes)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="/home/mahidhar14/amazonml2026/student_resource/dataset/test")
    parser.add_argument("--output-dir", default="/home/mahidhar14/amazonml2026/output")
    parser.add_argument("--model-path", default="/home/mahidhar14/amazonml2026/experiments/best_model.joblib")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    run_inference(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        max_candidates=args.max_candidates,
        chunk_size=args.chunk_size,
        n_jobs=args.n_jobs,
        dry_run=args.dry_run
    )
