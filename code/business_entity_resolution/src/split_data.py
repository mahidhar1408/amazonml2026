import os
import random
import polars as pl

DATA_DIR = "/home/mahidhar14/amazonml2026/student_resource/dataset/train"
OUTPUT_DIR = "/home/mahidhar14/amazonml2026/experiments"

def create_splits(n_train_s1: int = 70000, n_val_s1: int = 30000, seed: int = 42):
    random.seed(seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading train_source1.tsv and train_ground_truth.tsv...")
    s1_df = pl.read_csv(os.path.join(DATA_DIR, "train_source1.tsv"), separator="\t")
    gt_df = pl.read_csv(os.path.join(DATA_DIR, "train_ground_truth.tsv"), separator="\t")
    
    # Join to get country and match status
    joined = s1_df.join(gt_df, left_on="entity_id", right_on="source1_entity_id")
    
    # Calculate match count per entity for stratification
    def get_match_count(m):
        if m is None or str(m).strip() == "" or str(m).strip() == "None":
            return 0
        return len(str(m).split(","))
        
    match_counts = [get_match_count(m) for m in joined["matched_entity_ids"].to_list()]
    joined = joined.with_columns(pl.Series("match_count", match_counts))
    
    # Group by (country, is_singleton) and sample
    val_ids = []
    train_ids = []
    
    grouped = joined.group_by(["country", "match_count"])
    for (country, mcount), group_df in grouped:
        all_ids = group_df["entity_id"].to_list()
        random.shuffle(all_ids)
        
        # Calculate proportional allocations
        total_in_stratum = len(all_ids)
        frac = (n_train_s1 + n_val_s1) / joined.shape[0]
        allocated = int(total_in_stratum * frac)
        
        val_size = int(allocated * (n_val_s1 / (n_train_s1 + n_val_s1)))
        train_size = allocated - val_size
        
        val_ids.extend(all_ids[:val_size])
        train_ids.extend(all_ids[val_size:val_size + train_size])
        
    print(f"Selected {len(train_ids):,} train S1 entities and {len(val_ids):,} validation S1 entities.")
    
    val_s1 = s1_df.filter(pl.col("entity_id").is_in(val_ids))
    val_gt = gt_df.filter(pl.col("source1_entity_id").is_in(val_ids))
    
    train_s1 = s1_df.filter(pl.col("entity_id").is_in(train_ids))
    train_gt = gt_df.filter(pl.col("source1_entity_id").is_in(train_ids))
    
    val_s1.write_parquet(os.path.join(OUTPUT_DIR, "val_s1.parquet"))
    val_gt.write_parquet(os.path.join(OUTPUT_DIR, "val_gt.parquet"))
    train_s1.write_parquet(os.path.join(OUTPUT_DIR, "train_s1.parquet"))
    train_gt.write_parquet(os.path.join(OUTPUT_DIR, "train_gt.parquet"))
    
    print(f"Saved stratified splits to {OUTPUT_DIR}:")
    print(f"  val_s1.parquet: {val_s1.shape[0]:,} rows")
    print(f"  val_gt.parquet: {val_gt.shape[0]:,} rows")
    print(f"  train_s1.parquet: {train_s1.shape[0]:,} rows")
    print(f"  train_gt.parquet: {train_gt.shape[0]:,} rows")

if __name__ == "__main__":
    create_splits()
