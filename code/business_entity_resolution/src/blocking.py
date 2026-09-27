from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import polars as pl

from src.normalize import normalize_text, extract_core_name, extract_address_keys, extract_house_numbers


class RankedBlockingIndex:
    """Shared multi-signal index used by training, validation, and inference.

    Country is deliberately not a blocking constraint: cross-country matches remain
    discoverable, and country relationship is left to the pair features/model.
    """
    def __init__(self, max_bucket_size: int = 500):
        self.max_bucket_size = max_bucket_size
        self.idx_exact_name = defaultdict(list)
        self.idx_sorted_name = defaultdict(list)
        self.idx_unspaced_name = defaultdict(list)
        self.idx_prefix_name = defaultdict(list)
        self.idx_address = defaultdict(list)
        self.idx_postal = defaultdict(list)
        self.idx_house = defaultdict(list)

    def add_target(self, tid: str, name: Optional[str], addr: Optional[str], country: Optional[str] = None):
        # The challenge target universe contains only S2/S3. Keep the guard here
        # so an accidental S1 input can never become a candidate.
        if not tid or not str(tid).startswith(("S2-", "S3-")):
            return
        if name:
            norm = normalize_text(name)
            if norm:
                self.idx_exact_name[norm].append(tid)
            core = extract_core_name(name)
            if core:
                words = core.split()
                self.idx_sorted_name[" ".join(sorted(set(words)))].append(tid)
                compact = core.replace(" ", "")
                if len(compact) >= 4:
                    self.idx_unspaced_name[compact].append(tid)
                prefix = " ".join(words[:2]) if len(words) >= 2 else words[0]
                if len(prefix) >= 4:
                    self.idx_prefix_name[prefix].append(tid)
        if addr:
            for key in extract_address_keys(addr):
                self.idx_address[key].append(tid)
            norm_addr = normalize_text(addr)
            for code in _postal_codes(norm_addr):
                self.idx_postal[code].append(tid)
            for number in extract_house_numbers(None, norm_text=norm_addr):
                self.idx_house[number].append(tid)

    def query_ranked(self, name: Optional[str], addr: Optional[str], max_candidates: int = 100) -> List[str]:
        scores = defaultdict(float)
        def add(index, key, weight):
            if key and key in index:
                bucket = index[key]
                for tid in bucket[:self.max_bucket_size]:
                    scores[tid] += weight
        norm = normalize_text(name)
        add(self.idx_exact_name, norm, 100.0)
        core = extract_core_name(name)
        if core:
            words = core.split()
            add(self.idx_sorted_name, " ".join(sorted(set(words))), 80.0)
            compact = core.replace(" ", "")
            if len(compact) >= 4:
                add(self.idx_unspaced_name, compact, 70.0)
            prefix = " ".join(words[:2]) if len(words) >= 2 else words[0]
            if len(prefix) >= 4:
                add(self.idx_prefix_name, prefix, 30.0)
        if addr:
            norm_addr = normalize_text(addr)
            for key in extract_address_keys(None, norm_text=norm_addr):
                add(self.idx_address, key, 50.0)
            for code in _postal_codes(norm_addr):
                add(self.idx_postal, code, 35.0)
            for number in extract_house_numbers(None, norm_text=norm_addr):
                add(self.idx_house, number, 10.0)
        if max_candidates <= 0:
            return []
        return [tid for tid, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:max_candidates]]


def _postal_codes(text: str) -> List[str]:
    import re
    return re.findall(r"\b\d{5,6}\b", text or "")


def generate_candidates(s1_records: Iterable[dict], target_records: Iterable[dict],
                        max_candidates: int = 100, max_bucket_size: int = 500,
                        same_country_only: bool = True) -> Dict[str, List[str]]:
    """Return candidates for every S1, including address-only rows.

    Default country partitioning matches current inference. Disable it only if
    validation confirms genuine cross-country links.
    """
    s1_rows = list(s1_records)
    indexes = {}
    global_index = RankedBlockingIndex(max_bucket_size=max_bucket_size) if not same_country_only else None
    for row in target_records:
        country = row.get("country")
        index = indexes.setdefault(country, RankedBlockingIndex(max_bucket_size=max_bucket_size)) if same_country_only else global_index
        index.add_target(row.get("entity_id"), row.get("business_name"),
                         row.get("business_address"), country)
    output = {}
    for row in s1_rows:
        sid = row.get("entity_id")
        index = indexes.get(row.get("country")) if same_country_only else global_index
        output[sid] = index.query_ranked(row.get("business_name"), row.get("business_address"), max_candidates) if index else []
    return output


def candidate_diagnostics(candidates: Dict[str, List[str]], ground_truth: Optional[Dict[str, Set[str]]] = None) -> dict:
    counts = [len(v) for v in candidates.values()]
    n = len(counts)
    diag = {
        "s1_count": n,
        "avg_candidates": (sum(counts) / n) if n else 0.0,
        "median_candidates": (sorted(counts)[n // 2] if n else 0),
        "max_candidates": max(counts, default=0),
        "zero_candidate_pct": (100.0 * sum(c == 0 for c in counts) / n) if n else 0.0,
    }
    if ground_truth is not None:
        true_count = sum(len(v) for v in ground_truth.values())
        found = sum(len(set(candidates.get(sid, [])) & true) for sid, true in ground_truth.items())
        diag["true_match_candidate_recall"] = found / true_count if true_count else 1.0
        diag["true_match_missing_pct"] = 100.0 * (1.0 - diag["true_match_candidate_recall"])
    return diag


# Backwards-compatible name for older scripts.
MultiStageBlockingIndex = RankedBlockingIndex
