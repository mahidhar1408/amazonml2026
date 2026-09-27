import re
from typing import Dict, List, Any, Optional, Set
import numpy as np
from rapidfuzz import fuzz, distance

from src.normalize import (
    normalize_text,
    extract_core_name,
    get_name_tokens_sorted,
    get_unspaced_name,
    extract_house_numbers,
    COMMON_ADDR_STOPWORDS
)

FEATURE_NAMES = [
    # 1. Name Features (16)
    "name_exact_match",
    "name_norm_exact",
    "name_sorted_exact",
    "name_unspaced_exact",
    "name_fuzz_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_jaro_winkler",
    "name_token_jaccard",
    "name_token_overlap_count",
    "name_char_len_diff",
    "name_char_len_ratio",
    "name_prefix_match",
    "name_char_ngram_jaccard",
    "name_acronym_match",
    
    # 2. Address Features (15)
    "is_addr_missing",
    "addr_exact_match",
    "addr_norm_exact",
    "addr_fuzz_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_jaccard",
    "addr_token_overlap_count",
    "addr_char_ngram_jaccard",
    "house_num_exact",
    "house_num_both_present",
    "house_num_mismatch",
    "postal_exact",
    "postal_both_present",
    "postal_mismatch",
    
    # 3. Cross-Field & Structural (6)
    "target_source_is_s2",
    "target_source_is_s3",
    "name_addr_harmonic_mean",
    "name_addr_product",
    "country_matches_target",
    "country_differs_from_target",
]

POSTAL_REGEX = re.compile(r"\b(\d{5,6})\b")

def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}

def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Compute character n-gram Jaccard similarity."""
    if not s1 or not s2:
        return 0.0
    ng1 = get_char_ngrams(s1, n)
    ng2 = get_char_ngrams(s2, n)
    inter = len(ng1 & ng2)
    union = len(ng1 | ng2)
    return (inter / union) if union > 0 else 0.0

def extract_postal_codes(text: Optional[str]) -> List[str]:
    """Extract 5 or 6 digit postal / PIN codes."""
    if not text:
        return []
    return POSTAL_REGEX.findall(text)

def check_acronym(tokens1: List[str], text2: str) -> float:
    """Check if the initials of tokens in text1 form an acronym in text2 or vice versa."""
    if len(tokens1) >= 2:
        acr = "".join(w[0] for w in tokens1 if w)
        if acr and (acr in text2.split() or acr == text2.strip()):
            return 1.0
    return 0.0

def extract_pair_features(
    s1_name_raw: str,
    s1_addr_raw: Optional[str],
    t_name_raw: str,
    t_addr_raw: Optional[str],
    country: str,
    target_id: str,
    # S1 precomputed representations
    s1_norm_name: Optional[str] = None,
    s1_sorted_name: Optional[str] = None,
    s1_unspaced_name: Optional[str] = None,
    s1_norm_addr: Optional[str] = None,
    s1_house_nums: Optional[List[str]] = None,
    s1_addr_tokens: Optional[set] = None,
    s1_postal_codes: Optional[List[str]] = None,
    # Target precomputed representations (for high-throughput inference)
    t_norm_name: Optional[str] = None,
    t_sorted_name: Optional[str] = None,
    t_unspaced_name: Optional[str] = None,
    t_norm_addr: Optional[str] = None,
    t_house_nums: Optional[List[str]] = None,
    t_addr_tokens: Optional[set] = None,
    t_postal_codes: Optional[List[str]] = None,
    target_country: Optional[str] = None,
) -> List[float]:
    """Extract all named features for a pair with optional precomputations."""
    norm_n1 = s1_norm_name if s1_norm_name is not None else normalize_text(s1_name_raw)
    sorted_n1 = s1_sorted_name if s1_sorted_name is not None else get_name_tokens_sorted(s1_name_raw)
    unspaced_n1 = s1_unspaced_name if s1_unspaced_name is not None else get_unspaced_name(s1_name_raw)
    
    norm_n2 = t_norm_name if t_norm_name is not None else normalize_text(t_name_raw)
    sorted_n2 = t_sorted_name if t_sorted_name is not None else get_name_tokens_sorted(t_name_raw)
    unspaced_n2 = t_unspaced_name if t_unspaced_name is not None else get_unspaced_name(t_name_raw)
    
    # 1. Name Features
    name_exact = 1.0 if s1_name_raw == t_name_raw else 0.0
    name_norm_exact = 1.0 if (norm_n1 == norm_n2 and norm_n1) else 0.0
    name_sorted_exact = 1.0 if (sorted_n1 == sorted_n2 and sorted_n1) else 0.0
    name_unspaced_exact = 1.0 if (unspaced_n1 == unspaced_n2 and unspaced_n1) else 0.0
    
    name_fuzz_ratio = fuzz.ratio(norm_n1, norm_n2) / 100.0
    name_token_sort = fuzz.token_sort_ratio(norm_n1, norm_n2) / 100.0
    name_token_set = fuzz.token_set_ratio(norm_n1, norm_n2) / 100.0
    name_partial = fuzz.partial_ratio(norm_n1, norm_n2) / 100.0
    name_jw = distance.JaroWinkler.similarity(norm_n1, norm_n2)
    
    t1_set = set(norm_n1.split())
    t2_set = set(norm_n2.split())
    inter_len = len(t1_set & t2_set)
    union_len = len(t1_set | t2_set)
    name_jaccard = (inter_len / union_len) if union_len > 0 else 0.0
    
    l1, l2 = len(norm_n1), len(norm_n2)
    name_len_diff = abs(l1 - l2)
    name_len_ratio = (min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 0.0
    
    w1 = norm_n1.split()
    w2 = norm_n2.split()
    name_prefix = 1.0 if (w1 and w2 and w1[0] == w2[0]) else 0.0
    name_ngram = char_ngram_jaccard(norm_n1, norm_n2, n=3)
    name_acr = max(check_acronym(w1, norm_n2), check_acronym(w2, norm_n1))
    
    # 2. Address Features
    is_addr_missing = 1.0 if (t_addr_raw is None or not str(t_addr_raw).strip()) else 0.0
    if is_addr_missing:
        addr_exact = 0.0
        addr_norm_exact = 0.0
        addr_fuzz_ratio = 0.0
        addr_token_sort = 0.0
        addr_token_set = 0.0
        addr_jaccard = 0.0
        addr_overlap_count = 0.0
        addr_ngram = 0.0
        house_exact = 0.0
        house_both = 0.0
        house_mismatch = 0.0
        postal_exact = 0.0
        postal_both = 0.0
        postal_mismatch = 0.0
    else:
        norm_a1 = s1_norm_addr if s1_norm_addr is not None else normalize_text(s1_addr_raw)
        norm_a2 = t_norm_addr if t_norm_addr is not None else normalize_text(t_addr_raw)
        
        addr_exact = 1.0 if s1_addr_raw == t_addr_raw else 0.0
        addr_norm_exact = 1.0 if (norm_a1 == norm_a2 and norm_a1) else 0.0
        
        addr_fuzz_ratio = fuzz.ratio(norm_a1, norm_a2) / 100.0
        addr_token_sort = fuzz.token_sort_ratio(norm_a1, norm_a2) / 100.0
        addr_token_set = fuzz.token_set_ratio(norm_a1, norm_a2) / 100.0
        
        a1_set = s1_addr_tokens if s1_addr_tokens is not None else set(norm_a1.split())
        a2_set = t_addr_tokens if t_addr_tokens is not None else set(norm_a2.split())
        a_inter = len(a1_set & a2_set)
        a_union = len(a1_set | a2_set)
        addr_jaccard = (a_inter / a_union) if a_union > 0 else 0.0
        addr_overlap_count = float(a_inter)
        addr_ngram = char_ngram_jaccard(norm_a1, norm_a2, n=3)
        
        # House number agreement
        h1 = s1_house_nums if s1_house_nums is not None else extract_house_numbers(s1_addr_raw)
        h2 = t_house_nums if t_house_nums is not None else extract_house_numbers(t_addr_raw)
        if h1 and h2:
            house_both = 1.0
            if set(h1) & set(h2):
                house_exact = 1.0
                house_mismatch = 0.0
            else:
                house_exact = 0.0
                house_mismatch = 1.0
        else:
            house_both = 0.0
            house_exact = 0.0
            house_mismatch = 0.0
            
        # Postal code agreement
        p1 = s1_postal_codes if s1_postal_codes is not None else extract_postal_codes(s1_addr_raw)
        p2 = t_postal_codes if t_postal_codes is not None else extract_postal_codes(t_addr_raw)
        if p1 and p2:
            postal_both = 1.0
            if set(p1) & set(p2):
                postal_exact = 1.0
                postal_mismatch = 0.0
            else:
                postal_exact = 0.0
                postal_mismatch = 1.0
        else:
            postal_both = 0.0
            postal_exact = 0.0
            postal_mismatch = 0.0
            
    # 3. Cross-field and Metadata
    is_s2 = 1.0 if target_id.startswith("S2-") else 0.0
    is_s3 = 1.0 if target_id.startswith("S3-") else 0.0
    
    n_score = name_token_set
    a_score = addr_token_set
    hm = (2.0 * n_score * a_score) / (n_score + a_score + 1e-5)
    prod = n_score * a_score
    
    c_same = 1.0 if country and target_country and country == target_country else 0.0
    c_diff = 1.0 if country and target_country and country != target_country else 0.0
    
    return [
        name_exact,
        name_norm_exact,
        name_sorted_exact,
        name_unspaced_exact,
        name_fuzz_ratio,
        name_token_sort,
        name_token_set,
        name_partial,
        name_jw,
        name_jaccard,
        float(inter_len),
        float(name_len_diff),
        name_len_ratio,
        name_prefix,
        name_ngram,
        name_acr,
        
        is_addr_missing,
        addr_exact,
        addr_norm_exact,
        addr_fuzz_ratio,
        addr_token_sort,
        addr_token_set,
        addr_jaccard,
        addr_overlap_count,
        addr_ngram,
        house_exact,
        house_both,
        house_mismatch,
        postal_exact,
        postal_both,
        postal_mismatch,
        
        is_s2,
        is_s3,
        hm,
        prod,
        c_same,
        c_diff,
    ]
