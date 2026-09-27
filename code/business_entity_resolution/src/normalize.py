import re
import anyascii
from typing import List, Set, Optional

# Compiled regexes for fast performance
DOMAIN_REGEX = re.compile(r"\.(com|in|org|net|co|io|fr|us|biz|info)\b", re.IGNORECASE)
NOISE_PREFIX_REGEX = re.compile(
    r"\b(f/?k/?a|formerly known as|formerly|d/?b/?a|dba|fka|aka|trading as|t/?a|a/?k/?a)\b", 
    re.IGNORECASE
)
LEET_ZERO = re.compile(r"(?<=[a-z])0(?=[a-z])")
LEET_ONE = re.compile(r"(?<=[a-z])1(?=[a-z])")
NON_WORD_SPACE = re.compile(r"[^\w\s]")
WHITESPACE = re.compile(r"\s+")
HOUSE_NUM_REGEX = re.compile(r"\b(\d{1,6})([a-z])?\b")

LEGAL_TERMS: Set[str] = {
    "inc", "incorporated", "corp", "corporation", "llc", "pllc", "pc", "ltd", 
    "limited", "pvt", "private", "co", "company", "lp", "llp", "opc",
    "sarl", "sas", "sasu", "eurl", "sci", "sa", "gmbh", "cie", "fils", 
    "freres", "holding", "holdings", "group", "services", "enterprises"
}

COMMON_ADDR_STOPWORDS: Set[str] = {
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive", "dr", 
    "blvd", "boulevard", "court", "ct", "circle", "cir", "highway", "hwy",
    "way", "place", "pl", "floor", "flr", "suite", "ste", "building", "bldg",
    "near", "opp", "opposite", "beside", "behind", "post", "po", "box", "pobox",
    "north", "south", "east", "west", "n", "s", "e", "w", "c/o", "co", "s/o", "d/o",
    "city", "town", "village", "dist", "district", "state", "sector", "phase",
    "block", "plot", "no", "door", "apt", "unit", "rue", "av", "des", "du", "de",
    "null", "na"
}

def normalize_text(text: Optional[str]) -> str:
    """Normalize general text by transliterating Unicode to ASCII, lowercasing,
    stripping domains, normalizing & to and, replacing leetspeak digits, and
    stripping punctuation."""
    if not text:
        return ""
    # Transliterate Unicode scripts (Indic, French accents, etc.) to ASCII
    text = anyascii.anyascii(text).lower()
    text = DOMAIN_REGEX.sub(" ", text)
    text = text.replace("&", " and ")
    text = LEET_ZERO.sub("o", text)
    text = LEET_ONE.sub("l", text)
    text = NON_WORD_SPACE.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()

def extract_core_name(text: Optional[str], norm_text: Optional[str] = None) -> str:
    """Extract the core business name by isolating the entity from DBA/FKA noise
    and stripping common legal suffixes."""
    if not text and not norm_text:
        return ""
    if text:
        match = NOISE_PREFIX_REGEX.search(text)
        if match:
            after_part = text[match.end():].strip()
            if len(after_part) > 2:
                text = after_part
                norm_text = None
    norm = norm_text if norm_text is not None else normalize_text(text)
    tokens = [w for w in norm.split() if w not in LEGAL_TERMS and len(w) > 1]
    return " ".join(tokens) if tokens else norm

def get_name_tokens_sorted(text: Optional[str]) -> str:
    """Return sorted unique tokens of the core business name."""
    core = extract_core_name(text)
    return " ".join(sorted(set(core.split())))

def get_unspaced_name(text: Optional[str]) -> str:
    """Return core business name without any spaces (handles concatenated URLs)."""
    core = extract_core_name(text)
    return core.replace(" ", "")

def get_name_prefix(text: Optional[str], n_words: int = 2) -> str:
    """Return the first n words of the core name."""
    core = extract_core_name(text)
    words = core.split()
    return " ".join(words[:n_words]) if len(words) >= n_words else (words[0] if words else "")

def extract_house_numbers(text: Optional[str], norm_text: Optional[str] = None) -> List[str]:
    """Extract normalized house / building numbers (leading zeros stripped)."""
    if not text and not norm_text:
        return []
    norm = norm_text if norm_text is not None else normalize_text(text)
    numbers = []
    for m in HOUSE_NUM_REGEX.finditer(norm):
        num_str = m.group(1).lstrip("0")
        if num_str and len(num_str) <= 6:
            numbers.append(num_str)
    return list(dict.fromkeys(numbers))  # preserve order, unique

def extract_address_keys(text: Optional[str], norm_text: Optional[str] = None) -> List[str]:
    """Extract discriminative address keys for multi-stage blocking."""
    if not text and not norm_text:
        return []
    norm = norm_text if norm_text is not None else normalize_text(text)
    tokens = norm.split()
    house_nums = extract_house_numbers(None, norm_text=norm)
    sig_words = [w for w in tokens if w not in COMMON_ADDR_STOPWORDS and not w.isdigit() and len(w) >= 4]
    
    keys = []
    # Key combination: House number + significant address word
    if house_nums and sig_words:
        h = house_nums[0]
        for w in sig_words[:3]:
            keys.append(f"{h}_{w}")
            
    # Key combination: Pair of consecutive significant words
    if len(sig_words) >= 2:
        keys.append(f"{sig_words[0]}_{sig_words[1]}")
        
    # Key: Highly distinctive long word (>= 7 chars)
    for w in sig_words:
        if len(w) >= 7:
            keys.append(f"rare_{w}")
            
    return list(dict.fromkeys(keys))
