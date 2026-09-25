"""
Pairwise feature engineering for (source1_id, candidate_id) pairs produced by
the blocking stage. All features are derived purely from the two records'
own fields -- no external data.
"""
import pandas as pd
try:
    from rapidfuzz import fuzz
except ImportError:  # fallback for diagnostics; production requirements pin rapidfuzz
    from difflib import SequenceMatcher
    class _Fuzz:
        @staticmethod
        def ratio(a, b): return 100 * SequenceMatcher(None, a, b).ratio()
        @staticmethod
        def token_sort_ratio(a, b): return 100 * SequenceMatcher(None, " ".join(sorted(a.split())), " ".join(sorted(b.split()))).ratio()
        @staticmethod
        def token_set_ratio(a, b): return 100 * SequenceMatcher(None, " ".join(sorted(set(a.split()))), " ".join(sorted(set(b.split())))).ratio()
    fuzz = _Fuzz()

FEATURE_COLUMNS = [
    "blocking_score",
    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaccard",
    "name_exact",
    "name_first_token_match",
    "addr_ratio",
    "addr_token_sort_ratio",
    "addr_jaccard",
    "postal_match",
    "postal_both_present",
    "country_match",
    "name_len_diff",
    "addr_len_diff",
]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def build_feature_frame(pairs, s1_store, pool_lookup):
    """
    pairs: iterable of (s1_id, cand_id, blocking_score)
    s1_store: RecordStore for Source-1
    pool_lookup: dict-like giving RecordStore-style attribute dicts for any
        candidate id, i.e. an object with .name_norm/.name_tokens/.name_core/
        .addr_norm/.addr_core/.postal/.country merged across Source-2 and
        Source-3 (see predict.py / train.py for how this is built).

    Returns a pandas DataFrame with one row per pair: source1_entity_id,
    candidate_entity_id, and all FEATURE_COLUMNS.
    """
    rows = []
    s1_ids, cand_ids = [], []

    for s1_id, cand_id, block_score in pairs:
        s1_name = s1_store.name_norm[s1_id]
        c_name = pool_lookup.name_norm[cand_id]
        s1_addr = s1_store.addr_norm[s1_id]
        c_addr = pool_lookup.addr_norm[cand_id]

        s1_name_core = s1_store.name_core[s1_id]
        c_name_core = pool_lookup.name_core[cand_id]
        s1_addr_core = s1_store.addr_core[s1_id]
        c_addr_core = pool_lookup.addr_core[cand_id]

        s1_first = s1_store.name_tokens[s1_id][0] if s1_store.name_tokens[s1_id] else ""
        c_first = pool_lookup.name_tokens[cand_id][0] if pool_lookup.name_tokens[cand_id] else ""

        s1_postal = s1_store.postal[s1_id]
        c_postal = pool_lookup.postal[cand_id]

        row = {
            "blocking_score": block_score,
            "name_ratio": fuzz.ratio(s1_name, c_name) / 100.0,
            "name_token_sort_ratio": fuzz.token_sort_ratio(s1_name, c_name) / 100.0,
            "name_token_set_ratio": fuzz.token_set_ratio(s1_name, c_name) / 100.0,
            "name_jaccard": _jaccard(s1_name_core, c_name_core),
            "name_exact": float(s1_name == c_name and s1_name != ""),
            "name_first_token_match": float(s1_first == c_first and s1_first != ""),
            "addr_ratio": fuzz.ratio(s1_addr, c_addr) / 100.0,
            "addr_token_sort_ratio": fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0,
            "addr_jaccard": _jaccard(s1_addr_core, c_addr_core),
            "postal_match": float(bool(s1_postal) and bool(c_postal) and s1_postal == c_postal),
            "postal_both_present": float(bool(s1_postal) and bool(c_postal)),
            "country_match": float(s1_store.country[s1_id] == pool_lookup.country[cand_id]),
            "name_len_diff": abs(len(s1_name) - len(c_name)),
            "addr_len_diff": abs(len(s1_addr) - len(c_addr)),
        }
        rows.append(row)
        s1_ids.append(s1_id)
        cand_ids.append(cand_id)

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    df.insert(0, "source1_entity_id", s1_ids)
    df.insert(1, "candidate_entity_id", cand_ids)
    return df


class MergedPool:
    """Merges several RecordStore objects into one dict-of-dicts lookup so
    features.py can treat Source-2 and Source-3 candidates uniformly."""

    def __init__(self, stores):
        self.name_norm, self.name_tokens, self.name_core = {}, {}, {}
        self.addr_norm, self.addr_core, self.postal, self.country = {}, {}, {}, {}
        for s in stores:
            self.name_norm.update(s.name_norm)
            self.name_tokens.update(s.name_tokens)
            self.name_core.update(s.name_core)
            self.addr_norm.update(s.addr_norm)
            self.addr_core.update(s.addr_core)
            self.postal.update(s.postal)
            self.country.update(s.country)
