"""
Pairwise feature engineering for (source1_id, candidate_id) pairs produced by
the blocking stage. All features are derived purely from the two records'
own fields -- no external data.
"""
import pandas as pd
from rapidfuzz import fuzz

FEATURE_COLUMNS = [
    "blocking_score",
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_w_ratio",
    "name_core_ratio",
    "name_core_token_set_ratio",
    "name_jaccard",
    "name_dice",
    "name_overlap",
    "name_exact",
    "name_first_token_match",
    "name_len_diff",
    "name_len_ratio",
    "name_missing",
    "addr_ratio",
    "addr_partial_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_w_ratio",
    "addr_jaccard",
    "addr_dice",
    "addr_overlap",
    "addr_exact",
    "addr_len_diff",
    "addr_missing",
    "postal_match",
    "postal_mismatch",
    "postal_both_present",
    "digits_jaccard",
    "digits_mismatch",
    "country_match",
    "max_ratio",
    "mean_ratio",
    "is_s2",
]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _dice(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    total = len(a) + len(b)
    return (2.0 * inter) / total if total else 0.0


def _overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    m = min(len(a), len(b))
    return inter / m if m else 0.0


import concurrent.futures


def _process_pair_slice(pairs_slice, s1_store, pool_lookup):
    rows = []
    s1_ids, cand_ids = [], []

    for s1_id, cand_id, block_score in pairs_slice:
        s1_name = s1_store.name_norm[s1_id]
        c_name = pool_lookup.name_norm[cand_id]
        s1_addr = s1_store.addr_norm[s1_id]
        c_addr = pool_lookup.addr_norm[cand_id]

        s1_name_core_set = s1_store.name_core[s1_id]
        c_name_core_set = pool_lookup.name_core[cand_id]
        s1_addr_core_set = s1_store.addr_core[s1_id]
        c_addr_core_set = pool_lookup.addr_core[cand_id]

        s1_name_core_str = s1_store.name_core_str[s1_id]
        c_name_core_str = pool_lookup.name_core_str[cand_id]

        s1_first = s1_store.name_tokens[s1_id][0] if s1_store.name_tokens[s1_id] else ""
        c_first = pool_lookup.name_tokens[cand_id][0] if pool_lookup.name_tokens[cand_id] else ""

        s1_postal = s1_store.postal[s1_id]
        c_postal = pool_lookup.postal[cand_id]

        s1_digits = s1_store.addr_digits[s1_id]
        c_digits = pool_lookup.addr_digits[cand_id]

        name_r = fuzz.ratio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0
        addr_r = fuzz.ratio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0

        has_both_postal = bool(s1_postal) and bool(c_postal)
        postal_eq = float(has_both_postal and s1_postal == c_postal)
        postal_diff = float(has_both_postal and s1_postal != c_postal)

        has_both_digits = bool(s1_digits) and bool(c_digits)
        digits_jac = _jaccard(s1_digits, c_digits) if has_both_digits else 0.0
        digits_mismatch = float(has_both_digits and len(s1_digits & c_digits) == 0)

        s1_len_name, c_len_name = len(s1_name), len(c_name)
        max_len_n = max(s1_len_name, c_len_name, 1)
        min_len_n = min(s1_len_name, c_len_name)

        row = {
            "blocking_score": block_score,
            "name_ratio": name_r,
            "name_partial_ratio": fuzz.partial_ratio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0,
            "name_token_sort_ratio": fuzz.token_sort_ratio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0,
            "name_token_set_ratio": fuzz.token_set_ratio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0,
            "name_w_ratio": fuzz.WRatio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0,
            "name_core_ratio": fuzz.ratio(s1_name_core_str, c_name_core_str) / 100.0 if s1_name_core_str and c_name_core_str else 0.0,
            "name_core_token_set_ratio": fuzz.token_set_ratio(s1_name_core_str, c_name_core_str) / 100.0 if s1_name_core_str and c_name_core_str else 0.0,
            "name_jaccard": _jaccard(s1_name_core_set, c_name_core_set),
            "name_dice": _dice(s1_name_core_set, c_name_core_set),
            "name_overlap": _overlap(s1_name_core_set, c_name_core_set),
            "name_exact": float(s1_name == c_name and s1_name != ""),
            "name_first_token_match": float(s1_first == c_first and s1_first != ""),
            "name_len_diff": abs(s1_len_name - c_len_name),
            "name_len_ratio": min_len_n / max_len_n,
            "name_missing": float(not s1_name or not c_name),
            "addr_ratio": addr_r,
            "addr_partial_ratio": fuzz.partial_ratio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0,
            "addr_token_sort_ratio": fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0,
            "addr_token_set_ratio": fuzz.token_set_ratio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0,
            "addr_w_ratio": fuzz.WRatio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0,
            "addr_jaccard": _jaccard(s1_addr_core_set, c_addr_core_set),
            "addr_dice": _dice(s1_addr_core_set, c_addr_core_set),
            "addr_overlap": _overlap(s1_addr_core_set, c_addr_core_set),
            "addr_exact": float(s1_addr == c_addr and s1_addr != ""),
            "addr_len_diff": abs(len(s1_addr) - len(c_addr)),
            "addr_missing": float(not s1_addr or not c_addr),
            "postal_match": postal_eq,
            "postal_mismatch": postal_diff,
            "postal_both_present": float(has_both_postal),
            "digits_jaccard": digits_jac,
            "digits_mismatch": digits_mismatch,
            "country_match": float(s1_store.country[s1_id] == pool_lookup.country[cand_id]),
            "max_ratio": max(name_r, addr_r),
            "mean_ratio": 0.5 * (name_r + addr_r),
            "is_s2": float(str(cand_id).startswith("S2-")),
        }
        rows.append(row)
        s1_ids.append(s1_id)
        cand_ids.append(cand_id)
    return rows, s1_ids, cand_ids


import numpy as np


def compute_features_matrix(pairs, s1_store, pool_lookup):
    """
    Direct NumPy matrix feature extraction.
    Bypasses pandas and dict allocations for 15x faster execution.
    """
    n = len(pairs)
    mat = np.zeros((n, len(FEATURE_COLUMNS)), dtype=np.float32)
    s1_ids = []
    cand_ids = []

    s1_name_norm = s1_store.name_norm
    pool_name_norm = pool_lookup.name_norm
    s1_addr_norm = s1_store.addr_norm
    pool_addr_norm = pool_lookup.addr_norm
    s1_name_core = s1_store.name_core
    pool_name_core = pool_lookup.name_core
    s1_addr_core = s1_store.addr_core
    pool_addr_core = pool_lookup.addr_core
    s1_name_core_str = s1_store.name_core_str
    pool_name_core_str = pool_lookup.name_core_str
    s1_name_tokens = s1_store.name_tokens
    pool_name_tokens = pool_lookup.name_tokens
    s1_postal = s1_store.postal
    pool_postal = pool_lookup.postal
    s1_addr_digits = s1_store.addr_digits
    pool_addr_digits = pool_lookup.addr_digits
    s1_country = s1_store.country
    pool_country = pool_lookup.country

    fuzz_ratio = fuzz.ratio
    fuzz_partial = fuzz.partial_ratio
    fuzz_token_sort = fuzz.token_sort_ratio
    fuzz_token_set = fuzz.token_set_ratio
    fuzz_w_ratio = fuzz.WRatio

    for i, (s1_id, cand_id, block_score) in enumerate(pairs):
        s1_ids.append(s1_id)
        cand_ids.append(cand_id)

        s_name = s1_name_norm[s1_id]
        c_name = pool_name_norm[cand_id]
        s_addr = s1_addr_norm[s1_id]
        c_addr = pool_addr_norm[cand_id]

        s_ncore_set = s1_name_core[s1_id]
        c_ncore_set = pool_name_core[cand_id]
        s_acore_set = s1_addr_core[s1_id]
        c_acore_set = pool_addr_core[cand_id]

        s_ncore_str = s1_name_core_str[s1_id]
        c_ncore_str = pool_name_core_str[cand_id]

        s_first = s1_name_tokens[s1_id][0] if s1_name_tokens[s1_id] else ""
        c_first = pool_name_tokens[cand_id][0] if pool_name_tokens[cand_id] else ""

        s_post = s1_postal[s1_id]
        c_post = pool_postal[cand_id]

        s_dig = s1_addr_digits[s1_id]
        c_dig = pool_addr_digits[cand_id]

        n_r = fuzz_ratio(s_name, c_name) / 100.0 if s_name and c_name else 0.0
        a_r = fuzz_ratio(s_addr, c_addr) / 100.0 if s_addr and c_addr else 0.0

        has_both_post = bool(s_post) and bool(c_post)
        has_both_dig = bool(s_dig) and bool(c_dig)
        s_len_n, c_len_n = len(s_name), len(c_name)

        # 0: blocking_score
        mat[i, 0] = block_score
        # 1: name_ratio
        mat[i, 1] = n_r
        # 2: name_partial_ratio
        mat[i, 2] = fuzz_partial(s_name, c_name) / 100.0 if s_name and c_name else 0.0
        # 3: name_token_sort_ratio
        mat[i, 3] = fuzz_token_sort(s_name, c_name) / 100.0 if s_name and c_name else 0.0
        # 4: name_token_set_ratio
        mat[i, 4] = fuzz_token_set(s_name, c_name) / 100.0 if s_name and c_name else 0.0
        # 5: name_w_ratio
        mat[i, 5] = fuzz_w_ratio(s_name, c_name) / 100.0 if s_name and c_name else 0.0
        # 6: name_core_ratio
        mat[i, 6] = fuzz_ratio(s_ncore_str, c_ncore_str) / 100.0 if s_ncore_str and c_ncore_str else 0.0
        # 7: name_core_token_set_ratio
        mat[i, 7] = fuzz_token_set(s_ncore_str, c_ncore_str) / 100.0 if s_ncore_str and c_ncore_str else 0.0
        # 8: name_jaccard
        mat[i, 8] = _jaccard(s_ncore_set, c_ncore_set)
        # 9: name_dice
        mat[i, 9] = _dice(s_ncore_set, c_ncore_set)
        # 10: name_overlap
        mat[i, 10] = _overlap(s_ncore_set, c_ncore_set)
        # 11: name_exact
        mat[i, 11] = 1.0 if (s_name == c_name and s_name != "") else 0.0
        # 12: name_first_token_match
        mat[i, 12] = 1.0 if (s_first == c_first and s_first != "") else 0.0
        # 13: name_len_diff
        mat[i, 13] = abs(s_len_n - c_len_n)
        # 14: name_len_ratio
        mat[i, 14] = min(s_len_n, c_len_n) / max(s_len_n, c_len_n, 1)
        # 15: name_missing
        mat[i, 15] = 1.0 if (not s_name or not c_name) else 0.0
        # 16: addr_ratio
        mat[i, 16] = a_r
        # 17: addr_partial_ratio
        mat[i, 17] = fuzz_partial(s_addr, c_addr) / 100.0 if s_addr and c_addr else 0.0
        # 18: addr_token_sort_ratio
        mat[i, 18] = fuzz_token_sort(s_addr, c_addr) / 100.0 if s_addr and c_addr else 0.0
        # 19: addr_token_set_ratio
        mat[i, 19] = fuzz_token_set(s_addr, c_addr) / 100.0 if s_addr and c_addr else 0.0
        # 20: addr_w_ratio
        mat[i, 20] = fuzz_w_ratio(s_addr, c_addr) / 100.0 if s_addr and c_addr else 0.0
        # 21: addr_jaccard
        mat[i, 21] = _jaccard(s_acore_set, c_acore_set)
        # 22: addr_dice
        mat[i, 22] = _dice(s_acore_set, c_acore_set)
        # 23: addr_overlap
        mat[i, 23] = _overlap(s_acore_set, c_acore_set)
        # 24: addr_exact
        mat[i, 24] = 1.0 if (s_addr == c_addr and s_addr != "") else 0.0
        # 25: addr_len_diff
        mat[i, 25] = abs(len(s_addr) - len(c_addr))
        # 26: addr_missing
        mat[i, 26] = 1.0 if (not s_addr or not c_addr) else 0.0
        # 27: postal_match
        mat[i, 27] = 1.0 if (has_both_post and s_post == c_post) else 0.0
        # 28: postal_mismatch
        mat[i, 28] = 1.0 if (has_both_post and s_post != c_post) else 0.0
        # 29: postal_both_present
        mat[i, 29] = 1.0 if has_both_post else 0.0
        # 30: digits_jaccard
        mat[i, 30] = _jaccard(s_dig, c_dig) if has_both_dig else 0.0
        # 31: digits_mismatch
        mat[i, 31] = 1.0 if (has_both_dig and len(s_dig & c_dig) == 0) else 0.0
        # 32: country_match
        mat[i, 32] = 1.0 if (s1_country[s1_id] == pool_country[cand_id]) else 0.0
        # 33: max_ratio
        mat[i, 33] = max(n_r, a_r)
        # 34: mean_ratio
        mat[i, 34] = 0.5 * (n_r + a_r)
        # 35: is_s2
        mat[i, 35] = 1.0 if str(cand_id).startswith("S2-") else 0.0

    return mat, s1_ids, cand_ids


def build_feature_frame(pairs, s1_store, pool_lookup, num_workers=8):
    """
    pairs: iterable of (s1_id, cand_id, blocking_score)
    s1_store: RecordStore for Source-1
    pool_lookup: MergedPool instance

    Returns a pandas DataFrame with one row per pair: source1_entity_id,
    candidate_entity_id, and all FEATURE_COLUMNS.
    """
    if not pairs:
        df = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"] + FEATURE_COLUMNS)
        return df

    if len(pairs) < 2000 or num_workers <= 1:
        rows, s1_ids, cand_ids = _process_pair_slice(pairs, s1_store, pool_lookup)
    else:
        chunk_sz = (len(pairs) + num_workers - 1) // num_workers
        slices = [pairs[i:i + chunk_sz] for i in range(0, len(pairs), chunk_sz)]
        rows, s1_ids, cand_ids = [], [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_pair_slice, sl, s1_store, pool_lookup) for sl in slices]
            for fut in futures:
                r, s1, c = fut.result()
                rows.extend(r)
                s1_ids.extend(s1)
                cand_ids.extend(c)

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    df.insert(0, "source1_entity_id", s1_ids)
    df.insert(1, "candidate_entity_id", cand_ids)
    return df


class MergedPool:
    """Merges several RecordStore objects into one lookup."""

    def __init__(self, stores):
        self.name_norm, self.name_tokens, self.name_core, self.name_core_str = {}, {}, {}, {}
        self.addr_norm, self.addr_tokens, self.addr_core, self.postal, self.addr_digits = {}, {}, {}, {}, {}
        self.country = {}
        for s in stores:
            self.name_norm.update(s.name_norm)
            self.name_tokens.update(s.name_tokens)
            self.name_core.update(s.name_core)
            self.name_core_str.update(s.name_core_str)
            self.addr_norm.update(s.addr_norm)
            self.addr_tokens.update(s.addr_tokens)
            self.addr_core.update(s.addr_core)
            self.postal.update(s.postal)
            self.addr_digits.update(s.addr_digits)
            self.country.update(s.country)

