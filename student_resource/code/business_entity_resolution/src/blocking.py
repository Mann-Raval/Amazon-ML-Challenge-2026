"""
Candidate generation (blocking).

Strategy: token-based inverted index, partitioned by country.
  1. Index every Source-2/3 record under each of its "informative" normalized
     name/address tokens (informative = not too frequent -- see config).
  2. For a Source-1 entity, look up its own tokens in the index to get a
     *candidate pool* (union of postings) cheaply, without comparing against
     every record in the dataset.
  3. Precisely score only that (small) pool with weighted name/address token
     Jaccard and keep the top MAX_CANDIDATES_PER_ENTITY.

This keeps blocking O(records) for the index build and O(pool size) per S1
entity for scoring, instead of the O(n * m) full cross product.

Country is used as a hard partition (index and lookups are scoped per
country) since a business match essentially never crosses countries in this
data, and it keeps postings lists (and therefore blocking latency) small.
The pipeline never filters/hardcodes the *set* of countries -- whatever
string appears in the `country` column (including "France" at test time,
which never appears in training) becomes its own partition automatically.
"""
from collections import defaultdict

from .config import (
    MAX_TOKEN_DOC_FREQ,
    MAX_TOKEN_POSTINGS,
    MAX_CANDIDATES_PER_ENTITY,
    BLOCKING_NAME_WEIGHT,
    BLOCKING_ADDRESS_WEIGHT,
    MIN_POSTINGS_FLOOR,
)


def _build_index(pool_ids, core_token_dict, country_dict, max_doc_freq, max_postings):
    """token -> (country) -> list[entity_id], skipping over-common tokens."""
    raw_postings = defaultdict(list)
    for eid in pool_ids:
        country = country_dict[eid]
        for tok in core_token_dict[eid]:
            raw_postings[(country, tok)].append(eid)

    n_by_country = defaultdict(int)
    for eid in pool_ids:
        n_by_country[country_dict[eid]] += 1

    index = {}
    for key, ids in raw_postings.items():
        country, _tok = key
        # Floor of MIN_POSTINGS_FLOOR keeps small country partitions usable --
        # without it, `max_doc_freq * n` rounds to 0-1 on small partitions and
        # every token gets dropped as "too common".
        cap = min(max_postings, max(MIN_POSTINGS_FLOOR, int(max_doc_freq * max(n_by_country[country], 1))))
        if len(ids) > cap:
            continue  # too common to be informative -- drop, not truncate
        index[key] = ids
    return index


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def generate_candidates(s1_store, pool_stores, max_candidates=MAX_CANDIDATES_PER_ENTITY):
    """
    s1_store: RecordStore for Source-1 entities.
    pool_stores: list of RecordStore (Source-2, Source-3) to draw candidates from.

    Returns: dict[s1_entity_id] -> list[(candidate_id, score)] sorted desc,
    trimmed to max_candidates. score is the blocking score.
    """
    pool_ids = []
    name_core, addr_core, country, postal = {}, {}, {}, {}
    for store in pool_stores:
        pool_ids.extend(store.ids)
        name_core.update(store.name_core)
        addr_core.update(store.addr_core)
        country.update(store.country)
        postal.update(store.postal)

    name_index = _build_index(pool_ids, name_core, country, MAX_TOKEN_DOC_FREQ, MAX_TOKEN_POSTINGS)
    addr_index = _build_index(pool_ids, addr_core, country, MAX_TOKEN_DOC_FREQ, MAX_TOKEN_POSTINGS)

    candidates = {}
    for s1_id in s1_store.ids:
        c = s1_store.country[s1_id]
        
        # Token overlap count mapping for fast candidate pruning
        cand_counts = defaultdict(int)
        for tok in s1_store.name_core[s1_id]:
            for cid in name_index.get((c, tok), ()):
                cand_counts[cid] += 2  # name token match gets higher weight
        for tok in s1_store.addr_core[s1_id]:
            for cid in addr_index.get((c, tok), ()):
                cand_counts[cid] += 1

        if not cand_counts:
            candidates[s1_id] = []
            continue

        # Fast partition candidates by source (S2 and S3)
        s2_pool = []
        s3_pool = []
        for cid, cnt in cand_counts.items():
            if str(cid).startswith("S2-"):
                s2_pool.append((cid, cnt))
            else:
                s3_pool.append((cid, cnt))
        
        # Keep top 10 from each source before jaccard
        if len(s2_pool) > 10:
            s2_pool.sort(key=lambda x: x[1], reverse=True)
            s2_pool = s2_pool[:10]
        if len(s3_pool) > 10:
            s3_pool.sort(key=lambda x: x[1], reverse=True)
            s3_pool = s3_pool[:10]

        eval_pool = [cid for cid, _ in s2_pool] + [cid for cid, _ in s3_pool]

        s1_name_set = s1_store.name_core[s1_id]
        s1_addr_set = s1_store.addr_core[s1_id]
        s1_p = s1_store.postal[s1_id]
        
        s2_scored = []
        s3_scored = []
        for cid in eval_pool:
            c_name_set = name_core.get(cid, set())
            c_addr_set = addr_core.get(cid, set())
            name_sim = _jaccard(s1_name_set, c_name_set)
            addr_sim = _jaccard(s1_addr_set, c_addr_set)
            
            weighted = BLOCKING_NAME_WEIGHT * name_sim + BLOCKING_ADDRESS_WEIGHT * addr_sim
            score = max(weighted, name_sim * 0.9, addr_sim * 0.85)
            
            # Postal match bonus
            c_p = postal.get(cid)
            if s1_p and c_p and s1_p == c_p:
                score += 0.05
            
            if str(cid).startswith("S2-"):
                s2_scored.append((cid, score))
            else:
                s3_scored.append((cid, score))

        top_cands = []
        if s2_scored:
            s2_scored.sort(key=lambda x: x[1], reverse=True)
            top_cands.append(s2_scored[0])
        if s3_scored:
            s3_scored.sort(key=lambda x: x[1], reverse=True)
            top_cands.append(s3_scored[0])

        candidates[s1_id] = top_cands

    return candidates