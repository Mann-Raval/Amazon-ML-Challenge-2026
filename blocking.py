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
    trimmed to max_candidates. score is the weighted-Jaccard blocking score.
    """
    # Merge pool stores into flat lookup dicts keyed by entity_id.
    pool_ids = []
    name_core, addr_core, country = {}, {}, {}
    for store in pool_stores:
        pool_ids.extend(store.ids)
        name_core.update(store.name_core)
        addr_core.update(store.addr_core)
        country.update(store.country)

    name_index = _build_index(pool_ids, name_core, country, MAX_TOKEN_DOC_FREQ, MAX_TOKEN_POSTINGS)
    addr_index = _build_index(pool_ids, addr_core, country, MAX_TOKEN_DOC_FREQ, MAX_TOKEN_POSTINGS)

    candidates = {}
    for s1_id in s1_store.ids:
        c = s1_store.country[s1_id]
        pool = set()
        for tok in s1_store.name_core[s1_id]:
            pool.update(name_index.get((c, tok), []))
        for tok in s1_store.addr_core[s1_id]:
            pool.update(addr_index.get((c, tok), []))

        if not pool:
            candidates[s1_id] = []
            continue

        s1_name_set = s1_store.name_core[s1_id]
        s1_addr_set = s1_store.addr_core[s1_id]
        scored = []
        for cid in pool:
            name_sim = _jaccard(s1_name_set, name_core.get(cid, set()))
            addr_sim = _jaccard(s1_addr_set, addr_core.get(cid, set()))
            score = BLOCKING_NAME_WEIGHT * name_sim + BLOCKING_ADDRESS_WEIGHT * addr_sim
            scored.append((cid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        candidates[s1_id] = scored[:max_candidates]

    return candidates