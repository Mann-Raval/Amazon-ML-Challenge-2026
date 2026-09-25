"""
Loading and pre-normalizing the source TSVs into a common in-memory structure
used by blocking.py, features.py, train.py and predict.py.
"""
import pandas as pd

from . import normalize as norm
from .config import MIN_TOKEN_LEN


class RecordStore:
    """Holds normalized fields for one source file, keyed by entity_id.

    Attributes (all dict[entity_id] -> value):
        name_norm, name_tokens, name_core (set, len-filtered)
        addr_norm, addr_tokens, addr_core (set, len-filtered), postal
        country (raw string, only whitespace/case normalized)
        ids: list of entity_id in file order
    """

    def __init__(self, df: pd.DataFrame):
        required = {"entity_id", "business_name", "business_address", "country"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Input file is missing columns: {missing}")

        self.ids = df["entity_id"].tolist()
        self.name_norm, self.name_tokens, self.name_core = {}, {}, {}
        self.addr_norm, self.addr_tokens, self.addr_core, self.postal = {}, {}, {}, {}
        self.country = {}
        self.raw_name = dict(zip(df["entity_id"], df["business_name"]))
        self.raw_addr = dict(zip(df["entity_id"], df["business_address"]))

        for eid, name, addr, country in zip(
            df["entity_id"], df["business_name"], df["business_address"], df["country"]
        ):
            n_norm, n_tok = norm.normalize_name(name)
            a_norm, a_tok, postal = norm.normalize_address(addr)
            self.name_norm[eid] = n_norm
            self.name_tokens[eid] = n_tok
            self.name_core[eid] = norm.core_tokens(n_tok, MIN_TOKEN_LEN)
            self.addr_norm[eid] = a_norm
            self.addr_tokens[eid] = a_tok
            self.addr_core[eid] = norm.core_tokens(a_tok, MIN_TOKEN_LEN)
            self.postal[eid] = postal
            self.country[eid] = (str(country).strip().lower() if pd.notna(country) else "")

    def __len__(self):
        return len(self.ids)


def load_source(path: str) -> RecordStore:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    df["entity_id"] = df["entity_id"].astype(str)
    return RecordStore(df)


def load_ground_truth(path: str):
    """Return dict[source1_entity_id] -> set(matched_entity_ids) (possibly empty set)."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    gt = {}
    for s1, matches in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        ids = set(x for x in str(matches).split(",") if x) if pd.notna(matches) and matches else set()
        gt[str(s1)] = ids
    return gt