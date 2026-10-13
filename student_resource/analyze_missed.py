import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution"))
from src.data import load_source, load_ground_truth, RecordStore
from src.blocking import generate_candidates

s1_raw = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", nrows=2000, dtype=str, keep_default_na=False)
s1 = RecordStore(s1_raw)
s2_raw = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", nrows=100000, dtype=str, keep_default_na=False)
s2 = RecordStore(s2_raw)
s3_raw = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", nrows=100000, dtype=str, keep_default_na=False)
s3 = RecordStore(s3_raw)

gt_full = load_ground_truth("dataset/train/train_ground_truth.tsv")
s1_set = set(s1.ids)
gt = {k: v for k, v in gt_full.items() if k in s1_set}

# Find pool of available matches
available_cands = set(s2.ids) | set(s3.ids)

candidates = generate_candidates(s1, [s2, s3], max_candidates=50)

missed = []
for s1_id, clist in candidates.items():
    true_m = gt.get(s1_id, set()) & available_cands
    cset = {c for c, _ in clist}
    for m in (true_m - cset):
        missed.append((s1_id, m))

print(f"Total missed available matches in sample: {len(missed)}")
for s1_id, m in missed[:10]:
    s1_name = s1.raw_name[s1_id]
    s1_addr = s1.raw_addr[s1_id]
    c_store = s2 if m.startswith("S2-") else s3
    c_name = c_store.raw_name[m]
    c_addr = c_store.raw_addr[m]
    print("--------------------------------------------------")
    print(f"S1:   Name='{s1_name}' | Addr='{s1_addr}'")
    print(f"Cand: Name='{c_name}' | Addr='{c_addr}'")
