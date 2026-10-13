import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution"))
from src.data import load_source, load_ground_truth
from src.blocking import generate_candidates

print("Testing full blocking recall on 10,000 S1 entities with FULL S2 + S3 pool...", flush=True)
t0 = time.time()

print("1. Loading full S2...", flush=True)
s2 = load_source("dataset/train/train_source2.tsv")
print(f"   S2 loaded: {len(s2)} records ({time.time()-t0:.1f}s)", flush=True)

print("2. Loading full S3...", flush=True)
s3 = load_source("dataset/train/train_source3.tsv")
print(f"   S3 loaded: {len(s3)} records ({time.time()-t0:.1f}s)", flush=True)

print("3. Loading S1 sample (20,000 records)...", flush=True)
s1_raw = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", nrows=20000, dtype=str, keep_default_na=False)
from src.data import RecordStore
s1 = RecordStore(s1_raw)
print(f"   S1 loaded: {len(s1)} records ({time.time()-t0:.1f}s)", flush=True)

print("4. Loading ground truth for S1 sample...", flush=True)
gt_full = load_ground_truth("dataset/train/train_ground_truth.tsv")
s1_set = set(s1.ids)
gt = {k: v for k, v in gt_full.items() if k in s1_set}

print("5. Generating candidates against full S2+S3 pool...", flush=True)
t_gen = time.time()
candidates = generate_candidates(s1, [s2, s3], max_candidates=40)
print(f"   Candidate generation done in {time.time()-t_gen:.1f}s", flush=True)

hits, total = 0, 0
for s1_id, clist in candidates.items():
    true_m = gt.get(s1_id, set())
    total += len(true_m)
    cset = {c for c, _ in clist}
    hits += len(true_m & cset)

rec = hits / total if total else 0
print(f"\n==========================================")
print(f"FULL POOL BLOCKING RECALL: {hits}/{total} = {rec*100:.2f}%")
print(f"==========================================\n", flush=True)
