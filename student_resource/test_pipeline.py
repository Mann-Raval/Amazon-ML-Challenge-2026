import json
import os
import random
import time
import lightgbm as lgb
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution"))

from src import config
from src.blocking import generate_candidates
from src.data import load_source, load_ground_truth, RecordStore
from src.features import build_feature_frame, MergedPool, FEATURE_COLUMNS
from src.metrics import macro_f_beta

print("Starting test pipeline...", flush=True)
t0 = time.time()

# Load 50,000 S1 records
s1_raw = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", nrows=50000, dtype=str, keep_default_na=False)
s1 = RecordStore(s1_raw)
print(f"Loaded S1: {len(s1)} records ({time.time()-t0:.1f}s)", flush=True)

# Load S2 and S3 pool (250,000 each for fast test)
s2_raw = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", nrows=250000, dtype=str, keep_default_na=False)
s2 = RecordStore(s2_raw)
s3_raw = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", nrows=250000, dtype=str, keep_default_na=False)
s3 = RecordStore(s3_raw)
print(f"Loaded S2: {len(s2)}, S3: {len(s3)} ({time.time()-t0:.1f}s)", flush=True)

gt_full = load_ground_truth("dataset/train/train_ground_truth.tsv")
s1_set = set(s1.ids)
gt = {k: v for k, v in gt_full.items() if k in s1_set}
print(f"Filtered GT: {len(gt)} entities ({time.time()-t0:.1f}s)", flush=True)

# Split into train/val
rng = random.Random(42)
s1_ids = list(s1.ids)
rng.shuffle(s1_ids)
n_val = int(len(s1_ids) * 0.2)
val_ids = set(s1_ids[:n_val])
train_ids = set(s1_ids[n_val:])
print(f"Train S1: {len(train_ids)}, Val S1: {len(val_ids)}", flush=True)

# Generate candidates
print("Generating candidates...", flush=True)
t_block = time.time()
candidates = generate_candidates(s1, [s2, s3], max_candidates=40)
print(f"Candidate generation done in {time.time()-t_block:.1f}s", flush=True)

# Labeled pairs
def get_pairs(cands, max_neg=25):
    pairs = []
    hits, total = 0, 0
    for s1_id, clist in cands.items():
        true_m = gt.get(s1_id, set())
        total += len(true_m)
        cset = {c for c, _ in clist}
        hits += len(true_m & cset)
        pos = [(s1_id, c, sc, 1) for c, sc in clist if c in true_m]
        neg = [(s1_id, c, sc, 0) for c, sc in clist if c not in true_m]
        if max_neg is not None and len(neg) > max_neg:
            neg = rng.sample(neg, max_neg)
        pairs.extend(pos)
        pairs.extend(neg)
    rec = hits / total if total else 0
    print(f"Blocking recall on sample: {hits}/{total} = {rec:.4f}", flush=True)
    return pairs

train_cands = {k: v for k, v in candidates.items() if k in train_ids}
val_cands = {k: v for k, v in candidates.items() if k in val_ids}

train_pairs = get_pairs(train_cands, max_neg=20)
val_pairs = get_pairs(val_cands, max_neg=None)

pool = MergedPool([s2, s3])
print(f"Computing features for {len(train_pairs)} train pairs, {len(val_pairs)} val pairs...", flush=True)
t_feat = time.time()

def to_feat_df(plist):
    triples = [(a, b, c) for a, b, c, _ in plist]
    labels = [lbl for _, _, _, lbl in plist]
    fdf = build_feature_frame(triples, s1, pool)
    fdf["label"] = labels
    return fdf

train_df = to_feat_df(train_pairs)
val_df = to_feat_df(val_pairs)
print(f"Features computed in {time.time()-t_feat:.1f}s", flush=True)

X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
X_val, y_val = val_df[FEATURE_COLUMNS], val_df["label"]

pos = max(int(y_train.sum()), 1)
neg = max(len(y_train) - pos, 1)
params = dict(config.LGB_PARAMS)
params["scale_pos_weight"] = neg / pos

train_set = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLUMNS)
val_set = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLUMNS, reference=train_set)

booster = lgb.train(
    params,
    train_set,
    num_boost_round=1000,
    valid_sets=[val_set],
    callbacks=[lgb.early_stopping(30, verbose=True), lgb.log_evaluation(50)],
)

val_df = val_df.copy()
val_df["prob"] = booster.predict(X_val, num_iteration=booster.best_iteration)

best_t, best_f = 0.5, -1.0
for t in np.arange(0.1, 0.95, 0.02):
    preds = {}
    for s1_id, group in val_df[val_df["prob"] >= t].groupby("source1_entity_id"):
        preds[s1_id] = set(group["candidate_entity_id"])
    score = macro_f_beta(preds, gt, val_ids, beta=0.5)
    if score > best_f:
        best_f, best_t = score, float(t)

print(f"\n==========================================")
print(f"RESULT: Best Val Macro F0.5 = {best_f:.4f} at threshold = {best_t:.2f}")
print(f"==========================================\n")
