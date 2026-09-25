"""
Train the matching classifier end-to-end from the training TSVs.

Usage:
    python -m src.train \
        --train-dir dataset/train \
        --model-out models/model.txt \
        --threshold-out models/threshold.json

Steps:
    1. Load + normalize train_source1/2/3.tsv and train_ground_truth.tsv.
    2. Run blocking to get candidate_pairs for the training pool (also
       reports blocking recall -- the ceiling on achievable recall).
    3. Label each candidate pair: positive if in ground truth, else negative.
    4. Split by Source-1 entity id (not by row) into train/val.
    5. Train a LightGBM binary classifier with early stopping on val AUC.
    6. Grid-search the decision threshold on val to maximize macro F0.5
       (the actual leaderboard metric), evaluated over ALL val Source-1
       entities (including ones blocking never produced candidates for).
    7. Save the model and chosen threshold.
"""
import argparse
import json
import os
import random

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from .blocking import generate_candidates
from .data import load_source, load_ground_truth
from .features import build_feature_frame, MergedPool, FEATURE_COLUMNS
from .metrics import macro_f_beta


def build_labeled_pairs(candidates, ground_truth, max_negatives_per_entity):
    """Return list[(s1_id, cand_id, block_score, label)]."""
    rng = random.Random(config.RANDOM_SEED)
    pairs = []
    recall_hits, recall_total = 0, 0

    for s1_id, cand_list in candidates.items():
        true_matches = ground_truth.get(s1_id, set())
        recall_total += len(true_matches)
        cand_ids_in_pool = {c for c, _ in cand_list}
        recall_hits += len(true_matches & cand_ids_in_pool)

        pos, neg = [], []
        for cand_id, score in cand_list:
            if cand_id in true_matches:
                pos.append((s1_id, cand_id, score, 1))
            else:
                neg.append((s1_id, cand_id, score, 0))

        if max_negatives_per_entity is not None and len(neg) > max_negatives_per_entity:
            neg = rng.sample(neg, max_negatives_per_entity)

        pairs.extend(pos)
        pairs.extend(neg)

    recall = recall_hits / recall_total if recall_total else float("nan")
    print(f"[train] Blocking recall on training pool: {recall_hits}/{recall_total} = {recall:.4f}")
    return pairs


def run(train_dir, model_out, threshold_out, max_candidates, max_negatives, val_fraction):
    print("[train] Loading sources...")
    s1 = load_source(os.path.join(train_dir, "train_source1.tsv"))
    s2 = load_source(os.path.join(train_dir, "train_source2.tsv"))
    s3 = load_source(os.path.join(train_dir, "train_source3.tsv"))
    gt = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))
    print(f"[train] S1={len(s1)} S2={len(s2)} S3={len(s3)} ground-truth rows={len(gt)}")

    print("[train] Splitting Source-1 entities into train/val...")
    rng = random.Random(config.RANDOM_SEED)
    s1_ids = list(s1.ids)
    rng.shuffle(s1_ids)
    n_val = max(1, int(len(s1_ids) * val_fraction))
    val_ids = set(s1_ids[:n_val])
    train_ids = set(s1_ids[n_val:])
    print(f"[train] train S1 entities={len(train_ids)} val S1 entities={len(val_ids)}")

    print("[train] Running blocking on full training pool...")
    candidates = generate_candidates(s1, [s2, s3], max_candidates=max_candidates)

    train_candidates = {k: v for k, v in candidates.items() if k in train_ids}
    val_candidates = {k: v for k, v in candidates.items() if k in val_ids}

    print("[train] Building labeled pairs (train split)...")
    train_pairs = build_labeled_pairs(train_candidates, gt, max_negatives)
    print("[train] Building labeled pairs (val split, unsampled)...")
    val_pairs = build_labeled_pairs(val_candidates, gt, None)  # keep all for honest eval

    pool = MergedPool([s2, s3])

    def to_feature_df(pair_list):
        triples = [(a, b, c) for a, b, c, _ in pair_list]
        labels = [lbl for _, _, _, lbl in pair_list]
        fdf = build_feature_frame(triples, s1, pool)
        fdf["label"] = labels
        return fdf

    print("[train] Computing features...")
    train_df = to_feature_df(train_pairs)
    val_df = to_feature_df(val_pairs)
    print(f"[train] train pairs={len(train_df)} (pos={train_df['label'].sum()}) "
          f"val pairs={len(val_df)} (pos={val_df['label'].sum()})")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df["label"]

    pos = max(int(y_train.sum()), 1)
    neg = max(len(y_train) - pos, 1)
    params = dict(config.LGB_PARAMS)
    params["scale_pos_weight"] = neg / pos

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLUMNS)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLUMNS, reference=train_set)

    print("[train] Training LightGBM...")
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=config.LGB_NUM_BOOST_ROUND,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(config.LGB_EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    print("[train] Scoring val candidates for threshold search...")
    val_df = val_df.copy()
    val_df["prob"] = booster.predict(X_val, num_iteration=booster.best_iteration)

    best_thresh, best_f = 0.5, -1.0
    grid = np.arange(config.THRESHOLD_GRID_START, config.THRESHOLD_GRID_STOP, config.THRESHOLD_GRID_STEP)
    for t in grid:
        preds = {}
        for s1_id, group in val_df[val_df["prob"] >= t].groupby("source1_entity_id"):
            preds[s1_id] = set(group["candidate_entity_id"])
        score = macro_f_beta(preds, gt, val_ids, beta=config.F_BETA)
        if score > best_f:
            best_f, best_thresh = score, float(t)

    print(f"[train] Best val macro F{config.F_BETA}: {best_f:.4f} at threshold={best_thresh:.2f}")

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    booster.save_model(model_out)
    with open(threshold_out, "w") as f:
        json.dump(
            {
                "threshold": best_thresh,
                "val_macro_f_beta": best_f,
                "beta": config.F_BETA,
                "max_candidates": max_candidates,
                "feature_columns": FEATURE_COLUMNS,
            },
            f,
            indent=2,
        )
    print(f"[train] Saved model to {model_out} and threshold to {threshold_out}")

    print("[train] Feature importance (gain):")
    importances = booster.feature_importance(importance_type="gain")
    for name, imp in sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1]):
        print(f"    {name}: {imp:.1f}")


def main():
    p = argparse.ArgumentParser(description="Train the entity-matching classifier.")
    p.add_argument("--train-dir", default="dataset/train")
    p.add_argument("--model-out", default="models/model.txt")
    p.add_argument("--threshold-out", default="models/threshold.json")
    p.add_argument("--max-candidates", type=int, default=config.MAX_CANDIDATES_PER_ENTITY)
    p.add_argument("--max-negatives", type=int, default=config.MAX_NEGATIVES_PER_ENTITY)
    p.add_argument("--val-fraction", type=float, default=config.VAL_FRACTION)
    args = p.parse_args()
    run(args.train_dir, args.model_out, args.threshold_out, args.max_candidates,
        args.max_negatives, args.val_fraction)


if __name__ == "__main__":
    main()