"""
Run inference on the test set: blocking -> feature scoring -> thresholding,
and write both required output files.

Usage:
    python -m src.predict \
        --test-dir dataset/test \
        --model models/model.txt \
        --threshold-file models/threshold.json \
        --output-dir output
"""
import argparse
import gc
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from .blocking import generate_candidates
from .data import load_source
from .features import compute_features_matrix, MergedPool, FEATURE_COLUMNS


def write_id_list_tsv(path, header, mapping, all_s1_ids):
    """mapping: dict[s1_id] -> list/set of ids (already deduped)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for s1_id in all_s1_ids:
            ids = mapping.get(s1_id, [])
            seen = set()
            ordered = []
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    ordered.append(i)
            f.write(f"{s1_id}\t{','.join(ordered)}\n")


def run(test_dir, model_path, threshold_file, output_dir, max_candidates_override=None, batch_size=200_000):
    print("[predict] Loading test sources...", flush=True)
    s1 = load_source(os.path.join(test_dir, "test_source1.tsv"))
    s2 = load_source(os.path.join(test_dir, "test_source2.tsv"))
    s3 = load_source(os.path.join(test_dir, "test_source3.tsv"))
    print(f"[predict] S1={len(s1)} S2={len(s2)} S3={len(s3)}", flush=True)

    with open(threshold_file) as f:
        meta = json.load(f)
    threshold = meta["threshold"]
    max_candidates = max_candidates_override or meta.get("max_candidates", config.MAX_CANDIDATES_PER_ENTITY)
    print(f"[predict] Using threshold={threshold} max_candidates={max_candidates}", flush=True)

    print("[predict] Running blocking on test pool...", flush=True)
    candidates = generate_candidates(s1, [s2, s3], max_candidates=max_candidates)

    # candidate_pairs.tsv -- the exact set fed to the model.
    candidate_mapping = {s1_id: [c for c, _ in lst] for s1_id, lst in candidates.items()}
    n_with_candidates = sum(1 for v in candidate_mapping.values() if v)
    print(f"[predict] {n_with_candidates}/{len(s1)} S1 entities have >=1 candidate.", flush=True)

    pairs = [(s1_id, cand_id, score) for s1_id, lst in candidates.items() for cand_id, score in lst]
    print(f"[predict] Scoring {len(pairs)} candidate pairs...", flush=True)
    pool = MergedPool([s2, s3])
    booster = lgb.Booster(model_file=model_path)

    matching_mapping = {}

    # Score in batches to avoid high memory spikes
    for start_idx in range(0, len(pairs), batch_size):
        end_idx = min(start_idx + batch_size, len(pairs))
        batch_pairs = pairs[start_idx:end_idx]
        mat, b_s1_ids, b_cand_ids = compute_features_matrix(batch_pairs, s1, pool)
        if len(mat):
            probs = booster.predict(mat)
            above_idx = np.where(probs >= threshold)[0]
            for idx in above_idx:
                s1_id = b_s1_ids[idx]
                cid = b_cand_ids[idx]
                if s1_id not in matching_mapping:
                    matching_mapping[s1_id] = []
                matching_mapping[s1_id].append(cid)
        print(f"[predict] Processed {end_idx:,}/{len(pairs):,} pairs", flush=True)
        gc.collect()

    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    write_id_list_tsv(matching_path, ["source1_entity_id", "matched_entity_ids"], matching_mapping, s1.ids)
    write_id_list_tsv(candidate_path, ["source1_entity_id", "candidate_entity_ids"], candidate_mapping, s1.ids)

    n_matched = sum(1 for v in matching_mapping.values() if v)
    print(f"[predict] {n_matched}/{len(s1)} S1 entities predicted >=1 match.", flush=True)
    print(f"[predict] Wrote {matching_path} and {candidate_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Run inference and write submission files.")
    p.add_argument("--test-dir", default="dataset/test")
    p.add_argument("--model", default="models/model.txt")
    p.add_argument("--threshold-file", default="models/threshold.json")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--max-candidates", type=int, default=None,
                    help="Override max candidates per entity (default: value used at train time).")
    args = p.parse_args()
    run(args.test_dir, args.model, args.threshold_file, args.output_dir, args.max_candidates)


if __name__ == "__main__":
    main()