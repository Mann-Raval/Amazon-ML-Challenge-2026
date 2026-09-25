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
import json
import os

import lightgbm as lgb
import pandas as pd

from . import config
from .blocking import generate_candidates
from .data import load_source
from .features import build_feature_frame, MergedPool, FEATURE_COLUMNS


def write_id_list_tsv(path, header, mapping, all_s1_ids):
    """mapping: dict[s1_id] -> list/set of ids (already deduped)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for s1_id in all_s1_ids:
            ids = mapping.get(s1_id, [])
            # dedupe while preserving order
            seen = set()
            ordered = []
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    ordered.append(i)
            f.write(f"{s1_id}\t{','.join(ordered)}\n")


def run(test_dir, model_path, threshold_file, output_dir, max_candidates_override=None):
    print("[predict] Loading test sources...")
    s1 = load_source(os.path.join(test_dir, "test_source1.tsv"))
    s2 = load_source(os.path.join(test_dir, "test_source2.tsv"))
    s3 = load_source(os.path.join(test_dir, "test_source3.tsv"))
    print(f"[predict] S1={len(s1)} S2={len(s2)} S3={len(s3)}")

    with open(threshold_file) as f:
        meta = json.load(f)
    threshold = meta["threshold"]
    max_candidates = max_candidates_override or meta.get("max_candidates", config.MAX_CANDIDATES_PER_ENTITY)
    print(f"[predict] Using threshold={threshold} max_candidates={max_candidates}")

    print("[predict] Running blocking on test pool...")
    candidates = generate_candidates(s1, [s2, s3], max_candidates=max_candidates)

    # candidate_pairs.tsv -- the exact set fed to the model.
    candidate_mapping = {s1_id: [c for c, _ in lst] for s1_id, lst in candidates.items()}
    n_with_candidates = sum(1 for v in candidate_mapping.values() if v)
    print(f"[predict] {n_with_candidates}/{len(s1)} S1 entities have >=1 candidate.")

    pairs = [(s1_id, cand_id, score) for s1_id, lst in candidates.items() for cand_id, score in lst]
    print(f"[predict] Scoring {len(pairs)} candidate pairs...")
    pool = MergedPool([s2, s3])
    feat_df = build_feature_frame(pairs, s1, pool)

    booster = lgb.Booster(model_file=model_path)
    if len(feat_df):
        feat_df["prob"] = booster.predict(feat_df[FEATURE_COLUMNS])
    else:
        feat_df["prob"] = pd.Series(dtype=float)

    matched = feat_df[feat_df["prob"] >= threshold]
    matching_mapping = {}
    for s1_id, group in matched.groupby("source1_entity_id"):
        matching_mapping[s1_id] = list(group["candidate_entity_id"])

    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    write_id_list_tsv(matching_path, ["source1_entity_id", "matched_entity_ids"], matching_mapping, s1.ids)
    write_id_list_tsv(candidate_path, ["source1_entity_id", "candidate_entity_ids"], candidate_mapping, s1.ids)

    n_matched = sum(1 for v in matching_mapping.values() if v)
    print(f"[predict] {n_matched}/{len(s1)} S1 entities predicted >=1 match.")
    print(f"[predict] Wrote {matching_path} and {candidate_path}")


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