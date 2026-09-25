"""
Score a matching_results.tsv-format predictions file against a ground-truth
file of the same shape (source1_entity_id \t matched_entity_ids), using the
exact macro F_0.5 the leaderboard uses. Handy for scoring held-out splits you
carve out of train_ground_truth.tsv yourself.

Usage:
    python -m src.evaluate --predictions output/matching_results.tsv \
        --ground-truth dataset/train/train_ground_truth.tsv
"""
import argparse

from .data import load_ground_truth
from .metrics import macro_f_beta
from . import config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--ground-truth", required=True)
    p.add_argument(
        "--source1",
        default=None,
        help="Optional Source-1 TSV. When supplied, every Source-1 row is included in the macro average, including entities whose prediction and truth are both empty.",
    )
    p.add_argument("--beta", type=float, default=config.F_BETA)
    args = p.parse_args()

    preds = load_ground_truth(args.predictions)  # same two-column shape
    truth = load_ground_truth(args.ground_truth)
    if args.source1:
        all_ids = set(load_source(args.source1).ids)
    else:
        all_ids = set(preds) | set(truth)
        print("Warning: --source1 was not supplied; entities absent from both files cannot be included.")
    score = macro_f_beta(preds, truth, all_ids, beta=args.beta)
    print(f"Macro F_{args.beta} over {len(all_ids)} entities: {score:.4f}")


if __name__ == "__main__":
    main()
