"""
Macro F-beta scoring, matching the challenge's evaluation exactly: computed
per Source-1 entity (empty prediction vs empty truth = 1.0), then averaged
across all Source-1 entities in the evaluation set.
"""


def f_beta_per_entity(pred: set, true: set, beta: float = 0.5) -> float:
    if not true and not pred:
        return 1.0
    if not pred:
        return 0.0
    if not true:
        return 0.0  # any predicted match on a true singleton scores 0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(true)
    b2 = beta ** 2
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def macro_f_beta(predictions: dict, ground_truth: dict, all_s1_ids, beta: float = 0.5) -> float:
    """
    predictions: dict[s1_id] -> set(matched_ids) (missing key treated as empty)
    ground_truth: dict[s1_id] -> set(true_ids) (missing key treated as empty)
    all_s1_ids: the full set of Source-1 ids to average over (every entity
        counts, even ones absent from both dicts).
    """
    scores = []
    for s1_id in all_s1_ids:
        pred = predictions.get(s1_id, set())
        true = ground_truth.get(s1_id, set())
        scores.append(f_beta_per_entity(pred, true, beta))
    return sum(scores) / len(scores) if scores else 0.0