# ML Challenge 2026: Business Entity Resolution

**Team Name:** [Fill in team name]  
**Team Members:** [Fill in team members]  
**Submission Date:** 2026-09-25

## 1. Executive Summary

The solution uses deterministic, offline normalization followed by country-scoped token blocking and a LightGBM binary classifier. Name, address, postal-code, country, and blocking features are combined, and the decision threshold is selected on a held-out Source-1 validation split to optimize macro F0.5. The country partition is open-set and therefore does not exclude France at test time.

## 2. Methodology

### 2.1 Problem Analysis

The data contains punctuation and case variation, legal suffix and address abbreviations, reordered tokens, missing address components, URLs and landmark text, and transliterated scripts. Source 1 can have zero, one, or multiple matches in Sources 2 and 3, so the model must allow an empty prediction.

### 2.2 Solution Strategy

**Approach Type:** Blocking + supervised pair classifier  
**Core Innovation:** A conservative, open-country candidate pipeline that combines normalized token retrieval with precision-oriented threshold selection.

## 3. Candidate Generation (Blocking)

Names and addresses are lowercased, transliterated, stripped of URLs and punctuation, and expanded using common business/address abbreviations. The Source-2/3 pool is indexed by informative name and address tokens within the normalized country label. Overly frequent tokens are ignored. For each Source-1 row, the union of name and address postings is ranked by weighted token Jaccard similarity and capped at the configured candidate limit.

The candidate file is exactly the set passed to the classifier. The training command reports blocking recall against the supplied ground truth; this value is the upper bound for the final matcher and should be recorded from the run.

## 4. Matching Model

Features include RapidFuzz name ratios, token-sort/set ratios, normalized token Jaccard, exact and first-token indicators, address ratios/Jaccard, postal-code equality, country equality, and length differences. The model is LightGBM binary classification with class weighting and early stopping. The threshold is grid-searched on validation entities using the competition's macro F0.5 metric, including entities with no candidates.

## 5. Results & Error Analysis

- **F0.5 Score (macro):** Record the value printed by `src.train` after execution.
- **Blocking recall:** Record the value printed by `src.train`.
- **Common false positives:** Similar generic business names or short addresses with insufficient distinguishing tokens.
- **Common false negatives:** Matches whose name and address share no surviving indexed token, transliteration/typo pairs, and candidates beyond the cap.

## 6. Conclusion

The pipeline is fully offline and reproducible, produces an explicit candidate set, and preserves singleton predictions through thresholding. Open-set country handling applies the same logic to France without a country-specific rule.

## Appendix A. Code Artefacts

The runnable package is under `code/business_entity_resolution/`. `src/train.py` creates the model and threshold; `src/predict.py` writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`; `requirements.txt` pins dependencies.

## Appendix B. Additional Results

Add per-country validation results and representative false-positive/false-negative examples from the final training run.
