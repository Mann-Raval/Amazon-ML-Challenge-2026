# Business Entity Resolution

This package implements the competition solution using only the supplied TSV files. It normalizes names and addresses, generates country-scoped token candidates, trains a LightGBM pair classifier, tunes a precision-oriented F0.5 threshold, and writes both required output files.

## Run

From this directory, create a Python 3.10+ environment and install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
python -m src.train --train-dir ../../dataset/train --model-out models/model.txt --threshold-out models/threshold.json
python -m src.predict --test-dir ../../dataset/test --model models/model.txt --threshold-file models/threshold.json --output-dir ../../output
```

Validate the generated files from the `student_resource` directory:

```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

For a complete local score, include the full Source-1 file when evaluating a prediction file:

```bash
python -m src.evaluate --predictions predictions.tsv --ground-truth ../../dataset/train/train_ground_truth.tsv --source1 ../../dataset/train/train_source1.tsv
```

Countries are treated as open string labels. No country list is hard-coded, so the test-only France partition is processed by the same code path as training countries.

## Low-memory deterministic baseline

If the full LightGBM pipeline exceeds available RAM, run the SQLite-backed baseline. It indexes Source 2/3 on disk and processes Source 1 one row at a time:

```bash
python -m src.rule_based \
  --source1 ../../dataset/test/test_source1.tsv \
  --source2 ../../dataset/test/test_source2.tsv \
  --source3 ../../dataset/test/test_source3.tsv \
  --matching-out ../../output/matching_results.tsv \
  --candidate-out ../../output/candidate_pairs.tsv \
  --db models/entity_index.sqlite
```

This conservative baseline uses exact normalized names plus matching addresses or postal codes. It is useful for a first valid score and uses much less RAM, while the LightGBM pipeline can recover additional typo and abbreviation matches when sufficient memory is available.
