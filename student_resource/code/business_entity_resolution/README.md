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
