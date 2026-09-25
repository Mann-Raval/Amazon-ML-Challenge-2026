"""
Central configuration for the Business Entity Resolution pipeline.
Tune these values here rather than hardcoding them inside the modules.
"""

# ---- Blocking ----
# Max fraction of records a name/address token may appear in before it is
# considered "too common" to be useful for blocking (e.g. "private", "road").
MAX_TOKEN_DOC_FREQ = 0.02
# Absolute cap on postings list length per token, as a second safety net on
# top of MAX_TOKEN_DOC_FREQ for very large datasets.
MAX_TOKEN_POSTINGS = 2000
# Minimum postings-list cap regardless of MAX_TOKEN_DOC_FREQ, so small country
# partitions (e.g. a smaller country's slice of the data) don't have every
# token dropped as "too common" just because the partition itself is small.
MIN_POSTINGS_FLOOR = 25
# How many candidates to keep per Source-1 entity after ranking (this is the
# set written to candidate_pairs.tsv and fed to the classifier).
MAX_CANDIDATES_PER_ENTITY = 50
# Relative weight of name vs address token-overlap in the blocking rank score.
BLOCKING_NAME_WEIGHT = 0.7
BLOCKING_ADDRESS_WEIGHT = 0.3
# Minimum token length to index (drop 1-2 char tokens, too noisy/common).
MIN_TOKEN_LEN = 3

# ---- Train/val split ----
# Fraction of Source-1 training entities held out for validation (split by
# entity, never by row, to avoid leaking a pair's sibling into both sets).
VAL_FRACTION = 0.15
RANDOM_SEED = 42

# ---- Negative sampling ----
# Cap on negative (non-matching) candidate pairs kept per Source-1 entity
# during training, to keep the training set from being dominated by easy
# negatives from entities with huge candidate pools. None = keep all.
MAX_NEGATIVES_PER_ENTITY = 30

# ---- LightGBM ----
LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": RANDOM_SEED,
}
LGB_NUM_BOOST_ROUND = 2000
LGB_EARLY_STOPPING_ROUNDS = 50

# ---- Threshold search (maximizing macro F0.5 on validation) ----
THRESHOLD_GRID_START = 0.05
THRESHOLD_GRID_STOP = 0.95
THRESHOLD_GRID_STEP = 0.01

# beta for F-beta scoring (challenge uses 0.5, precision-weighted)
F_BETA = 0.5