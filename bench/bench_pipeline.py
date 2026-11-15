"""One-line JSON stats wrapper for repeatable pipeline benchmarks."""

import json
import os

from curation_lab.filters import FilterConfig
from curation_lab.pipeline import PipelineConfig, run

DATA_DIR = os.environ.get("DATA_DIR", "bench/data")
REPEAT = os.environ.get("REPEAT", "0")
NEAR_DEDUP = os.environ.get("NEAR_DEDUP", "1") != "0"
TRANSITIVE = os.environ.get("TRANSITIVE", "1") != "0"
EVAL = os.environ.get("EVAL", f"{DATA_DIR}/eval.jsonl")
eval_path = None if EVAL.lower() in ("", "0", "none") else EVAL

config = PipelineConfig(
    filters=FilterConfig(min_chars=50),
    near_dedup=NEAR_DEDUP,
    transitive_dedup=TRANSITIVE,
    num_perm=128,
    bands=16,
    rows=8,
    ngram=5,
    decontam_ngram=13,
    decontam_threshold=0.5,
)
stats = run(
    f"{DATA_DIR}/corpus.jsonl",
    f"{DATA_DIR}/out.jsonl",
    eval_path=eval_path,
    config=config,
)
stats["repeat"] = REPEAT
print(json.dumps(stats))
