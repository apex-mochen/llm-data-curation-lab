"""One-line JSON stats wrapper for repeatable pipeline benchmarks."""

import json
import os

from curation_lab.filters import FilterConfig
from curation_lab.pipeline import PipelineConfig, run

DATA_DIR = os.environ.get("DATA_DIR", "bench/data")
REPEAT = os.environ.get("REPEAT", "0")

config = PipelineConfig(
    filters=FilterConfig(min_chars=50),
    near_dedup=True,
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
    eval_path=f"{DATA_DIR}/eval.jsonl",
    config=config,
)
stats["repeat"] = REPEAT
print(json.dumps(stats))
