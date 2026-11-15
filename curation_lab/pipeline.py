"""Streaming end-to-end curation pipeline."""

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional

from .decontam import NGramDecontaminator
from .dedup import ExactDedup, MinHashLSHDedup
from .filters import FilterConfig, check, normalize_text
from .lsh_params import optimal_lsh_params
from .simhash_dedup import SimHashLSHDedup


@dataclass
class PipelineConfig:
    text_key: str = "text"
    filters: FilterConfig = field(default_factory=FilterConfig)
    near_dedup: bool = True
    near_algo: str = "minhash"
    transitive_dedup: bool = True
    num_perm: int = 128
    bands: int = 16
    rows: int = 8
    jaccard_threshold: Optional[float] = None
    simhash_hamming: int = 3
    ngram: int = 5
    mode: str = "word"
    decontam_ngram: int = 13
    decontam_threshold: float = 0.5


def read_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _peak_rss_sampler():
    try:
        import psutil
    except ImportError:
        return None, None, None
    proc = psutil.Process()
    peak = [proc.memory_info().rss]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.005)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    return peak, stop, thread


def run(
    input_path: str,
    output_path: str,
    eval_path: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
    limit: Optional[int] = None,
) -> Dict:
    """Filter, deduplicate and decontaminate a JSONL corpus.

    Returns a stats dict and writes the kept records to ``output_path``.
    """
    config = config or PipelineConfig()
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    exact = ExactDedup()
    lsh_params = {}
    near = None
    if config.near_dedup:
        if config.near_algo == "minhash":
            lsh_params = {
                "bands": config.bands,
                "rows": config.rows,
                "s_curve_threshold": round((1.0 / config.bands) ** (1.0 / config.rows), 6),
            }
            if config.jaccard_threshold is not None:
                lsh_params = optimal_lsh_params(config.jaccard_threshold, config.num_perm)
            near = MinHashLSHDedup(
                num_perm=config.num_perm,
                bands=lsh_params["bands"],
                rows=lsh_params["rows"],
                ngram=config.ngram,
                mode=config.mode,
                transitive=config.transitive_dedup,
            )
        elif config.near_algo == "simhash":
            near = SimHashLSHDedup(
                ngram=config.ngram,
                mode=config.mode,
                max_hamming=config.simhash_hamming,
            )
        else:
            raise ValueError(f"unknown near_algo: {config.near_algo}")
    decontam = None
    if eval_path:
        decontam = NGramDecontaminator(
            [record[config.text_key] for record in read_jsonl(eval_path)],
            n=config.decontam_ngram,
            threshold=config.decontam_threshold,
            mode=config.mode,
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "input": 0,
        "invalid_json": 0,
        "missing_text": 0,
        "filtered": {},
        "exact_duplicates": 0,
        "near_duplicates": 0,
        "contaminated": 0,
        "output": 0,
        "near_algo": config.near_algo,
        "lsh": lsh_params,
    }
    peak, stop, sampler = _peak_rss_sampler()
    started = time.perf_counter()

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if limit is not None and stats["input"] >= limit:
                break
            line = line.strip()
            if not line:
                continue
            stats["input"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_json"] += 1
                continue

            text = record.get(config.text_key)
            if not isinstance(text, str) or not text:
                stats["missing_text"] += 1
                continue

            text = normalize_text(text)
            reason = check(text, config.filters)
            if reason is not None:
                stats["filtered"][reason] = stats["filtered"].get(reason, 0) + 1
                continue
            if exact.check_and_add(text):
                stats["exact_duplicates"] += 1
                continue
            if near is not None and near.check_and_add(text):
                stats["near_duplicates"] += 1
                continue

            curation = {}
            if decontam is not None:
                ratio = decontam.match_ratio(text)
                curation["decontamination_ratio"] = round(ratio, 6)
                if ratio >= config.decontam_threshold:
                    stats["contaminated"] += 1
                    continue

            record[config.text_key] = text
            if curation:
                record["curation"] = curation
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["output"] += 1

    elapsed = time.perf_counter() - started
    if stop is not None:
        stop.set()
        sampler.join(timeout=1)
    stats["elapsed_seconds"] = round(elapsed, 4)
    stats["docs_per_second"] = round(stats["input"] / elapsed, 1) if elapsed > 0 else 0.0
    stats["peak_rss_mb"] = round(peak[0] / 1024 / 1024, 2) if peak is not None else None
    stats["near_dedup_buckets"] = near.bucket_count if near is not None else 0
    stats["config"] = asdict(config)
    return stats
