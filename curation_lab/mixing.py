"""Weighted data mixing across JSONL sources.

Quotas are allocated proportionally to weights, then each source contributes a
uniform reservoir sample of its quota. Sampling is deterministic for a given
seed, and memory is O(total output documents) because the reservoir must be
held before writing.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class MixSource:
    path: str
    weight: float


def allocate_quotas(weights: List[float], total: int) -> List[int]:
    """Allocate integer quotas proportionally, with deterministic tie-breaking."""
    if total <= 0:
        raise ValueError("total must be positive")
    if not weights:
        raise ValueError("at least one source is required")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("at least one weight must be positive")

    raw = [total * weight / weight_sum for weight in weights]
    quotas = [int(value) for value in raw]
    remainder = total - sum(quotas)
    order = sorted(range(len(weights)), key=lambda i: (-(raw[i] - quotas[i]), i))
    for index in order[:remainder]:
        quotas[index] += 1
    return quotas


def reservoir_sample_jsonl(path: str, k: int, seed: int) -> List[Tuple[int, dict]]:
    """Uniform reservoir sample of ``k`` records, returned in source order."""
    if k <= 0:
        return []
    rng = random.Random(seed)
    reservoir: List[Tuple[int, dict]] = []
    with open(path, "r", encoding="utf-8") as fin:
        index = 0
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            if index < k:
                reservoir.append((index, record))
            else:
                slot = rng.randrange(index + 1)
                if slot < k:
                    reservoir[slot] = (index, record)
            index += 1
    reservoir.sort(key=lambda item: item[0])
    return reservoir


def mix_jsonl(sources: List[MixSource], output_path: str, total: int, seed: int = 42) -> Dict:
    """Write a weighted mixture of the sources to ``output_path``.

    A source that has fewer documents than its quota underfills; the remainder
    is not redistributed in this version.
    """
    quotas = allocate_quotas([source.weight for source in sources], total)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "requested": total,
        "quotas": quotas,
        "written": [],
        "output": 0,
    }
    with open(output_path, "w", encoding="utf-8") as fout:
        for index, (source, quota) in enumerate(zip(sources, quotas)):
            sample = reservoir_sample_jsonl(source.path, quota, seed + index)
            for _, record in sample:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"].append(len(sample))
            stats["output"] += len(sample)
    return stats
