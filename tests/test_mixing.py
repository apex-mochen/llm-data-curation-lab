import json
import pathlib
import uuid

from curation_lab.mixing import (
    MixSource,
    allocate_quotas,
    mix_jsonl,
    reservoir_sample_jsonl,
)


def _tmpdir():
    path = pathlib.Path(__file__).parent / f"_tmp_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _write_jsonl(path, n, source):
    with open(path, "w", encoding="utf-8") as fout:
        for i in range(n):
            fout.write(json.dumps({"id": f"{source}-{i}", "source": source}) + "\n")


def _read_ids(path):
    with open(path, encoding="utf-8") as fin:
        return [json.loads(line)["id"] for line in fin if line.strip()]


def test_allocate_quotas_proportional():
    assert allocate_quotas([0.5, 0.3, 0.2], 10) == [5, 3, 2]
    assert allocate_quotas([0.5, 0.3, 0.2], 7) == [4, 2, 1]
    assert allocate_quotas([0.0, 1.0], 5) == [0, 5]


def test_allocate_quotas_rejects_invalid():
    for weights, total in [([], 10), ([0.0, 0.0], 10), ([1.0], 0), ([-1.0, 2.0], 10)]:
        try:
            allocate_quotas(weights, total)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {weights}, {total}")


def test_reservoir_sample_is_deterministic():
    tmp = _tmpdir()
    path = tmp / "a.jsonl"
    _write_jsonl(path, 100, "a")
    first = [record["id"] for _, record in reservoir_sample_jsonl(str(path), 10, seed=7)]
    second = [record["id"] for _, record in reservoir_sample_jsonl(str(path), 10, seed=7)]
    assert first == second
    assert len(first) == 10


def test_mix_jsonl_respects_weights():
    tmp = _tmpdir()
    paths = []
    for name in ("a", "b", "c"):
        path = tmp / f"{name}.jsonl"
        _write_jsonl(path, 10, name)
        paths.append(path)

    output = tmp / "mixed.jsonl"
    stats = mix_jsonl(
        [
            MixSource(str(paths[0]), 0.5),
            MixSource(str(paths[1]), 0.3),
            MixSource(str(paths[2]), 0.2),
        ],
        str(output),
        total=20,
        seed=42,
    )
    assert stats["quotas"] == [10, 6, 4]
    assert stats["written"] == [10, 6, 4]
    assert stats["output"] == 20

    counts = {"a": 0, "b": 0, "c": 0}
    for record_id in _read_ids(output):
        counts[record_id.split("-")[0]] += 1
    assert counts == {"a": 10, "b": 6, "c": 4}


def test_mix_jsonl_is_deterministic():
    tmp = _tmpdir()
    path = tmp / "a.jsonl"
    _write_jsonl(path, 50, "a")
    out1 = tmp / "one.jsonl"
    out2 = tmp / "two.jsonl"
    mix_jsonl([MixSource(str(path), 1.0)], str(out1), total=10, seed=3)
    mix_jsonl([MixSource(str(path), 1.0)], str(out2), total=10, seed=3)
    assert _read_ids(out1) == _read_ids(out2)


def test_mix_jsonl_underfills_short_source():
    tmp = _tmpdir()
    path = tmp / "short.jsonl"
    _write_jsonl(path, 2, "short")
    output = tmp / "out.jsonl"
    stats = mix_jsonl([MixSource(str(path), 1.0)], str(output), total=5)
    assert stats["output"] == 2
    assert stats["written"] == [2]
