import json
import pathlib
import uuid

from curation_lab.mixing import (
    MixSource,
    allocate_quotas,
    count_jsonl,
    mix_jsonl,
    mix_jsonl_quality,
    mix_jsonl_temperature,
    quality_temperature_quotas,
    reservoir_sample_jsonl,
    temperature_quotas,
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


def test_count_jsonl_skips_blank_lines():
    tmp = _tmpdir()
    path = tmp / "a.jsonl"
    _write_jsonl(path, 7, "a")
    with open(path, "a", encoding="utf-8") as fout:
        fout.write("\n")
    assert count_jsonl(str(path)) == 7


def test_temperature_quotas_endpoints():
    assert temperature_quotas([100, 300], total=40, alpha=1.0) == [10, 30]
    assert temperature_quotas([100, 300], total=40, alpha=0.0) == [20, 20]


def test_temperature_alpha_below_one_upweights_small_source():
    proportional = temperature_quotas([10, 90], total=10, alpha=1.0)
    tempered = temperature_quotas([10, 90], total=10, alpha=0.5)
    assert sum(tempered) == 10
    assert tempered[0] > proportional[0]


def test_mix_jsonl_temperature_end_to_end():
    tmp = _tmpdir()
    small = tmp / "small.jsonl"
    large = tmp / "large.jsonl"
    _write_jsonl(small, 10, "small")
    _write_jsonl(large, 90, "large")
    output = tmp / "mixed.jsonl"
    stats = mix_jsonl_temperature(
        [MixSource(str(small), 1.0), MixSource(str(large), 1.0)],
        str(output),
        total=10,
        alpha=0.5,
    )
    assert stats["counts"] == [10, 90]
    assert stats["quotas"] == [3, 7]
    assert stats["output"] == 10


def test_quality_beta_zero_matches_temperature():
    counts = [1000, 200]
    qualities = [1.0, 0.25]
    assert quality_temperature_quotas(counts, qualities, 400, alpha=1.0, beta=0.0) == temperature_quotas(
        counts, 400, 1.0
    )


def test_lower_quality_source_gets_smaller_quota():
    baseline = temperature_quotas([1000, 200], 400, alpha=1.0)
    weighted = quality_temperature_quotas([1000, 200], [1.0, 0.25], 400, alpha=1.0, beta=1.0)
    assert weighted[1] < baseline[1]
    assert sum(weighted) == 400


def test_mix_jsonl_quality_end_to_end():
    tmp = _tmpdir()
    good = tmp / "good.jsonl"
    noisy = tmp / "noisy.jsonl"
    _write_jsonl(good, 1000, "good")
    _write_jsonl(noisy, 200, "noisy")
    output = tmp / "mixed.jsonl"
    stats = mix_jsonl_quality(
        [MixSource(str(good), 1.0, 1.0), MixSource(str(noisy), 1.0, 0.25)],
        str(output),
        total=400,
        alpha=1.0,
        beta=1.0,
    )
    assert stats["counts"] == [1000, 200]
    assert stats["qualities"] == [1.0, 0.25]
    assert stats["quotas"][1] < 67
    assert stats["output"] == 400
