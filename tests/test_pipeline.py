import json
import pathlib
import uuid

from curation_lab.filters import FilterConfig
from curation_lab.pipeline import PipelineConfig, run


def _tmpdir():
    path = pathlib.Path(__file__).parent / f"_tmp_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def test_pipeline_filters_dedups_and_decontaminates():
    tmp = _tmpdir()
    base = " ".join(f"alpha{i}" for i in range(100))
    near = " ".join(["changed"] + [f"alpha{i}" for i in range(1, 100)])
    unrelated = " ".join(f"gamma{i}" for i in range(100))
    eval_text = " ".join(f"eval{i}" for i in range(100))

    input_records = [
        {"id": 1, "text": base},
        {"id": 2, "text": base},
        {"id": 3, "text": near},
        {"id": 4, "text": "too short"},
        {"id": 5, "text": "!!! ??? *** " * 10},
        {"id": 6, "text": unrelated},
        {"id": 7, "text": eval_text},
    ]
    input_path = tmp / "input.jsonl"
    eval_path = tmp / "eval.jsonl"
    output_path = tmp / "output.jsonl"
    _write_jsonl(input_path, input_records)
    _write_jsonl(eval_path, [{"text": eval_text}])

    config = PipelineConfig(
        filters=FilterConfig(min_chars=20),
        near_dedup=True,
        num_perm=128,
        bands=16,
        rows=8,
        ngram=5,
        decontam_ngram=5,
        decontam_threshold=0.5,
    )
    stats = run(str(input_path), str(output_path), eval_path=str(eval_path), config=config)

    assert stats["input"] == 7
    assert stats["exact_duplicates"] == 1
    assert stats["near_duplicates"] == 1
    assert stats["contaminated"] == 1
    assert stats["filtered"]["too_short"] == 1
    assert stats["output"] == 2

    kept = _read_jsonl(output_path)
    assert {record["id"] for record in kept} == {1, 6}
    assert kept[0]["text"] == base


def test_pipeline_auto_selects_lsh_params():
    tmp = _tmpdir()
    docs = [
        {"id": 0, "text": " ".join(f"alpha{i}" for i in range(100))},
        {"id": 1, "text": " ".join(f"beta{i}" for i in range(100))},
        {"id": 2, "text": " ".join(f"gamma{i}" for i in range(100))},
    ]
    input_path = tmp / "input.jsonl"
    output_path = tmp / "output.jsonl"
    _write_jsonl(input_path, docs)

    config = PipelineConfig(
        filters=FilterConfig(min_chars=20),
        jaccard_threshold=0.7,
    )
    stats = run(str(input_path), str(output_path), config=config)

    assert stats["lsh"]["bands"] == 12
    assert stats["lsh"]["rows"] == 10
    assert stats["lsh"]["target_threshold"] == 0.7
    assert stats["output"] == 3


def test_pipeline_simhash_near_dedup():
    tmp = _tmpdir()
    tokens = [f"token{i}alpha" for i in range(100)]
    base = " ".join(tokens)
    changed = " ".join(["changed"] + tokens[1:])
    docs = [{"id": 0, "text": base}, {"id": 1, "text": changed}]
    input_path = tmp / "input.jsonl"
    output_path = tmp / "output.jsonl"
    _write_jsonl(input_path, docs)

    config = PipelineConfig(
        filters=FilterConfig(min_chars=20),
        near_algo="simhash",
        simhash_hamming=12,
    )
    stats = run(str(input_path), str(output_path), config=config)

    assert stats["near_algo"] == "simhash"
    assert stats["near_duplicates"] == 1
    assert stats["output"] == 1
