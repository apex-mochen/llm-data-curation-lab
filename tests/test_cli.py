import json
import pathlib
import uuid

from curation_lab.cli import main


def _tmpdir():
    path = pathlib.Path(__file__).parent / f"_tmp_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record) + "\n")


def test_cli_run_end_to_end():
    tmp = _tmpdir()
    input_path = tmp / "in.jsonl"
    output_path = tmp / "out.jsonl"
    _write_jsonl(
        input_path,
        [
            {"id": 1, "text": "alpha beta gamma delta epsilon zeta eta theta"},
            {"id": 2, "text": "one two three four five six seven eight nine ten"},
        ],
    )
    code = main(
        [
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--min-chars",
            "10",
            "--no-near-dedup",
        ]
    )
    assert code == 0
    with open(output_path, encoding="utf-8") as fin:
        assert len([line for line in fin if line.strip()]) == 2


def test_cli_mix_end_to_end():
    tmp = _tmpdir()
    a = tmp / "a.jsonl"
    b = tmp / "b.jsonl"
    output = tmp / "mixed.jsonl"
    _write_jsonl(a, [{"id": f"a-{i}"} for i in range(5)])
    _write_jsonl(b, [{"id": f"b-{i}"} for i in range(5)])
    code = main(
        [
            "mix",
            "--input",
            f"{a}:1.0",
            "--input",
            f"{b}:1.0",
            "--output",
            str(output),
            "--total",
            "6",
        ]
    )
    assert code == 0
    with open(output, encoding="utf-8") as fin:
        assert len([line for line in fin if line.strip()]) == 6
