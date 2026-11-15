"""Build a JSONL corpus from a local repository (real text, not synthetic).

Used to exercise the pipeline on real Markdown/Python documents without network
access. The corpus is a convenience sample of one repository, not a public
benchmark; label results accordingly.
"""

import argparse
import json
from pathlib import Path

DEFAULT_EXTS = (".md", ".txt", ".py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--exts", nargs="+", default=list(DEFAULT_EXTS))
    parser.add_argument("--min-chars", type=int, default=200)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exts = tuple(args.exts)

    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if len(text) < args.min_chars:
            continue
        records.append({"id": str(path.relative_to(root)), "text": text})

    with open(out_dir / "corpus.jsonl", "w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_chars = sum(len(record["text"]) for record in records)
    print(
        json.dumps(
            {
                "root": str(root),
                "docs": len(records),
                "total_chars": total_chars,
                "exts": exts,
                "out": str(out_dir / "corpus.jsonl"),
            }
        )
    )


if __name__ == "__main__":
    main()
