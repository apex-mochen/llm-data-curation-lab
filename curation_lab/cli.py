"""Command-line entry point for the curation lab."""

import argparse
import json

from .filters import FilterConfig
from .pipeline import PipelineConfig, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="curation-lab")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run the curation pipeline on JSONL")
    run_parser.add_argument("--input", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--eval", default=None, help="optional decontamination reference JSONL")
    run_parser.add_argument("--stats", default=None, help="write stats JSON here")
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--min-chars", type=int, default=50)
    run_parser.add_argument("--max-chars", type=int, default=200_000)
    run_parser.add_argument("--no-near-dedup", action="store_true")
    run_parser.add_argument("--decontam-ngram", type=int, default=13)
    run_parser.add_argument("--decontam-threshold", type=float, default=0.5)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "run":
        raise SystemExit(f"unknown command: {args.command}")

    config = PipelineConfig(
        filters=FilterConfig(min_chars=args.min_chars, max_chars=args.max_chars),
        near_dedup=not args.no_near_dedup,
        decontam_ngram=args.decontam_ngram,
        decontam_threshold=args.decontam_threshold,
    )
    stats = run(args.input, args.output, eval_path=args.eval, config=config, limit=args.limit)
    text = json.dumps(stats, indent=2, ensure_ascii=False)
    print(text)
    if args.stats:
        with open(args.stats, "w", encoding="utf-8") as fout:
            fout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
