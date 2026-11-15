"""Command-line entry point for the curation lab."""

import argparse
import json

from .filters import FilterConfig
from .mixing import MixSource, mix_jsonl
from .pipeline import PipelineConfig, run


def _parse_source(spec: str) -> MixSource:
    path, separator, weight = spec.rpartition(":")
    if not separator or not path:
        raise argparse.ArgumentTypeError(f"expected path:weight, got {spec!r}")
    try:
        return MixSource(path=path, weight=float(weight))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid weight in {spec!r}") from exc


def _emit(stats: dict, stats_path: str) -> None:
    text = json.dumps(stats, indent=2, ensure_ascii=False)
    print(text)
    if stats_path:
        with open(stats_path, "w", encoding="utf-8") as fout:
            fout.write(text)


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

    mix_parser = sub.add_parser("mix", help="weighted mix of JSONL sources")
    mix_parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=_parse_source,
        help="path:weight; repeat for each source",
    )
    mix_parser.add_argument("--output", required=True)
    mix_parser.add_argument("--total", type=int, required=True)
    mix_parser.add_argument("--seed", type=int, default=42)
    mix_parser.add_argument("--stats", default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        config = PipelineConfig(
            filters=FilterConfig(min_chars=args.min_chars, max_chars=args.max_chars),
            near_dedup=not args.no_near_dedup,
            decontam_ngram=args.decontam_ngram,
            decontam_threshold=args.decontam_threshold,
        )
        stats = run(args.input, args.output, eval_path=args.eval, config=config, limit=args.limit)
    elif args.command == "mix":
        stats = mix_jsonl(args.input, args.output, args.total, seed=args.seed)
    else:
        raise SystemExit(f"unknown command: {args.command}")
    _emit(stats, args.stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
