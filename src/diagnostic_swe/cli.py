"""Command-line entry point for a reproducible diagnostic SWE-bench run."""

import argparse
import json
from pathlib import Path

from .benchmark.compare import compare
from .benchmark.evaluate import evaluate
from .benchmark.manifest import load_manifest, prepare
from .benchmark.run import VARIANTS, run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="diagnostic-swe")
    commands = parser.add_subparsers(dest="command", required=True)

    make = commands.add_parser("prepare", help="freeze a SWE-bench task list")
    make.add_argument("--dataset", default="lite", help="mini alias, HF dataset, or local JSON/JSONL")
    make.add_argument("--split", default="dev")
    make.add_argument("--instance-id", action="append", dest="instance_ids")
    make.add_argument("--limit", type=int)
    make.add_argument("--test-commands", type=Path, help="trusted JSON map: instance_id to focused test command")
    make.add_argument("--output", type=Path, required=True)

    run = commands.add_parser("run", help="run the enhanced agent on the frozen tasks")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--variant", choices=VARIANTS, default="full")
    run.add_argument("--model", help="main mini-SWE-agent model name")
    run.add_argument("--diagnosis-model", help="independent diagnostic model; defaults to main model")
    run.add_argument("--config", action="append", dest="config_specs", help="mini config spec; repeat to merge")

    grade = commands.add_parser("evaluate", help="invoke the official SWE-bench harness")
    grade.add_argument("--manifest", type=Path, required=True)
    grade.add_argument("--predictions", type=Path, required=True)
    grade.add_argument("--run-id", required=True)
    grade.add_argument("--output", type=Path, required=True, help="directory for official logs")
    grade.add_argument("--workers", type=int, default=1)

    diff = commands.add_parser("compare", help="compare paired official harness reports")
    diff.add_argument("--manifest", type=Path, required=True)
    diff.add_argument("--baseline-report", type=Path, required=True)
    diff.add_argument("--enhanced-report", type=Path, required=True)
    diff.add_argument("--enhanced-run-dir", type=Path)
    diff.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        commands = None
        if args.test_commands:
            commands = json.loads(args.test_commands.read_text(encoding="utf-8"))
            if not isinstance(commands, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in commands.items()
            ):
                raise ValueError("test-commands must be a JSON object of strings")
        result = prepare(
            dataset_name=args.dataset, split=args.split, output=args.output,
            instance_ids=args.instance_ids, limit=args.limit, test_commands=commands,
        )
        print(f"Saved {len(result.tasks)} tasks to {args.output}")
    elif args.command == "run":
        path = run_benchmark(
            manifest=load_manifest(args.manifest), output=args.output,
            variant=args.variant, config_specs=args.config_specs,
            model_name=args.model, diagnosis_model_name=args.diagnosis_model,
        )
        print(f"Predictions: {path}")
    elif args.command == "evaluate":
        path = evaluate(
            manifest=load_manifest(args.manifest), predictions=args.predictions,
            run_id=args.run_id, output=args.output, workers=args.workers,
        )
        print(f"Official report: {path}")
    elif args.command == "compare":
        result = compare(
            manifest=load_manifest(args.manifest), baseline_report=args.baseline_report,
            enhanced_report=args.enhanced_report, output=args.output,
            enhanced_run_dir=args.enhanced_run_dir,
        )
        print(f"Resolved: {result['baseline_resolved']} -> {result['enhanced_resolved']} / {result['total']}")
        print(f"Comparison: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
