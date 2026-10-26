"""Invoke the official SWE-bench harness; never implement grading locally."""

import json
import re
import subprocess
import sys
from pathlib import Path

from .audit import audit_artifacts
from .manifest import BenchmarkManifest


_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def read_predictions(path: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    if path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            rows = list(raw.values())
        elif isinstance(raw, list):
            rows = raw
        else:
            raise ValueError("predictions JSON must be a list or instance-ID mapping")
    elif path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raise ValueError("predictions must be .json or .jsonl")
    for number, row in enumerate(rows, 1):
        if (not isinstance(row, dict)
                or not isinstance(row.get("instance_id"), str)
                or not isinstance(row.get("model_name_or_path"), str)
                or row.get("model_patch", "missing") is not None
                and not isinstance(row.get("model_patch"), str)):
            raise ValueError(f"invalid prediction at line {number}")
        iid = row["instance_id"]
        if iid in result:
            raise ValueError(f"duplicate prediction for {iid}")
        result[iid] = row
    return result


def evaluate(
    *, manifest: BenchmarkManifest, predictions: Path, run_id: str,
    output: Path, workers: int = 1, python: str = sys.executable,
) -> Path:
    """Run official Docker grading in output and return its results.json path."""
    if not _SAFE_RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError("run_id must be one safe path component")
    if workers < 1:
        raise ValueError("workers must be positive")
    rows = read_predictions(predictions)
    if set(rows) != set(manifest.instance_ids):
        raise ValueError("prediction IDs must equal the frozen manifest IDs")
    output.mkdir(parents=True, exist_ok=True)
    report = output / "logs" / "evaluation" / run_id / "results.json"
    if report.exists():
        raise FileExistsError(f"run_id already has an evaluation report: {report}")
    command = [
        python, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", manifest.dataset_name,
        "--split", manifest.split,
        "--predictions_path", str(predictions.resolve()),
        "--max_workers", str(workers),
        "--run_id", run_id,
        "--instance_ids", *manifest.instance_ids,
    ]
    subprocess.run(command, cwd=output, check=True)
    if not report.is_file():
        raise RuntimeError(f"official harness finished without {report}")
    audit_artifacts(report.parent, report.parent / "diagnostic-analysis.json")
    return report
