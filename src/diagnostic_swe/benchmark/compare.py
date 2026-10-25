"""Paired, official-score comparisons for a frozen set of task IDs."""

import json
from pathlib import Path

from .manifest import BenchmarkManifest


_OUTCOME_FIELDS = (
    "resolved_ids", "unresolved_ids", "error_ids", "empty_patch_ids", "incomplete_ids"
)


def _outcomes(path: Path, wanted: set[str]) -> dict[str, str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"invalid official report: {path}")
    outcomes: dict[str, str] = {}
    for field in _OUTCOME_FIELDS:
        value = report.get(field, [])
        if not isinstance(value, list) or not all(isinstance(iid, str) for iid in value):
            raise ValueError(f"invalid {field} in {path}")
        for iid in value:
            if iid not in wanted:
                continue
            if iid in outcomes:
                raise ValueError(f"conflicting outcomes for {iid} in {path}")
            outcomes[iid] = field.removesuffix("_ids")
    if set(outcomes) != wanted:
        raise ValueError(f"official report does not cover the manifest: {sorted(wanted - outcomes.keys())}")
    return outcomes


def _interventions(run_dir: Path | None, wanted: set[str]) -> dict[str, int] | None:
    if run_dir is None:
        return None
    counts = {"diagnoses": 0, "recoveries": 0, "restarts": 0}
    for iid in wanted:
        path = run_dir / "tasks" / iid / "manifest.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        counts["diagnoses"] += len(data.get("diagnosis_report_paths", []))
        attempts = data.get("attempts", [])
        counts["restarts"] += sum(item.get("exit_status") == "DiagnosticRestart" for item in attempts)
        from diagnostic_swe.schemas import EventKind
        from diagnostic_swe.storage import read_events

        for attempt in attempts:
            events_path = path.parent / f"attempt-{int(attempt['number']):03d}" / "events.jsonl"
            for event in read_events(events_path):
                if event.kind == EventKind.RECOVERY_APPLIED:
                    counts["recoveries"] += 1
    return counts


def compare(
    *, manifest: BenchmarkManifest, baseline_report: Path,
    enhanced_report: Path, output: Path,
    enhanced_run_dir: Path | None = None,
) -> dict:
    """Compare exactly the frozen IDs, while keeping harness errors visible."""
    ids = set(manifest.instance_ids)
    baseline = _outcomes(baseline_report, ids)
    enhanced = _outcomes(enhanced_report, ids)
    paired = [
        {"instance_id": iid, "baseline": baseline[iid], "enhanced": enhanced[iid]}
        for iid in manifest.instance_ids
    ]
    base_solved = sum(item["baseline"] == "resolved" for item in paired)
    enhanced_solved = sum(item["enhanced"] == "resolved" for item in paired)
    result = {
        "dataset_name": manifest.dataset_name,
        "split": manifest.split,
        "total": len(paired),
        "baseline_resolved": base_solved,
        "enhanced_resolved": enhanced_solved,
        "baseline_resolved_rate": base_solved / len(paired),
        "enhanced_resolved_rate": enhanced_solved / len(paired),
        "improved_ids": [item["instance_id"] for item in paired if item["baseline"] != "resolved" and item["enhanced"] == "resolved"],
        "regressed_ids": [item["instance_id"] for item in paired if item["baseline"] == "resolved" and item["enhanced"] != "resolved"],
        "enhanced_interventions": _interventions(enhanced_run_dir, ids),
        "paired": paired,
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
