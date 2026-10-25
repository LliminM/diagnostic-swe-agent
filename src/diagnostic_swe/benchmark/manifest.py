"""Freeze one SWE-bench split and task selection before either run."""

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class BenchmarkTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str
    problem_statement: str
    repo: str | None = None
    base_commit: str | None = None
    image_name: str | None = None
    docker_image: str | None = None
    test_command: str | None = None

    @model_validator(mode="after")
    def validate_task(self):
        if not _SAFE_ID.fullmatch(self.instance_id) or self.instance_id in {".", ".."}:
            raise ValueError("instance_id must be one safe path component")
        if not self.problem_statement.strip():
            raise ValueError("problem_statement is required")
        return self


class BenchmarkManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1)
    dataset_name: str
    split: str
    tasks: list[BenchmarkTask]

    @model_validator(mode="after")
    def validate_manifest(self):
        if not self.dataset_name.strip() or not self.split.strip() or not self.tasks:
            raise ValueError("dataset, split and at least one task are required")
        ids = [task.instance_id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate instance_id in manifest")
        return self

    @property
    def instance_ids(self) -> list[str]:
        return [task.instance_id for task in self.tasks]


def _load_local_dataset(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("local dataset JSON must be an array")
    return data


def prepare(
    *, dataset_name: str, split: str, output: Path,
    instance_ids: list[str] | None = None, limit: int | None = None,
    test_commands: dict[str, str] | None = None,
) -> BenchmarkManifest:
    """Resolve IDs once and save the selected task text and image metadata.

    A local JSON/JSONL path avoids a network request; otherwise Hugging Face
    datasets uses the same dataset naming convention as mini-SWE-agent.
    """
    from minisweagent.run.benchmarks.swebench import DATASET_MAPPING

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if instance_ids and limit is not None:
        raise ValueError("choose explicit instance IDs or a limit, not both")
    if output.exists():
        raise FileExistsError(output)
    resolved_name = DATASET_MAPPING.get(dataset_name, dataset_name)
    dataset_path = Path(resolved_name)
    if dataset_path.is_file() and dataset_path.suffix in {".json", ".jsonl"}:
        rows = _load_local_dataset(dataset_path)
    else:
        from datasets import load_dataset

        rows = list(load_dataset(resolved_name, split=split))
    by_id = {str(row["instance_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("dataset contains duplicate instance IDs")
    if instance_ids:
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError("duplicate requested instance IDs")
        missing = set(instance_ids) - by_id.keys()
        if missing:
            raise ValueError(f"unknown instance IDs: {sorted(missing)}")
        selected = instance_ids
    else:
        selected = sorted(by_id)
        if limit is not None:
            selected = selected[:limit]
    commands = test_commands or {}
    if set(commands) - set(selected):
        raise ValueError("test commands contain an unselected instance ID")
    tasks = [BenchmarkTask.model_validate({
        "instance_id": iid,
        "problem_statement": by_id[iid]["problem_statement"],
        "repo": by_id[iid].get("repo"),
        "base_commit": by_id[iid].get("base_commit"),
        "image_name": by_id[iid].get("image_name"),
        "docker_image": by_id[iid].get("docker_image"),
        "test_command": commands.get(iid),
    }) for iid in selected]
    manifest = BenchmarkManifest(dataset_name=resolved_name, split=split, tasks=tasks)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def load_manifest(path: Path) -> BenchmarkManifest:
    return BenchmarkManifest.model_validate_json(path.read_text(encoding="utf-8"))
