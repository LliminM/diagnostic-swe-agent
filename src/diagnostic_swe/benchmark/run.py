"""Run frozen SWE-bench tasks with mini-SWE-agent's own Docker setup."""

import json
from pathlib import Path

from minisweagent.config import get_config_from_spec
from minisweagent.models import get_model, get_model_name
from minisweagent.run.benchmarks.swebench import DEFAULT_CONFIG_FILE, get_sb_environment
from minisweagent.utils.serialize import recursive_merge

from diagnostic_swe.diagnosis.model import LiteLLMDiagnosticModel
from diagnostic_swe.runner import run_diagnostic_task

from .manifest import BenchmarkManifest


VARIANTS = ("full", "no-diagnosis", "no-recovery")


def build_mini_config(config_specs: list[str], model_name: str | None) -> dict:
    specs = config_specs or [str(DEFAULT_CONFIG_FILE)]
    configs = [get_config_from_spec(spec) for spec in specs]
    config = recursive_merge(*configs)
    if model_name:
        config.setdefault("model", {})["model_name"] = model_name
    agent_config = config.get("agent", {})
    if not agent_config.get("system_template") or not agent_config.get("instance_template"):
        raise ValueError("mini config must provide system_template and instance_template")
    get_model_name(config=config.get("model", {}))
    return config


def run_benchmark(
    *, manifest: BenchmarkManifest, output: Path, variant: str,
    config_specs: list[str] | None = None, model_name: str | None = None,
    diagnosis_model_name: str | None = None,
) -> Path:
    """Write one official predictions JSONL plus per-instance run records.

    Instances run sequentially to keep a task container and its artifacts easy
    to inspect. A failed instance gets an empty prediction and an error record;
    the rest of the fixed task list continues.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if output.exists():
        raise FileExistsError(output)
    config = build_mini_config(config_specs or [], model_name)
    main_name = get_model_name(config=config.get("model", {}))
    if variant != "no-diagnosis" and not (diagnosis_model_name or main_name):
        raise ValueError("a diagnosis model is required")
    experiment_name = f"diagnostic-swe-{variant}-{main_name}".replace("/", "__")
    output.mkdir(parents=True)
    (output / "run_config.json").write_text(json.dumps({
        "dataset_name": manifest.dataset_name, "split": manifest.split,
        "instance_ids": manifest.instance_ids, "variant": variant,
        "model_name": main_name,
        "diagnosis_model_name": diagnosis_model_name or main_name,
        "config_specs": config_specs or [str(DEFAULT_CONFIG_FILE)],
        "model_name_or_path": experiment_name,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    predictions_path = output / "predictions.jsonl"
    status_path = output / "task_status.jsonl"
    for task in manifest.tasks:
        env = None
        status = "error"
        error = None
        patch = ""
        try:
            env = get_sb_environment(config, task.model_dump(exclude_none=True))
            if task.base_commit:
                head = env.execute({"command": "git rev-parse HEAD"})
                if head.get("returncode") != 0 or str(head.get("output", "")).strip() != task.base_commit:
                    raise RuntimeError("task container HEAD differs from frozen base_commit")
            result = run_diagnostic_task(
                task_id=task.instance_id,
                task=task.problem_statement,
                model_factory=lambda: get_model(config=config.get("model", {})),
                environment=env,
                diagnosis_model=(
                    LiteLLMDiagnosticModel(diagnosis_model_name or main_name)
                    if variant != "no-diagnosis" else None
                ),
                system_template=config["agent"]["system_template"],
                instance_template=config["agent"]["instance_template"],
                output_root=output / "tasks",
                test_command=task.test_command,
                isolated_workspace=True,
                diagnosis_enabled=variant != "no-diagnosis",
                recovery_enabled=variant == "full",
            )
            status = result.exit_status
            if status == "Submitted":
                # Grade the same submitted patch that the upstream mini runner
                # writes to preds.json; the full worktree diff remains an artifact.
                patch = result.submission
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if env is not None:
                env.cleanup()
                env.container_id = None
        prediction = {
            "instance_id": task.instance_id,
            "model_name_or_path": experiment_name,
            "model_patch": patch,
        }
        with predictions_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        with status_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "instance_id": task.instance_id, "exit_status": status,
                "error": error, "has_patch": bool(patch),
            }, ensure_ascii=False) + "\n")
    return predictions_path
