"""Explain official SWE-bench failures without changing official scores."""

import json
import re
from pathlib import Path


_PYTEST_VERSION = re.compile(r"\bpytest[- ](\d+)(?:\.\d+)*\b", re.IGNORECASE)
_MISSING_TEST_FIELD = re.compile(
    r"AttributeError:\s*'([A-Za-z_][A-Za-z0-9_]*)' object has no attribute '[^']+'"
)


def analyze_official_instance(instance_id: str, report: dict, test_output: str) -> dict:
    """Separate target failures from preserved-test failures using official IDs."""

    entry = report.get(instance_id)
    if not isinstance(entry, dict) or not isinstance(entry.get("tests_status"), dict):
        raise ValueError(f"missing tests_status for {instance_id}")
    status = entry["tests_status"]

    def failures(group: str) -> list[str]:
        value = status.get(group)
        if not isinstance(value, dict) or not isinstance(value.get("failure"), list):
            raise ValueError(f"missing {group}.failure for {instance_id}")
        items = value["failure"]
        if not all(isinstance(item, str) for item in items):
            raise ValueError(f"invalid {group}.failure for {instance_id}")
        return items

    target_failures = failures("FAIL_TO_PASS")
    preserved_failures = failures("PASS_TO_PASS")
    version = _PYTEST_VERSION.search(test_output)
    pytest_major = int(version.group(1)) if version else None
    missing_classes = sorted(set(_MISSING_TEST_FIELD.findall(test_output)))
    environment_hypotheses: list[str] = []

    if target_failures:
        diagnosis = "target_test_failure"
        recommendation = (
            "目标测试仍失败：检查失败断言和候选补丁覆盖的行为；修复后重新运行 agent "
            "并以新的 predictions 评测。"
        )
    elif preserved_failures:
        diagnosis = "preserved_test_failure"
        recommendation = (
            "目标测试已通过，但保留测试失败。先在原始代码和相同镜像上复测这些节点，"
            "确认是否为预存的测试环境问题，再决定修改补丁或镜像。"
        )
        if pytest_major is not None and pytest_major >= 8 and missing_classes:
            environment_hypotheses.append("legacy_setup_runner_mismatch")
            recommendation = (
                "目标测试已通过。检查失败类是否依靠旧式 setup() 初始化缺失属性，"
                "并在相同环境中用原始代码复测；确认是测试框架兼容问题后，再选择"
                "与该项目兼容的测试依赖版本。"
            )
    elif entry.get("resolved") is True:
        diagnosis = "resolved"
        recommendation = "无需修复。"
    else:
        diagnosis = "unclassified"
        recommendation = "官方报告没有列出目标或保留测试失败；检查测试输出与评测错误。"

    return {
        "instance_id": instance_id,
        "official_resolved": entry.get("resolved"),
        "official_infra_failure": entry.get("infra_failure"),
        "diagnosis": diagnosis,
        "failed_target_tests": target_failures,
        "failed_preserved_tests": preserved_failures,
        "pytest_major": pytest_major,
        "missing_test_field_classes": missing_classes,
        "environment_hypotheses": environment_hypotheses,
        "recommendation": recommendation,
        "note": "本报告只解释官方结果，不改变 SWE-bench 评分，也不将隐藏测试反馈提供给同轮 Agent。",
    }


def audit_artifacts(root: Path, output: Path) -> list[dict]:
    """Read per-instance report.json/test_output.txt pairs under a run folder."""

    if output.exists():
        raise FileExistsError(output)
    analyses: list[dict] = []
    for report_path in sorted(root.rglob("report.json")):
        test_path = report_path.with_name("test_output.txt")
        if not test_path.is_file():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                raise ValueError("per-instance report must be an object")
            test_output = test_path.read_text(encoding="utf-8", errors="replace")
            for instance_id in report:
                analysis = analyze_official_instance(instance_id, report, test_output)
                analysis["report_path"] = str(report_path)
                analysis["test_output_path"] = str(test_path)
                analyses.append(analysis)
        except (OSError, ValueError) as error:
            analyses.append({"diagnosis": "audit_error", "report_path": str(report_path),
                             "test_output_path": str(test_path),
                             "error": f"{type(error).__name__}: {error}"})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analyses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return analyses
