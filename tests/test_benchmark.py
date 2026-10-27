import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from diagnostic_swe.benchmark.audit import analyze_official_instance, audit_artifacts
from diagnostic_swe.benchmark.compare import compare
from diagnostic_swe.benchmark.evaluate import evaluate, read_predictions
from diagnostic_swe.benchmark.manifest import BenchmarkManifest, BenchmarkTask, load_manifest, prepare
from diagnostic_swe.benchmark.run import run_benchmark
from diagnostic_swe.cli import main


class BenchmarkTests(unittest.TestCase):
    def test_prepare_freezes_explicit_order_and_rejects_missing_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "dataset.json"
            source.write_text(json.dumps([
                {"instance_id": "repo__2", "problem_statement": "Fix two", "base_commit": "a" * 40},
                {"instance_id": "repo__1", "problem_statement": "Fix one", "base_commit": "b" * 40},
            ]), encoding="utf-8")
            output = root / "manifest.json"
            manifest = prepare(dataset_name=str(source), split="test", output=output,
                               instance_ids=["repo__2", "repo__1"])
            self.assertEqual(manifest.instance_ids, ["repo__2", "repo__1"])
            self.assertEqual(load_manifest(output), manifest)
            with self.assertRaises(FileExistsError):
                prepare(dataset_name=str(source), split="test", output=output)
            with self.assertRaises(ValueError):
                prepare(dataset_name=str(source), split="test", output=root / "other.json",
                        instance_ids=["unknown"])

    def test_official_evaluation_arguments_and_complete_prediction_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = BenchmarkManifest(dataset_name="princeton-nlp/SWE-bench_Lite", split="dev",
                tasks=[BenchmarkTask(instance_id="repo__1", problem_statement="Fix")])
            predictions = root / "predictions.jsonl"
            predictions.write_text(json.dumps({"instance_id": "repo__1",
                "model_name_or_path": "agent", "model_patch": "diff --git a/x b/x"}) + "\n",
                encoding="utf-8")
            def fake_run(command, *, cwd, check):
                self.assertEqual(command[1:3], ["-m", "swebench.harness.run_evaluation"])
                self.assertEqual(command[-2:], ["--instance_ids", "repo__1"])
                self.assertIn("princeton-nlp/SWE-bench_Lite", command)
                report = cwd / "logs/evaluation/trial/results.json"
                report.parent.mkdir(parents=True)
                report.write_text("{}", encoding="utf-8")
            with patch("diagnostic_swe.benchmark.evaluate.subprocess.run", side_effect=fake_run):
                report = evaluate(manifest=manifest, predictions=predictions, run_id="trial",
                                  output=root / "official")
            self.assertTrue(report.is_file())
            self.assertEqual(json.loads((report.parent / "diagnostic-analysis.json").read_text()), [])
            with self.assertRaises(FileExistsError):
                evaluate(manifest=manifest, predictions=predictions, run_id="trial",
                         output=root / "official")
            predictions.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate(manifest=manifest, predictions=predictions, run_id="other",
                         output=root / "official")

    def test_paired_comparison_uses_official_resolved_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = BenchmarkManifest(dataset_name="fixture", split="test", tasks=[
                BenchmarkTask(instance_id=iid, problem_statement="Fix")
                for iid in ("one", "two", "three")
            ])
            base = root / "base.json"
            enhanced = root / "enhanced.json"
            base.write_text(json.dumps({"resolved_ids": ["one"],
                "unresolved_ids": ["two"], "error_ids": ["three"]}), encoding="utf-8")
            enhanced.write_text(json.dumps({"resolved_ids": ["two"],
                "unresolved_ids": ["one"], "empty_patch_ids": ["three"]}), encoding="utf-8")
            result = compare(manifest=manifest, baseline_report=base,
                enhanced_report=enhanced, output=root / "comparison.json")
            self.assertEqual(result["baseline_resolved"], 1)
            self.assertEqual(result["enhanced_resolved"], 1)
            self.assertEqual(result["improved_ids"], ["two"])
            self.assertEqual(result["regressed_ids"], ["one"])
            self.assertEqual(result["paired"][2]["enhanced"], "empty_patch")

    def test_official_audit_separates_preserved_failures_from_target_failures(self):
        preserved = {"pydicom__case": {"resolved": False, "infra_failure": False,
            "tests_status": {"FAIL_TO_PASS": {"failure": []},
                             "PASS_TO_PASS": {"failure": ["tests/test_vr.py::TestOld::test_value"]}}}}
        old_log = ("platform linux -- Python 3.9, pytest-8.3.5\n"
                   "E AttributeError: 'TestOld' object has no attribute 'tag'\n")
        old = analyze_official_instance("pydicom__case", preserved, old_log)
        self.assertEqual(old["diagnosis"], "preserved_test_failure")
        self.assertEqual(old["failed_target_tests"], [])
        self.assertEqual(old["environment_hypotheses"], ["legacy_setup_runner_mismatch"])
        self.assertNotIn("7.4.4", old["recommendation"])

        target = {"astroid__case": {"resolved": False, "infra_failure": False,
            "tests_status": {"FAIL_TO_PASS": {"failure": ["tests/test_output.py::test_capture"]},
                             "PASS_TO_PASS": {"failure": []}}}}
        target_log = "capsys.readouterr()\nassert expected_out in caplog.text\n"
        actual = analyze_official_instance("astroid__case", target, target_log)
        self.assertEqual(actual["diagnosis"], "target_test_failure")
        self.assertIn("目标测试仍失败", actual["recommendation"])
        self.assertEqual(actual["environment_hypotheses"], [])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "task"
            folder.mkdir()
            (folder / "report.json").write_text(json.dumps(preserved), encoding="utf-8")
            (folder / "test_output.txt").write_text(old_log, encoding="utf-8")
            output = root / "analysis.json"
            self.assertEqual(len(audit_artifacts(root, output)), 1)
            self.assertEqual(json.loads(output.read_text())[0]["diagnosis"],
                             "preserved_test_failure")
            with self.assertRaises(FileExistsError):
                audit_artifacts(root, output)

    def test_cli_help_and_duplicate_predictions(self):
        with self.assertRaises(SystemExit) as captured:
            main(["--help"])
        self.assertEqual(captured.exception.code, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preds.jsonl"
            row = {"instance_id": "one", "model_name_or_path": "model", "model_patch": ""}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_predictions(path)

    def test_upstream_mini_predictions_json_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preds.json"
            row = {"instance_id": "one", "model_name_or_path": "model", "model_patch": "diff"}
            path.write_text(json.dumps({"one": row}), encoding="utf-8")
            self.assertEqual(read_predictions(path), {"one": row})
            row["model_patch"] = None
            path.write_text(json.dumps({"one": row}), encoding="utf-8")
            self.assertEqual(read_predictions(path), {"one": row})

    def test_run_writes_official_predictions_and_passes_variant_switches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patch_path = root / "candidate.patch"
            patch_path.write_text("diff --git a/x b/x\n", encoding="utf-8")
            manifest = BenchmarkManifest(dataset_name="fixture", split="test", tasks=[
                BenchmarkTask(instance_id="repo__1", problem_statement="Fix",
                              base_commit="a" * 40),
            ])
            class FakeEnvironment:
                container_id = "container"
                cleaned = False
                def execute(self, action):
                    return {"returncode": 0, "output": "b" * 40 + "\n"}
                def cleanup(self):
                    self.cleaned = True
            env = FakeEnvironment()
            fake_result = SimpleNamespace(exit_status="Submitted", submission=patch_path.read_text())
            with patch("diagnostic_swe.benchmark.run.get_sb_environment", return_value=env), \
                 patch("diagnostic_swe.benchmark.run.run_diagnostic_task", return_value=fake_result) as runner:
                predictions = run_benchmark(manifest=manifest, output=root / "run",
                    variant="no-recovery", model_name="fake/main")
            self.assertTrue(env.cleaned)
            self.assertEqual(runner.call_args.kwargs["diagnosis_enabled"], True)
            self.assertEqual(runner.call_args.kwargs["recovery_enabled"], False)
            row = read_predictions(predictions)["repo__1"]
            self.assertEqual(row["model_patch"], "diff --git a/x b/x\n")
            status = json.loads((root / "run" / "task_status.jsonl").read_text())
            self.assertEqual(status["exit_status"], "Submitted")
            self.assertEqual(status["base_commit_check"], {
                "manifest_base_commit": "a" * 40,
                "container_head": "b" * 40,
                "matches": False,
            })


if __name__ == "__main__":
    unittest.main()
