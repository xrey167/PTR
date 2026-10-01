import copy
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("select_m002_v5_pilot", ROOT / "scripts/select_m002_v5_pilot.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def outputs():
    result = []
    for candidate in mod.CANDIDATES:
        rank, limit, dropout = candidate
        for seed in mod.SEEDS:
            records = [{"row": "run-v5", "seed": seed, "rank": rank, "bias_limit": limit, "metadata_dropout": dropout}]
            bonus = rank / 1000 + limit / 1000 + dropout / 100
            for fold in mod.FOLDS:
                for arm in mod.ARMS:
                    treatment = arm == mod.ARMS[0]
                    records.append({
                        "row": "final-v5", "fold": fold, "arm": arm, "split": "test_ood",
                        "accuracy": 0.8 + (bonus if treatment else 0.0),
                        "nll": 0.4 if treatment else 0.5,
                        "ece15": 0.04 if treatment else 0.05,
                    })
            result.append((f"{rank}-{limit}-{dropout}-{seed}", "\n".join(json.dumps(row) for row in records)))
    return result


def artifact_grid(directory: Path, source_sha: str = "a" * 40):
    digests = mod.locked_fold_digests()
    paths = []
    for index, (name, text) in enumerate(outputs()):
        protocol = json.loads(text.splitlines()[0])
        candidate = {
            "rank": protocol["rank"],
            "bias_limit": int(protocol["bias_limit"]),
            "metadata_dropout": float(protocol["metadata_dropout"]),
        }
        seed = protocol["seed"]
        records = [json.loads(line) for line in text.splitlines()]
        records[1:1] = [
            {"row": "data-v2", "fold": fold, "data_fnv64": digest}
            for fold, digest in digests.items()
        ]
        raw = ("\n".join(json.dumps(row) for row in records) + "\n").encode()
        path = directory / f"cell-{index:02}.stdout"
        path.write_bytes(raw)
        metadata = {
            "schema_version": 1,
            "source_sha": source_sha,
            "command": mod.command_for(candidate, seed, digests),
            "candidate": candidate,
            "seed": seed,
            "fold_digests": digests,
            "exit_code": 0,
            "stdout_file": path.name,
            "stdout_sha256": mod.sha256(raw),
        }
        path.with_suffix(".json").write_text(
            json.dumps(metadata, sort_keys=True), encoding="utf-8"
        )
        paths.append(path)
    return paths


class PilotSelectionTests(unittest.TestCase):
    def test_fixed_rule_selects_highest_minimum_accuracy_candidate(self):
        decision = mod.select(outputs())
        self.assertEqual(decision["decision"], "SELECTED")
        self.assertEqual(decision["selection"]["rank"], 16)
        self.assertEqual(decision["selection"]["bias_limit"], 2.0)
        self.assertEqual(decision["selection"]["metadata_dropout"], 0.1)
        self.assertFalse(decision["claimable"])

    def test_ece_regression_eliminates_candidate_before_accuracy_ranking(self):
        changed = copy.deepcopy(outputs())
        target = (16, 2.0, 0.1)
        rewritten = []
        for name, text in changed:
            rows = [json.loads(line) for line in text.splitlines()]
            protocol = rows[0]
            if (protocol["rank"], protocol["bias_limit"], protocol["metadata_dropout"]) == target:
                for row in rows[1:]:
                    if row["arm"] == mod.ARMS[0]:
                        row["ece15"] = 0.08
            rewritten.append((name, "\n".join(json.dumps(row) for row in rows)))
        decision = mod.select(rewritten)
        self.assertNotEqual(
            (decision["selection"]["rank"], decision["selection"]["bias_limit"], decision["selection"]["metadata_dropout"]),
            target,
        )

    def test_missing_grid_cell_is_a_hard_error(self):
        with self.assertRaisesRegex(mod.PilotError, "incomplete pilot grid"):
            mod.select(outputs()[:-1])

    def test_cli_emits_bound_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = artifact_grid(Path(directory))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(mod.main([str(path) for path in reversed(paths)]), 0)
            decision = json.loads(stdout.getvalue())
            self.assertEqual(decision["source_sha"], "a" * 40)
            self.assertEqual(decision["fold_digests"], mod.locked_fold_digests())
            self.assertEqual(decision["fixed_protocol"], mod.fixed_protocol())
            self.assertEqual(
                list(decision["input_stdout_sha256"]),
                sorted(path.name for path in paths),
            )
            for path in paths:
                self.assertEqual(
                    decision["input_stdout_sha256"][path.name], mod.sha256(path.read_bytes())
                )

    def test_cli_rejects_tampered_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = artifact_grid(Path(directory))
            paths[0].write_bytes(paths[0].read_bytes() + b"tampered\n")
            with self.assertRaisesRegex(mod.PilotError, "stdout SHA256"):
                mod.load_provenanced_outputs(paths)

    def test_cli_rejects_missing_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = artifact_grid(Path(directory))
            paths[0].with_suffix(".json").unlink()
            with self.assertRaisesRegex(mod.PilotError, "missing sibling metadata"):
                mod.load_provenanced_outputs(paths)

    def test_cli_rejects_mixed_commits(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = artifact_grid(Path(directory))
            sidecar = paths[0].with_suffix(".json")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            metadata["source_sha"] = "b" * 40
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(mod.PilotError, "mixed source commits"):
                mod.load_provenanced_outputs(paths)

    def test_cli_rejects_mixed_fold_digests(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = artifact_grid(Path(directory))
            path = paths[0]
            sidecar = path.with_suffix(".json")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            changed = dict(metadata["fold_digests"])
            changed[mod.FOLDS[0]] = "0" * 16
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for row in rows:
                if row.get("row") == "data-v2" and row.get("fold") == mod.FOLDS[0]:
                    row["data_fnv64"] = changed[mod.FOLDS[0]]
            raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
            path.write_bytes(raw)
            metadata["fold_digests"] = changed
            metadata["command"] = mod.command_for(metadata["candidate"], metadata["seed"], changed)
            metadata["stdout_sha256"] = mod.sha256(raw)
            sidecar.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(mod.PilotError, "mixed fold digests"):
                mod.load_provenanced_outputs(paths)

    def test_cli_rejects_metadata_row_disagreement(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = artifact_grid(Path(directory))
            sidecar = paths[0].with_suffix(".json")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            metadata["seed"] = 13 if metadata["seed"] == 7 else 7
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(mod.PilotError, "metadata and stdout row identity disagree"):
                mod.load_provenanced_outputs(paths)


if __name__ == "__main__":
    unittest.main()
