import json
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import build_manifest  # noqa: E402
import verify_manifest  # noqa: E402
from emb_common import sha256_file  # noqa: E402

DIM = 4


def fixture(directory: Path) -> Path:
    """A copy of the tooling with tiny vectors, so the verifier is run on a tree it can fail."""
    for name in build_manifest.SCRIPT_FILES:
        shutil.copyfile(HERE / name, directory / name)
    for pool in ("q003", "m008"):
        shutil.copyfile(HERE / f"pool-{pool}.jsonl", directory / f"pool-{pool}.jsonl")
    rows = 12000
    vector = struct.pack(f"<{DIM}e", 0.5, 0.5, 0.5, 0.5)
    (directory / "vectors-q003.f16").write_bytes(vector * rows)
    record = {
        "model": "minilm", "pool": "q003", "file": "vectors-q003.f16", "rows": rows, "dimension": DIM,
        "threads": 4, "batch_size": 64, "seconds": [1.0, 1.0],
        "run_digests": ["a" * 64, "a" * 64], "deterministic_on_this_machine": True,
        "vectors_sha256": sha256_file(directory / "vectors-q003.f16"),
        "tools": {"python": "3.11.0", "torch": "x"},
    }
    manifest = build_manifest.build(directory, [record])
    (directory / "MANIFEST.toml").write_text("\n".join(build_manifest.emit(manifest)) + "\n", encoding="utf-8")
    return directory


class VerifyManifestTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        fixture(self.dir)

    def test_a_consistent_tree_verifies(self):
        self.assertEqual(verify_manifest.verify(self.dir), [])
        self.assertEqual(verify_manifest.verify(self.dir, deep=False, regenerate=True), [])

    def test_the_committed_tree_verifies(self):
        self.assertEqual(verify_manifest.verify(HERE, regenerate=True), [])

    def test_a_flipped_vector_byte_is_caught(self):
        path = self.dir / "vectors-q003.f16"
        data = bytearray(path.read_bytes())
        data[10] ^= 1
        path.write_bytes(bytes(data))
        self.assertTrue(any("digest mismatch" in p for p in verify_manifest.verify(self.dir)))

    def test_a_truncated_vector_file_is_caught(self):
        path = self.dir / "vectors-q003.f16"
        path.write_bytes(path.read_bytes()[:-DIM * 2])
        problems = verify_manifest.verify(self.dir)
        self.assertTrue(any("size" in p for p in problems))

    def test_an_edited_pool_is_caught_by_digest_and_by_token_recount(self):
        path = self.dir / "pool-q003.jsonl"
        lines = path.read_text("ascii").splitlines()
        row = json.loads(lines[0])
        row["text"] += " extra words"
        lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        problems = verify_manifest.verify(self.dir)
        self.assertTrue(any("pools.q003: digest mismatch" in p for p in problems), problems)
        self.assertTrue(any("token count" in p for p in problems), problems)

    def test_an_edited_script_is_caught(self):
        with open(self.dir / "embed.py", "a", encoding="utf-8") as handle:
            handle.write("\n# tampered\n")
        self.assertTrue(any("scripts.embed.py" in p for p in verify_manifest.verify(self.dir)))

    def test_a_changed_model_pin_is_caught(self):
        path = self.dir / "models.toml"
        text = path.read_text(encoding="utf-8").replace("1110a243fdf4706b3f48f1d95db1a4f5529b4d41", "0" * 40)
        path.write_text(text, encoding="utf-8")
        problems = verify_manifest.verify(self.dir)
        self.assertTrue(any("models.minilm.revision" in p for p in problems), problems)

    def test_model_files_are_rehashed_when_a_root_is_given(self):
        root = self.dir / "models"
        (root / "minilm").mkdir(parents=True)
        problems = verify_manifest.verify(self.dir, models_root=root)
        self.assertTrue(any("model file minilm/model.safetensors: missing" in p for p in problems))
        (root / "minilm" / "model.safetensors").write_bytes(b"not the model")
        problems = verify_manifest.verify(self.dir, models_root=root)
        self.assertTrue(any("model.safetensors: digest mismatch" in p for p in problems))

    def test_non_unit_rows_fail_the_deep_check(self):
        problems = verify_manifest.verify(self.dir, deep=True)
        self.assertEqual(problems, [])  # 4 x 0.5 has norm 1
        path = self.dir / "vectors-q003.f16"
        bad = struct.pack(f"<{DIM}e", 2.0, 0.0, 0.0, 0.0) + path.read_bytes()[DIM * 2:]
        path.write_bytes(bad)
        self.assertTrue(any("norm" in p for p in verify_manifest.verify(self.dir, deep=True)))

    def test_a_missing_manifest_is_reported(self):
        (self.dir / "MANIFEST.toml").unlink()
        self.assertTrue(verify_manifest.verify(self.dir)[0].startswith("MANIFEST.toml unreadable"))

    def test_an_unreproducible_recorded_run_fails(self):
        path = self.dir / "MANIFEST.toml"
        text = path.read_text(encoding="utf-8").replace(
            "deterministic_on_this_machine = true", "deterministic_on_this_machine = false")
        path.write_text(text, encoding="utf-8")
        self.assertTrue(any("not identical" in p for p in verify_manifest.verify(self.dir)))

    def test_a_fresh_process_repeat_that_differs_fails(self):
        path = self.dir / "MANIFEST.toml"
        text = path.read_text(encoding="utf-8")
        self.assertNotIn("fresh_process_repeat_sha256", text)
        digest = verify_manifest.tomllib.loads(text)["vectors"]["minilm-q003"]["sha256"]
        path.write_text(text + 'fresh_process_repeat_sha256 = "' + digest + '"\n', encoding="utf-8")
        self.assertEqual(verify_manifest.verify(self.dir), [])
        path.write_text(text + 'fresh_process_repeat_sha256 = "' + "0" * 64 + '"\n', encoding="utf-8")
        self.assertTrue(any("fresh-process repeat" in p for p in verify_manifest.verify(self.dir)))

    def test_the_cli_exits_nonzero_on_mismatch(self):
        with open(self.dir / "pool-m008.jsonl", "a", encoding="ascii") as handle:
            handle.write("\n")
        self.assertEqual(verify_manifest.main(["--root", str(self.dir)]), 1)


if __name__ == "__main__":
    unittest.main()
