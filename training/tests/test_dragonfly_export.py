import filecmp
import json
import struct
import tempfile
import unittest
from pathlib import Path

from ptr_training.dragonfly import (
    HashedBagOfWordsEncoder,
    PodIdentityDescriptor,
    PodTrainingSpec,
    TrainConfig,
    artifact_stem,
    export_address,
    train_pod_address,
    verify_artifact,
)
from ptr_training.dragonfly.fixture import build_world

FIXTURES = Path(__file__).resolve().parents[2] / "crates/ptr-router/tests/fixtures"
ENCODER = HashedBagOfWordsEncoder(16)


def trained(**config):
    descriptor = PodIdentityDescriptor(
        domain="supply",
        kind="Retrieval",
        semantic_role="reader",
        predicate="lead_time",
        aliases=frozenset({"Alpha Corp"}),
    )
    spec = PodTrainingSpec(
        pod_id="pod-alpha",
        descriptor=descriptor,
        generation=2,
        positives=("Delivery time of Alpha Corp?", "Is Alpha Corp late?"),
        negatives=("Delivery time of Beta Corp?", "Is Beta Corp late?"),
        negative_aliases=("Beta Corp",),
    )
    return train_pod_address(spec, ENCODER, TrainConfig(**config))


class ExportTests(unittest.TestCase):
    def test_export_writes_a_verifiable_pair_named_after_the_identity(self):
        address = trained()
        with tempfile.TemporaryDirectory() as directory:
            json_path, bin_path = export_address(address, Path(directory))
            self.assertEqual(json_path.stem, artifact_stem(address.spec.descriptor.identity()))
            self.assertNotIn(":", json_path.name)
            manifest = verify_artifact(json_path)
            self.assertEqual(manifest["pod_identity"], address.spec.descriptor.identity())
            self.assertEqual(manifest["cluster_key"], address.spec.descriptor.cluster_key().hex())
            self.assertEqual(manifest["generation"], 2)
            self.assertEqual(manifest["gate"]["status"], address.gate.status)
            self.assertEqual(len(bin_path.read_bytes()), (address.dims + 1) * 4)

    def test_weights_file_is_z_then_bias_as_little_endian_float32(self):
        address = trained()
        with tempfile.TemporaryDirectory() as directory:
            _, bin_path = export_address(address, Path(directory))
            values = struct.unpack(f"<{address.dims + 1}f", bin_path.read_bytes())
            self.assertEqual(values[:-1], address.z)
            self.assertEqual(values[-1], address.bias)

    def test_a_changed_weight_or_field_is_detected(self):
        address = trained()
        with tempfile.TemporaryDirectory() as directory:
            json_path, bin_path = export_address(address, Path(directory))
            original = bin_path.read_bytes()
            bin_path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            with self.assertRaises(ValueError):
                verify_artifact(json_path)
            bin_path.write_bytes(original)
            manifest = json.loads(json_path.read_text(encoding="utf-8"))
            manifest["generation"] = 3
            json_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_artifact(json_path)

    def test_a_not_activated_address_is_exported_with_that_status(self):
        address = trained(steps=1, learning_rate=0.0)
        with tempfile.TemporaryDirectory() as directory:
            json_path, _ = export_address(address, Path(directory))
            manifest = verify_artifact(json_path)
            self.assertEqual(manifest["gate"]["status"], address.gate.status)


@unittest.skipUnless(FIXTURES.is_dir(), "fixture directory not present")
class SharedFixtureTests(unittest.TestCase):
    """The committed Rust fixture is exactly what this package produces today."""

    def test_the_committed_fixture_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            build_world(Path(directory))
            produced = sorted(p.name for p in Path(directory).iterdir())
            committed = sorted(p.name for p in FIXTURES.iterdir())
            self.assertEqual(produced, committed)
            for name in produced:
                self.assertTrue(
                    filecmp.cmp(Path(directory) / name, FIXTURES / name, shallow=False),
                    f"{name} differs: regenerate with `python -m ptr_training.dragonfly.fixture`",
                )

    def test_every_committed_artifact_verifies(self):
        manifests = [p for p in FIXTURES.glob("dragonfly-pod-*.json")]
        self.assertEqual(len(manifests), 3)
        for path in manifests:
            verify_artifact(path)

    def test_exactly_one_committed_pod_is_not_activated(self):
        statuses = sorted(
            json.loads(p.read_text(encoding="utf-8"))["gate"]["status"]
            for p in FIXTURES.glob("dragonfly-pod-*.json")
        )
        self.assertEqual(statuses, ["activated", "activated", "not-activated"])


if __name__ == "__main__":
    unittest.main()
