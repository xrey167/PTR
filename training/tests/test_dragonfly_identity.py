import json
import unittest
from pathlib import Path

from ptr_training.dragonfly.identity import PodIdentityDescriptor

FIXTURES = Path(__file__).resolve().parents[2] / "crates/ptr-router/tests/fixtures"


def descriptor(**changes):
    base = dict(
        domain="supply",
        kind="Retrieval",
        semantic_role="reader",
        predicate="lead_time",
        aliases=frozenset({"lead time", "delivery time"}),
    )
    base.update(changes)
    return PodIdentityDescriptor(**base)


class PodIdentityTests(unittest.TestCase):
    def test_identity_and_cluster_key_are_deterministic(self):
        self.assertEqual(descriptor().identity(), descriptor().identity())
        self.assertEqual(descriptor().cluster_key(), descriptor().cluster_key())

    def test_alias_change_moves_the_identity_but_not_the_cluster(self):
        base = descriptor()
        changed = descriptor(aliases=base.aliases | {"shipping delay"})
        self.assertNotEqual(base.identity(), changed.identity())
        self.assertEqual(base.cluster_key(), changed.cluster_key())

    def test_role_domain_kind_and_predicate_move_both(self):
        base = descriptor()
        for changes in (
            {"semantic_role": "writer"},
            {"domain": "finance"},
            {"kind": "Memory"},
            {"predicate": None},
        ):
            changed = descriptor(**changes)
            self.assertNotEqual(base.identity(), changed.identity(), changes)
            self.assertNotEqual(base.cluster_key(), changed.cluster_key(), changes)

    def test_field_boundaries_cannot_be_shifted(self):
        left = descriptor(domain="ab", kind="c")
        right = descriptor(domain="a", kind="bc")
        self.assertNotEqual(left.identity(), right.identity())

    def test_identity_is_prefixed_hex(self):
        identity = descriptor().identity()
        self.assertTrue(identity.startswith("pod:"))
        self.assertEqual(len(identity), len("pod:") + 64)

    def test_validate_rejects_empty_fields(self):
        for changes in (
            {"domain": ""},
            {"kind": ""},
            {"semantic_role": ""},
            {"predicate": ""},
            {"aliases": frozenset({""})},
            {"schema_version": 2},
        ):
            with self.assertRaises(ValueError, msg=str(changes)):
                descriptor(**changes).validate()
        descriptor().validate()

    @unittest.skipUnless(FIXTURES.is_dir(), "fixture directory not present")
    def test_matches_the_vectors_the_rust_test_checks(self):
        expected = json.loads((FIXTURES / "dragonfly-expected.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(expected["identity_vectors"]), 3)
        for vector in expected["identity_vectors"]:
            current = PodIdentityDescriptor.from_json(vector["descriptor"])
            self.assertEqual(current.identity(), vector["identity"])
            self.assertEqual(current.cluster_key().hex(), vector["cluster_key"])


if __name__ == "__main__":
    unittest.main()
