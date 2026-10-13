import math
import unittest

from ptr_training.dragonfly import (
    ALIAS_THRESHOLD_PERMILLE,
    NEGATIVE_REJECT_THRESHOLD_PERMILLE,
    HashedBagOfWordsEncoder,
    PodIdentityDescriptor,
    PodTrainingSpec,
    TrainConfig,
    evaluate_gate,
    score,
    train_pod_address,
)
from ptr_training.dragonfly.encoder import SentenceTransformerEncoder, l2_normalize, normalized
from ptr_training.dragonfly.gate import round_f32
from ptr_training.dragonfly.train import expand_alias_questions

ENCODER = HashedBagOfWordsEncoder(32)

QUESTIONS = (
    "How long is the delivery time of {name}?",
    "Is {name} late with the X12 shipment?",
    "What did {name} quote for the lead time?",
    "Delay of the procurement order at {name}",
    "Which delivery date did {name} confirm?",
)
ALIASES = {
    "Mueller GmbH": ("Mueller GmbH", "Muller Werke"),
    "Schmidt AG": ("Schmidt AG", "Schmitt Logistik"),
}


def spec(name="Mueller GmbH", *, extra_negatives=(), negative_aliases=None, generation=1):
    other = next(n for n in ALIASES if n != name)
    return PodTrainingSpec(
        pod_id=f"pod-{name.split()[0].lower()}",
        descriptor=PodIdentityDescriptor(
            domain="supply",
            kind="Retrieval",
            predicate="lead_time",
            semantic_role=f"supplier-{name.split()[0].lower()}",
            aliases=frozenset(ALIASES[name]),
        ),
        generation=generation,
        positives=tuple(q.format(name=name) for q in QUESTIONS),
        negatives=tuple(q.format(name=other) for q in QUESTIONS) + tuple(extra_negatives),
        negative_aliases=tuple(ALIASES[other]) if negative_aliases is None else negative_aliases,
    )


class EncoderTests(unittest.TestCase):
    def test_hashed_encoder_is_deterministic_and_unit_length(self):
        first = ENCODER.encode("Delivery time of Mueller GmbH")
        self.assertEqual(first, ENCODER.encode("Delivery time of Mueller GmbH"))
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in first)), 1.0, places=12)
        self.assertEqual(len(first), ENCODER.dims)

    def test_hashed_encoder_ignores_case_accents_and_spacing(self):
        self.assertEqual(ENCODER.encode("Müller  GMBH"), ENCODER.encode("muller gmbh"))

    def test_near_spellings_are_closer_than_unrelated_names(self):
        a, b, c = (ENCODER.encode(t) for t in ("Mueller", "Muller", "Kowalski"))
        dot = lambda x, y: sum(p * q for p, q in zip(x, y))
        self.assertGreater(dot(a, b), dot(a, c))

    def test_text_without_words_is_refused(self):
        with self.assertRaises(ValueError):
            ENCODER.encode("  ?! ")

    def test_normalized_strips_accents_and_case(self):
        self.assertEqual(normalized("  Ärger  ÜBER Ølen "), "arger uber ølen")

    def test_zero_vector_cannot_be_normalised(self):
        with self.assertRaises(ValueError):
            l2_normalize([0.0, 0.0])

    def test_production_encoder_needs_its_extra(self):
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            with self.assertRaises(ImportError) as context:
                SentenceTransformerEncoder("model", "rev")
            self.assertIn("dragonfly", str(context.exception))
        else:  # pragma: no cover - only with the extra installed
            self.skipTest("sentence-transformers is installed")


class AliasExpansionTests(unittest.TestCase):
    def test_aliases_substitute_into_questions_and_join_as_bare_names(self):
        aliases, expanded = expand_alias_questions(
            ["Mueller GmbH", "Muller Werke"], ["Delivery time of Mueller GmbH?"]
        )
        self.assertEqual(aliases, ["Mueller GmbH", "Muller Werke"])
        self.assertIn("Delivery time of Muller Werke?", expanded)
        self.assertIn("Muller Werke", expanded)
        self.assertIn("muller werke", expanded)

    def test_no_trusted_alias_is_an_error(self):
        with self.assertRaises(ValueError):
            expand_alias_questions([], ["anything"])


class TrainingTests(unittest.TestCase):
    def test_a_well_separated_pod_is_activated(self):
        trained = train_pod_address(spec(), ENCODER)
        self.assertTrue(trained.activated)
        self.assertEqual(trained.gate.status, "activated")
        self.assertGreaterEqual(trained.gate.alias_min_permille, ALIAS_THRESHOLD_PERMILLE)
        self.assertGreaterEqual(
            trained.gate.negative_reject_permille, NEGATIVE_REJECT_THRESHOLD_PERMILLE
        )
        self.assertLess(trained.loss_after, trained.loss_before)
        self.assertEqual(len(trained.z), ENCODER.dims)

    def test_training_is_deterministic(self):
        first = train_pod_address(spec(), ENCODER)
        second = train_pod_address(spec(), ENCODER)
        self.assertEqual(first.z, second.z)
        self.assertEqual(first.bias, second.bias)

    def test_parameters_are_exactly_float32(self):
        trained = train_pod_address(spec(), ENCODER)
        self.assertTrue(all(round_f32(v) == v for v in trained.z))
        self.assertEqual(round_f32(trained.bias), trained.bias)

    def test_an_untrained_pod_does_not_pass_the_gate(self):
        trained = train_pod_address(spec(), ENCODER, TrainConfig(steps=1, learning_rate=0.0))
        self.assertFalse(trained.activated)
        self.assertEqual(trained.gate.status, "not-activated")
        self.assertLess(trained.gate.negative_reject_permille, NEGATIVE_REJECT_THRESHOLD_PERMILLE)

    def test_pods_are_trained_independently(self):
        mueller = train_pod_address(spec("Mueller GmbH"), ENCODER)
        schmidt = train_pod_address(spec("Schmidt AG"), ENCODER)
        alone = train_pod_address(spec("Mueller GmbH"), ENCODER)
        # Training another pod changes nothing about this one: no shared trainable state.
        self.assertEqual(mueller.z, alone.z)
        self.assertNotEqual(mueller.z, schmidt.z)

    def test_the_address_prefers_its_own_pod_name(self):
        mueller = train_pod_address(spec("Mueller GmbH"), ENCODER)
        own = ENCODER.encode("Which delivery date did Mueller GmbH confirm?")
        foreign = ENCODER.encode("Which delivery date did Schmidt AG confirm?")
        args = dict(
            cosine_weight=mueller.cosine_weight, metadata_weight=mueller.metadata_weight
        )
        self.assertGreater(
            score(own, mueller.z, mueller.bias, **args),
            score(foreign, mueller.z, mueller.bias, **args),
        )

    def test_inputs_are_validated(self):
        base = spec()
        with self.assertRaises(ValueError):
            train_pod_address(PodTrainingSpec(**{**vars(base), "negatives": (), "negative_aliases": ()}), ENCODER)
        with self.assertRaises(ValueError):
            train_pod_address(PodTrainingSpec(**{**vars(base), "positives": ()}), ENCODER)
        with self.assertRaises(ValueError):
            train_pod_address(PodTrainingSpec(**{**vars(base), "generation": 0}), ENCODER)
        with self.assertRaises(ValueError):
            train_pod_address(spec(), ENCODER, TrainConfig(steps=0))
        with self.assertRaises(ValueError):
            train_pod_address(PodTrainingSpec(**{**vars(base), "pod_id": ""}), ENCODER)

    def test_conflicting_labels_are_refused(self):
        base = spec()
        conflicting = PodTrainingSpec(
            **{**vars(base), "negatives": base.negatives + (base.positives[0],)}
        )
        with self.assertRaises(ValueError):
            train_pod_address(conflicting, ENCODER)

    def test_a_trusted_alias_may_not_be_a_negative_example(self):
        with self.assertRaises(ValueError):
            train_pod_address(spec(negative_aliases=("Muller Werke",)), ENCODER)


class GateTests(unittest.TestCase):
    def test_gate_floors_permille_and_requires_both_halves(self):
        z = [1.0, 0.0, 0.0]
        result = evaluate_gate(
            alias_vectors=[[1.0, 0.0, 0.0]],
            negative_vectors=[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            z=z,
            bias=10.0,
        )
        self.assertEqual(result.alias_count, 1)
        self.assertEqual(result.negative_count, 2)
        self.assertTrue(result.activated)
        self.assertEqual(result.negative_reject_permille, 1000)

    def test_a_gate_that_measured_nothing_does_not_pass(self):
        result = evaluate_gate(alias_vectors=[], negative_vectors=[], z=[1.0, 0.0], bias=0.0)
        self.assertFalse(result.activated)
        self.assertEqual(result.status, "not-activated")

    def test_a_weak_alias_fails_the_gate(self):
        result = evaluate_gate(
            alias_vectors=[[-1.0, 0.0, 0.0]],
            negative_vectors=[[0.0, 1.0, 0.0]],
            z=[1.0, 0.0, 0.0],
            bias=10.0,
        )
        self.assertFalse(result.activated)


if __name__ == "__main__":
    unittest.main()
