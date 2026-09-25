"""Hand-checked label-rule cases, run against both implementations.

Every expected label below was worked out by hand from README.md's rule table,
not by running either implementation. Notation used in the comments:

  operator codes / costs: semantic 0 (0.2), deductive 1 (0.3), probabilistic 2 (0.5),
    statistical 3 (0.5), temporal 4 (0.5), causal 5 (0.6), search 6 (0.6),
    optimization 7 (0.8), simulation 8 (0.8), symbolic 9 (0.4), external_pod 10 (0.9)
  per usable fact, the budget penalty pen_k = max(0, COST_k - BUDGET_B):
    B=0 (0.4): prob/stat/temp 0.1, causal/search 0.2, optim/sim 0.4, ext 0.5, else 0
    B=1 (0.7): optim/sim 0.1, ext 0.2, else 0
    B=2 (1.0): 0 everywhere
  G by bucket: cb0 0, cb1 0.5, cb2 1, cb3 1, cb4 1.25
  W / U: unknown .30/.35, assumed .55/.60, hypothesis .80/.85,
         observed 1.30/0, inferred 1.05/0, verified 1.55/0
  confidence draws: k=100 -> c .1005 (cb0), k=300 -> .3005 (cb1), k=500 -> .5005 (cb2),
                    k=700 -> .7005 (cb3), k=900 -> .9005 (cb4)
"""

from __future__ import annotations

import unittest

from _support import gen, score

K_FOR_BUCKET = {0: 100, 1: 300, 2: 500, 3: 700, 4: 900}
REGIME = {"tabular": 0, "temporal": 1, "interventional": 2, "textual": 3}


def fact(role, epistemic, bucket, validity="live", entity=0):
    return (role, epistemic, bucket, validity, entity)


def gen_label(facts, regime, budget, **kw):
    encoded = [
        (gen.ROLE[r], gen.EPI[e], gen.confidence(K_FOR_BUCKET[cb]), gen.VALIDITY[v], ent)
        for r, e, cb, v, ent in facts
    ]
    return gen.label_of(encoded, REGIME[regime], budget, **kw)


def score_label(facts, regime, budget):
    encoded = []
    for r, e, cb, v, ent in facts:
        text = f"{gen.confidence(K_FOR_BUCKET[cb]):.4f}"
        encoded.append(score.Fact(r, e, text, score.bucket_of(text), v, ent))
    return score.winner(score.scores(encoded, regime, score.BUDGETS[budget]))


# (name, facts, regime, budget index, expected operator)
CASES = [
    (
        # Only usable-free facts: a Live fact with cb 0 and a revoked one. z = 0 for all
        # 11 operators, an exact 11-way tie; the lowest COST is semantic (0.2).
        "all-zero exact tie goes to the cheapest operator",
        [fact("goal", "observed", 0), fact("claim", "verified", 4, "revoked")],
        "tabular", 2, "semantic",
    ),
    (
        # goal/observed/cb2: search 1*1.30*2.0 = 2.60, optimization 1.30*0.9 = 1.17. No penalty.
        "one fact votes for its primary",
        [fact("goal", "observed", 2)],
        "tabular", 2, "search",
    ),
    (
        # resource/unknown/cb1, G .5: external_pod .5*.3*2 = 0.30, search .5*.3*.9 = 0.135,
        # probabilistic .5*.35 = 0.175. B=2: external_pod 0.30 wins.
        "budget 1.0: no penalty, external_pod wins",
        [fact("resource", "unknown", 1)],
        "textual", 2, "external_pod",
    ),
    (
        # Same fact, B=0: ext 0.30-0.5 = -0.20, search 0.135-0.2 = -0.065, prob 0.175-0.1 = 0.075,
        # semantic/deductive/symbolic 0. probabilistic 0.075 wins.
        "budget 0.4: the penalty flips external_pod to probabilistic",
        [fact("resource", "unknown", 1)],
        "textual", 0, "probabilistic",
    ),
    (
        # Same fact, B=1: ext 0.30-0.2 = 0.10, search 0.135, prob 0.175. probabilistic wins.
        "budget 0.7: probabilistic still beats search and external_pod",
        [fact("resource", "unknown", 1)],
        "textual", 1, "probabilistic",
    ),
    (
        # As the B=0 case plus a Live goal/observed fact with cb 0. The cb 0 fact is not usable,
        # so it adds neither a vote nor a penalty: probabilistic 0.075 still wins. (Penalising it
        # would give prob 0.175-0.2 = -0.025 < semantic 0 and flip the label to semantic.)
        "a cb-0 fact is not penalised",
        [fact("resource", "unknown", 1), fact("goal", "observed", 0)],
        "textual", 0, "probabilistic",
    ),
    (
        # goal/observed/cb2 -> search 2.60; relation/observed/cb2 -> causal 2.60 (+ optimization
        # 1.17, deductive 1.17). search and causal tie exactly and both cost 0.6, so the lower
        # code wins: causal (5) < search (6).
        "exact tie at equal cost goes to the lower code",
        [fact("goal", "observed", 2), fact("relation", "observed", 2)],
        "tabular", 2, "causal",
    ),
    (
        # procedure/observed/cb2 -> symbolic 2.60; evidence/observed/cb2 under tabular ->
        # statistical 2.60 (+ probabilistic 1.17). symbolic costs 0.4 < statistical 0.5, so
        # symbolic (code 9) beats the lower code statistical (3).
        "exact tie goes to the lower cost before the lower code",
        [fact("procedure", "observed", 2), fact("evidence", "observed", 2)],
        "tabular", 2, "symbolic",
    ),
    (
        # claim/verified/cb4 is disputed, so only goal/inferred/cb2 counts: search 1.05*2 = 2.10,
        # optimization 0.945. (Admitted, the claim would give deductive 1.25*1.55*2 = 3.875.)
        "a non-Live fact is not admitted",
        [fact("claim", "verified", 4, "disputed"), fact("goal", "inferred", 2)],
        "tabular", 2, "search",
    ),
    (
        # evidence/verified/cb3 under temporal: temporal 1.55*2 = 3.10, probabilistic 1.395.
        "Evidence votes for op(R): temporal",
        [fact("evidence", "verified", 3)],
        "temporal", 2, "temporal",
    ),
    (
        # The same fact under textual: semantic 3.10, probabilistic 1.395.
        "Evidence votes for op(R): textual -> semantic",
        [fact("evidence", "verified", 3)],
        "textual", 2, "semantic",
    ),
    (
        # constraint/assumed/cb4: optimization 1.25*.55*2 = 1.375, symbolic 1.25*.55*.9 = 0.61875,
        # probabilistic 1.25*.60 = 0.75. action/observed/cb1: external_pod .5*1.3*2 = 1.30,
        # temporal .5*1.3*.9 = 0.585. B=2: optimization 1.375 > external_pod 1.30.
        "two facts, budget 1.0: optimization",
        [fact("constraint", "assumed", 4), fact("action", "observed", 1)],
        "tabular", 2, "optimization",
    ),
    (
        # Same facts, B=0, two usable facts so each penalty counts twice:
        # optimization 1.375-0.8 = 0.575, external_pod 1.30-1.0 = 0.30,
        # probabilistic 0.75-0.2 = 0.55, temporal 0.585-0.2 = 0.385, symbolic 0.61875-0 = 0.61875,
        # semantic/deductive 0. symbolic 0.61875 wins.
        "two facts, budget 0.4: the doubled penalty makes symbolic win",
        [fact("constraint", "assumed", 4), fact("action", "observed", 1)],
        "tabular", 0, "symbolic",
    ),
    (
        # claim/unknown/cb2 under tabular: deductive .3*2 = 0.60, statistical .3*.9 = 0.27,
        # probabilistic 0.35. evidence/unknown/cb2 under tabular: statistical 0.60,
        # probabilistic .3*.9 + .35 = 0.62. Totals: deductive 0.60, statistical 0.87,
        # probabilistic 0.97. probabilistic wins through the U term.
        "the U term decides for probabilistic",
        [fact("claim", "unknown", 2), fact("evidence", "unknown", 2)],
        "tabular", 2, "probabilistic",
    ),
    (
        # claim/observed/cb2 under interventional: deductive 2.60, causal (op(R)) 1.17.
        # capability/inferred/cb4: simulation 1.25*1.05*2 = 2.625, external_pod 1.25*1.05*.9 = 1.18125.
        # B=1: simulation 2.625-0.2 = 2.425, deductive 2.60, external_pod 1.18125-0.4 = 0.78125.
        # deductive 2.60 wins.
        "budget 0.7 penalty flips simulation to deductive",
        [fact("claim", "observed", 2), fact("capability", "inferred", 4)],
        "interventional", 1, "deductive",
    ),
    (
        # Same facts, B=2: simulation 2.625 > deductive 2.60.
        "budget 1.0 keeps simulation",
        [fact("claim", "observed", 2), fact("capability", "inferred", 4)],
        "interventional", 2, "simulation",
    ),
]


class HandCheckedRuleCases(unittest.TestCase):
    def test_there_are_at_least_twelve_cases(self):
        self.assertGreaterEqual(len(CASES), 12)

    def test_generator_rule(self):
        for name, facts, regime, budget, expected in CASES:
            with self.subTest(name):
                self.assertEqual(gen.OP_NAMES[gen_label(facts, regime, budget)], expected)

    def test_score_rule(self):
        for name, facts, regime, budget, expected in CASES:
            with self.subTest(name):
                self.assertEqual(score.OPERATOR_BY_CODE[score_label(facts, regime, budget)], expected)


class TieTolerance(unittest.TestCase):
    """Values within 1e-9 of the maximum are tied; ties go to lower COST, then lower code."""

    def vector(self, **values):
        z = [0.0] * 11
        for name, value in values.items():
            z[gen.OP[name]] = value
        return z

    def both(self, z):
        return gen.OP_NAMES[gen.decide(z)[0]], score.OPERATOR_BY_CODE[score.winner(z)]

    def test_within_tolerance_is_a_tie(self):
        # symbolic 1.0 + 5e-10 is within 1e-9 of the max, so it ties with deductive 1.0;
        # deductive costs 0.3 < symbolic 0.4.
        z = self.vector(deductive=1.0, symbolic=1.0 + 5e-10)
        self.assertEqual(self.both(z), ("deductive", "deductive"))
        self.assertTrue(gen.decide(z)[1])

    def test_outside_tolerance_is_not(self):
        z = self.vector(deductive=1.0, symbolic=1.0 + 2e-9)
        self.assertEqual(self.both(z), ("symbolic", "symbolic"))
        self.assertFalse(gen.decide(z)[1])

    def test_float_noise_is_a_tie(self):
        # 0.1 + 0.2 != 0.3 in float64; the tolerance makes them equal, and temporal (0.5)
        # loses to statistical (0.5) on code.
        z = self.vector(temporal=0.1 + 0.2, statistical=0.3)
        self.assertEqual(self.both(z), ("statistical", "statistical"))


class Counterfactuals(unittest.TestCase):
    """Tags on hand-checked examples, from both implementations."""

    def gen_tags(self, facts, regime, budget, split="test_iid"):
        encoded = [
            (gen.ROLE[r], gen.EPI[e], gen.confidence(K_FOR_BUCKET[cb]), gen.VALIDITY[v], ent)
            for r, e, cb, v, ent in facts
        ]
        example = gen.Example(split, 0, REGIME[regime], budget, encoded, [])
        return gen.tags_of(example, gen.label_of(encoded, REGIME[regime], budget))

    def score_tags(self, facts, regime, budget, split="test_iid"):
        item = score.Item()
        item.id, item.split, item.regime, item.budget, item.tokens = "x", split, REGIME[regime], budget, []
        item.facts = []
        for r, e, cb, v, ent in facts:
            text = f"{gen.confidence(K_FOR_BUCKET[cb]):.4f}"
            item.facts.append(score.Fact(r, e, text, score.bucket_of(text), v, ent))
        score.derive(item)
        return item.tags

    def test_validity_tag(self):
        # Admitting the disputed claim turns search (2.10) into deductive (3.875).
        facts = [fact("claim", "verified", 4, "disputed"), fact("goal", "inferred", 2)]
        self.assertIn("validity", self.gen_tags(facts, "tabular", 2))
        self.assertIn("validity", self.score_tags(facts, "tabular", 2))

    def test_regime_tag_only_through_claim_and_evidence(self):
        # A lone goal fact ignores the regime; a lone evidence fact does not.
        goal = [fact("goal", "observed", 2)]
        evidence = [fact("evidence", "verified", 3)]
        self.assertNotIn("regime", self.gen_tags(goal, "tabular", 2))
        self.assertIn("regime", self.gen_tags(evidence, "temporal", 2))
        self.assertNotIn("regime", self.score_tags(goal, "tabular", 2))
        self.assertIn("regime", self.score_tags(evidence, "temporal", 2))

    def test_budget_tag(self):
        # resource/unknown/cb1: external_pod at B=2, probabilistic at B=0 and B=1.
        facts = [fact("resource", "unknown", 1)]
        self.assertIn("budget", self.gen_tags(facts, "textual", 2))
        self.assertIn("budget", self.score_tags(facts, "textual", 2))

    def test_confidence_tag_includes_cb0_facts(self):
        # resource/unknown/cb1 at B=0 gives probabilistic 0.075. With G = 1 the cb-0
        # goal/observed fact joins: search .3*.9 + 1.3*2 - 2*0.2 = 2.47 wins, so the tag is set.
        facts = [fact("resource", "unknown", 1), fact("goal", "observed", 0)]
        self.assertIn("confidence", self.gen_tags(facts, "textual", 0))
        self.assertIn("confidence", self.score_tags(facts, "textual", 0))

    def test_epistemic_tag(self):
        # The U-term case. With W = 1 and U = 0 the claim votes deductive 2.0 / statistical
        # 0.9 and the evidence votes statistical 2.0 / probabilistic 0.9: statistical 2.9
        # wins instead of probabilistic.
        facts = [fact("claim", "unknown", 2), fact("evidence", "unknown", 2)]
        self.assertIn("epistemic", self.gen_tags(facts, "tabular", 2))
        self.assertIn("epistemic", self.score_tags(facts, "tabular", 2))

    def test_heldout_tag(self):
        # (constraint, hypothesis) is held out.
        # No tag: constraint/hypothesis/cb1 votes optimization .5*.8*2 = 0.8; goal/observed/cb4
        # votes search 1.25*1.3*2 = 3.25 and optimization 1.25*1.3*.9 = 1.4625. search 3.25 beats
        # optimization 2.2625, and without the held-out fact search still wins.
        # Tag: constraint/hypothesis/cb4 votes optimization 1.25*.8*2 = 2.0 (probabilistic
        # 1.0625); goal/unknown/cb2 votes search 0.6, optimization 0.27, probabilistic 0.35.
        # optimization 2.27 wins; without the held-out fact search 0.6 wins.
        no_tag = [fact("constraint", "hypothesis", 1), fact("goal", "observed", 4)]
        tag = [fact("constraint", "hypothesis", 4), fact("goal", "unknown", 2)]
        self.assertNotIn("heldout", self.gen_tags(no_tag, "tabular", 2, "ood_compose_epi"))
        self.assertIn("heldout", self.gen_tags(tag, "tabular", 2, "ood_compose_epi"))
        self.assertNotIn("heldout", self.score_tags(no_tag, "tabular", 2, "ood_compose_epi"))
        self.assertIn("heldout", self.score_tags(tag, "tabular", 2, "ood_compose_epi"))

    def test_transfer_tag(self):
        # evidence/verified/cb3 under interventional votes causal 3.10: label causal. Ignored,
        # the label is semantic (all zero); as tabular/temporal/textual it is statistical,
        # temporal, semantic. None is causal, so the example is in the strict transfer subset.
        facts = [fact("evidence", "verified", 3)]
        self.assertIn("transfer", self.gen_tags(facts, "interventional", 2, "ood_compose_regime"))
        self.assertIn("transfer", self.score_tags(facts, "interventional", 2, "ood_compose_regime"))


if __name__ == "__main__":
    unittest.main()
