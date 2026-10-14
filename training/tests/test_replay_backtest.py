import math
import unittest

from ptr_training.replay import (
    ExtrapolationAbort,
    HistoryError,
    HistoryPool,
    NotPredictive,
    Outcome,
    ReplaySimulator,
    backtest,
    choose_winner,
)

RULE = {"min_within_bound": 2, "min_within_fraction": 0.5}
PINNED = {"error_bound": 0.05, **RULE}
TIGHT = {"error_bound": 0.01, **RULE}
LOOSE = {"error_bound": 0.5, **RULE}


def outcome(rate_permille: int, family: str = "typed") -> dict[str, Outcome]:
    return {family: Outcome(correct=rate_permille, total=1000)}


def linear_pool(rates=(200, 300, 400, 500)) -> HistoryPool:
    """One decision `x` moved by one each generation; the rate is linear in it."""
    pool = HistoryPool()
    for order, rate in enumerate(rates):
        pool.add_generation(f"g{order}", order=order, decisions={"x": order}, outcome=outcome(rate))
    return pool


def research_pool() -> HistoryPool:
    """The shape of the real history: decisions move together, and the last swaps a base model."""
    pool = HistoryPool()
    pool.add_generation(
        "gen3", order=0, decisions={"oversample": 0, "anchor": 0}, outcome=outcome(300)
    )
    pool.add_generation(
        "gen4", order=1, decisions={"oversample": 3, "anchor": 0}, outcome=outcome(400)
    )
    pool.add_generation(
        "gen5", order=2, decisions={"oversample": 3, "anchor": 1}, outcome=outcome(450)
    )
    pool.add_generation(
        "gen6",
        order=3,
        decisions={"oversample": 3, "anchor": 1, "base_model": "other"},
        outcome=outcome(420),
    )
    return pool


class PoolTests(unittest.TestCase):
    def test_a_repeated_name_or_a_backward_order_is_refused_and_changes_nothing(self):
        pool = linear_pool()
        before = pool.fingerprint()
        for name, order in (("g1", 10), ("fresh", 3), ("fresh", 2), ("fresh", True)):
            with self.assertRaises(HistoryError, msg=f"{name} {order}"):
                pool.add_generation(name, order=order, decisions={"x": 9}, outcome=outcome(100))
        self.assertEqual(pool.fingerprint(), before)
        self.assertEqual(len(pool.generations), 4)

    def test_a_malformed_generation_is_refused(self):
        pool = linear_pool()
        bad = (
            {"decisions": {"x": math.nan}, "outcome": outcome(1)},
            {"decisions": {"x": 1}, "outcome": {"other": Outcome(1, 2)}},
            {"decisions": {"x": 1}, "outcome": {"typed": (1, 2)}},
            {"decisions": {"x": object()}, "outcome": outcome(1)},
            {"decisions": {"": 1}, "outcome": outcome(1)},
        )
        for case in bad:
            with self.assertRaises(HistoryError, msg=str(case)):
                pool.add_generation("next", order=10, **case)
        self.assertEqual(len(pool.generations), 4)

    def test_an_outcome_of_no_cases_or_more_correct_than_total_is_refused(self):
        for correct, total in ((0, 0), (5, 4), (-1, 4)):
            with self.assertRaises(HistoryError):
                Outcome(correct=correct, total=total)
        with self.assertRaises(HistoryError):
            Outcome(correct=1.0, total=2)

    def test_the_pool_is_bounded(self):
        pool = HistoryPool()
        for order in range(256):
            pool.add_generation(f"g{order}", order=order, decisions={}, outcome=outcome(1))
        with self.assertRaises(HistoryError):
            pool.add_generation("one-more", order=999, decisions={}, outcome=outcome(1))

    def test_the_fingerprint_follows_every_outcome(self):
        self.assertEqual(linear_pool().fingerprint(), linear_pool().fingerprint())
        self.assertNotEqual(linear_pool().fingerprint(), linear_pool((200, 300, 400, 501)).fingerprint())


class BacktestTests(unittest.TestCase):
    def test_a_linear_history_is_predicted_and_the_extrapolating_end_is_flagged(self):
        report = backtest(linear_pool(), error_bound=0.01, **RULE)
        self.assertTrue(report.pipeline_ran)
        self.assertTrue(report.predictive)
        # g0 has no level-zero generation left once it is held out, so it is not identified.
        self.assertEqual((report.predicted, report.within_bound), (3, 3))
        first = report.per_generation[0]
        self.assertFalse(first["predicted"])
        self.assertTrue(first["not_identified"])
        last = report.per_generation[-1]
        self.assertTrue(last["predicted"])
        self.assertEqual(last["extrapolation"][0]["decision"], "x")

    def test_one_good_generation_among_many_bad_ones_is_not_predictive(self):
        # Under the old rule (at least one within the bound) this history was predictive.
        pool = linear_pool((200, 300, 400, 800, 500))
        loose = backtest(pool, error_bound=0.15, min_within_bound=1, min_within_fraction=0.01)
        strict = backtest(pool, error_bound=0.15, min_within_bound=1, min_within_fraction=0.9)
        self.assertGreaterEqual(loose.within_bound, 1)
        self.assertLess(loose.within_bound, loose.predicted)
        self.assertTrue(loose.predictive)
        self.assertEqual((strict.predicted, strict.within_bound), (loose.predicted, loose.within_bound))
        self.assertFalse(strict.predictive)

    def test_too_few_generations_within_the_bound_is_not_predictive(self):
        pool = linear_pool()
        report = backtest(pool, error_bound=0.01, min_within_bound=4, min_within_fraction=0.5)
        self.assertEqual(report.within_bound, 3)
        self.assertFalse(report.predictive)

    def test_nothing_predicted_means_not_predictive_although_the_pipeline_ran(self):
        report = backtest(research_pool(), error_bound=0.5, **RULE)
        self.assertTrue(report.pipeline_ran)
        self.assertFalse(report.predictive)
        self.assertEqual(report.predicted, 0)
        self.assertTrue(report.degenerate)
        self.assertTrue(all(not row["predicted"] for row in report.per_generation))

    def test_a_transition_that_moves_two_decisions_is_ambiguous_and_charged_to_neither(self):
        pool = HistoryPool()
        pool.add_generation("g0", order=0, decisions={"a": 0, "b": 0}, outcome=outcome(200))
        pool.add_generation("g1", order=1, decisions={"a": 1, "b": 1}, outcome=outcome(500))
        simulator = ReplaySimulator(pool)
        self.assertEqual(simulator.responses, {})
        self.assertEqual(simulator.ambiguous[0]["decisions"], ["a", "b"])
        self.assertFalse(simulator.simulate({"a": 1})["typed"].identified)

    def test_predictions_outside_the_bound_are_not_predictive(self):
        report = backtest(linear_pool((200, 300, 500, 900)), error_bound=0.01, **RULE)
        self.assertTrue(report.pipeline_ran)
        self.assertGreater(report.predicted, 0)
        self.assertEqual(report.within_bound, 0)
        self.assertFalse(report.predictive)

    def test_the_in_sample_fit_is_reported_but_flagged_degenerate(self):
        report = backtest(linear_pool(), error_bound=0.01, **RULE)
        self.assertEqual(report.in_sample_max_abs_error, 0.0)
        self.assertFalse(report.degenerate)
        self.assertTrue(backtest(research_pool(), error_bound=0.5, **RULE).degenerate)

    def test_the_error_bound_has_no_default_and_must_be_a_proper_fraction(self):
        pool = linear_pool()
        with self.assertRaises(TypeError):
            backtest(pool)  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            backtest(pool, error_bound=0.1)  # type: ignore[call-arg]
        for bound in (0, -0.1, 1.5, math.nan, math.inf, True, "0.1", None):
            with self.assertRaises(ValueError, msg=repr(bound)):
                backtest(pool, error_bound=bound, **RULE)
        for count in (0, -1, True, 1.5, None):
            with self.assertRaises(ValueError, msg=repr(count)):
                backtest(pool, error_bound=0.1, min_within_bound=count, min_within_fraction=0.5)
        for fraction in (0, -0.5, 1.5, math.nan, True, None):
            with self.assertRaises(ValueError, msg=repr(fraction)):
                backtest(pool, error_bound=0.1, min_within_bound=1, min_within_fraction=fraction)

    def test_an_empty_history_is_refused(self):
        with self.assertRaises(HistoryError):
            backtest(HistoryPool(), error_bound=0.1, **RULE)

    def test_the_report_serialises_both_claims_separately(self):
        record = backtest(research_pool(), error_bound=0.5, **RULE).to_dict()
        self.assertIs(record["pipeline_ran"], True)
        self.assertIs(record["predictive"], False)
        self.assertEqual(record["pool_fingerprint"], research_pool().fingerprint())


class WinnerTests(unittest.TestCase):
    def setUp(self):
        self.pool = linear_pool()
        self.simulator = ReplaySimulator(self.pool)
        self.report = backtest(self.pool, error_bound=0.01, **RULE)

    def test_the_best_identified_policy_in_the_observed_range_wins(self):
        winner = choose_winner(self.simulator.pool, self.report, [{"x": 1}, {"x": 3}, {"x": 2}], **TIGHT)
        self.assertEqual(winner["decisions"], {"x": 3})

    def test_a_winner_beyond_observed_levels_aborts(self):
        with self.assertRaises(ExtrapolationAbort):
            choose_winner(self.simulator.pool, self.report, [{"x": 1}, {"x": 9}], **TIGHT)

    def test_no_policy_is_chosen_from_a_history_that_is_not_predictive(self):
        pool = research_pool()
        report = backtest(pool, error_bound=0.5, **RULE)
        with self.assertRaises(NotPredictive):
            choose_winner(pool, report, [{"oversample": 3, "anchor": 1}], **LOOSE)

    def test_a_report_of_another_history_is_refused(self):
        other = backtest(linear_pool((200, 300, 400, 501)), error_bound=0.01, **RULE)
        with self.assertRaises(ValueError):
            choose_winner(self.simulator.pool, other, [{"x": 1}], **TIGHT)

    def test_a_policy_using_a_decision_the_history_never_moved_is_not_identified(self):
        with self.assertRaises(ValueError):
            choose_winner(self.simulator.pool, self.report, [{"unseen": 1}], **TIGHT)
        ranked = self.simulator.rank([{"unseen": 1}, {"x": 1}])
        self.assertTrue(ranked[0]["identified"])
        self.assertFalse(ranked[1]["identified"])


if __name__ == "__main__":
    unittest.main()


class CategoricalAndPrecisionTests(unittest.TestCase):
    def test_a_categorical_change_is_one_decision_that_can_be_attributed(self):
        pool = HistoryPool()
        pool.add_generation("g0", order=0, decisions={}, outcome=outcome(200))
        pool.add_generation("g1", order=1, decisions={"mode": "off"}, outcome=outcome(300))
        pool.add_generation("g2", order=2, decisions={"mode": "on"}, outcome=outcome(500))
        simulator = ReplaySimulator(pool)
        self.assertEqual(simulator.ambiguous, [])
        self.assertEqual(set(simulator.responses), {"mode"})
        self.assertEqual(pool.categories(), {"mode": ["off", "on"]})
        # not set -> off -> on: the single decision `mode` moved by one level each time.
        self.assertAlmostEqual(simulator.responses["mode"]["typed"], 0.15)
        self.assertTrue(simulator.simulate({"mode": "on"})["typed"].identified)

    def test_an_unseen_category_is_extrapolation_and_never_a_silent_winner(self):
        pool = HistoryPool()
        pool.add_generation("g0", order=0, decisions={}, outcome=outcome(200))
        pool.add_generation("g1", order=1, decisions={"mode": "off"}, outcome=outcome(300))
        pool.add_generation("g2", order=2, decisions={"mode": "on"}, outcome=outcome(500))
        simulator = ReplaySimulator(pool)
        beyond = simulator.extrapolation({"mode": "turbo"})
        self.assertEqual(beyond[0]["decision"], "mode")
        ranked = simulator.rank([{"mode": "turbo"}])
        self.assertTrue(ranked[0]["extrapolation"])
        # Encoding a policy never registers its value in the pool.
        self.assertEqual(pool.categories(), {"mode": ["off", "on"]})

    def test_a_refused_generation_leaves_the_registered_categories_alone(self):
        pool = HistoryPool()
        pool.add_generation("g0", order=0, decisions={"mode": "off"}, outcome=outcome(200))
        with self.assertRaises(HistoryError):
            pool.add_generation("g0", order=1, decisions={"mode": "on"}, outcome=outcome(300))
        with self.assertRaises(HistoryError):
            pool.add_generation("g1", order=1, decisions={"mode": "on", "x": math.nan}, outcome=outcome(300))
        self.assertEqual(pool.categories(), {"mode": ["off"]})

    def test_the_decision_uses_the_exact_error_and_only_the_report_is_rounded(self):
        total = 10_000_000

        def exact(correct):
            return {"typed": Outcome(correct=correct, total=total)}

        pool = HistoryPool()
        # g2 is predicted .4 from g0 and g1 and observed .4500004: an error of .0500004.
        for order, correct in enumerate((2_000_000, 3_000_000, 4_500_004)):
            pool.add_generation(
                f"g{order}", order=order, decisions={"x": order}, outcome=exact(correct)
            )
        report = backtest(pool, error_bound=0.05, min_within_bound=1, min_within_fraction=0.5)
        row = next(r for r in report.per_generation if r["generation"] == "g2")
        self.assertEqual(row["abs_error"]["typed"], 0.05, "the report rounds to six places")
        self.assertFalse(row["within_bound"], "the decision does not")
        # g1 loses both its edges when held out; g2 is the one predicted generation.
        self.assertEqual((report.predicted, report.within_bound), (1, 0))
        self.assertFalse(report.predictive)

    def test_an_alternating_categorical_history_anchors_its_intercept_and_predicts(self):
        pool = HistoryPool()
        rates = (200, 500, 210, 490, 205, 495)
        for order, (rate, mode) in enumerate(zip(rates, ("off", "on") * 3)):
            pool.add_generation(
                f"g{order}", order=order, decisions={"mode": mode}, outcome=outcome(rate)
            )
        self.assertEqual(pool.categories(), {"mode": ["off", "on"]})
        report = backtest(pool, error_bound=0.05, **RULE)
        self.assertGreaterEqual(report.predicted, 3)
        self.assertTrue(report.predictive)

    def test_a_step_that_overflows_the_slope_is_not_identified_and_leaves_no_nan(self):
        import json

        pool = HistoryPool()
        pool.add_generation("g0", order=0, decisions={"x": 0}, outcome=outcome(200))
        pool.add_generation("g1", order=1, decisions={"x": 5e-324}, outcome=outcome(300))
        pool.add_generation("g2", order=2, decisions={"x": 1}, outcome=outcome(400))
        simulator = ReplaySimulator(pool)
        self.assertIn("x", simulator.nonfinite)
        self.assertNotIn("x", simulator.responses)
        for family in simulator.simulate({"x": 1}).values():
            self.assertFalse(family.identified)
            self.assertIsNone(family.rate)
        report = backtest(pool, error_bound=0.1, **RULE)
        self.assertTrue(report.pipeline_ran)
        json.dumps(report.to_dict(), allow_nan=False)

    def test_a_response_times_a_huge_level_cannot_make_a_nan_prediction(self):
        pool = HistoryPool()
        pool.add_generation("g0", order=0, decisions={"x": 0}, outcome=outcome(200))
        pool.add_generation("g1", order=1, decisions={"x": 1}, outcome=outcome(300))
        simulator = ReplaySimulator(pool)
        for family in simulator.simulate({"x": 1e308}).values():
            self.assertTrue(family.rate is None or 0.0 <= family.rate <= 1.0)


class RankingNeighbourhoodAndRangeTests(unittest.TestCase):
    def test_the_ranking_uses_the_exact_score_and_only_the_report_rounds(self):
        pool = linear_pool()
        simulator = ReplaySimulator(pool)
        report = backtest(pool, error_bound=0.05, **RULE)
        # Both scores round to 0.3 at six places; the exact ones differ.
        winner = choose_winner(simulator.pool, report, [{"x": 1.0000036}, {"x": 1.000004}], **PINNED)
        self.assertEqual(winner["decisions"], {"x": 1.000004})
        self.assertEqual(winner["score_reported"], round(winner["score"], 6))
        self.assertGreater(winner["score"], simulator.rank([{"x": 1.0000036}])[0]["score"])

    def test_a_held_out_generation_takes_its_edges_with_it(self):
        pool = HistoryPool()
        for order in range(3):
            pool.add_generation(
                f"g{order}", order=order, decisions={"x": order}, outcome=outcome(200 + 100 * order)
            )
        report = backtest(pool, error_bound=0.5, min_within_bound=1, min_within_fraction=0.1)
        row = next(r for r in report.per_generation if r["generation"] == "g1")
        # Without g1 the only pair left is g0 -> g2, which is not a transition of the history:
        # x is not identified, and g1 is not predicted from an edge that spans it.
        self.assertFalse(row["predicted"])
        self.assertTrue(row["not_identified"])
        self.assertEqual(ReplaySimulator(pool, exclude="g1").responses, {})

    def test_a_decision_that_appears_later_has_level_zero_inside_its_observed_range(self):
        pool = HistoryPool()
        pool.add_generation("g0", order=0, decisions={}, outcome=outcome(200))
        pool.add_generation("g1", order=1, decisions={"x": 1}, outcome=outcome(300))
        pool.add_generation("g2", order=2, decisions={"x": 2}, outcome=outcome(400))
        simulator = ReplaySimulator(pool)
        self.assertEqual(simulator.limits["x"], (0.0, 2.0))
        self.assertEqual(simulator.extrapolation({"x": 0}), [])
        self.assertNotEqual(simulator.extrapolation({"x": 3}), [])


class ImmutableHistoryTests(unittest.TestCase):
    def test_a_generation_taken_from_the_pool_cannot_be_edited(self):
        pool = linear_pool()
        before = pool.fingerprint()
        generation = pool.generations[0]
        for mapping in (generation.decisions, generation.outcome, generation.levels):
            with self.assertRaises(TypeError):
                mapping["x"] = None  # type: ignore[index]
        self.assertEqual(pool.fingerprint(), before)

    def test_the_callers_dicts_are_copied_on_the_way_in(self):
        pool = HistoryPool()
        decisions = {"x": 1}
        pool.add_generation("g0", order=0, decisions=decisions, outcome=outcome(200))
        before = pool.fingerprint()
        decisions["x"] = 99
        decisions["y"] = 3
        self.assertEqual(pool.fingerprint(), before)
        self.assertEqual(dict(pool.generations[0].decisions), {"x": 1})


class FoldEncodingTests(unittest.TestCase):
    def test_a_category_seen_only_in_the_held_out_generation_does_not_reach_the_training_encoding(self):
        pool = HistoryPool()
        # "mode" is categorical; "beta" appears only in g1, the generation held out below, and
        # sits between "alpha" (g0) and "gamma" (g2) in the pool's own order of appearance.
        for order, (mode, rate) in enumerate([("alpha", 200), ("beta", 300), ("gamma", 400)]):
            pool.add_generation(
                f"g{order}", order=order, decisions={"mode": mode}, outcome=outcome(rate)
            )
        self.assertEqual(pool.categories()["mode"], ["alpha", "beta", "gamma"])
        fold = ReplaySimulator(pool, exclude="g1")
        # The fold knows alpha and gamma only; gamma is level 1, not the pool's level 2.
        self.assertEqual(fold._categories["mode"], ["alpha", "gamma"])
        self.assertNotIn("beta", fold._categories["mode"])
        self.assertEqual(fold._levels["g2"]["mode"], 1.0)
        # The held-out value is unseen in the fold: it falls outside the observed range.
        self.assertNotEqual(fold.extrapolation({"mode": "beta"}), [])
        # The fold with nothing excluded still sees all three.
        self.assertEqual(ReplaySimulator(pool)._categories["mode"], ["alpha", "beta", "gamma"])


class OversizedLevelTests(unittest.TestCase):
    def test_an_int_too_large_for_a_float_is_a_history_error_and_leaves_the_pool_unchanged(self):
        pool = HistoryPool()
        pool.add_generation("g0", order=0, decisions={"x": 1}, outcome=outcome(200))
        before = pool.fingerprint()
        with self.assertRaises(HistoryError):
            pool.add_generation("g1", order=1, decisions={"x": 10**10000}, outcome=outcome(300))
        self.assertEqual(pool.fingerprint(), before)
        self.assertEqual(len(pool.generations), 1)
        with self.assertRaises(HistoryError):
            ReplaySimulator(pool).simulate({"x": 10**10000})


class WinnerIsRebuiltTests(unittest.TestCase):
    def setUp(self):
        self.pool = linear_pool()
        self.report = backtest(self.pool, error_bound=0.05, **RULE)
        self.assertTrue(self.report.predictive)

    def test_a_simulator_cannot_be_handed_in(self):
        with self.assertRaises(TypeError):
            choose_winner(ReplaySimulator(self.pool, exclude="g3"), self.report, [{"x": 1}], **PINNED)

    def test_edited_fitted_state_has_no_effect_on_the_winner(self):
        expected = choose_winner(self.pool, self.report, [{"x": 1}, {"x": 3}, {"x": 2}], **PINNED)
        tampered = ReplaySimulator(self.pool, exclude="g3")
        tampered.responses["x"] = {"typed": -9.0}
        again = choose_winner(tampered.pool, self.report, [{"x": 1}, {"x": 3}, {"x": 2}], **PINNED)
        self.assertEqual(again["decisions"], expected["decisions"])
        self.assertEqual(again["score"], expected["score"])


class CountBoundsTests(unittest.TestCase):
    def test_huge_order_and_counts_are_refused_and_leave_the_pool_unchanged(self):
        pool = HistoryPool()
        pool.add_generation("g0", order=0, decisions={"x": 1}, outcome=outcome(200))
        before = pool.fingerprint()
        with self.assertRaises(HistoryError):
            pool.add_generation("g1", order=10**10000, decisions={"x": 2}, outcome=outcome(300))
        with self.assertRaises(HistoryError):
            Outcome(correct=10**10000, total=10**10001)
        with self.assertRaises(HistoryError):
            Outcome(correct=1, total=2**53)
        self.assertEqual(pool.fingerprint(), before)
        self.assertEqual(len(pool.generations), 1)


class ForgedReportTests(unittest.TestCase):
    def test_a_report_forged_as_predictive_selects_nothing(self):
        import dataclasses

        # Two generations: nothing can be predicted out of sample, so the history is not
        # predictive; a caller flips the verdict on the report.
        pool = HistoryPool()
        pool.add_generation("g0", order=0, decisions={"x": 0}, outcome=outcome(200))
        pool.add_generation("g1", order=1, decisions={"x": 1}, outcome=outcome(300))
        honest = backtest(pool, **PINNED)
        self.assertFalse(honest.predictive)
        forged = dataclasses.replace(honest, predictive=True)
        with self.assertRaises(HistoryError):
            choose_winner(pool, forged, [{"x": 1}], **PINNED)
        # The honest report is a NotPredictive abort, as before.
        with self.assertRaises(NotPredictive):
            choose_winner(pool, honest, [{"x": 1}], **PINNED)


class LevelRangeTests(unittest.TestCase):
    def test_integer_levels_beyond_the_exact_float_range_are_refused(self):
        pool = HistoryPool()
        pool.add_generation("g0", order=0, decisions={"x": 2**53 - 1}, outcome=outcome(200))
        before = pool.fingerprint()
        for bad in (2**53, 2**53 + 1, -(2**53)):
            with self.assertRaises(HistoryError):
                pool.add_generation("g1", order=1, decisions={"x": bad}, outcome=outcome(300))
        self.assertEqual(pool.fingerprint(), before)
        with self.assertRaises(HistoryError):
            ReplaySimulator(pool).simulate({"x": 2**53 + 1})


class CategoryLengthTests(unittest.TestCase):
    def test_a_long_categorical_value_is_refused_and_the_pool_is_unchanged(self):
        pool = HistoryPool()
        pool.add_generation("g0", order=0, decisions={"mode": "a" * 256}, outcome=outcome(200))
        before = pool.fingerprint()
        # 257 characters; 257 four-byte characters; and 342 three-byte characters (1026 bytes).
        for bad in ("a" * 257, "\U0001F600" * 257, "\u20ac" * 342):
            with self.assertRaises(HistoryError):
                pool.add_generation("g1", order=1, decisions={"mode": bad}, outcome=outcome(300))
        self.assertEqual(pool.fingerprint(), before)
        self.assertEqual(pool.categories(), {"mode": ["a" * 256]})
        with self.assertRaises(HistoryError):
            ReplaySimulator(pool).simulate({"mode": "b" * 257})
