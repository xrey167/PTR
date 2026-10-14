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
        report = backtest(linear_pool(), error_bound=0.01)
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

    def test_nothing_predicted_means_not_predictive_although_the_pipeline_ran(self):
        report = backtest(research_pool(), error_bound=0.5)
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
        report = backtest(linear_pool((200, 300, 500, 900)), error_bound=0.01)
        self.assertTrue(report.pipeline_ran)
        self.assertGreater(report.predicted, 0)
        self.assertEqual(report.within_bound, 0)
        self.assertFalse(report.predictive)

    def test_the_in_sample_fit_is_reported_but_flagged_degenerate(self):
        report = backtest(linear_pool(), error_bound=0.01)
        self.assertEqual(report.in_sample_max_abs_error, 0.0)
        self.assertFalse(report.degenerate)
        self.assertTrue(backtest(research_pool(), error_bound=0.5).degenerate)

    def test_the_error_bound_has_no_default_and_must_be_a_proper_fraction(self):
        pool = linear_pool()
        with self.assertRaises(TypeError):
            backtest(pool)  # type: ignore[call-arg]
        for bound in (0, -0.1, 1.5, math.nan, math.inf, True, "0.1", None):
            with self.assertRaises(ValueError, msg=repr(bound)):
                backtest(pool, error_bound=bound)

    def test_an_empty_history_is_refused(self):
        with self.assertRaises(HistoryError):
            backtest(HistoryPool(), error_bound=0.1)

    def test_the_report_serialises_both_claims_separately(self):
        record = backtest(research_pool(), error_bound=0.5).to_dict()
        self.assertIs(record["pipeline_ran"], True)
        self.assertIs(record["predictive"], False)
        self.assertEqual(record["pool_fingerprint"], research_pool().fingerprint())


class WinnerTests(unittest.TestCase):
    def setUp(self):
        self.pool = linear_pool()
        self.simulator = ReplaySimulator(self.pool)
        self.report = backtest(self.pool, error_bound=0.01)

    def test_the_best_identified_policy_in_the_observed_range_wins(self):
        winner = choose_winner(self.simulator, self.report, [{"x": 1}, {"x": 3}, {"x": 2}])
        self.assertEqual(winner["decisions"], {"x": 3})

    def test_a_winner_beyond_observed_levels_aborts(self):
        with self.assertRaises(ExtrapolationAbort):
            choose_winner(self.simulator, self.report, [{"x": 1}, {"x": 9}])

    def test_no_policy_is_chosen_from_a_history_that_is_not_predictive(self):
        pool = research_pool()
        report = backtest(pool, error_bound=0.5)
        with self.assertRaises(NotPredictive):
            choose_winner(ReplaySimulator(pool), report, [{"oversample": 3, "anchor": 1}])

    def test_a_report_of_another_history_is_refused(self):
        other = backtest(linear_pool((200, 300, 400, 501)), error_bound=0.01)
        with self.assertRaises(ValueError):
            choose_winner(self.simulator, other, [{"x": 1}])

    def test_a_policy_using_a_decision_the_history_never_moved_is_not_identified(self):
        with self.assertRaises(ValueError):
            choose_winner(self.simulator, self.report, [{"unseen": 1}])
        ranked = self.simulator.rank([{"unseen": 1}, {"x": 1}])
        self.assertTrue(ranked[0]["identified"])
        self.assertFalse(ranked[1]["identified"])


if __name__ == "__main__":
    unittest.main()
