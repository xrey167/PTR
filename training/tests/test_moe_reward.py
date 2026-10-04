import json
import unittest

from ptr_training.moe import reward


def arith_spec(v):
    return json.dumps({"value": v})


def json_spec(values):
    return json.dumps({"values": values})


class MoeRewardTests(unittest.TestCase):
    def test_arith_pass(self):
        r = reward.verify("arith", arith_spec(19), "19")
        self.assertEqual((r.status, r.score, r.findings), ("Pass", 1.0, []))
        self.assertEqual(r.level, "Deterministic")

    def test_arith_pass_ignores_whitespace_and_trailing_lines(self):
        self.assertEqual(reward.reward("arith", arith_spec(-3), " -3 \njunk"), 1.0)

    def test_arith_wrong_value_has_diagnostic_score_hard_finding_zero_reward(self):
        r = reward.verify("arith", arith_spec(19), "20")
        self.assertEqual(r.status, "Fail")
        self.assertEqual(r.score, reward.SHAPE_CREDIT)
        self.assertEqual([f.code for f in r.findings], ["wrong-value"])
        self.assertTrue(r.findings[0].hard)

    def test_arith_not_integer(self):
        for text in ("abc", "1.5", "1 2"):
            r = reward.verify("arith", arith_spec(1), text)
            self.assertEqual((r.status, r.score), ("Fail", 0.0), text)
            self.assertEqual(r.findings[0].code, "not-integer")

    def test_json_pass(self):
        r = reward.verify("json", json_spec({"a": 3, "b": 5}), '{"a":3,"b":5}')
        self.assertEqual((r.status, r.score), ("Pass", 1.0))

    def test_json_graded_failures(self):
        spec = json_spec({"a": 3})
        cases = {
            "nonsense": (0.0, "invalid-json"),
            "[1]": (reward.JSON_PARSE_CREDIT, "not-object"),
            '{"z":3}': (reward.JSON_PARSE_CREDIT, "wrong-keys"),
            '{"a":4}': (reward.JSON_PARSE_CREDIT + reward.JSON_KEYS_CREDIT, "wrong-values"),
            '{"a":"3"}': (reward.JSON_PARSE_CREDIT + reward.JSON_KEYS_CREDIT, "wrong-values"),
        }
        for text, (score, code) in cases.items():
            r = reward.verify("json", spec, text)
            self.assertEqual(r.status, "Fail", text)
            self.assertEqual(reward.rl_reward(r), 0.0, text)
            self.assertAlmostEqual(r.score, score, msg=text)
            self.assertEqual(r.findings[0].code, code, text)

    def test_all_four_statuses_and_reward_mapping(self):
        cases = {
            "Pass": ("19", 1.0),
            "Fail": ("20", 0.0),
            "Disputed": ("19\n20", 0.0),
            "Unknown": ("  \n", 0.0),
        }
        for status, (text, expected_reward) in cases.items():
            r = reward.verify("arith", arith_spec(19), text)
            self.assertEqual(r.status, status, text)
            self.assertEqual(reward.rl_reward(r), expected_reward, status)
            self.assertEqual(reward.reward("arith", arith_spec(19), text), expected_reward)
        self.assertEqual(reward.STATUSES, ("Pass", "Fail", "Disputed", "Unknown"))

    def test_json_unknown_on_empty(self):
        r = reward.verify("json", json_spec({"a": 1}), "")
        self.assertEqual(r.status, "Unknown")
        self.assertEqual(reward.rl_reward(r), 0.0)

    def test_non_pass_independent_score_is_never_reward(self):
        for status in ("Fail", "Disputed", "Unknown"):
            r = reward.VerificationReport(status, reward.LEVEL, 0.9, [])
            self.assertEqual(reward.rl_reward(r), 0.0, status)
        # graded diagnostics stay in the report but not in the reward
        r = reward.verify("arith", arith_spec(19), "20")
        self.assertEqual(r.score, reward.SHAPE_CREDIT)
        self.assertEqual(reward.rl_reward(r), 0.0)

    def test_hard_finding_forces_fail(self):
        hard = [reward.Finding("x", "bad", True)]
        r = reward.VerificationReport("Pass", reward.LEVEL, 1.0, hard)
        self.assertEqual(r.status, "Fail")
        self.assertEqual(reward.rl_reward(r), 0.0)
        soft = [reward.Finding("x", "note", False)]
        r = reward.VerificationReport("Pass", reward.LEVEL, 0.7, soft)
        self.assertEqual((r.status, reward.rl_reward(r)), ("Pass", 0.7))

    def test_pass_requires_positive_score(self):
        r = reward.VerificationReport("Pass", reward.LEVEL, 0.0, [])
        self.assertEqual((r.status, reward.rl_reward(r)), ("Fail", 0.0))

    def test_invalid_status_or_score_rejected(self):
        with self.assertRaises(ValueError):
            reward.VerificationReport("Maybe", reward.LEVEL, 0.5, [])
        with self.assertRaises(ValueError):
            reward.VerificationReport("Fail", reward.LEVEL, 1.5, [])

    def test_score_always_in_unit_interval(self):
        for text in ("", "1", "{}", '{"a":3}', "\u0000"):
            for kind, spec in (("arith", arith_spec(1)), ("json", json_spec({"a": 3}))):
                self.assertTrue(0.0 <= reward.reward(kind, spec, text) <= 1.0)

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            reward.verify("nope", "{}", "x")

    def test_batch_matches_trl_calling_convention(self):
        out = reward.reward_batch(
            prompts=["p", "q"],
            completions=["19", "x"],
            completion_ids=[[1], [2]],
            kind=["arith", "arith"],
            spec=[arith_spec(19), arith_spec(19)],
            answer=["19", "19"],
        )
        self.assertEqual(out, [1.0, 0.0])

    def test_batch_accepts_conversational_completions(self):
        out = reward.reward_batch(
            completions=[[{"role": "assistant", "content": "7"}]],
            kind=["arith"],
            spec=[arith_spec(7)],
        )
        self.assertEqual(out, [1.0])

    def test_report_to_dict_is_json_serialisable(self):
        d = reward.verify("arith", arith_spec(1), "2").to_dict()
        self.assertEqual(json.loads(json.dumps(d))["findings"][0]["hard"], True)


if __name__ == "__main__":
    unittest.main()
