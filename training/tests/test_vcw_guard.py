import unittest

from ptr_training.vcw import guard
from ptr_training.vcw.guard import SelfReportError

FIELDS = {"prompt": "input", "completion": "model-output", "reward": "verifier", "trace": "routing-trace"}


class GuardFieldTests(unittest.TestCase):
    def test_clean_schema_passes(self):
        guard.check_fields(FIELDS, ["reward"])

    def test_every_self_report_origin_spelling_is_refused(self):
        for origin in guard.ORIGINS_SELF_REPORT + ("Self_Report", "SELF-REPORT"):
            with self.subTest(origin=origin):
                with self.assertRaisesRegex(SelfReportError, "self-report"):
                    guard.check_fields({**FIELDS, "x": origin}, ["reward"])

    def test_unknown_origin_fails_closed(self):
        with self.assertRaisesRegex(SelfReportError, "unknown origin"):
            guard.check_fields({**FIELDS, "x": "who-knows"}, ["reward"])

    def test_field_name_marker_refused_even_with_innocent_origin(self):
        for name in ("self_report", "introspection_text", "verbal_confidence", "consciousness_label", "qualia_note"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(SelfReportError, "named like a self-report"):
                    guard.check_fields({**FIELDS, name: "verifier"}, ["reward"])

    def test_targets_must_be_verifier_like(self):
        for bad in ("prompt", "completion"):
            with self.subTest(target=bad):
                with self.assertRaisesRegex(SelfReportError, "targets must come from"):
                    guard.check_fields(FIELDS, [bad])

    def test_undeclared_or_missing_targets_refused(self):
        with self.assertRaisesRegex(SelfReportError, "not a declared field"):
            guard.check_fields(FIELDS, ["nope"])
        with self.assertRaisesRegex(SelfReportError, "no targets"):
            guard.check_fields(FIELDS, [])
        with self.assertRaisesRegex(SelfReportError, "no fields"):
            guard.check_fields({}, ["reward"])


class GuardRecordTests(unittest.TestCase):
    def test_record_ok(self):
        guard.check_record({"prompt": "p", "reward": 1.0}, FIELDS)

    def test_record_flag_refused(self):
        with self.assertRaisesRegex(SelfReportError, "marked as a self-report"):
            guard.check_record({"prompt": "p", guard.SELF_REPORT_FLAG: True}, FIELDS)

    def test_undeclared_field_refused(self):
        with self.assertRaisesRegex(SelfReportError, "undeclared"):
            guard.check_record({"prompt": "p", "extra": 1}, FIELDS)

    def test_screen_text_counts_rows(self):
        rows = ["2 + 2 = 4", "I feel aware of my own consciousness", "wrong-value", "subjective experience"]
        self.assertEqual(guard.screen_text(rows), 2)
        self.assertEqual(guard.screen_text(["def f(): return 1"]), 0)


if __name__ == "__main__":
    unittest.main()
