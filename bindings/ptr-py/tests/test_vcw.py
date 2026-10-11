"""Tests for the VCW-MoE part of ptr_py: verifiers, finding router, bounded workspace and the
in-process loop simulation. Expected values are hand-derived or taken from the Rust docs."""

import unittest

import ptr_py

TWO = 2**53


def bad_report():
    return ptr_py.verify_arithmetic("1+2*3", "8")


def good_report():
    return ptr_py.verify_arithmetic("1+2*3", "7")


def unknown_report():
    return ptr_py.verify_facts([], [("a", "b", "c")])


def new_router(**kwargs):
    return ptr_py.FindingRouter(
        7, ["gen"], [("fa", ["arith.value"]), ("fb", ["arith.value"])], **kwargs
    )


class VerifierTests(unittest.TestCase):
    def test_arithmetic(self):
        report = good_report()
        self.assertEqual((report.status, report.level, report.score), ("pass", "deterministic", 1.0))
        self.assertEqual(report.findings, [])
        report = bad_report()
        self.assertEqual(report.status, "fail")
        self.assertEqual(report.score, 0.0)
        finding = report.findings[0]
        self.assertEqual((finding.code, finding.severity, finding.hard), ("arith.value", "hard", True))
        self.assertEqual(report.hard_codes(), ["arith.value"])
        self.assertEqual(report.to_dict()["findings"][0]["code"], "arith.value")
        self.assertIn("VerificationReport", repr(report))

    def test_arithmetic_unusable_task_is_a_finding_not_an_exception(self):
        self.assertNotEqual(ptr_py.verify_arithmetic("1+", "2").status, "pass")

    def test_json_schema(self):
        schema = '{"type":"object","required":["a"]}'
        self.assertEqual(ptr_py.verify_json_schema(schema, '{"a":1}').status, "pass")
        self.assertEqual(ptr_py.verify_json_schema(schema, "{}").findings[0].code, "json.required")
        self.assertEqual(ptr_py.verify_json_schema(schema, "{").findings[0].code, "json.parse")

    def test_code(self):
        tests = [{"id": "t1", "args": [2], "expected": 4}]
        self.assertEqual(ptr_py.verify_code("f", tests, "fn f(x){return x*x;}").status, "pass")
        failed = ptr_py.verify_code("f", tests, "fn f(x){return x+1;}")
        self.assertEqual(failed.findings[0].code, "code.test_failed")
        self.assertNotIn("expected", failed.findings[0].message)  # hidden by default
        visible = [{"id": "t1", "args": [2], "expected": 4, "visible": True}]
        shown = ptr_py.verify_code("f", visible, "fn f(x){return x+1;}")
        self.assertIn("expected 4", shown.findings[0].message)
        output = [{"id": "t1", "args": [2], "expected": 4, "expected_output": "x"}]
        self.assertEqual(ptr_py.verify_code("f", output, "fn f(x){return x*x;}").status, "fail")

    def test_code_negative(self):
        empty = ptr_py.verify_code("f", [], "fn f(x){return x;}")
        self.assertEqual((empty.status, empty.findings[0].code), ("unknown", "code.task"))
        self.assertEqual(empty.findings[0].severity, "soft")
        with self.assertRaises(ptr_py.PtrVerifierError):
            ptr_py.verify_code("f", [{"id": "t"}], "fn f(){return 1;}")
        with self.assertRaises(ptr_py.PtrVerifierError):
            ptr_py.verify_code("f", [{"id": 1, "args": [], "expected": 1}], "x")
        with self.assertRaises(ValueError):
            ptr_py.verify_code("f", [{"id": "t", "args": [], "expected": 1, "visible": "yes"}], "x")

    def test_facts(self):
        facts = [("paris", "capital-of", "france")]
        self.assertEqual(ptr_py.verify_facts(facts, facts).status, "pass")
        wrong = ptr_py.verify_facts(facts, [("paris", "capital-of", "lyon")])
        self.assertEqual((wrong.status, wrong.findings[0].code), ("fail", "fact.contradicted"))
        absent = unknown_report()
        self.assertEqual((absent.status, absent.findings[0].code), ("unknown", "fact.unknown"))
        stale = ptr_py.verify_facts(facts, facts, live_revision=9)
        self.assertEqual((stale.status, stale.findings[0].code), ("unknown", "fact.stale_snapshot"))
        malformed = ptr_py.verify_facts([], [("a/b", "c", "d")])
        self.assertEqual((malformed.status, malformed.findings[0].code), ("fail", "fact.malformed"))

    def test_code_table(self):
        table = ptr_py.finding_code_table()
        by_code = {row["code"]: row for row in table}
        self.assertEqual(by_code["arith.value"]["role"], "value-repair")
        self.assertTrue(by_code["arith.value"]["hard"])
        self.assertTrue(any(not row["hard"] and row["role"] is None for row in table))
        self.assertEqual(len(by_code), len(table))
        self.assertEqual(ptr_py.repair_role_for("arith.value"), "value-repair")
        self.assertIsNone(ptr_py.repair_role_for("fact.unknown"))
        self.assertIsNone(ptr_py.repair_role_for("no.such.code"))
        for role in ptr_py.repair_roles():
            for code in ptr_py.codes_for_role(role):
                self.assertEqual(ptr_py.repair_role_for(code), role)
        self.assertIn("arith.format", ptr_py.codes_for_role("format-repair"))
        with self.assertRaises(ptr_py.PtrVerifierError):
            ptr_py.codes_for_role("nonsense")

    def test_reports_cannot_be_forged(self):
        with self.assertRaises(TypeError):
            ptr_py.VerificationReport()
        with self.assertRaises(AttributeError):
            good_report().status = "pass"


class RouterTests(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(ptr_py.classify_report(good_report()), {"class": "accept"})
        self.assertEqual(
            ptr_py.classify_report(bad_report()), {"class": "repair", "codes": ["arith.value"]}
        )
        self.assertEqual(
            ptr_py.classify_report(unknown_report()), {"class": "fallback", "reason": "unknown"}
        )
        with self.assertRaises(TypeError):
            ptr_py.classify_report(good_report(), required="unverified")  # level is fixed

    def test_propensity_equals_support_entry(self):
        router = new_router()
        for i in range(40):
            decision = router.route(bad_report(), request_id=f"r{i}")
            self.assertEqual(decision["kind"], "repair")
            route = decision["routes"][0]
            support = dict(route["support"])
            self.assertEqual(sum(support.values()), TWO)
            self.assertEqual(route["support_total"], TWO)
            self.assertEqual(route["propensity"], support[route["pod"]] / TWO)
            self.assertGreater(route["propensity"], 0.0)
            self.assertLessEqual(route["propensity"], 1.0)
            self.assertEqual(dict(route["support_probabilities"])[route["pod"]], route["propensity"])

    def test_seeded_draw_replays(self):
        def pods(seed):
            router = ptr_py.FindingRouter(
                seed, ["gen"], [("fa", ["arith.value"]), ("fb", ["arith.value"])]
            )
            return [
                router.route(bad_report(), request_id=f"r{i}")["routes"][0]["pod"] for i in range(60)
            ]

        self.assertEqual(pods(1), pods(1))
        self.assertNotEqual(pods(1), pods(2))
        self.assertEqual(set(pods(1)), {"fa", "fb"})

    def test_accept_is_the_verifiers_alone(self):
        router = new_router()
        self.assertEqual(router.route(good_report(), request_id="a"), {"kind": "accept"})
        for report in (bad_report(), unknown_report()):
            self.assertNotEqual(router.route(report, request_id="b")["kind"], "accept")
        facts = ptr_py.verify_facts([("a", "b", "c")], [("a", "b", "c")])
        self.assertEqual(router.route(facts, request_id="c")["kind"], "accept")
        # The required level is fixed at deterministic: no argument can lower it. The
        # below-level and forged-pass classifications are pinned by Rust unit tests in
        # routing.rs (a Python caller cannot build such a report).
        for call in (
            lambda: router.route(unknown_report(), request_id="d", required="unverified"),
            lambda: ptr_py.classify_report(unknown_report(), required="unverified"),
        ):
            with self.assertRaises(TypeError):
                call()

    def test_fallbacks_and_unroutable(self):
        router = new_router()
        decision = router.route(unknown_report(), request_id="u")
        self.assertEqual((decision["kind"], decision["reason"]), ("regenerate", "unknown"))
        self.assertEqual(decision["route"]["pod"], "gen")
        self.assertEqual(decision["route"]["propensity"], 1.0)
        no_repairer = ptr_py.FindingRouter(1, ["gen"], [("fa", ["other.code"])])
        decision = no_repairer.route(bad_report(), request_id="x")
        self.assertEqual(decision["routes"], [{"unroutable": "arith.value"}])
        self.assertEqual(no_repairer.unroutable_counts(), {"arith.value": 1})
        no_proposer = ptr_py.FindingRouter(1, [], [("fa", ["arith.value"])])
        decision = no_proposer.route(unknown_report(), request_id="y")
        self.assertEqual((decision["kind"], decision["cause"]), ("escalate", "no_proposer"))

    def test_roster_and_generations_are_fixed_at_construction(self):
        router = ptr_py.FindingRouter(
            1, ["gen"], [("fa", ["arith.value"]), ("fb", ["arith.value"])],
            generations={"fa": 3},
        )
        generations = {}
        for i in range(60):
            route = router.route(bad_report(), request_id=f"g{i}")["routes"][0]
            generations[route["pod"]] = route["pod_generation"]
            self.assertIn(route["pod"], router.pods())
            self.assertEqual({pod for pod, _ in route["support"]}, {"fa", "fb"})
        self.assertEqual(generations, {"fa": 3, "fb": 1})
        # No route call can introduce a Pod, a generation or an eligibility view.
        for extra in ({"eligible": {"ghost": 9}}, {"generations": {"fa": 9}}, {"pods": ["x"]}):
            with self.assertRaises(TypeError):
                router.route(bad_report(), request_id="z", **extra)
        with self.assertRaises(ptr_py.PtrRouterError):
            ptr_py.FindingRouter(1, ["gen"], [], generations={"ghost": 1})
        with self.assertRaises(ptr_py.PtrRouterError):
            ptr_py.FindingRouter(1, ["gen"], [], generations={"gen": 0})
        self.assertEqual(router.pods(), ["gen", "fa", "fb"])

    def test_settle_verifies_the_candidate_inside_rust(self):
        router = new_router()
        first = router.route(bad_report(), request_id="s1")["routes"][0]["decision_id"]
        second = router.route(bad_report(), request_id="s2")["routes"][0]["decision_id"]
        self.assertTrue(router.settle(first, "7"))  # the task is the routed report's: 1+2*3
        self.assertFalse(router.settle(second, "8"))  # a failing candidate: reward 0
        self.assertEqual(router.log_len(), 2)
        digests = router.log_digests()
        self.assertEqual(len(digests), 2)
        self.assertNotEqual(digests[0]["candidate"], digests[1]["candidate"])  # "7" vs "8"
        self.assertEqual(len(digests[0]["report"]), 64)
        import hashlib
        self.assertEqual(digests[0]["candidate"], hashlib.sha256(b"7").hexdigest())
        self.assertNotEqual(digests[0]["report"], digests[1]["report"])
        with self.assertRaises(ptr_py.PtrRouterError):
            router.settle(first, "7")  # settled once
        with self.assertRaises(ptr_py.PtrRouterError):
            router.settle(10_000, "7")
        third = router.route(bad_report(), request_id="s3")["routes"][0]["decision_id"]
        for forged in (good_report(), True, 1, None):
            with self.assertRaises((TypeError, ptr_py.PtrVerifierError)):
                router.settle(third, forged)  # a report or a number is not a candidate
        self.assertEqual(router.log_len(), 2)
        with self.assertRaises(ptr_py.PtrVerifierError):
            router.settle(third, [("a", "b", "c")])  # wrong shape for an arithmetic task
        self.assertFalse(router.settle(third, "not a number"))

    def test_settled_decisions_free_their_slot_and_unsettled_ones_are_capped(self):
        router = new_router()
        bad = bad_report()
        for i in range(66_000):  # more than the 65,536 cap, all settled
            decision = router.route(bad, request_id="r")["routes"][0]["decision_id"]
            router.settle(decision, "7")
        self.assertEqual(router.log_len(), 66_000)
        ids = set()
        router = new_router()
        for i in range(65_536):
            ids.add(router.route(bad, request_id="u")["routes"][0]["decision_id"])
        self.assertEqual(len(ids), 65_536)  # ids are unique
        with self.assertRaises(ptr_py.PtrRouterError) as raised:
            router.route(bad, request_id="u")
        self.assertIn("await a candidate", str(raised.exception))
        router.settle(min(ids), "7")  # one settled decision frees one place, ids are not reused
        fresh = router.route(bad, request_id="u")["routes"][0]["decision_id"]
        self.assertNotIn(fresh, ids)
        with self.assertRaises(ptr_py.PtrRouterError):
            router.settle(min(ids), "7")

    def test_settle_uses_the_routed_task_kind(self):
        facts = [("paris", "capital-of", "france")]
        router = ptr_py.FindingRouter(2, ["gen"], [("fa", ["fact.contradicted"])])
        wrong = ptr_py.verify_facts(facts, [("paris", "capital-of", "lyon")])
        decision = router.route(wrong, request_id="f")["routes"][0]["decision_id"]
        self.assertTrue(router.settle(decision, [("paris", "capital-of", "france")]))
        other = router.route(wrong, request_id="g")["routes"][0]["decision_id"]
        self.assertFalse(router.settle(other, [("paris", "capital-of", "rome")]))
        code = ptr_py.verify_code("f", [{"id": "t", "args": [2], "expected": 4}], "fn f(x){return x;}")
        coder = ptr_py.FindingRouter(2, ["gen"], [("fc", ["code.test_failed"])])
        decision = coder.route(code, request_id="c")["routes"][0]["decision_id"]
        self.assertTrue(coder.settle(decision, "fn f(x){return x*x;}"))

    def test_estimators(self):
        router = new_router()
        with self.assertRaises(ptr_py.PtrRouterError):
            router.estimate({})  # empty log
        for i in range(300):
            for route in router.route(bad_report(), request_id=f"q{i}")["routes"]:
                # fa always clears, fb never does
                router.settle(route["decision_id"], "7" if route["pod"] == "fa" else "8")
        uniform = router.estimate({})
        always_fa = router.estimate({"repair/arith.value": {"fa": 1.0}})
        always_fb = router.estimate({"repair/arith.value": {"fb": 1.0}})
        self.assertEqual(uniform["n"], 300)
        for key in ("ips", "snips", "doubly_robust"):
            self.assertAlmostEqual(always_fa[key], 1.0, delta=0.15)
            self.assertAlmostEqual(always_fb[key], 0.0, delta=0.15)
            self.assertAlmostEqual(uniform[key], 0.5, delta=0.15)
        self.assertGreater(always_fa["effective_sample_size"], 0)
        with self.assertRaises(ptr_py.PtrRouterError):
            router.estimate({"repair/arith.value": {"ghost": 1.0}})
        with self.assertRaises(ptr_py.PtrRouterError):
            router.estimate({"repair/arith.value": {"fa": 0.0}})
        with self.assertRaises(ptr_py.PtrRouterError):
            router.estimate({}, default_reward=2.0)

    def test_no_floor_means_no_log(self):
        router = new_router(exploration=0.0)
        decision = router.route(bad_report(), request_id="z")["routes"][0]
        with self.assertRaises(ptr_py.PtrRouterError):
            router.settle(decision["decision_id"], "7")

    def test_construction_errors(self):
        for exploration in (-0.1, 1.5, 1e-9, float("nan")):
            with self.assertRaises(ptr_py.PtrRouterError):
                new_router(exploration=exploration)
        with self.assertRaises(ptr_py.PtrRouterError):
            ptr_py.FindingRouter(1, ["gen", "gen"], [])  # already registered

    def test_router_has_no_mutator(self):
        public = {name for name in dir(ptr_py.FindingRouter) if not name.startswith("_")}
        self.assertEqual(
            public,
            {"route", "settle", "log_len", "log_digests", "estimate", "unroutable_counts",
             "exploration", "pods"},
        )
        router = new_router()
        for name in ("exploration", "pods", "required", "scorer", "registry", "fallback", "generations"):
            with self.assertRaises(AttributeError):
                setattr(router, name, 0)


class WorkspaceTests(unittest.TestCase):
    def test_admission_is_the_verifiers(self):
        ws = ptr_py.BoundedWorkspace("arithmetic", "1+2*3", slots=2)
        subscriber = ws.subscribe("listener")
        ws.submit("a", "7")
        ws.submit("b", "8")
        ws.submit("c", "9")
        self.assertEqual(ws.pending, 3)
        snapshot = ws.run_cycle()
        self.assertEqual(snapshot["authority"], "provisional")
        self.assertEqual([slot["candidate"] for slot in snapshot["slots"]], ["7"])
        self.assertEqual(snapshot["occupied"], 1)
        reasons = {r["author"]: r["reason"] for r in snapshot["refusals"]}
        self.assertEqual(reasons, {"b": "not_passed", "c": "not_passed"})
        self.assertEqual(snapshot["refusals"][0]["findings"][0]["code"], "arith.value")
        self.assertNotIn("message", snapshot["refusals"][0]["findings"][0])  # host-only
        self.assertEqual(snapshot["delivered"], [subscriber])
        self.assertEqual(ws.revision, 1)
        self.assertEqual(ws.pending, 0)
        received = ws.receive(subscriber)
        self.assertEqual(received["digest"], snapshot["digest"])
        self.assertIsNone(ws.receive(subscriber))

    def test_slot_bound_duplicate_and_bid_cannot_buy_admission(self):
        ws = ptr_py.BoundedWorkspace("arithmetic", "2", slots=1)
        ws.submit("a", "2")
        ws.submit("b", "3", bid=2**32 - 1)  # a high bid does not rescue a failing candidate
        snapshot = ws.run_cycle()
        self.assertEqual([s["author"] for s in snapshot["slots"]], ["a"])
        ws.submit("c", "2")
        again = ws.run_cycle()
        self.assertIn("duplicate", [r["reason"] for r in again["refusals"]])

    def test_json_schema_kind(self):
        ws = ptr_py.BoundedWorkspace("json_schema", '{"type":"object","required":["a"]}')
        ws.submit("a", '{"a":1}')
        ws.submit("b", "{}")
        snapshot = ws.run_cycle()
        self.assertEqual(snapshot["occupied"], 1)
        self.assertEqual(snapshot["refusals"][0]["findings"][0]["code"], "json.required")

    def test_capacity_errors_are_typed(self):
        self.assertTrue(issubclass(ptr_py.PtrWorkspaceCapacityError, ptr_py.PtrWorkspaceError))
        ws = ptr_py.BoundedWorkspace("arithmetic", "1")
        with self.assertRaises(ptr_py.PtrWorkspaceCapacityError):
            ws.submit("a", "1" * (64 * 1024 + 1))  # over the per-text byte cap
        ws.submit("a", "1" * 64 * 1024)  # exactly at the cap is accepted
        ws = ptr_py.BoundedWorkspace("arithmetic", "1")
        with self.assertRaises(ptr_py.PtrWorkspaceCapacityError):
            for i in range(100):  # one author holds at most 16 pending proposals
                ws.submit("a", str(i))
        self.assertEqual(ws.pending, 16)
        ws.run_cycle()  # only the passing text ("1") keeps a slot; the others are evicted
        self.assertEqual((ws.occupied, ws.held_texts), (1, 1))
        for i in range(10):  # pending proposals are consumed by a cycle
            for j in range(16):
                ws.submit(f"w{i}-{j}", str(1000 + i * 16 + j))
            self.assertEqual(ws.held_texts, 16 + ws.occupied)
            ws.run_cycle()
            self.assertEqual(ws.held_texts, ws.occupied)  # failing texts are evicted
        self.assertEqual(ws.pending, 0)
        self.assertEqual((ws.occupied, ws.held_texts), (1, 1))

    def test_eviction_and_stale_mailbox_snapshots(self):
        ws = ptr_py.BoundedWorkspace("arithmetic", "7", slots=1)
        subscriber = ws.subscribe("listener")
        ws.submit("a", "7")
        first = ws.run_cycle()
        self.assertEqual(first["slots"][0]["candidate"], "7")
        self.assertEqual(ws.held_texts, 1)  # a slot holds it
        for _ in range(70):  # the slot ages out after 64 cycles; the mailbox keeps the first snapshot
            ws.run_cycle()
        self.assertEqual(ws.occupied, 0)
        self.assertEqual(ws.held_texts, 0)  # the text is evicted with its slot
        stale = ws.receive(subscriber)
        self.assertEqual(stale["digest"], first["digest"])
        # Candidate texts are read from the current text map: the evicted text is gone,
        # the content digest is not.
        self.assertIsNone(stale["slots"][0]["candidate"])
        self.assertEqual(stale["slots"][0]["content"], first["slots"][0]["content"])

    def test_negative_cases(self):
        with self.assertRaises(ptr_py.PtrWorkspaceError):
            ptr_py.BoundedWorkspace("nonsense", "x")
        with self.assertRaises(ptr_py.PtrWorkspaceError):
            ptr_py.BoundedWorkspace("arithmetic", "1", slots=0)
        ws = ptr_py.BoundedWorkspace("arithmetic", "1")
        ws.submit("a", "1")
        with self.assertRaises(ptr_py.PtrWorkspaceError):
            ws.submit("a", "1")  # identical proposal pending
        with self.assertRaises(ptr_py.PtrWorkspaceError):
            ws.submit("  ", "1")  # blank author
        with self.assertRaises(ptr_py.PtrWorkspaceError):
            ws.receive(99)
        ws.subscribe("p")
        with self.assertRaises(ptr_py.PtrWorkspaceError):
            ws.subscribe("p")

    def test_no_way_to_change_how_proposals_are_judged(self):
        public = {name for name in dir(ptr_py.BoundedWorkspace) if not name.startswith("_")}
        self.assertEqual(
            public,
            {"submit", "subscribe", "run_cycle", "receive", "revision", "occupied", "pending", "slots",
             "held_texts"},
        )
        ws = ptr_py.BoundedWorkspace("arithmetic", "1")
        for name in ("slots", "verifier", "required", "level", "config"):
            with self.assertRaises(AttributeError):
                setattr(ws, name, 0)
        with self.assertRaises(TypeError):
            ptr_py.BoundedWorkspace("arithmetic", "1", required="unverified")
        with self.assertRaises(TypeError):
            ws.submit("a", "1", report=good_report())
        # A candidate cannot carry its own verdict: the verifier decides alone.
        ws.submit("a", "2")
        self.assertEqual(ws.run_cycle()["occupied"], 0)


class SimulationTests(unittest.TestCase):
    ROSTER = (["gen"], [("fa", ["a", "b"]), ("fb", ["a", "b"])])

    def run_loop(self, seed=1, **kwargs):
        kwargs.setdefault("error_rates", {"a": 0.9, "b": 0.9})
        kwargs.setdefault("repair_success", {"a": 0.5, "b": 0.5})
        return ptr_py.simulate_vcw_run(seed, *self.ROSTER, **kwargs)

    def test_clean_proposals_are_accepted_at_once(self):
        run = self.run_loop(error_rates={})
        self.assertEqual((run["outcome"], run["reason"]), ("Accepted", None))
        self.assertEqual(run["answer"]["author"], "gen")
        self.assertEqual(run["answer"]["report"]["status"], "pass")
        self.assertEqual(run["answer"]["defects"], [])
        self.assertEqual(run["trace"]["spent"], {"verifier_calls": 1, "rounds": 0, "pod_calls": 1})
        self.assertEqual(run["trace"]["outcome"], "Verified")

    def test_exhausted_when_repair_never_succeeds(self):
        run = ptr_py.simulate_vcw_run(
            1, ["gen"], [("fa", ["a"])], error_rates={"a": 1.0}, repair_success={"a": 0.0}, rounds=2
        )
        self.assertEqual(run["outcome"], "Exhausted")
        self.assertIsNone(run["answer"])
        self.assertIn("Exhausted", run["trace"]["outcome"])
        self.assertLessEqual(run["trace"]["spent"]["rounds"], run["trace"]["budget"]["rounds"])
        self.assertLessEqual(
            run["trace"]["spent"]["verifier_calls"], run["trace"]["budget"]["verifier_calls"]
        )

    def test_escalated_without_a_proposer(self):
        run = ptr_py.simulate_vcw_run(1, [], [("fa", ["a"])], error_rates={"a": 1.0})
        self.assertIn(run["outcome"], {"Escalated", "Exhausted"})
        self.assertIsNone(run["answer"])

    def test_repairs_are_routed_with_propensities(self):
        run = self.run_loop(seed=3)
        routes = [route for step in run["trace"]["steps"] for route in step["routes"]]
        self.assertTrue(any(route["class"] == "repair" for route in routes))
        for route in routes:
            self.assertEqual(route["propensity"], dict(route["support"])[route["pod"]] / TWO)
        for step in run["trace"]["steps"]:
            candidate = step["candidate"]
            if candidate and candidate["class"] == "accept":
                self.assertEqual(candidate["report"]["hard_codes"], [])
                self.assertEqual(run["outcome"], "Accepted")

    def test_replay_and_seed_dependence(self):
        self.assertEqual(self.run_loop(seed=5), self.run_loop(seed=5))
        digests = {self.run_loop(seed=s)["trace"]["digest"] for s in range(1, 6)}
        self.assertGreater(len(digests), 1)

    def test_budget_is_enforced_and_validated(self):
        run = self.run_loop(seed=2, verifier_calls=1, rounds=1)
        self.assertLessEqual(run["trace"]["spent"]["verifier_calls"], 1)
        for kwargs in ({"verifier_calls": 0}, {"rounds": 0}, {"error_rates": {"a": 2.0}},
                       {"exploration": 1.5}, {"default_repair_success": -1.0}):
            with self.subTest(**kwargs):
                with self.assertRaises(ptr_py.PtrVcwError):
                    self.run_loop(**kwargs)
        with self.assertRaises(ptr_py.PtrVcwError):
            ptr_py.simulate_vcw_run(1, ["gen"], [("fa", ["a"])], repair_success={"a": float("nan")})

    def test_the_gate_is_not_a_parameter(self):
        for name in ("required", "level", "threshold", "verifier", "eligible"):
            with self.assertRaises(TypeError):
                self.run_loop(**{name: "unverified"})


class SafetyTests(unittest.TestCase):
    def test_exceptions_are_value_errors(self):
        for error in (ptr_py.PtrVerifierError, ptr_py.PtrRouterError, ptr_py.PtrWorkspaceError,
                      ptr_py.PtrVcwError):
            self.assertTrue(issubclass(error, ValueError))

    def test_no_function_takes_a_gate_threshold_or_verifier_for_the_vcw_surface(self):
        names = [n for n in dir(ptr_py) if n.startswith(("verify_", "classify", "simulate_"))]
        self.assertEqual(
            sorted(names),
            ["classify_report", "simulate_vcw_run", "verify_arithmetic", "verify_code",
             "verify_facts", "verify_json_schema"],
        )
        self.assertFalse(hasattr(ptr_py, "promote"))
        self.assertFalse(hasattr(ptr_py, "PromotionGate"))
        self.assertFalse(hasattr(ptr_py, "update_registry"))

    def test_docstrings(self):
        for obj in (ptr_py.verify_arithmetic, ptr_py.verify_code, ptr_py.verify_facts,
                    ptr_py.simulate_vcw_run, ptr_py.classify_report, ptr_py.FindingRouter,
                    ptr_py.BoundedWorkspace, ptr_py.VerificationReport):
            self.assertTrue(obj.__doc__)


if __name__ == "__main__":
    unittest.main()
