use std::process::Command;

#[test]
fn benchmark_binary_emits_json_lines() {
    let exe = env!("CARGO_BIN_EXE_ptr-bench");
    let output = Command::new(exe)
        .args(["all", "10"])
        .output()
        .expect("run ptr-bench");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains(r#""benchmark":"semdb""#));
    assert!(stdout.contains(r#""benchmark":"mailbox""#));
}

#[test]
fn ledger_recovery_smoke_has_zero_false_accepts() {
    let exe = env!("CARGO_BIN_EXE_ptr-bench");
    let output = Command::new(exe)
        .args(["ledger-recovery", "3", "17"])
        .output()
        .expect("run ledger recovery smoke");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains(r#""false_accepts":0"#));
    assert!(stdout.contains(r#""recovery_errors":0"#));
    assert!(stdout.contains(r#""tail_trim_errors":0"#));
}

#[test]
fn certified_branches_runs_one_case_clean() {
    let exe = env!("CARGO_BIN_EXE_ptr-bench");
    let output = Command::new(exe)
        .args(["certified-branches", "1", "17"])
        .output()
        .expect("run the S003 smoke");
    assert!(
        output.status.success(),
        "a hard counter is above zero: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 1, "exactly one JSON line on stdout");
    assert!(stdout.starts_with(r#"{"benchmark":"certified-branches","iterations":1,"seed":17,"server":"in-memory","preregistration":"#));
    assert!(stdout.contains(r#""hard_failures":0"#));
    for name in [
        "lost_updates",
        "provenance_mismatches",
        "canary_misses",
        "nondeterminism",
        "harness_errors",
    ] {
        assert!(stdout.contains(&format!(r#""{name}":0"#)), "{name}");
    }
    // The probes, the canaries and the round trips ran.
    assert!(stdout.contains(r#""probe_p26_exercised":1"#));
    assert!(stdout.contains(r#""canaries_run":9"#));
    assert!(stdout.contains(r#""durable_roundtrips":"#));
    assert!(stdout.contains(r#""cases":[{"case":0,"level":0,"groups":4"#));
}

#[test]
fn certified_branches_zero_cases_emits_an_empty_run_for_boundary_seeds() {
    for seed in ["0", "18446744073709551615"] {
        let output = Command::new(env!("CARGO_BIN_EXE_ptr-bench"))
            .args(["certified-branches", "0", seed])
            .output()
            .expect("run S003 with no cases");
        assert!(output.status.success(), "{seed}: {output:?}");
        assert!(
            output.stderr.is_empty(),
            "no cases or reruns should execute"
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert_eq!(stdout.lines().count(), 1);
        assert!(stdout.starts_with(&format!(
            r#"{{"benchmark":"certified-branches","iterations":0,"seed":{seed},"server":"in-memory","preregistration":"#
        )));
        assert!(stdout.contains(r#""cases":[],"hard_failures":0"#));
        for counter in [
            "merge_calls",
            "canaries_run",
            "probe_p26_exercised",
            "nondeterminism",
        ] {
            assert!(stdout.contains(&format!(r#""{counter}":0,"#)), "{counter}");
        }
    }
}

#[test]
fn kv_reuse_smoke_is_bitwise_equal_and_reuses_the_shared_prefix() {
    let exe = env!("CARGO_BIN_EXE_ptr-bench");
    let output = Command::new(exe)
        .args(["kv-reuse-smoke", "3"])
        .output()
        .expect("run the KV reuse smoke");
    assert!(
        output.status.success(),
        "a warm run differed from its cold twin: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 1, "exactly one JSON line on stdout");
    assert!(
        stdout.starts_with(r#"{"benchmark":"kv-reuse-smoke","requests":3,"hit_tokens_total":128,"#)
    );
    assert!(stdout.contains(r#""bitwise_equal":true"#));
    assert!(stdout.contains("not an R005 result"));
}

#[test]
fn kv_reuse_smoke_refuses_a_bad_request_count() {
    let exe = env!("CARGO_BIN_EXE_ptr-bench");
    for bad in ["0", "1000", "many"] {
        let output = Command::new(exe)
            .args(["kv-reuse-smoke", bad])
            .output()
            .expect("run the KV reuse smoke");
        assert_eq!(output.status.code(), Some(2), "{bad}");
        assert!(output.stdout.is_empty(), "{bad}");
    }
}

#[test]
fn exact_prefix_reuse_needs_one_numeric_seed_and_prints_nothing_otherwise() {
    let exe = env!("CARGO_BIN_EXE_ptr-bench");
    // No seed, a typo and a negative seed stop the run before it starts: R005 has no default
    // seed, and a typo must not run another seed than the one asked for.
    for args in [
        vec!["exact-prefix-reuse"],
        vec!["exact-prefix-reuse", "1O1"],
        vec!["exact-prefix-reuse", "-1"],
        vec!["exact-prefix-reuse", ""],
    ] {
        let output = Command::new(exe)
            .args(&args)
            .output()
            .expect("run the R005 harness");
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        assert!(output.stdout.is_empty(), "{args:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("seed"),
            "{args:?}"
        );
    }
}

#[test]
fn the_usage_lists_the_r005_harness_the_manifest_names() {
    let exe = env!("CARGO_BIN_EXE_ptr-bench");
    let output = Command::new(exe)
        .arg("no-such-command")
        .output()
        .expect("run ptr-bench");
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("exact-prefix-reuse"));
    // The manifest names this harness as its entrypoint; the run writes no record itself.
    let manifest =
        include_str!("../../../experiments/runtime/R005-exact-prefix-reuse/experiment.toml");
    assert!(manifest.contains(
        "entrypoint = \"cargo run --release --locked -p ptr-bench -- exact-prefix-reuse <seed>\""
    ));
}
