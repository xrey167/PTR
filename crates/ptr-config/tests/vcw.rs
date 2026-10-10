use ptr_config::{
    CliOverrides, PtrConfig, VcwBackend, VcwHintMode, VcwTaskKind, MAX_VCW_BUDGET,
    MAX_VCW_REQUEST_MS,
};

#[test]
fn the_vcw_path_is_off_by_default_and_the_repository_default_says_so() {
    let config = PtrConfig::default();
    assert!(!config.vcw.enabled);
    assert_eq!(config.vcw.experts.backend, None, "no implicit backend");
    assert_eq!(config.vcw.max_request_ms, 30_000);
    assert_eq!(config.vcw.hint_mode, VcwHintMode::TraceOnly);
    assert_eq!(config.vcw.kinds, VcwTaskKind::ALL.to_vec());
    let text = include_str!("../../../config/default.toml");
    let from_file = PtrConfig::from_toml_str(text).unwrap();
    assert_eq!(
        from_file.vcw, config.vcw,
        "default.toml spells out the defaults"
    );
}

#[test]
fn a_full_section_parses_and_validates() {
    let text = r#"
[vcw]
enabled = true
project = "demo-vcw"
kinds = ["arithmetic", "fact"]
max_verifier_calls = 12
max_rounds = 5
hint_mode = "apply-to-schema"
exploration_floor = 0.1

[vcw.pods]
proposer = "p"
value_repair = "v"
format_repair = "v"

[vcw.experts]
backend = "simulated"

[vcw.experts.simulated]
seed = 9
default_repair_success = 0.5
decline_on_failure = true

[vcw.experts.simulated.error_rates]
"arith.value" = 0.25

[vcw.experts.simulated.pod_repair_success.v]
"arith.value" = 0.75
"#;
    let config = PtrConfig::from_toml_str(text).unwrap();
    config.validate().unwrap();
    assert!(config.vcw.enabled);
    assert_eq!(
        config.vcw.kinds,
        [VcwTaskKind::Arithmetic, VcwTaskKind::Fact]
    );
    assert_eq!(config.vcw.hint_mode, VcwHintMode::ApplyToSchema);
    assert_eq!(config.vcw.experts.simulated.seed, 9);
    assert_eq!(
        config.vcw.experts.simulated.error_rates["arith.value"],
        0.25
    );
    assert_eq!(
        config.vcw.experts.simulated.pod_repair_success["v"]["arith.value"],
        0.75
    );
}

#[test]
fn unknown_keys_are_refused_at_every_level_of_the_section() {
    for text in [
        "[vcw]\nenabeld = true\n",
        "[vcw.pods]\nproposr = \"x\"\n",
        "[vcw.experts]\nbackends = \"simulated\"\n",
        "[vcw.experts.simulated]\nsed = 1\n",
        "[vcw.experts.qwen_cpu]\nmodel = \"x\"\n",
        "[vcw]\nkinds = [\"poetry\"]\n",
        "[vcw]\nhint_mode = \"learn\"\n",
        "[vcw.experts]\nbackend = \"gpt\"\n",
    ] {
        assert!(PtrConfig::from_toml_str(text).is_err(), "{text}");
    }
}

#[test]
fn invalid_values_fail_validation_even_while_the_section_is_off() {
    for text in [
        format!("[vcw]\nmax_rounds = {}\n", MAX_VCW_BUDGET + 1),
        "[vcw]\nmax_verifier_calls = 0\n".to_owned(),
        "[vcw]\nmax_request_ms = 0\n".to_owned(),
        format!("[vcw]\nmax_request_ms = {}\n", MAX_VCW_REQUEST_MS + 1),
        "[vcw]\nexploration_floor = 3.0\n".to_owned(),
        "[vcw.pods]\nproposer = \"\"\n".to_owned(),
        "[vcw.pods]\nvalue_repair = \"vcw-proposer\"\n".to_owned(),
        "[vcw.experts.simulated]\ndefault_repair_success = 2.0\n".to_owned(),
        "[vcw.experts.simulated.error_rates]\n\"a\" = -1.0\n".to_owned(),
    ] {
        let config = PtrConfig::from_toml_str(&text).unwrap();
        assert!(config.validate().is_err(), "{text}");
    }
}

#[test]
fn the_model_backend_needs_its_directory_and_four_digests_from_file_or_environment() {
    let mut config = PtrConfig::from_toml_str(
        "[vcw]\nenabled = true\nkinds = [\"arithmetic\"]\n[vcw.experts]\nbackend = \"qwen-cpu\"\n",
    )
    .unwrap();
    assert!(config.validate().is_err());
    let digest = "0f".repeat(32);
    config
        .apply_env([
            ("PTR_VCW_QWEN_MODEL_DIR", "/models/qwen"),
            ("PTR_VCW_QWEN_WEIGHTS_SHA256", digest.as_str()),
            ("PTR_VCW_QWEN_CONFIG_SHA256", digest.as_str()),
            ("PTR_VCW_QWEN_TOKENIZER_SHA256", digest.as_str()),
        ])
        .expect_err("three digests are not four");
    config
        .apply_env([("PTR_VCW_QWEN_TOKENIZER_CONFIG_SHA256", digest.as_str())])
        .unwrap();
    assert_eq!(config.vcw.experts.qwen_cpu.model_dir, "/models/qwen");
    assert_eq!(config.vcw.experts.qwen_cpu.max_new_tokens, 0);
    assert_eq!(config.vcw.experts.qwen_cpu.effective_max_new_tokens(), 24);
}

#[test]
fn the_enabled_flag_has_an_environment_switch_but_no_cli_switch() {
    let mut config = PtrConfig::default();
    // The switch alone is not enough: the backend has no default.
    let error = config.apply_env([("PTR_VCW_ENABLED", "true")]).unwrap_err();
    assert!(error.contains("vcw.experts.backend"), "{error}");
    assert!(config.vcw.enabled);
    config.vcw.experts.backend = Some(VcwBackend::Simulated);
    config.validate().unwrap();
    assert!(config.apply_env([("PTR_VCW_ENABLED", "yes")]).is_err());
    assert!(CliOverrides::parse(["--vcw-enabled", "true"]).is_err());
}

#[test]
fn an_enabled_section_must_name_its_backend() {
    let config = PtrConfig::from_toml_str("[vcw]\nenabled = true\n").unwrap();
    let error = config.validate().unwrap_err();
    assert!(error.contains("vcw.experts.backend"), "{error}");
    let config =
        PtrConfig::from_toml_str("[vcw]\nenabled = true\n[vcw.experts]\nbackend = \"simulated\"\n")
            .unwrap();
    config.validate().unwrap();
    let config =
        PtrConfig::from_toml_str("[vcw]\nenabled = true\n[vcw.experts]\nbackend = \"unset\"\n")
            .unwrap();
    assert!(config.validate().is_err(), "unset is not a backend");
    // Off, the backend may stay unset.
    PtrConfig::from_toml_str("[vcw]\nenabled = false\n")
        .unwrap()
        .validate()
        .unwrap();
}
