use ptr_config::PtrConfig;

#[test]
fn environment_overrides_are_typed_and_validated() {
    let mut config = PtrConfig::default();
    config
        .apply_env([
            ("PTR_RUNTIME_MODE", "cluster"),
            ("PTR_MAILBOX_CAPACITY", "256"),
            ("PTR_REQUIRE_CURRENT_REVISION", "false"),
        ])
        .unwrap();

    assert_eq!(config.runtime.mode, "cluster");
    assert_eq!(config.runtime.mailbox_capacity, 256);
    assert!(!config.action_boundary.require_current_revision);
}

#[test]
fn invalid_environment_override_fails_closed() {
    let mut config = PtrConfig::default();
    assert!(config
        .apply_env([("PTR_MAILBOX_CAPACITY", "not-a-number")])
        .is_err());
}

#[test]
fn the_environment_can_complete_a_qwen_section_the_file_leaves_empty() {
    let directory =
        std::env::temp_dir().join(format!("ptr-config-qwen-env-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("ptr.toml");
    std::fs::write(
        &path,
        "[vcw]\nenabled = true\nkinds = [\"arithmetic\"]\n\n[vcw.experts]\nbackend = \"qwen-cpu\"\n",
    )
    .unwrap();
    assert!(
        PtrConfig::from_path(&path).is_err(),
        "the file alone lacks the qwen-cpu pins"
    );

    let mut config = PtrConfig::parse_path(&path).unwrap();
    let digest = "ab".repeat(32);
    config
        .apply_env([
            ("PTR_VCW_QWEN_MODEL_DIR", "/models/qwen"),
            ("PTR_VCW_QWEN_WEIGHTS_SHA256", digest.as_str()),
            ("PTR_VCW_QWEN_CONFIG_SHA256", digest.as_str()),
            ("PTR_VCW_QWEN_TOKENIZER_SHA256", digest.as_str()),
            ("PTR_VCW_QWEN_TOKENIZER_CONFIG_SHA256", digest.as_str()),
        ])
        .unwrap();
    assert_eq!(config.vcw.experts.qwen_cpu.model_dir, "/models/qwen");

    let mut incomplete = PtrConfig::parse_path(&path).unwrap();
    assert!(
        incomplete
            .apply_env([("PTR_VCW_QWEN_MODEL_DIR", "/models/qwen")])
            .is_err(),
        "apply_env still validates the completed configuration"
    );
    std::fs::remove_dir_all(&directory).unwrap();
}
