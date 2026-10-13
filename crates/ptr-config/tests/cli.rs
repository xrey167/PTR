use ptr_config::{CliOverrides, PtrConfig};

#[test]
fn cli_overrides_apply_after_file_and_environment_layers() {
    let overrides = CliOverrides::parse([
        "--config",
        "custom.toml",
        "--mode=cluster",
        "--mailbox-capacity",
        "512",
        "--require-current-revision=false",
    ])
    .unwrap();

    assert_eq!(overrides.config_path.as_deref(), Some("custom.toml"));
    let mut config = PtrConfig::default();
    config.apply_env([("PTR_MAILBOX_CAPACITY", "256")]).unwrap();
    overrides.apply_to(&mut config).unwrap();

    assert_eq!(config.runtime.mode, "cluster");
    assert_eq!(config.runtime.mailbox_capacity, 512);
    assert!(!config.action_boundary.require_current_revision);
}

#[test]
fn unknown_cli_argument_is_rejected() {
    assert!(CliOverrides::parse(["--surprise", "1"]).is_err());
}

#[test]
fn a_higher_layer_corrects_a_lower_one_before_the_single_validation() {
    // The file and the environment leave mailbox_capacity at 0, which is invalid on its own;
    // the CLI corrects it, and only the final configuration is validated.
    let mut config = PtrConfig::from_toml_str("[runtime]\nmailbox_capacity = 0\n").unwrap();
    assert!(config.validate().is_err());
    config
        .overlay_env([("PTR_MAILBOX_CAPACITY", "0")])
        .expect("overlay_env does not validate the whole configuration");
    assert!(
        config
            .overlay_env([("PTR_MAILBOX_CAPACITY", "not-a-number")])
            .is_err(),
        "a value that does not parse is still refused"
    );
    CliOverrides::parse(["--mailbox-capacity", "512"])
        .unwrap()
        .overlay(&mut config);
    assert_eq!(config.runtime.mailbox_capacity, 512);
    config.validate().unwrap();

    let mut uncorrected = PtrConfig::from_toml_str("[runtime]\nmailbox_capacity = 0\n").unwrap();
    CliOverrides::parse(["--bind", "127.0.0.1:9000"])
        .unwrap()
        .overlay(&mut uncorrected);
    assert!(
        uncorrected.validate().is_err(),
        "the single validation still refuses"
    );
}
