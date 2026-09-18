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
    config
        .apply_env([("PTR_MAILBOX_CAPACITY", "256")])
        .unwrap();
    overrides.apply_to(&mut config).unwrap();

    assert_eq!(config.runtime.mode, "cluster");
    assert_eq!(config.runtime.mailbox_capacity, 512);
    assert!(!config.action_boundary.require_current_revision);
}

#[test]
fn unknown_cli_argument_is_rejected() {
    assert!(CliOverrides::parse(["--surprise", "1"]).is_err());
}
