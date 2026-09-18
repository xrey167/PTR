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
