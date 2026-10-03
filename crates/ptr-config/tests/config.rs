use ptr_config::PtrConfig;

#[test]
fn repository_default_config_parses_and_validates() {
    let text = include_str!("../../../config/default.toml");
    let config = PtrConfig::from_toml_str(text).expect("default config parses");
    config.validate().expect("default config validates");
    assert_eq!(config.runtime.mode, "standalone");
}

#[test]
fn daemon_validation_rejects_non_loopback_and_missing_persistence() {
    let mut config = PtrConfig::default();
    config.server.bind = "0.0.0.0:8080".into();
    config.server.data_dir = "definitely-missing-ptr-data".into();
    assert!(config.validate_daemon().is_err());

    config.server.bind = "127.0.0.1:8080".into();
    config.server.data_dir = std::env::temp_dir().to_string_lossy().into_owned();
    assert!(config.validate_daemon().is_ok());
}
