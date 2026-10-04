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

#[test]
fn a_misspelled_key_is_refused_instead_of_silently_ignored() {
    let text = "[server]\nbind = \"127.0.0.1:8080\"\ndata_dirr = \"typo\"\n";
    assert!(PtrConfig::from_toml_str(text).is_err());
    let text = "[runtim]\nmode = \"cluster\"\n";
    assert!(PtrConfig::from_toml_str(text).is_err());
}

#[test]
fn loading_from_a_path_validates_the_result() {
    let directory = std::env::temp_dir().join(format!("ptr-config-load-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let invalid = directory.join("invalid.toml");
    std::fs::write(&invalid, "[runtime]\nmailbox_capacity = 0\n").unwrap();
    assert!(PtrConfig::from_path(&invalid).is_err());
    let valid = directory.join("valid.toml");
    std::fs::write(&valid, "[runtime]\nmailbox_capacity = 8\n").unwrap();
    assert!(PtrConfig::from_path(&valid).is_ok());
    std::fs::remove_dir_all(&directory).unwrap();
}
