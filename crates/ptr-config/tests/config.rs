use ptr_config::PtrConfig;

#[test]
fn repository_default_config_parses_and_validates() {
    let text = include_str!("../../../config/default.toml");
    let config = PtrConfig::from_toml_str(text).expect("default config parses");
    config.validate().expect("default config validates");
    assert_eq!(config.runtime.mode, "standalone");
}
