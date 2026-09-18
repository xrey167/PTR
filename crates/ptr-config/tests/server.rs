use ptr_config::{CliOverrides, PtrConfig};

#[test]
fn server_bind_can_be_overridden_by_env_and_cli() {
    let mut config = PtrConfig::default();
    config.apply_env([("PTR_BIND", "127.0.0.1:9000")]).unwrap();
    assert_eq!(config.server.bind, "127.0.0.1:9000");

    let cli = CliOverrides::parse(["--bind", "0.0.0.0:8081"]).unwrap();
    cli.apply_to(&mut config).unwrap();
    assert_eq!(config.server.bind, "0.0.0.0:8081");
}
