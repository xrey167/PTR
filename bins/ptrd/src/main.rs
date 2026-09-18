use ptr_config::PtrConfig;
use ptr_runtime::PtrRuntime;
use ptr_types::RequestId;

fn main() {
    let path = std::env::var("PTR_CONFIG").unwrap_or_else(|_| "config/default.toml".into());
    let config = PtrConfig::from_path(&path).unwrap_or_else(|err| {
        eprintln!("failed to load PTR config {path}: {err}; using defaults");
        PtrConfig::default()
    });
    let mut runtime = PtrRuntime::new(config).expect("valid PTR runtime configuration");
    let revision = runtime.ingest_text(RequestId::from("bootstrap"), "ptrd ready");
    println!("ptrd ready at semantic revision {}", revision.0);
}
