use ptr_config::{CliOverrides, PtrConfig};
use ptr_runtime::PtrRuntime;
use ptr_types::RequestId;

fn main() {
    let cli = match CliOverrides::parse(std::env::args().skip(1)) {
        Ok(value) => value,
        Err(err) if err == "help requested" => {
            println!(
                "usage: ptrd [--config PATH] [--mode standalone|cluster] \
                 [--mailbox-capacity N] [--max-parallel-candidates N] \
                 [--require-current-revision true|false] \
                 [--require-live-generation true|false]"
            );
            return;
        }
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(2);
        }
    };

    let path = cli
        .config_path
        .clone()
        .or_else(|| std::env::var("PTR_CONFIG").ok())
        .unwrap_or_else(|| "config/default.toml".into());

    let mut config = PtrConfig::from_path(&path).unwrap_or_else(|err| {
        eprintln!("failed to load PTR config {path}: {err}; using defaults");
        PtrConfig::default()
    });

    if let Err(err) = config.apply_env(std::env::vars()) {
        eprintln!("invalid PTR environment override: {err}");
        std::process::exit(2);
    }
    if let Err(err) = cli.apply_to(&mut config) {
        eprintln!("invalid PTR CLI override: {err}");
        std::process::exit(2);
    }

    let mut runtime = PtrRuntime::new(config).expect("valid PTR runtime configuration");
    let revision = runtime.ingest_text(RequestId::from("bootstrap"), "ptrd ready");
    println!("ptrd ready at semantic revision {}", revision.0);
}
