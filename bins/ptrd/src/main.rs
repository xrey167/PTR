use ptr_config::{CliOverrides, PtrConfig};
use ptr_ledger::LedgerEvent;
use ptr_model_api::{ReferenceEchoBackend, ScenarioBackend};
use ptr_pods::PodRegistry;
use ptr_router::PodRouter;
use ptr_runtime::PtrRuntime;
use ptr_types::{CapabilityId, CapsuleId, Generation};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let cli = match CliOverrides::parse(std::env::args().skip(1)) {
        Ok(value) => value,
        Err(err) if err == "help requested" => {
            println!(
                "usage: ptrd [--config PATH] [--bind HOST:PORT] [--mode standalone|cluster] \
                 [--data-dir PATH] [--backend NAME] \
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
        eprintln!("failed to load PTR config {path}: {err}");
        std::process::exit(2);
    });

    if let Err(err) = config.apply_env(std::env::vars()) {
        eprintln!("invalid PTR environment override: {err}");
        std::process::exit(2);
    }
    if let Err(err) = cli.apply_to(&mut config) {
        eprintln!("invalid PTR CLI override: {err}");
        std::process::exit(2);
    }

    if let Err(err) = config.validate_daemon() {
        eprintln!("refusing to start ptrd: {err}");
        std::process::exit(2);
    }
    if !matches!(config.model.backend.as_str(), "reference-echo" | "scenario") {
        eprintln!(
            "refusing to start ptrd: unsupported explicit backend {}",
            config.model.backend
        );
        std::process::exit(2);
    }

    let bind = config.server.bind.clone();
    let ledger_path = std::path::Path::new(&config.server.data_dir).join("ledger.log");
    let backend_name = config.model.backend.clone();
    let data_dir = config.server.data_dir.clone();
    let mut runtime = PtrRuntime::open_durable(config, ledger_path)
        .unwrap_or_else(|err| panic!("failed to open durable PTR runtime: {err:?}"));
    if runtime.live_generation("demo-note").is_none() {
        runtime
            .commit(LedgerEvent::CapsuleCommitted {
                project: "demo".into(),
                capsule: CapsuleId::from("demo-note"),
                generation: Generation(1),
            })
            .unwrap_or_else(|err| panic!("failed to install demo-note target: {err:?}"));
    }
    runtime.permissions_mut().allow_mutation = true;
    runtime
        .permissions_mut()
        .capabilities
        .insert(CapabilityId::from("demo.local-note.create"));
    let backend: Arc<dyn ptr_model_api::ResumableInferenceBackend> = match backend_name.as_str() {
        "reference-echo" => Arc::new(ReferenceEchoBackend),
        "scenario" => Arc::new(ScenarioBackend),
        _ => unreachable!("backend validated above"),
    };
    let effect_grants = ptr_server::demo_effect_grants(data_dir);
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .unwrap_or_else(|err| panic!("failed to bind {bind}: {err}"));

    println!("ptrd listening on http://{bind}");
    if let Err(err) = ptr_server::serve_with_dependencies(
        listener,
        runtime,
        backend,
        PodRouter,
        Arc::new(PodRegistry::default()),
        ptr_server::default_verifier(),
        effect_grants,
    )
    .await
    {
        eprintln!("ptrd server error: {err}");
        std::process::exit(1);
    }
}
