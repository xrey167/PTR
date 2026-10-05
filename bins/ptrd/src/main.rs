use ptr_config::{CliOverrides, PtrConfig};
use ptr_ledger::LedgerEvent;
use ptr_model_api::{ReferenceEchoBackend, ScenarioBackend};
use ptr_pods::PodRegistry;
use ptr_router::SamplingKey;
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
    let sampling_key =
        load_or_create_sampling_key(std::path::Path::new(&data_dir)).unwrap_or_else(|err| {
            eprintln!("refusing to start ptrd: {err}");
            std::process::exit(2);
        });
    let instance = random_bytes::<8>().unwrap_or_else(|err| {
        eprintln!("refusing to start ptrd: {err}");
        std::process::exit(2);
    });
    // Only the non-secret identifier of the key is ever printed.
    println!(
        "pod router sampling key id {}",
        sampling_key
            .key_id()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    let pod_router = ptr_server::checked_pod_router(sampling_key, instance);
    let effect_grants = ptr_server::demo_effect_grants(data_dir);
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .unwrap_or_else(|err| panic!("failed to bind {bind}: {err}"));

    println!("ptrd listening on http://{bind}");
    if let Err(err) = ptr_server::serve_with_dependencies(
        listener,
        runtime,
        backend,
        pod_router,
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

/// File name of the router sampling key inside the data directory.
const SAMPLING_KEY_FILE: &str = "router-sampling.key";
const SAMPLING_KEY_LEN: usize = 32;

/// `N` bytes of operating-system entropy.
#[cfg(unix)]
fn random_bytes<const N: usize>() -> Result<[u8; N], String> {
    use std::io::Read;
    let mut bytes = [0u8; N];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut bytes))
        .map_err(|err| format!("cannot read /dev/urandom: {err}"))?;
    Ok(bytes)
}

/// Without `/dev/urandom` no entropy source is available to this binary. The nonce instance id
/// only has to be unique per process (it is not secret), so it falls back to the clock mixed
/// with the process id; the sampling key can never be generated here and must be provisioned.
#[cfg(not(unix))]
fn random_bytes<const N: usize>() -> Result<[u8; N], String> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|err| format!("system clock before the epoch: {err}"))?
        .as_nanos();
    let mixed = nanos ^ (u128::from(std::process::id()) << 64);
    let mut bytes = [0u8; N];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = mixed.to_be_bytes()[16 - 1 - (index % 16)];
    }
    Ok(bytes)
}

/// Loads `<data_dir>/router-sampling.key` (exactly 32 bytes) or creates it from OS entropy.
///
/// Creation uses `create_new` (no race, never overwrites) and mode 0600, and is available on
/// unix only; elsewhere a missing file is a startup error that says how to provision it. The
/// key bytes are never printed or included in an error.
fn load_or_create_sampling_key(data_dir: &std::path::Path) -> Result<SamplingKey, String> {
    let path = data_dir.join(SAMPLING_KEY_FILE);
    std::fs::create_dir_all(data_dir)
        .map_err(|err| format!("cannot create data dir {}: {err}", data_dir.display()))?;
    match std::fs::read(&path) {
        Ok(bytes) => return key_from_file_bytes(&path, &bytes),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(format!("cannot read {}: {err}", path.display())),
    }
    #[cfg(not(unix))]
    {
        Err(format!(
            "{} does not exist and ptrd can only generate it on unix; provision a file of exactly \
             {SAMPLING_KEY_LEN} random bytes readable only by the ptrd user, for example in \
             PowerShell: [IO.File]::WriteAllBytes('{}', \
             [Security.Cryptography.RandomNumberGenerator]::GetBytes({SAMPLING_KEY_LEN}))",
            path.display(),
            path.display()
        ))
    }
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let key = random_bytes::<SAMPLING_KEY_LEN>()?;
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true).mode(0o600);
        match options.open(&path) {
            Ok(mut file) => {
                file.write_all(&key)
                    .and_then(|()| file.sync_all())
                    .map_err(|err| format!("cannot write {}: {err}", path.display()))?;
                Ok(SamplingKey::from_bytes(key))
            }
            // Another process created it first: use its key.
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                let bytes = std::fs::read(&path)
                    .map_err(|err| format!("cannot read {}: {err}", path.display()))?;
                key_from_file_bytes(&path, &bytes)
            }
            Err(err) => Err(format!("cannot create {}: {err}", path.display())),
        }
    }
}

fn key_from_file_bytes(path: &std::path::Path, bytes: &[u8]) -> Result<SamplingKey, String> {
    let key: [u8; SAMPLING_KEY_LEN] = bytes.try_into().map_err(|_| {
        format!(
            "{} must contain exactly {SAMPLING_KEY_LEN} bytes, found {}",
            path.display(),
            bytes.len()
        )
    })?;
    Ok(SamplingKey::from_bytes(key))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir =
                std::env::temp_dir().join(format!("ptrd-key-{tag}-{}-{nanos}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[cfg(unix)]
    #[test]
    fn creates_a_32_byte_owner_only_key_and_reuses_it() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new("create");
        let first = load_or_create_sampling_key(&dir.0).unwrap();
        let path = dir.0.join(SAMPLING_KEY_FILE);
        assert_eq!(std::fs::read(&path).unwrap().len(), 32);
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let again = load_or_create_sampling_key(&dir.0).unwrap();
        assert_eq!(first.key_id(), again.key_id());
    }

    #[test]
    fn reuses_an_existing_key_file_and_refuses_a_wrong_length() {
        let dir = TempDir::new("existing");
        let path = dir.0.join(SAMPLING_KEY_FILE);
        std::fs::write(&path, [7u8; 32]).unwrap();
        let key = load_or_create_sampling_key(&dir.0).unwrap();
        assert_eq!(key.key_id(), SamplingKey::from_bytes([7; 32]).key_id());

        for bad in [&[1u8; 31][..], &[1u8; 33][..], &[][..]] {
            std::fs::write(&path, bad).unwrap();
            let err = load_or_create_sampling_key(&dir.0).err().unwrap();
            assert!(err.contains("exactly 32 bytes"), "{err}");
            // The message never carries key material.
            assert!(!err.contains("[1, 1"), "{err}");
        }
    }

    #[test]
    fn instance_ids_differ_between_calls() {
        assert_ne!(random_bytes::<8>().unwrap(), random_bytes::<8>().unwrap());
    }
}
