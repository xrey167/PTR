use ptr_config::{CliOverrides, PtrConfig};
use ptr_ledger::LedgerEvent;
use ptr_model_api::{
    ByteTokenizer, KvReuseHint, KvReusePolicy, ReferenceEchoBackend, ScenarioBackend,
};
use ptr_pods::PodRegistry;
use ptr_router::SamplingKey;
use ptr_runtime::{PtrRuntime, ReferenceKvLimits, ReferenceKvReuseBackend, ReferenceKvScope};
use ptr_server::{KvBackend, ServerBackend};
use ptr_types::{CapabilityId, CapsuleId, Generation, PrincipalId, ProjectId};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let cli = match CliOverrides::parse(std::env::args().skip(1)) {
        Ok(value) => value,
        Err(err) if err == "help requested" => {
            println!(
                "usage: ptrd [--config PATH] [--bind HOST:PORT] [--mode standalone|cluster] \
                 [--data-dir PATH] [--backend NAME] \
                 (backends: reference-echo, scenario, reference-kv; reference-kv is a CPU \
                 reference with seeded random weights that reuses KV across requests, its \
                 output is deterministic byte text, not a language model) \
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
    if !matches!(
        config.model.backend.as_str(),
        "reference-echo" | "scenario" | "reference-kv"
    ) {
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
    let backend = match backend_name.as_str() {
        "reference-echo" => ServerBackend::Resumable(Arc::new(ReferenceEchoBackend)),
        "scenario" => ServerBackend::Resumable(Arc::new(ScenarioBackend)),
        "reference-kv" => {
            let limits = ReferenceKvLimits::default();
            let max_prompt_tokens = limits.max_prompt_tokens;
            let backend = ReferenceKvReuseBackend::new(
                limits,
                ReferenceKvScope {
                    principal: PrincipalId::from("ptrd"),
                    project: ProjectId::from("demo"),
                },
            )
            .unwrap_or_else(|err| {
                eprintln!("refusing to start ptrd: reference-kv backend: {err:?}");
                std::process::exit(2);
            });
            println!(
                "reference-kv: CPU reference with seeded random weights; output is \
                 deterministic byte text, not a language model"
            );
            ServerBackend::KvReusing(
                KvBackend::new(
                    Arc::new(backend),
                    Arc::new(ByteTokenizer::new()),
                    Some(KvReuseHint {
                        requested_prefix_tokens: 0,
                        policy: KvReusePolicy::ExactOnly,
                    }),
                )
                .with_max_prompt_tokens(max_prompt_tokens),
            )
        }
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
    if let Err(err) = ptr_server::serve_with_backend(
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

/// Without `/dev/urandom` this binary has no OS entropy source of its own. The nonce instance
/// id only has to be unique per process incarnation (it is not secret), so it is derived from
/// the clock, the process id and the per-process random keys the standard library seeds its
/// `RandomState` with; the sampling key can never be generated here and must be provisioned.
#[cfg(not(unix))]
fn random_bytes<const N: usize>() -> Result<[u8; N], String> {
    use std::hash::{BuildHasher, Hasher};
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|err| format!("system clock before the epoch: {err}"))?
        .as_nanos();
    let state = std::collections::hash_map::RandomState::new();
    let mut bytes = [0u8; N];
    for (index, chunk) in bytes.chunks_mut(8).enumerate() {
        let mut hasher = state.build_hasher();
        hasher.write_u128(nanos);
        hasher.write_u32(std::process::id());
        hasher.write_usize(index);
        chunk.copy_from_slice(&hasher.finish().to_be_bytes()[..chunk.len()]);
    }
    Ok(bytes)
}

/// Loads `<data_dir>/router-sampling.key` (exactly 32 bytes) or creates it from OS entropy.
///
/// Creation is atomic and unix only: the key is written and synced to a private 0600 temporary
/// file in the same directory and then hard-linked to its final name, which fails if the file
/// exists, so a concurrent starter or a crash leaves either no key file or a complete one and
/// an existing key is never replaced. Elsewhere a missing file is a startup error that says how
/// to provision it. On unix an existing file readable by group or others is refused. The key
/// bytes are never printed or included in an error.
fn load_or_create_sampling_key(data_dir: &std::path::Path) -> Result<SamplingKey, String> {
    let path = data_dir.join(SAMPLING_KEY_FILE);
    std::fs::create_dir_all(data_dir)
        .map_err(|err| format!("cannot create data dir {}: {err}", data_dir.display()))?;
    match read_key_file(&path) {
        Err(KeyFileError::Missing) => {}
        other => return other.map_err(|err| err.to_string()),
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
        let key = random_bytes::<SAMPLING_KEY_LEN>()?;
        match install_key_atomically(data_dir, &path, &key) {
            Ok(()) => Ok(SamplingKey::from_bytes(key)),
            // Another process installed its key first: use that one.
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                read_key_file(&path).map_err(|err| err.to_string())
            }
            Err(err) => Err(format!("cannot create {}: {err}", path.display())),
        }
    }
}

/// Writes `key` to a private temporary file next to `path`, syncs it, hard-links it to `path`
/// (`AlreadyExists` if `path` exists) and syncs the directory. The temporary file is removed on
/// every path.
#[cfg(unix)]
fn install_key_atomically(
    dir: &std::path::Path,
    path: &std::path::Path,
    key: &[u8; SAMPLING_KEY_LEN],
) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let suffix = random_bytes::<8>().map_err(std::io::Error::other)?;
    let suffix: String = suffix.iter().map(|byte| format!("{byte:02x}")).collect();
    let tmp = dir.join(format!(".{SAMPLING_KEY_FILE}.tmp.{suffix}"));
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)?;
        file.write_all(key)?;
        file.sync_all()?;
        std::fs::hard_link(&tmp, path)
    })();
    let _ = std::fs::remove_file(&tmp);
    result?;
    std::fs::File::open(dir)?.sync_all()
}

#[derive(Debug)]
enum KeyFileError {
    Missing,
    Unreadable(String),
    Invalid(String),
}

impl std::fmt::Display for KeyFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing => f.write_str("sampling key file is missing"),
            Self::Unreadable(message) | Self::Invalid(message) => f.write_str(message),
        }
    }
}

/// Reads and validates the key file. On unix it refuses a file that group or others can access.
fn read_key_file(path: &std::path::Path) -> Result<SamplingKey, KeyFileError> {
    use std::io::Read;
    let unreadable = |err: std::io::Error| {
        KeyFileError::Unreadable(format!("cannot read {}: {err}", path.display()))
    };
    let mut file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Err(KeyFileError::Missing)
        }
        Err(err) => return Err(unreadable(err)),
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = file.metadata().map_err(unreadable)?.permissions().mode();
        if mode & 0o077 != 0 {
            return Err(KeyFileError::Invalid(format!(
                "{} has mode {:03o}, which lets group or others access the router sampling key; \
                 run `chmod 600 {}` (and consider the key compromised if others could read it)",
                path.display(),
                mode & 0o777,
                path.display()
            )));
        }
    }
    let mut bytes = Vec::new();
    // One byte more than a key is enough to tell a file that is too long.
    (&mut file)
        .take(SAMPLING_KEY_LEN as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(unreadable)?;
    let key: [u8; SAMPLING_KEY_LEN] = bytes.as_slice().try_into().map_err(|_| {
        KeyFileError::Invalid(format!(
            "{} must contain exactly {SAMPLING_KEY_LEN} bytes, found {}",
            path.display(),
            if bytes.len() > SAMPLING_KEY_LEN {
                "more".to_string()
            } else {
                bytes.len().to_string()
            }
        ))
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

    fn write_private(path: &std::path::Path, bytes: &[u8]) {
        std::fs::write(path, bytes).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
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
        write_private(&path, &[7u8; 32]);
        let key = load_or_create_sampling_key(&dir.0).unwrap();
        assert_eq!(key.key_id(), SamplingKey::from_bytes([7; 32]).key_id());

        for bad in [&[1u8; 31][..], &[1u8; 33][..], &[][..]] {
            write_private(&path, bad);
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

    #[cfg(unix)]
    #[test]
    fn a_key_readable_by_group_or_others_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new("mode");
        let path = dir.0.join(SAMPLING_KEY_FILE);
        std::fs::write(&path, [9u8; 32]).unwrap();
        for mode in [0o644, 0o640, 0o604, 0o660] {
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
            let err = load_or_create_sampling_key(&dir.0).err().unwrap();
            assert!(err.contains("chmod 600"), "{err}");
            // The key bytes must not appear in any rendering (Debug list or hex); the
            // temp path may contain arbitrary digits, so match key-shaped text only.
            assert!(!err.contains("[9, 9") && !err.contains("0909"), "{err}");
        }
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o400)).unwrap();
        assert!(load_or_create_sampling_key(&dir.0).is_ok());
    }

    #[test]
    fn an_unreadable_key_path_errors_without_key_material() {
        let dir = TempDir::new("unreadable");
        // A directory in the key's place cannot be read as a file.
        let blocker = dir.0.join(SAMPLING_KEY_FILE);
        std::fs::create_dir(&blocker).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&blocker, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        let err = load_or_create_sampling_key(&dir.0).err().unwrap();
        assert!(err.contains("cannot read"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn concurrent_creation_yields_one_complete_key_and_no_stray_files() {
        let dir = TempDir::new("race");
        let path = dir.0.join(SAMPLING_KEY_FILE);
        let stop = std::sync::atomic::AtomicBool::new(false);
        let ids = std::thread::scope(|scope| {
            // A reader that sees the file at any moment must see all 32 bytes.
            let watcher = scope.spawn(|| {
                while !stop.load(std::sync::atomic::Ordering::SeqCst) {
                    if let Ok(bytes) = std::fs::read(&path) {
                        assert_eq!(bytes.len(), 32, "partial key file observed");
                    }
                }
            });
            let starters: Vec<_> = (0..8)
                .map(|_| scope.spawn(|| load_or_create_sampling_key(&dir.0).unwrap().key_id()))
                .collect();
            let ids: Vec<_> = starters.into_iter().map(|h| h.join().unwrap()).collect();
            stop.store(true, std::sync::atomic::Ordering::SeqCst);
            watcher.join().unwrap();
            ids
        });
        assert!(ids.windows(2).all(|pair| pair[0] == pair[1]));
        let names: Vec<_> = std::fs::read_dir(&dir.0)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(names, [std::ffi::OsString::from(SAMPLING_KEY_FILE)]);
    }

    #[cfg(unix)]
    #[test]
    fn install_never_replaces_an_existing_key() {
        let dir = TempDir::new("install");
        let path = dir.0.join(SAMPLING_KEY_FILE);
        install_key_atomically(&dir.0, &path, &[1; 32]).unwrap();
        let err = install_key_atomically(&dir.0, &path, &[2; 32]).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&path).unwrap(), [1u8; 32]);
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 1);
    }
}
