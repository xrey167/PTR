use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CliOverrides {
    pub config_path: Option<String>,
    pub bind: Option<String>,
    pub data_dir: Option<String>,
    pub model_backend: Option<String>,
    pub runtime_mode: Option<String>,
    pub mailbox_capacity: Option<usize>,
    pub max_parallel_candidates: Option<usize>,
    pub require_current_revision: Option<bool>,
    pub require_live_generation: Option<bool>,
}

impl CliOverrides {
    pub fn parse<I, S>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut out = Self::default();
        let mut args = args.into_iter().map(Into::into).peekable();
        while let Some(arg) = args.next() {
            let (key, inline) = match arg.split_once('=') {
                Some((key, value)) => (key.to_owned(), Some(value.to_owned())),
                None => (arg, None),
            };
            let mut value = || {
                inline
                    .clone()
                    .or_else(|| args.next())
                    .ok_or_else(|| format!("{key} requires a value"))
            };
            match key.as_str() {
                "--config" => out.config_path = Some(value()?),
                "--bind" => out.bind = Some(value()?),
                "--data-dir" => out.data_dir = Some(value()?),
                "--backend" => out.model_backend = Some(value()?),
                "--mode" => out.runtime_mode = Some(value()?),
                "--mailbox-capacity" => {
                    let v = value()?;
                    out.mailbox_capacity = Some(parse_usize("--mailbox-capacity", &v)?);
                }
                "--max-parallel-candidates" => {
                    let v = value()?;
                    out.max_parallel_candidates =
                        Some(parse_usize("--max-parallel-candidates", &v)?);
                }
                "--require-current-revision" => {
                    let v = value()?;
                    out.require_current_revision =
                        Some(parse_bool("--require-current-revision", &v)?);
                }
                "--require-live-generation" => {
                    let v = value()?;
                    out.require_live_generation =
                        Some(parse_bool("--require-live-generation", &v)?);
                }
                "--help" | "-h" => {
                    return Err("help requested".into());
                }
                other => return Err(format!("unknown PTR argument: {other}")),
            }
        }
        Ok(out)
    }

    pub fn apply_to(&self, config: &mut PtrConfig) -> Result<(), String> {
        if let Some(bind) = &self.bind {
            config.server.bind = bind.clone();
        }
        if let Some(data_dir) = &self.data_dir {
            config.server.data_dir = data_dir.clone();
        }
        if let Some(backend) = &self.model_backend {
            config.model.backend = backend.clone();
        }
        if let Some(mode) = &self.runtime_mode {
            config.runtime.mode = mode.clone();
        }
        if let Some(capacity) = self.mailbox_capacity {
            config.runtime.mailbox_capacity = capacity;
        }
        if let Some(max_parallel) = self.max_parallel_candidates {
            config.runtime.max_parallel_candidates = max_parallel;
        }
        if let Some(value) = self.require_current_revision {
            config.action_boundary.require_current_revision = value;
        }
        if let Some(value) = self.require_live_generation {
            config.action_boundary.require_live_generation = value;
        }
        config.validate()
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct PtrConfig {
    pub server: ServerConfig,
    pub runtime: RuntimeConfig,
    pub semantic: SemanticConfig,
    pub action_boundary: ActionBoundaryConfig,
    pub observability: ObservabilityConfig,
    pub research: ResearchConfig,
    pub model: ModelConfig,
    /// The network wires this host serves. Absent means none: there is no default
    /// wire, and a section that is present must decide every field (D4 in issue #23).
    pub wire: Option<WireConfig>,
}

impl PtrConfig {
    pub fn from_toml_str(input: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(input)
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(|e| e.to_string())?;
        let config = Self::from_toml_str(&text).map_err(|e| e.to_string())?;
        config.validate()?;
        Ok(config)
    }

    pub fn apply_env<I, K, V>(&mut self, values: I) -> Result<(), String>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        for (key, value) in values {
            let key = key.as_ref();
            let value = value.as_ref();
            match key {
                "PTR_BIND" => self.server.bind = value.to_owned(),
                "PTR_DATA_DIR" => self.server.data_dir = value.to_owned(),
                "PTR_MODEL_BACKEND" => self.model.backend = value.to_owned(),
                "PTR_RUNTIME_MODE" => self.runtime.mode = value.to_owned(),
                "PTR_MAILBOX_CAPACITY" => {
                    self.runtime.mailbox_capacity = parse_usize(key, value)?;
                }
                "PTR_MAX_PARALLEL_CANDIDATES" => {
                    self.runtime.max_parallel_candidates = parse_usize(key, value)?;
                }
                "PTR_CANCEL_STALE_REVISIONS" => {
                    self.semantic.cancel_stale_revisions = parse_bool(key, value)?;
                }
                "PTR_RETAIN_RAW_EVIDENCE" => {
                    self.semantic.retain_raw_evidence = parse_bool(key, value)?;
                }
                "PTR_REQUIRE_CAPABILITY" => {
                    self.action_boundary.require_capability = parse_bool(key, value)?;
                }
                "PTR_REQUIRE_LIVE_GENERATION" => {
                    self.action_boundary.require_live_generation = parse_bool(key, value)?;
                }
                "PTR_REQUIRE_CURRENT_REVISION" => {
                    self.action_boundary.require_current_revision = parse_bool(key, value)?;
                }
                "PTR_STRUCTURED_TRACING" => {
                    self.observability.structured_tracing = parse_bool(key, value)?;
                }
                "PTR_ASYNC_BACKTRACE" => {
                    self.observability.async_backtrace = parse_bool(key, value)?;
                }
                "PTR_ALLOW_UNLOCKED_COMPONENTS" => {
                    self.research.allow_unlocked_components = parse_bool(key, value)?;
                }
                "PTR_REQUIRE_EXPERIMENT_MANIFEST" => {
                    self.research.require_experiment_manifest = parse_bool(key, value)?;
                }
                _ => {}
            }
        }
        self.validate()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.server.bind.trim().is_empty() {
            return Err("server.bind must not be empty".into());
        }
        if self.runtime.mailbox_capacity == 0 {
            return Err("runtime.mailbox_capacity must be greater than zero".into());
        }
        if self.runtime.max_parallel_candidates == 0 {
            return Err("runtime.max_parallel_candidates must be greater than zero".into());
        }
        match self.runtime.mode.as_str() {
            "standalone" | "cluster" => {}
            other => return Err(format!("unsupported runtime.mode: {other}")),
        }
        if let Some(wire) = &self.wire {
            wire.validate().map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    /// Validation for the standalone daemon boundary. Library defaults remain
    /// usable for in-memory tests, but the daemon must never silently widen its
    /// deployment mode or invent persistence.
    pub fn validate_daemon(&self) -> Result<(), String> {
        self.validate()?;
        if self.runtime.mode != "standalone" {
            return Err("ptrd only permits standalone mode in this milestone".into());
        }
        if !is_loopback_bind(&self.server.bind) {
            return Err("ptrd only permits loopback server.bind".into());
        }
        if self.server.data_dir.trim().is_empty() {
            return Err("server.data_dir must be explicit".into());
        }
        if !std::path::Path::new(&self.server.data_dir).is_dir() {
            return Err(format!(
                "server.data_dir does not exist: {}",
                self.server.data_dir
            ));
        }
        if self.model.backend.trim().is_empty() {
            return Err("model.backend must be explicit".into());
        }
        Ok(())
    }
}

fn is_loopback_bind(bind: &str) -> bool {
    let Ok(address) = bind.parse::<std::net::SocketAddr>() else {
        return false;
    };
    address.ip().is_loopback()
}

fn parse_usize(key: &str, value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("{key} must be an unsigned integer"))
}

fn parse_bool(key: &str, value: &str) -> Result<bool, String> {
    value
        .parse::<bool>()
        .map_err(|_| format!("{key} must be true or false"))
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub bind: String,
    pub data_dir: String,
}
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8080".into(),
            data_dir: "data".into(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    pub backend: String,
}
impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            backend: "reference-echo".into(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeConfig {
    pub mode: String,
    pub mailbox_capacity: usize,
    pub max_parallel_candidates: usize,
}
impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            mode: "standalone".into(),
            mailbox_capacity: 128,
            max_parallel_candidates: 4,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct SemanticConfig {
    pub cancel_stale_revisions: bool,
    pub retain_raw_evidence: bool,
}
impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            cancel_stale_revisions: true,
            retain_raw_evidence: true,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ActionBoundaryConfig {
    pub require_capability: bool,
    pub require_live_generation: bool,
    pub require_current_revision: bool,
}
impl Default for ActionBoundaryConfig {
    fn default() -> Self {
        Self {
            require_capability: true,
            require_live_generation: true,
            require_current_revision: true,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ObservabilityConfig {
    pub structured_tracing: bool,
    pub async_backtrace: bool,
}
impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            structured_tracing: true,
            async_backtrace: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResearchConfig {
    pub allow_unlocked_components: bool,
    pub require_experiment_manifest: bool,
}
impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            allow_unlocked_components: true,
            require_experiment_manifest: true,
        }
    }
}

/// Upper bound on `wire.max_concurrent`. A limit is a decision about how much of the
/// host one peer population may hold at once; one this large is no longer a limit.
pub const MAX_WIRE_CONCURRENT: usize = 4096;

/// Shortest `wire.per_connection_deadline_ms`: below this a handshake on loopback
/// does not reliably finish, so every connection would be cut.
pub const MIN_WIRE_DEADLINE_MS: u64 = 10;

/// Longest `wire.per_connection_deadline_ms`: ten minutes. A connection allowed to
/// stay open longer holds a slot for as long as a peer cares to keep it.
pub const MAX_WIRE_DEADLINE_MS: u64 = 600_000;

/// The network wires a host serves, and where the peers it knows can be reached
/// (D4 in issue #23).
///
/// No field has a default, and the section itself is optional: a configuration
/// without `[wire]` serves no wire at all, and one with it has said, field by field,
/// how much it will hold and how long it will wait. "When do we give up" is the
/// host's written policy, never a number a library picked.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WireConfig {
    /// Connections handled at once, per service. One more is refused, not queued.
    pub max_concurrent: usize,
    /// How long one connection may take, handshake to answer, in milliseconds.
    pub per_connection_deadline_ms: u64,
    /// Which wires to serve: `"exec"` and/or `"pod"`.
    pub services: Vec<String>,
    /// Where each peer the deployment knows can be reached. May be empty.
    pub peers: Vec<WirePeerConfig>,
}

/// One peer's address, as the operator wrote it.
///
/// Addresses are deployment facts rather than history, so they come from here and
/// are rebuilt at every start; nothing journals them (C8, `34-address-authority.md`).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WirePeerConfig {
    /// The peer's endpoint id: its public key as 64 lowercase hex digits, exactly as
    /// the transport renders it. No other spelling is accepted, because every
    /// comparison of identities in this system is a comparison of this text.
    pub id: String,
    /// The socket address the peer is reachable at directly, as `IP:PORT`.
    pub direct: String,
}

/// Why a `[wire]` section was refused. Each its own code, so an operator can tell
/// which line to fix.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WireConfigError {
    /// `max_concurrent` is zero, or above [`MAX_WIRE_CONCURRENT`].
    Concurrency { value: usize },
    /// `per_connection_deadline_ms` is outside
    /// [`MIN_WIRE_DEADLINE_MS`]..=[`MAX_WIRE_DEADLINE_MS`].
    Deadline { value_ms: u64 },
    /// `services` is empty: a `[wire]` section that serves nothing is a mistake,
    /// not a configuration.
    NoServices,
    /// A service this build does not know.
    UnknownService { service: String },
    /// A service named twice.
    DuplicateService { service: String },
    /// A peer id that is not 64 lowercase hex digits.
    MalformedPeerId { id: String },
    /// A peer address that is not `IP:PORT`.
    MalformedPeerAddress { id: String, direct: String },
    /// Two entries for one peer id. The book keeps one address per peer, and which
    /// of two the operator meant is not this crate's guess.
    DuplicatePeer { id: String },
}

impl WireConfigError {
    /// Stable diagnostic code for this refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Concurrency { .. } => "PTR_CONFIG_WIRE_CONCURRENCY",
            Self::Deadline { .. } => "PTR_CONFIG_WIRE_DEADLINE",
            Self::NoServices => "PTR_CONFIG_WIRE_NO_SERVICES",
            Self::UnknownService { .. } => "PTR_CONFIG_WIRE_UNKNOWN_SERVICE",
            Self::DuplicateService { .. } => "PTR_CONFIG_WIRE_DUPLICATE_SERVICE",
            Self::MalformedPeerId { .. } => "PTR_CONFIG_WIRE_MALFORMED_PEER_ID",
            Self::MalformedPeerAddress { .. } => "PTR_CONFIG_WIRE_MALFORMED_PEER_ADDRESS",
            Self::DuplicatePeer { .. } => "PTR_CONFIG_WIRE_DUPLICATE_PEER",
        }
    }
}

impl std::fmt::Display for WireConfigError {
    /// The code, then what was wrong.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Concurrency { value } => write!(
                f,
                "{}: wire.max_concurrent must be 1..={MAX_WIRE_CONCURRENT}, got {value}",
                self.code()
            ),
            Self::Deadline { value_ms } => write!(
                f,
                "{}: wire.per_connection_deadline_ms must be \
                 {MIN_WIRE_DEADLINE_MS}..={MAX_WIRE_DEADLINE_MS}, got {value_ms}",
                self.code()
            ),
            Self::NoServices => write!(f, "{}: wire.services must name a wire", self.code()),
            Self::UnknownService { service } => {
                write!(f, "{}: unknown wire service {service:?}", self.code())
            }
            Self::DuplicateService { service } => {
                write!(f, "{}: wire service {service:?} named twice", self.code())
            }
            Self::MalformedPeerId { id } => write!(
                f,
                "{}: peer id {id:?} is not 64 lowercase hex digits",
                self.code()
            ),
            Self::MalformedPeerAddress { id, direct } => write!(
                f,
                "{}: peer {id} address {direct:?} is not IP:PORT",
                self.code()
            ),
            Self::DuplicatePeer { id } => {
                write!(f, "{}: peer {id} is listed twice", self.code())
            }
        }
    }
}

impl std::error::Error for WireConfigError {}

/// The services a `[wire]` section may name.
pub const WIRE_SERVICES: [&str; 2] = ["exec", "pod"];

impl WireConfig {
    /// Refuse a section that has not decided something it must, or decided it out of
    /// range. Checked in the order the fields are written.
    pub fn validate(&self) -> Result<(), WireConfigError> {
        if self.max_concurrent == 0 || self.max_concurrent > MAX_WIRE_CONCURRENT {
            return Err(WireConfigError::Concurrency {
                value: self.max_concurrent,
            });
        }
        if !(MIN_WIRE_DEADLINE_MS..=MAX_WIRE_DEADLINE_MS).contains(&self.per_connection_deadline_ms)
        {
            return Err(WireConfigError::Deadline {
                value_ms: self.per_connection_deadline_ms,
            });
        }
        if self.services.is_empty() {
            return Err(WireConfigError::NoServices);
        }
        let mut seen = std::collections::BTreeSet::new();
        for service in &self.services {
            if !WIRE_SERVICES.contains(&service.as_str()) {
                return Err(WireConfigError::UnknownService {
                    service: service.clone(),
                });
            }
            if !seen.insert(service.as_str()) {
                return Err(WireConfigError::DuplicateService {
                    service: service.clone(),
                });
            }
        }
        let mut ids = std::collections::BTreeSet::new();
        for peer in &self.peers {
            if !is_canonical_key(&peer.id) {
                return Err(WireConfigError::MalformedPeerId {
                    id: peer.id.clone(),
                });
            }
            if peer.direct.parse::<std::net::SocketAddr>().is_err() {
                return Err(WireConfigError::MalformedPeerAddress {
                    id: peer.id.clone(),
                    direct: peer.direct.clone(),
                });
            }
            if !ids.insert(peer.id.as_str()) {
                return Err(WireConfigError::DuplicatePeer {
                    id: peer.id.clone(),
                });
            }
        }
        Ok(())
    }

    /// Whether `service` is one this section serves.
    pub fn serves(&self, service: &str) -> bool {
        self.services.iter().any(|named| named == service)
    }
}

/// 64 lowercase hex digits: the only spelling of an endpoint id that compares equal
/// to the transport's.
fn is_canonical_key(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
