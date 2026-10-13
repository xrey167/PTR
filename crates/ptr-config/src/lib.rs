use serde::{Deserialize, Serialize};
use std::path::Path;

mod vcw;

pub use vcw::{
    VcwBackend, VcwExperts, VcwHintMode, VcwLearning, VcwPods, VcwQwenCpu, VcwServiceConfig,
    VcwSimulation, VcwTaskKind, MAX_VCW_BUDGET, MAX_VCW_NAME_BYTES, MAX_VCW_NEW_TOKENS,
    MAX_VCW_REQUEST_MS, VCW_REPAIR_ROLES,
};

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

    /// Applies the CLI overrides and validates the result.
    pub fn apply_to(&self, config: &mut PtrConfig) -> Result<(), String> {
        self.overlay(config);
        config.validate()
    }

    /// Applies the CLI overrides without validating, for a host that validates once after every
    /// layer (file, environment, CLI), so the highest-precedence layer can still correct a value
    /// a lower layer left invalid.
    pub fn overlay(&self, config: &mut PtrConfig) {
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
    /// The optional verifier-closed workspace request path; off by default.
    pub vcw: VcwServiceConfig,
}

impl PtrConfig {
    pub fn from_toml_str(input: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(input)
    }

    /// Reads and validates a configuration file that must be complete on its own.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let config = Self::parse_path(path)?;
        config.validate()?;
        Ok(config)
    }

    /// Reads and parses a configuration file without validating it, for a host that completes
    /// the file through overlays first (a `qwen-cpu` section whose model directory and digests
    /// come from the `PTR_VCW_QWEN_*` variables). The caller validates after the overlays;
    /// [`Self::apply_env`] does so itself.
    pub fn parse_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(|e| e.to_string())?;
        Self::from_toml_str(&text).map_err(|e| e.to_string())
    }

    /// Applies the `PTR_*` environment overrides and validates the result.
    pub fn apply_env<I, K, V>(&mut self, values: I) -> Result<(), String>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        self.overlay_env(values)?;
        self.validate()
    }

    /// Applies the `PTR_*` environment overrides without validating the whole configuration
    /// (a value that does not parse is still refused), for a host that validates once after the
    /// CLI layer too.
    pub fn overlay_env<I, K, V>(&mut self, values: I) -> Result<(), String>
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
                "PTR_VCW_ENABLED" => self.vcw.enabled = parse_bool(key, value)?,
                "PTR_VCW_QWEN_MODEL_DIR" => {
                    self.vcw.experts.qwen_cpu.model_dir = value.to_owned();
                }
                "PTR_VCW_QWEN_WEIGHTS_SHA256" => {
                    self.vcw.experts.qwen_cpu.weights_sha256 = value.to_owned();
                }
                "PTR_VCW_QWEN_CONFIG_SHA256" => {
                    self.vcw.experts.qwen_cpu.config_sha256 = value.to_owned();
                }
                "PTR_VCW_QWEN_TOKENIZER_SHA256" => {
                    self.vcw.experts.qwen_cpu.tokenizer_sha256 = value.to_owned();
                }
                "PTR_VCW_QWEN_TOKENIZER_CONFIG_SHA256" => {
                    self.vcw.experts.qwen_cpu.tokenizer_config_sha256 = value.to_owned();
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), String> {
        self.check_supported_settings().map_err(|e| e.to_string())?;
        if self.server.bind.trim().is_empty() {
            return Err("server.bind must not be empty".into());
        }
        if self.runtime.mailbox_capacity == 0 {
            return Err("runtime.mailbox_capacity must be greater than zero".into());
        }
        if self.runtime.max_parallel_candidates == 0 {
            return Err("runtime.max_parallel_candidates must be greater than zero".into());
        }
        self.vcw.validate()?;
        match self.runtime.mode.as_str() {
            "standalone" | "cluster" => Ok(()),
            other => Err(format!("unsupported runtime.mode: {other}")),
        }
    }

    /// Refuses every setting the system cannot honour. A field that no code
    /// reads must not be silently accepted with a non-default value, so each
    /// one is either refused here or listed as consumed in
    /// [`FIELD_DISPOSITIONS`].
    pub fn check_supported_settings(&self) -> Result<(), UnsupportedSetting> {
        let defaults = Self::default();
        if !self.semantic.retain_raw_evidence {
            return Err(UnsupportedSetting {
                field: "semantic.retain_raw_evidence",
                reason: "must be true: raw evidence is never replaced by typed state (Invariant)",
            });
        }
        let not_supported_yet = [
            (
                "observability.structured_tracing",
                self.observability.structured_tracing == defaults.observability.structured_tracing,
            ),
            (
                "observability.async_backtrace",
                self.observability.async_backtrace == defaults.observability.async_backtrace,
            ),
            (
                "research.allow_unlocked_components",
                self.research.allow_unlocked_components
                    == defaults.research.allow_unlocked_components,
            ),
            (
                "research.require_experiment_manifest",
                self.research.require_experiment_manifest
                    == defaults.research.require_experiment_manifest,
            ),
            (
                "runtime.max_parallel_candidates",
                self.runtime.max_parallel_candidates == defaults.runtime.max_parallel_candidates,
            ),
        ];
        for (field, is_default) in not_supported_yet {
            if !is_default {
                return Err(UnsupportedSetting {
                    field,
                    reason: "only the default value is supported; changing it is not supported yet (no component reads this setting)",
                });
            }
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

/// A configuration value the system cannot honour, refused at load time.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnsupportedSetting {
    pub field: &'static str,
    pub reason: &'static str,
}

impl std::fmt::Display for UnsupportedSetting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unsupported setting {}: {}", self.field, self.reason)
    }
}

impl std::error::Error for UnsupportedSetting {}

/// What happens to a configuration field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FieldDisposition {
    /// Read by the named component; any value valid per `validate` is honoured.
    Consumed(&'static str),
    /// Nothing reads it; any non-default value is refused by
    /// `check_supported_settings`.
    RefusedUnlessDefault,
}

/// Every leaf configuration field and its disposition. A test compares this
/// table against the serialized default configuration, so a new field without
/// a consumer or a validation rule fails the build.
pub const FIELD_DISPOSITIONS: &[(&str, FieldDisposition)] = &[
    (
        "server.bind",
        FieldDisposition::Consumed("ptrd, ptr-server"),
    ),
    ("server.data_dir", FieldDisposition::Consumed("ptrd")),
    ("model.backend", FieldDisposition::Consumed("ptrd")),
    (
        "runtime.mode",
        FieldDisposition::Consumed("ptrd (validate_daemon)"),
    ),
    (
        "runtime.mailbox_capacity",
        FieldDisposition::Consumed("ptr-server RequestAdmission"),
    ),
    (
        "runtime.max_parallel_candidates",
        FieldDisposition::RefusedUnlessDefault,
    ),
    (
        "semantic.cancel_stale_revisions",
        FieldDisposition::Consumed(
            "ptr-runtime SteppedRun only; no shipped host (ptr-server, ptrd) steps runs, so no production effect",
        ),
    ),
    (
        "semantic.retain_raw_evidence",
        FieldDisposition::RefusedUnlessDefault,
    ),
    (
        "action_boundary.require_capability",
        FieldDisposition::Consumed("ptr-runtime"),
    ),
    (
        "action_boundary.require_live_generation",
        FieldDisposition::Consumed("ptr-runtime"),
    ),
    (
        "action_boundary.require_current_revision",
        FieldDisposition::Consumed("ptr-runtime"),
    ),
    (
        "observability.structured_tracing",
        FieldDisposition::RefusedUnlessDefault,
    ),
    (
        "observability.async_backtrace",
        FieldDisposition::RefusedUnlessDefault,
    ),
    (
        "research.allow_unlocked_components",
        FieldDisposition::RefusedUnlessDefault,
    ),
    (
        "research.require_experiment_manifest",
        FieldDisposition::RefusedUnlessDefault,
    ),
    (
        "vcw.enabled",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.project",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.kinds",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.max_verifier_calls",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.max_rounds",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.max_request_ms",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.hint_mode",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.learning",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.exploration_floor",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.pods.proposer",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.pods.format_repair",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.pods.value_repair",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.pods.schema_repair",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.pods.code_repair",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.pods.resource_repair",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.pods.fact_repair",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.pods.generation",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.experts.backend",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.experts.simulated.seed",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.experts.simulated.default_repair_success",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.experts.simulated.decline_on_failure",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.experts.qwen_cpu.model_dir",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.experts.qwen_cpu.weights_sha256",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.experts.qwen_cpu.config_sha256",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.experts.qwen_cpu.tokenizer_sha256",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.experts.qwen_cpu.tokenizer_config_sha256",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
    (
        "vcw.experts.qwen_cpu.max_new_tokens",
        FieldDisposition::Consumed("ptr-server, ptrd (vcw path)"),
    ),
];

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
    /// Whether a stepped resumable run (`ptr_runtime::SteppedRun`) is cancelled, before the
    /// effect boundary, when the semantic revision moved for a reason other than the run's own
    /// writes. `false` switches the cooperative cancellation off; the hard effect boundary is
    /// unaffected by either value. **No shipped host steps runs** (`ptr-server` and `ptrd` call
    /// the blocking entry points, which force the cancellation off), so this setting has no
    /// effect on any production path today; only a host that drives `step_resumable_run` itself
    /// sees it. "Consumed" here means read by that API, not enforced anywhere in production.
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
