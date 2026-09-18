use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CliOverrides {
    pub config_path: Option<String>,
    pub bind: Option<String>,
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
#[serde(default)]
pub struct PtrConfig {
    pub server: ServerConfig,
    pub runtime: RuntimeConfig,
    pub semantic: SemanticConfig,
    pub action_boundary: ActionBoundaryConfig,
    pub observability: ObservabilityConfig,
    pub research: ResearchConfig,
}

impl PtrConfig {
    pub fn from_toml_str(input: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(input)
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(|e| e.to_string())?;
        Self::from_toml_str(&text).map_err(|e| e.to_string())
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
            "standalone" | "cluster" => Ok(()),
            other => Err(format!("unsupported runtime.mode: {other}")),
        }
    }
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
#[serde(default)]
pub struct ServerConfig {
    pub bind: String,
}
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8080".into(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default)]
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
#[serde(default)]
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
#[serde(default)]
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
#[serde(default)]
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
#[serde(default)]
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
