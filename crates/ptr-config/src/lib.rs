use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct PtrConfig {
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

    pub fn validate(&self) -> Result<(), String> {
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