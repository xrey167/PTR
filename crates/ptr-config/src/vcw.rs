//! The `[vcw]` section: the verifier-closed workspace loop as an optional request path.
//!
//! Everything here is off by default. With `enabled = false` no endpoint exists and nothing in
//! the section is read except for its shape, which is always validated so that a mistake does
//! not wait for the day the path is switched on.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Most verifier calls or rounds one request may be configured with. Lower than the runtime's
/// own bound (`ptr_runtime::vcw::MAX_VERIFIER_CALLS`, `MAX_ROUNDS`; this crate cannot depend on
/// the runtime) because the service holds the runtime lock for the whole request: with the
/// per-request deadline (`max_request_ms`) the lock is held for at most that deadline plus one
/// loop step. The server re-checks against the runtime's constants when it builds the budget.
pub const MAX_VCW_BUDGET: u32 = 64;
/// Longest wall-clock deadline one request may be configured with, in milliseconds.
pub const MAX_VCW_REQUEST_MS: u32 = 120_000;
/// Longest Pod id or project name, in bytes.
pub const MAX_VCW_NAME_BYTES: usize = 128;
/// Most new tokens one model call may be asked for (`ptr_pods::MAX_NEW_TOKENS_LIMIT`).
pub const MAX_VCW_NEW_TOKENS: u32 = 128;

/// The task kinds the endpoint can serve.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VcwTaskKind {
    /// An integer or rational result of an arithmetic expression.
    Arithmetic,
    /// A JSON object checked against a JSON Schema.
    JsonObject,
    /// A program checked against tests.
    Code,
    /// Claims checked against the runtime's semantic snapshot.
    Fact,
}

impl VcwTaskKind {
    /// Every kind, in a fixed order.
    pub const ALL: [Self; 4] = [Self::Arithmetic, Self::JsonObject, Self::Code, Self::Fact];

    /// The configuration and wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Arithmetic => "arithmetic",
            Self::JsonObject => "json_object",
            Self::Code => "code",
            Self::Fact => "fact",
        }
    }
}

/// Whether the loop passes the self-model's hints on to the attention schema.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum VcwHintMode {
    /// Hints go into the trace only (the default: the self-model observes).
    #[default]
    TraceOnly,
    /// Hints are applied to the schema, which uses them only inside the admissible set.
    ApplyToSchema,
}

/// Whether the loop learns from the verifier's verdicts.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum VcwLearning {
    /// Nothing is learned (the default).
    #[default]
    Off,
    /// Each served kind learns a routing policy from the verdicts on its routed Pods and logs
    /// off-policy estimates of how that policy would have done; routing stays uniform. Nothing
    /// learned is ever used to route by this setting: promoting a policy is a host action that
    /// has no configuration.
    Shadow,
}

/// Which experts answer the loop's calls.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum VcwBackend {
    /// Seeded programmatic experts with configured error and repair rates. They draw their
    /// candidates from the client's `reference`, so a simulated answer is client-authored; every
    /// admission and response says `simulated`.
    Simulated,
    /// A real language model on the CPU (needs the `qwen-cpu` build feature).
    QwenCpu,
}

/// The Pod ids of the roles. A repair Pod may hold several roles by repeating its id; the
/// proposer must be a different Pod.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct VcwPods {
    pub proposer: String,
    pub format_repair: String,
    pub value_repair: String,
    pub schema_repair: String,
    pub code_repair: String,
    pub resource_repair: String,
    pub fact_repair: String,
    /// The generation every Pod above is installed at when the lifecycle authority does not
    /// know it yet.
    pub generation: u64,
}

impl Default for VcwPods {
    fn default() -> Self {
        Self {
            proposer: "vcw-proposer".into(),
            format_repair: "vcw-format-repair".into(),
            value_repair: "vcw-value-repair".into(),
            schema_repair: "vcw-schema-repair".into(),
            code_repair: "vcw-code-repair".into(),
            resource_repair: "vcw-resource-repair".into(),
            fact_repair: "vcw-fact-repair".into(),
            generation: 1,
        }
    }
}

impl VcwPods {
    fn named(&self) -> [(&'static str, &str); 7] {
        [
            ("vcw.pods.proposer", &self.proposer),
            ("vcw.pods.format_repair", &self.format_repair),
            ("vcw.pods.value_repair", &self.value_repair),
            ("vcw.pods.schema_repair", &self.schema_repair),
            ("vcw.pods.code_repair", &self.code_repair),
            ("vcw.pods.resource_repair", &self.resource_repair),
            ("vcw.pods.fact_repair", &self.fact_repair),
        ]
    }
}

/// Rates of the simulated experts. Finding codes are those of the `ptr-verifier` code table; the
/// server refuses a code it cannot plant or repair when it builds the service.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct VcwSimulation {
    pub seed: u32,
    /// Probability that a proposal carries each defect code.
    pub error_rates: BTreeMap<String, f64>,
    /// Probability that a repairer clears a code, unless overridden per Pod.
    pub repair_success: BTreeMap<String, f64>,
    /// Probability for a code absent from `repair_success`.
    pub default_repair_success: f64,
    /// Per Pod id, per code overrides of the repair probability.
    pub pod_repair_success: BTreeMap<String, BTreeMap<String, f64>>,
    /// A failed repair returns nothing instead of the unchanged candidate.
    pub decline_on_failure: bool,
}

impl Default for VcwSimulation {
    fn default() -> Self {
        Self {
            seed: 1,
            error_rates: BTreeMap::new(),
            repair_success: BTreeMap::new(),
            default_repair_success: 1.0,
            pod_repair_success: BTreeMap::new(),
            decline_on_failure: false,
        }
    }
}

/// The CPU language model. The directory and the four file digests are all required when this
/// backend is selected; the model is loaded only if every digest matches.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct VcwQwenCpu {
    pub model_dir: String,
    pub weights_sha256: String,
    pub config_sha256: String,
    pub tokenizer_sha256: String,
    pub tokenizer_config_sha256: String,
    /// New tokens one call may generate; `0` selects the default of 24.
    pub max_new_tokens: u32,
}

impl VcwQwenCpu {
    /// The token bound in force.
    #[must_use]
    pub fn effective_max_new_tokens(&self) -> u32 {
        if self.max_new_tokens == 0 {
            24
        } else {
            self.max_new_tokens
        }
    }

    /// The four pinned file digests with their configuration field names.
    #[must_use]
    pub fn digests(&self) -> [(&'static str, &str); 4] {
        [
            ("vcw.experts.qwen_cpu.weights_sha256", &self.weights_sha256),
            ("vcw.experts.qwen_cpu.config_sha256", &self.config_sha256),
            (
                "vcw.experts.qwen_cpu.tokenizer_sha256",
                &self.tokenizer_sha256,
            ),
            (
                "vcw.experts.qwen_cpu.tokenizer_config_sha256",
                &self.tokenizer_config_sha256,
            ),
        ]
    }
}

/// `Option<VcwBackend>` as a file value: `"unset"` is `None`, so the default section lists the
/// key (every leaf of the configuration is accounted for) without choosing a backend.
mod backend_choice {
    use super::VcwBackend;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Deserialize)]
    #[serde(rename_all = "kebab-case")]
    enum Choice {
        Unset,
        Simulated,
        QwenCpu,
    }

    pub(super) fn serialize<S: Serializer>(
        value: &Option<VcwBackend>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            Some(backend) => backend.serialize(serializer),
            None => serializer.serialize_str("unset"),
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<VcwBackend>, D::Error> {
        Ok(match Choice::deserialize(deserializer)? {
            Choice::Unset => None,
            Choice::Simulated => Some(VcwBackend::Simulated),
            Choice::QwenCpu => Some(VcwBackend::QwenCpu),
        })
    }
}

/// The expert backend and its settings.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct VcwExperts {
    /// Which experts answer. Has no default: `unset` (the value of an absent key) is refused
    /// while the section is enabled, which must name `simulated` or `qwen-cpu`, so a deployment
    /// never serves simulated answers by omission.
    #[serde(with = "backend_choice")]
    pub backend: Option<VcwBackend>,
    pub simulated: VcwSimulation,
    pub qwen_cpu: VcwQwenCpu,
}

/// The `[vcw]` section.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct VcwServiceConfig {
    /// Serve `POST /v1/vcw/tasks`. Off by default; the endpoint does not exist when off.
    pub enabled: bool,
    /// The project of the scope, the Pods and the committed answers.
    pub project: String,
    /// The task kinds the endpoint accepts.
    pub kinds: Vec<VcwTaskKind>,
    /// Verifier calls one request may spend (one per candidate).
    pub max_verifier_calls: u32,
    /// Attempts after the first one request may make.
    pub max_rounds: u32,
    pub hint_mode: VcwHintMode,
    /// `off` (default) or `shadow`: learn a routing policy from verdicts and log off-policy
    /// estimates, without routing by it. `shadow` needs `exploration_floor > 0`, because the
    /// estimates rest on the positivity the floor guarantees.
    pub learning: VcwLearning,
    /// Wall-clock deadline of one request, in milliseconds, checked between loop steps (never
    /// inside a model call or a verifier run). After it the loop is given no further candidate and
    /// the request ends `exhausted` with reason `Deadline`. In `1..=MAX_VCW_REQUEST_MS`.
    pub max_request_ms: u32,
    /// The router's exploration floor: every eligible Pod keeps at least `floor / n`
    /// propensity. `0` or a value in `1e-6..=1`.
    pub exploration_floor: f64,
    pub pods: VcwPods,
    pub experts: VcwExperts,
}

impl Default for VcwServiceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            project: "vcw".into(),
            kinds: VcwTaskKind::ALL.to_vec(),
            max_verifier_calls: 8,
            max_rounds: 4,
            hint_mode: VcwHintMode::TraceOnly,
            learning: VcwLearning::Off,
            max_request_ms: 30_000,
            exploration_floor: 0.05,
            pods: VcwPods::default(),
            experts: VcwExperts::default(),
        }
    }
}

fn valid_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_VCW_NAME_BYTES
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn probability(field: &str, value: f64) -> Result<(), String> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(format!("{field} must be a probability in 0..=1"))
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl VcwServiceConfig {
    /// Checks the shape of the section. Backend requirements (the model directory and the four
    /// digests) are checked only while the section is enabled with that backend.
    ///
    /// # Errors
    /// The first violated rule, naming the field.
    pub fn validate(&self) -> Result<(), String> {
        if !valid_name(&self.project) {
            return Err("vcw.project must be a non-empty name without edge whitespace".into());
        }
        let mut seen = BTreeSet::new();
        for kind in &self.kinds {
            if !seen.insert(*kind) {
                return Err(format!("vcw.kinds lists {} twice", kind.as_str()));
            }
        }
        if self.enabled && self.kinds.is_empty() {
            return Err("vcw.kinds must name at least one task kind when vcw is enabled".into());
        }
        for (field, value) in [
            ("vcw.max_verifier_calls", self.max_verifier_calls),
            ("vcw.max_rounds", self.max_rounds),
        ] {
            if value == 0 || value > MAX_VCW_BUDGET {
                return Err(format!("{field} must be in 1..={MAX_VCW_BUDGET}"));
            }
        }
        if self.max_request_ms == 0 || self.max_request_ms > MAX_VCW_REQUEST_MS {
            return Err(format!(
                "vcw.max_request_ms must be in 1..={MAX_VCW_REQUEST_MS}"
            ));
        }
        let floor = self.exploration_floor;
        if !(floor == 0.0 || (floor.is_finite() && (1e-6..=1.0).contains(&floor))) {
            return Err("vcw.exploration_floor must be 0 or in 1e-6..=1".into());
        }
        if self.learning == VcwLearning::Shadow && floor == 0.0 {
            return Err("vcw.learning = \"shadow\" needs vcw.exploration_floor above 0".into());
        }
        if self.pods.generation == 0 {
            return Err("vcw.pods.generation must be at least 1".into());
        }
        let named = self.pods.named();
        for (field, value) in named {
            if !valid_name(value) {
                return Err(format!(
                    "{field} must be a non-empty id without edge whitespace of at most \
                     {MAX_VCW_NAME_BYTES} bytes"
                ));
            }
        }
        for (field, value) in &named[1..] {
            if *value == self.pods.proposer {
                return Err(format!(
                    "{field} names the proposer Pod; a proposer cannot also repair"
                ));
            }
        }
        let simulation = &self.experts.simulated;
        probability(
            "vcw.experts.simulated.default_repair_success",
            simulation.default_repair_success,
        )?;
        for (code, rate) in &simulation.error_rates {
            probability(&format!("vcw.experts.simulated.error_rates.{code}"), *rate)?;
        }
        for (code, rate) in &simulation.repair_success {
            probability(
                &format!("vcw.experts.simulated.repair_success.{code}"),
                *rate,
            )?;
        }
        for (pod, rates) in &simulation.pod_repair_success {
            for (code, rate) in rates {
                probability(
                    &format!("vcw.experts.simulated.pod_repair_success.{pod}.{code}"),
                    *rate,
                )?;
            }
        }
        if self.enabled {
            match self.experts.backend {
                None => {
                    return Err(
                        "vcw.experts.backend must be set explicitly when vcw is enabled \
                         (\"simulated\" or \"qwen-cpu\")"
                            .into(),
                    )
                }
                Some(VcwBackend::QwenCpu) => self.validate_qwen_cpu()?,
                Some(VcwBackend::Simulated) => {}
            }
        }
        Ok(())
    }

    fn validate_qwen_cpu(&self) -> Result<(), String> {
        let qwen = &self.experts.qwen_cpu;
        if qwen.model_dir.trim().is_empty() {
            return Err(
                "vcw.experts.qwen_cpu.model_dir is required for the qwen-cpu backend".into(),
            );
        }
        for (field, digest) in qwen.digests() {
            if !is_sha256_hex(digest) {
                return Err(format!(
                    "{field} is required for the qwen-cpu backend and must be 64 lowercase hex \
                     digits"
                ));
            }
        }
        if qwen.max_new_tokens > MAX_VCW_NEW_TOKENS {
            return Err(format!(
                "vcw.experts.qwen_cpu.max_new_tokens must be at most {MAX_VCW_NEW_TOKENS}"
            ));
        }
        if let Some(kind) = self
            .kinds
            .iter()
            .find(|kind| !matches!(kind, VcwTaskKind::Arithmetic | VcwTaskKind::JsonObject))
        {
            return Err(format!(
                "vcw.kinds lists {}, which the qwen-cpu backend has no prompts for (only \
                 arithmetic and json_object)",
                kind.as_str()
            ));
        }
        Ok(())
    }

    /// Whether `kind` is served.
    #[must_use]
    pub fn serves(&self, kind: VcwTaskKind) -> bool {
        self.kinds.contains(&kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_off_and_valid() {
        let config = VcwServiceConfig::default();
        assert!(!config.enabled);
        config.validate().unwrap();
    }

    #[test]
    fn budgets_floor_names_and_rates_are_validated() {
        type Mutation = Box<dyn Fn(&mut VcwServiceConfig)>;
        let bad: Vec<Mutation> = vec![
            Box::new(|c| c.max_verifier_calls = 0),
            Box::new(|c| c.max_rounds = MAX_VCW_BUDGET + 1),
            Box::new(|c| c.exploration_floor = 2.0),
            Box::new(|c| c.exploration_floor = f64::NAN),
            Box::new(|c| c.exploration_floor = 1e-9),
            Box::new(|c| c.project = " ".into()),
            Box::new(|c| c.pods.proposer = String::new()),
            Box::new(|c| c.pods.value_repair = c.pods.proposer.clone()),
            Box::new(|c| c.pods.generation = 0),
            Box::new(|c| c.kinds = vec![VcwTaskKind::Code, VcwTaskKind::Code]),
            Box::new(|c| c.experts.simulated.default_repair_success = 1.5),
            Box::new(|c| {
                c.experts
                    .simulated
                    .error_rates
                    .insert("arith.value".into(), -0.1);
            }),
            Box::new(|c| {
                c.experts
                    .simulated
                    .pod_repair_success
                    .entry("p".into())
                    .or_default()
                    .insert("c".into(), f64::INFINITY);
            }),
        ];
        for (index, mutate) in bad.iter().enumerate() {
            let mut config = VcwServiceConfig::default();
            mutate(&mut config);
            assert!(config.validate().is_err(), "case {index}");
        }
    }

    #[test]
    fn an_enabled_section_needs_a_kind_and_the_qwen_backend_needs_its_pins() {
        let mut config = VcwServiceConfig {
            enabled: true,
            kinds: Vec::new(),
            ..VcwServiceConfig::default()
        };
        assert!(config.validate().is_err());
        config.kinds = vec![VcwTaskKind::Arithmetic];
        assert!(
            config.validate().is_err(),
            "an enabled section must name its backend"
        );
        config.experts.backend = Some(VcwBackend::Simulated);
        config.validate().unwrap();

        config.experts.backend = Some(VcwBackend::QwenCpu);
        assert!(config.validate().is_err());
        config.experts.qwen_cpu.model_dir = "/models/qwen".into();
        assert!(config.validate().is_err());
        let digest = "ab".repeat(32);
        config.experts.qwen_cpu.weights_sha256 = digest.clone();
        config.experts.qwen_cpu.config_sha256 = digest.clone();
        config.experts.qwen_cpu.tokenizer_sha256 = digest.clone();
        assert!(config.validate().is_err(), "three of four digests");
        config.experts.qwen_cpu.tokenizer_config_sha256 = digest.to_uppercase();
        assert!(config.validate().is_err(), "digests are lowercase hex");
        config.experts.qwen_cpu.tokenizer_config_sha256 = digest;
        config.validate().unwrap();
        config.kinds.push(VcwTaskKind::Code);
        assert!(config.validate().is_err(), "no prompts for code");
        config.kinds.pop();
        config.experts.qwen_cpu.max_new_tokens = MAX_VCW_NEW_TOKENS + 1;
        assert!(config.validate().is_err());
        // The pins are not required while the section is off.
        config.enabled = false;
        config.validate().unwrap();
    }
}
