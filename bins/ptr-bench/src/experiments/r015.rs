//! Experiment R015, Dragonfly pod addressing, as a harness: `ptr-bench dragonfly-addressing
//! <seed> [--smoke]` is the one provenance-tracked entrypoint of a run. It generates the seed's
//! world itself, consumes it, and prints exactly one JSON object on stdout.
//!
//! The world comes from the training step (`python3 -m ptr_training.dragonfly.experiment`), which
//! the harness starts as a child process in a scratch directory of its own under the runner's
//! temporary directory and removes afterwards: nothing has to exist in the checkout beforehand,
//! so a clean checkout runs. The training step builds the encoder the experiment's `config.toml`
//! pins and stops with exit status 2 while that pin is unset or the encoder is unavailable; the
//! harness then exits 2 too. `--smoke` selects the deterministic stand-in encoder and marks the
//! output `"evidence": "none (stand-in encoder)"`; a smoke run is a check that the machinery
//! works, never a result.
//!
//! The arms, over the same probes:
//!
//! - `learned`: [`LearnedPodRouter`] over the activated addresses, through `route_checked`, with
//!   the preregistered thresholds. It may abstain, and an abstention is not a routed answer.
//! - `learned_no_abstain`: the same addresses with both abstention thresholds at zero (partial
//!   coverage still abstains: that is not a threshold).
//! - `learned_no_bias`: the same addresses with every bias set to zero.
//! - `sweep`: the same addresses at each pre-fixed `(abstain_below, min_margin)` pair.
//! - `cosine_only`: the pod whose description vector is closest to the query, always answering.
//! - `constant_score`: the registry's default [`PodRouter`] (every candidate scores 1.0).
//! - `oracle`: the ceiling; routes to the probe's label and declines a probe that names no pod.
//!
//! Metrics per arm, over all probes: `routing_accuracy` (labelled probes routed to their pod, of
//! all labelled probes), `abstain_rate` (probes the arm declined) and `false_route_rate` (probes
//! routed to a wrong pod, or to any pod when the probe names none), each also as an integer
//! permille rounded down. The harness only measures; `criteria.toml` judges, and the experiment
//! stays `planned` until its preregistration is frozen.
//!
//! The `reflex` block is a second and a third claim, kept apart from the routing comparison: the
//! `learned` router is put behind the runtime's real failover path (a fallback plan naming the
//! lexically smallest pod as the default, whose generation the lifecycle holds live) and every
//! probe is dispatched as a Pod request. `reflex_failover_coverage` is the mechanism (every miss
//! of the learned router is caught by the default pod); `reflex_dispatch_hits` is the benefit
//! (the dispatches the learned router makes itself are correct more often than sending every
//! request to the default). Neither is read from the routing arms, and a block that exercised
//! nothing (no miss, no hit) is reported as such so the criteria call the seed INVALID instead of
//! passing it.

use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_model_api::{
    InferenceBackend, ModelError, ModelEvent, ModelRequest, ModelResumeRequest,
    ResumableInferenceBackend,
};
use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_router::{
    route_checked, CheckedRouter, DecisionNonce, EncoderRef, FallbackPlan, LearnedPodRouter,
    LearnedRouterConfig, NonceSequence, PodRepresentation, PodRepresentationParts, PodRouter,
    RepresentationStore, RouteError, RouteFeatures, RouteRequest, RoutedChoice, RoutedError,
    Router, SamplingKey,
};
use ptr_runtime::{pod_lifecycle_target, PodSampler, PtrRuntime, RouteQuery, SetFallbackPlan};
use ptr_types::{
    CapabilityId, CapsuleId, Effect, Generation, NamespaceId, PodAddress, PodId,
    PodRevisionAddress, Probability, ProjectId, RequestId, TypeId, VerificationLevel,
};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::Mutex;

/// The preregistered thresholds of the `learned` arm, in permille. They restate
/// `LearnedRouterConfig::default()` and `config.toml`'s `abstain_below_permille` and
/// `abstain_margin_permille`.
const ABSTAIN_BELOW_PERMILLE: u32 = 500;
const MIN_MARGIN_PERMILLE: u32 = 150;

/// The pre-fixed sensitivity sweep, `(abstain_below, min_margin)` in permille. They restate
/// `config.toml`'s `sweep_abstain_below_permille` and `sweep_min_margin_permille`.
const SWEEP: [(u32, u32); 4] = [(400, 100), (500, 150), (600, 200), (500, 300)];

struct ProbePod(PodManifest);
impl DynPod for ProbePod {
    fn manifest(&self) -> &PodManifest {
        &self.0
    }

    fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String> {
        Ok(input)
    }
}

/// What one arm did with one probe.
#[derive(Clone, Debug, Eq, PartialEq)]
enum Outcome {
    Routed(PodId),
    Abstained,
}

#[derive(Clone, Default)]
struct Tally {
    labelled: u64,
    correct: u64,
    abstained: u64,
    false_routes: u64,
    probes: u64,
}

impl Tally {
    fn record(&mut self, label: Option<&PodId>, outcome: &Outcome) {
        self.probes += 1;
        if label.is_some() {
            self.labelled += 1;
        }
        match (outcome, label) {
            (Outcome::Abstained, _) => self.abstained += 1,
            (Outcome::Routed(pod), Some(expected)) if pod == expected => self.correct += 1,
            (Outcome::Routed(_), _) => self.false_routes += 1,
        }
    }

    fn rate(numerator: u64, denominator: u64) -> f64 {
        if denominator == 0 {
            0.0
        } else {
            numerator as f64 / denominator as f64
        }
    }

    /// `numerator / denominator` as an integer permille rounded down.
    fn permille(numerator: u64, denominator: u64) -> i64 {
        (numerator * 1000).checked_div(denominator).unwrap_or(0) as i64
    }

    fn accuracy_permille(&self) -> i64 {
        Self::permille(self.correct, self.labelled)
    }

    fn false_route_permille(&self) -> i64 {
        Self::permille(self.false_routes, self.probes)
    }

    fn metrics(&self) -> Value {
        json!({
            "routing_accuracy": Self::rate(self.correct, self.labelled),
            "abstain_rate": Self::rate(self.abstained, self.probes),
            "false_route_rate": Self::rate(self.false_routes, self.probes),
            "routing_accuracy_permille": self.accuracy_permille(),
            "abstain_permille": Self::permille(self.abstained, self.probes),
            "false_route_permille": self.false_route_permille(),
            "correct": self.correct,
            "abstained": self.abstained,
            "false_routes": self.false_routes,
        })
    }
}

/// `(left - right) / probes` in permille, rounded down (towards minus infinity).
fn gap_permille(left: u64, right: u64, probes: u64) -> i64 {
    ((left as i128 - right as i128) * 1000)
        .checked_div_euclid(probes as i128)
        .unwrap_or(0) as i64
}

/// A digest that differs for every probe index (a single byte would repeat after 255).
fn digest_of_index(index: usize) -> [u8; 32] {
    let mut digest = [0u8; 32];
    digest[..8].copy_from_slice(&(index as u64).to_le_bytes());
    digest
}

fn nonce_of_index(index: usize) -> [u8; 16] {
    let mut nonce = [0u8; 16];
    nonce[..8].copy_from_slice(&(index as u64).to_le_bytes());
    nonce
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("{} is not JSON: {error}", path.display()))
}

fn f32_vector(value: &Value, what: &str) -> Result<Vec<f32>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{what} is not an array"))?
        .iter()
        .map(|element| {
            element
                .as_f64()
                .map(|number| number as f32)
                .ok_or_else(|| format!("{what} holds a non-number"))
        })
        .collect()
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    let dot: f64 = left
        .iter()
        .zip(right)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (norm(left) * norm(right))
}

/// A scratch directory that is removed when it goes out of scope, also on an early return.
struct Scratch(PathBuf);

impl Scratch {
    fn create(seed: u64) -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!("ptr-r015-{seed}-{}", std::process::id()));
        if path.exists() {
            return Err(format!("refusing to reuse {}", path.display()));
        }
        std::fs::create_dir(&path)
            .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
        Ok(Self(path))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The command that runs the training step for `seed` into `out`.
///
/// A confirmatory run goes through the locked environment: `uv run --locked --project training
/// --extra dragonfly`, with the virtual environment in `venv` (inside the run's scratch directory,
/// so the checkout is not written to). `--locked` refuses a `uv.lock` that is out of date, and the
/// training step itself exits 2 when the installed `sentence-transformers`, `torch`,
/// `transformers` or `numpy` differ from the lock. A smoke run uses the plain `python3` and needs
/// no network: it only records the environment in the world.
fn training_command(seed: u64, out: &Path, smoke: bool, venv: &Path) -> Command {
    let training = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../training");
    let module_args = [
        "python",
        "-B",
        "-m",
        "ptr_training.dragonfly.experiment",
        "--seed",
    ];
    let mut command;
    if smoke {
        command = Command::new("python3");
        command.args(&module_args[1..]);
    } else {
        command = Command::new("uv");
        command
            .args(["run", "--locked", "--project"])
            .arg(&training)
            .args(["--extra", "dragonfly"])
            .args(module_args)
            .env("UV_PROJECT_ENVIRONMENT", venv);
    }
    command
        .arg(seed.to_string())
        .arg("--out")
        .arg(out)
        .env("PYTHONPATH", training.join("src"))
        .env("PYTHONDONTWRITEBYTECODE", "1")
        // The harness prints exactly one JSON object on stdout; the training step's own report is
        // not part of it (its world is read from the scratch directory).
        .stdout(std::process::Stdio::null());
    if smoke {
        command.arg("--smoke");
    }
    command
}

/// Runs the training step for `seed` into `out`; a non-zero exit is a harness failure.
fn generate_world(seed: u64, out: &Path, smoke: bool, venv: &Path) -> Result<(), String> {
    let mut command = training_command(seed, out, smoke, venv);
    let program = command.get_program().to_string_lossy().into_owned();
    let status = command
        .status()
        .map_err(|error| format!("cannot start the training step ({program}): {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "the training step failed ({status}); a confirmatory run needs the encoder the \
             preregistration pins and the environment training/uv.lock names"
        ))
    }
}

/// The same address with its bias replaced.
fn without_bias(representation: &PodRepresentation) -> Result<PodRepresentation, String> {
    PodRepresentation::new(PodRepresentationParts {
        project: representation.project().clone(),
        pod: representation.pod().clone(),
        identity: representation.identity().clone(),
        cluster_key: representation.cluster_key(),
        encoder: representation.encoder().clone(),
        generation: representation.generation(),
        z: representation.z().to_vec(),
        bias: 0.0,
        weights: representation.weights(),
        gate: representation.gate(),
        activated: representation.activated(),
    })
    .map_err(|error| error.to_string())
}

fn learned_router(
    encoder: &EncoderRef,
    representations: &[PodRepresentation],
    abstain_below_permille: u32,
    min_margin_permille: u32,
) -> Result<LearnedPodRouter, String> {
    let mut store = RepresentationStore::new(encoder.clone());
    for representation in representations {
        match store.insert(representation.clone()) {
            Ok(_) => {}
            // A representation that failed its gate is a result, not an error.
            Err(RouteError::InactiveRepresentation { .. }) => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    LearnedPodRouter::new(
        store,
        LearnedRouterConfig {
            abstain_below: f64::from(abstain_below_permille) / 1000.0,
            min_margin: f64::from(min_margin_permille) / 1000.0,
        },
    )
    .map_err(|error| error.to_string())
}

fn run_seed(seed: u64, smoke: bool) -> Result<Value, String> {
    let scratch = Scratch::create(seed)?;
    let dir = scratch.0.as_path();
    generate_world(seed, dir, smoke, &dir.join("venv"))?;
    let world = read_json(&dir.join("world.json"))?;
    let world_smoke = world["smoke"].as_bool().ok_or("world.smoke missing")?;
    if world_smoke != smoke {
        return Err("the world's smoke marker does not match the requested mode".into());
    }
    let encoder = EncoderRef {
        id: world["encoder"]["id"]
            .as_str()
            .ok_or("encoder.id missing")?
            .to_owned(),
        revision: world["encoder"]["revision"]
            .as_str()
            .ok_or("encoder.revision missing")?
            .to_owned(),
    };

    let project = ProjectId::from(world["project"].as_str().ok_or("project missing")?);
    let mut registry = PodRegistry::default();
    let mut representations: Vec<PodRepresentation> = Vec::new();
    let mut descriptions: Vec<(PodId, Vec<f32>)> = Vec::new();
    for pod in world["pods"].as_array().ok_or("pods missing")? {
        let pod_id = PodId::from(pod["pod_id"].as_str().ok_or("pod_id missing")?);
        registry.register(Arc::new(ProbePod(PodManifest {
            project: project.clone(),
            id: pod_id.clone(),
            capabilities: vec![CapabilityId::from("answer")],
            accepts: vec![TypeId::from("Text")],
            produces: vec![TypeId::from("Text")],
            effects: vec![Effect::Pure],
            protocol_version: 1,
        })));
        descriptions.push((
            pod_id,
            f32_vector(&pod["description_vector"], "description")?,
        ));
        let manifest_path = dir.join(pod["manifest"].as_str().ok_or("manifest missing")?);
        let manifest = std::fs::read_to_string(&manifest_path)
            .map_err(|error| format!("cannot read {}: {error}", manifest_path.display()))?;
        let manifest_json: Value = serde_json::from_str(&manifest)
            .map_err(|error| format!("{} is not JSON: {error}", manifest_path.display()))?;
        let weights_path = dir.join(
            manifest_json["weights_file"]
                .as_str()
                .ok_or("weights_file missing")?,
        );
        let weights = std::fs::read(&weights_path)
            .map_err(|error| format!("cannot read {}: {error}", weights_path.display()))?;
        representations.push(
            PodRepresentation::from_artifact(&manifest, &weights)
                .map_err(|error| format!("{}: {error}", manifest_path.display()))?,
        );
    }
    let loaded = learned_router(&encoder, &representations, 500, 150)?
        .store()
        .len() as u64;
    let refused = representations.len() as u64 - loaded;

    let learned = learned_router(
        &encoder,
        &representations,
        ABSTAIN_BELOW_PERMILLE,
        MIN_MARGIN_PERMILLE,
    )?;
    let no_abstain = learned_router(&encoder, &representations, 0, 0)?;
    let unbiased: Vec<PodRepresentation> = representations
        .iter()
        .map(without_bias)
        .collect::<Result<_, _>>()?;
    let no_bias = learned_router(
        &encoder,
        &unbiased,
        ABSTAIN_BELOW_PERMILLE,
        MIN_MARGIN_PERMILLE,
    )?;
    let sweep: Vec<LearnedPodRouter> = SWEEP
        .iter()
        .map(|(below, margin)| learned_router(&encoder, &representations, *below, *margin))
        .collect::<Result<_, _>>()?;

    let key = SamplingKey::from_bytes([0x15; 32]);
    let capability = CapabilityId::from("answer");
    let input_type = TypeId::from("Text");
    let mut learned_tally = Tally::default();
    let (mut no_abstain_tally, mut no_bias_tally) = (Tally::default(), Tally::default());
    let (mut cosine_tally, mut constant_tally, mut oracle_tally) =
        (Tally::default(), Tally::default(), Tally::default());
    let mut sweep_tallies = vec![Tally::default(); SWEEP.len()];
    let mut reflex_probes: Vec<ReflexProbe> = Vec::new();
    for (index, probe) in world["probes"]
        .as_array()
        .ok_or("probes missing")?
        .iter()
        .enumerate()
    {
        let label = probe["label"].as_str().map(PodId::from);
        let query = f32_vector(&probe["vector"], "probe vector")?;
        let features = RouteFeatures::new()
            .with_vector(query.clone())
            .map_err(|error| error.to_string())?;
        let request_id = RequestId::from(format!("probe-{index}").as_str());
        reflex_probes.push(ReflexProbe {
            request_id: request_id.0.clone(),
            vector: query.clone(),
            label: label.clone(),
        });
        let request = RouteRequest {
            registry: &registry,
            project: &project,
            capability: &capability,
            input_type: &input_type,
            request_id: &request_id,
            request_digest: digest_of_index(index),
            features: &features,
        };
        let nonce = DecisionNonce::from_bytes(nonce_of_index(index));

        learned_tally.record(label.as_ref(), &route(&learned, &request, &key, &nonce)?);
        no_abstain_tally.record(label.as_ref(), &route(&no_abstain, &request, &key, &nonce)?);
        no_bias_tally.record(label.as_ref(), &route(&no_bias, &request, &key, &nonce)?);
        for (router, tally) in sweep.iter().zip(&mut sweep_tallies) {
            tally.record(label.as_ref(), &route(router, &request, &key, &nonce)?);
        }
        constant_tally.record(label.as_ref(), &route(&PodRouter, &request, &key, &nonce)?);
        let best = descriptions
            .iter()
            .max_by(|a, b| {
                cosine(&query, &a.1)
                    .total_cmp(&cosine(&query, &b.1))
                    .then_with(|| b.0.cmp(&a.0))
            })
            .map(|(pod, _)| Outcome::Routed(pod.clone()))
            .unwrap_or(Outcome::Abstained);
        cosine_tally.record(label.as_ref(), &best);
        // The ceiling: the label when there is one, else a decline.
        let ceiling = label.clone().map_or(Outcome::Abstained, Outcome::Routed);
        oracle_tally.record(label.as_ref(), &ceiling);
    }

    let reflex = reflex_pass(
        learned_router(
            &encoder,
            &representations,
            ABSTAIN_BELOW_PERMILLE,
            MIN_MARGIN_PERMILLE,
        )?,
        &registry,
        &project,
        &reflex_probes,
        // The baseline of the dispatch claim: description cosine answers every probe.
        DispatchBaseline {
            correct: cosine_tally.correct,
            dispatches: cosine_tally.probes,
        },
    )?;
    let sweep_report: Vec<Value> = SWEEP
        .iter()
        .zip(&sweep_tallies)
        .map(|((below, margin), tally)| {
            json!({
                "abstain_below_permille": below,
                "min_margin_permille": margin,
                "metrics": tally.metrics(),
            })
        })
        .collect();
    let accuracy = |tally: &Tally| Tally::rate(tally.correct, tally.labelled);
    let false_rate = |tally: &Tally| Tally::rate(tally.false_routes, tally.probes);
    Ok(json!({
        "experiment": "R015",
        "seed": seed,
        "smoke": world_smoke,
        "evidence": world["evidence"],
        "encoder": world["encoder"],
        // Interpreter, package versions, the locked versions and the SHA-256 of training/uv.lock.
        "environment": world["environment"],
        "pods": world["pods"].as_array().map_or(0, Vec::len),
        "representations_loaded": loaded,
        "representations_refused": refused,
        "probes": learned_tally.probes,
        "labelled_probes": learned_tally.labelled,
        "arms": {
            "learned": learned_tally.metrics(),
            "learned_no_abstain": no_abstain_tally.metrics(),
            "learned_no_bias": no_bias_tally.metrics(),
            "cosine_only": cosine_tally.metrics(),
            "constant_score": constant_tally.metrics(),
            "oracle": oracle_tally.metrics(),
        },
        "sweep": sweep_report,
        "reflex": reflex,
        // Signed, rounded down: the primary criterion reads these as integers.
        "accuracy_gap_permille": gap_permille(
            learned_tally.correct,
            cosine_tally.correct,
            learned_tally.labelled,
        ),
        "false_route_gap_permille": gap_permille(
            cosine_tally.false_routes,
            learned_tally.false_routes,
            learned_tally.probes,
        ),
        "learned_accuracy_above_cosine": accuracy(&learned_tally) > accuracy(&cosine_tally),
        "learned_false_route_not_above_cosine":
            false_rate(&learned_tally) <= false_rate(&cosine_tally),
    }))
}

/// What the reflex pass needs of one probe: the query vector the learned router sees and the
/// pod the probe names (if any).
struct ReflexProbe {
    request_id: String,
    vector: Vec<f32>,
    label: Option<PodId>,
}

/// The learned router behind the runtime's trusted routing path. The runtime asks it for the
/// request features; here those are the probe's query vector, looked up by request id (PTR has no
/// Rust encoder: the host supplies the embedding, and in this harness the world file does).
struct HarnessSampler {
    router: CheckedRouter<LearnedPodRouter>,
    vectors: BTreeMap<String, Vec<f32>>,
    /// The pod the router chose, by request id.
    routed: Mutex<BTreeMap<String, PodId>>,
}

impl PodSampler for HarnessSampler {
    fn route(&self, request: &RouteRequest<'_>) -> Result<Option<RoutedChoice>, RoutedError> {
        let choice = self.router.route(request)?;
        if let (Some(choice), Ok(mut routed)) = (&choice, self.routed.lock()) {
            routed.insert(request.request_id.0.clone(), choice.route().pod_id.clone());
        }
        Ok(choice)
    }

    fn key_id(&self) -> [u8; 8] {
        self.router.key_id()
    }

    fn features(&self, query: &RouteQuery<'_>) -> Result<RouteFeatures, RouteError> {
        let vector = self
            .vectors
            .get(&query.request_id.0)
            .cloned()
            .ok_or_else(|| RouteError::InvalidFeatures {
                message: "no query vector for the request".into(),
            })?;
        RouteFeatures::new().with_vector(vector)
    }
}

/// Asks for one Pod of the experiment's capability and finishes after its answer.
struct AskOnce;

impl InferenceBackend for AskOnce {
    fn name(&self) -> &'static str {
        "r015-ask"
    }

    fn infer(&self, _: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::PodRequested {
            capability: CapabilityId::from("answer"),
            input_type: TypeId::from("Text"),
            payload: b"probe".to_vec(),
        }])
    }
}

impl ResumableInferenceBackend for AskOnce {
    fn resume(&self, _: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::Finished])
    }
}

/// The probe pods are echoes: every answer passes.
struct AcceptAll;

impl Verifier<TypedPayload> for AcceptAll {
    fn verify(&self, _: &TypedPayload) -> VerificationReport {
        VerificationReport {
            status: VerificationStatus::Pass,
            level: VerificationLevel::Deterministic,
            score: Probability::new(1.0).expect("one is a probability"),
            findings: vec![],
        }
    }
}

/// What the baseline of the dispatch claim got right, and how many dispatches it made.
struct DispatchBaseline {
    correct: u64,
    dispatches: u64,
}

impl DispatchBaseline {
    /// `learned_correct / learned_dispatches - correct / dispatches` in permille, rounded down
    /// (towards minus infinity); 0 when either side made no dispatch.
    fn precision_gap_permille(&self, learned_correct: u64, learned_dispatches: u64) -> i64 {
        let denominator = i128::from(learned_dispatches) * i128::from(self.dispatches);
        let numerator = i128::from(learned_correct) * i128::from(self.dispatches)
            - i128::from(self.correct) * i128::from(learned_dispatches);
        (numerator * 1000)
            .checked_div_euclid(denominator)
            .unwrap_or(0) as i64
    }
}

/// The default pod's attempt budget: the routed attempt and the default.
const REFLEX_MAX_ATTEMPTS: u8 = 2;

/// Dispatches every probe as a Pod request through the real runtime, with the learned router in
/// front and a fallback plan behind it, and reports what the reflex counters say. The default pod
/// is the lexically smallest pod id, fixed before any run (`config.toml`).
fn reflex_pass(
    learned: LearnedPodRouter,
    registry: &PodRegistry,
    project: &ProjectId,
    probes: &[ReflexProbe],
    baseline: DispatchBaseline,
) -> Result<Value, String> {
    let default_pod = registry
        .candidates(
            project,
            &CapabilityId::from("answer"),
            &TypeId::from("Text"),
        )
        .into_iter()
        .min()
        .ok_or("the registry has no pod")?;
    let revision = PodRevisionAddress {
        address: PodAddress::new(
            project.clone(),
            NamespaceId::from("r015"),
            default_pod.clone(),
        )
        .map_err(|error| format!("default pod address: {error:?}"))?,
        semantic_revision: [1; 32],
        generation: Generation(1),
    };
    let mut runtime = PtrRuntime::new(PtrConfig::default()).map_err(|e| format!("{e:?}"))?;
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: project.0.as_str().into(),
            capsule: CapsuleId::from(pod_lifecycle_target(&revision).as_str()),
            generation: revision.generation,
        })
        .map_err(|error| format!("default pod lifecycle: {error:?}"))?;
    runtime
        .set_fallback_plan(SetFallbackPlan {
            project: project.clone(),
            capability: CapabilityId::from("answer"),
            plan: FallbackPlan::new(Some(default_pod.clone()), REFLEX_MAX_ATTEMPTS)
                .map_err(|error| error.to_string())?,
            default_revision: Some(revision),
        })
        .map_err(|error| format!("fallback plan: {error:?}"))?;
    let sampler = HarnessSampler {
        router: CheckedRouter::new(
            learned,
            SamplingKey::from_bytes([0x15; 32]),
            NonceSequence::starting_at([0x15; 8], 0),
        ),
        vectors: probes
            .iter()
            .map(|probe| (probe.request_id.clone(), probe.vector.clone()))
            .collect(),
        routed: Mutex::new(BTreeMap::new()),
    };

    let (mut hits_correct, mut default_only_correct) = (0u64, 0u64);
    for probe in probes {
        let hits_before = runtime.reflex_counters().hits();
        // A refused request (nothing covered it) is a counted miss, not a harness failure.
        let _ = runtime.run_resumable_with_pods_using_router(
            RequestId::from(probe.request_id.as_str()),
            project,
            "probe",
            &AskOnce,
            registry,
            &sampler,
            &AcceptAll,
            2,
        );
        if probe.label.as_ref() == Some(&default_pod) {
            default_only_correct += 1;
        }
        // A hit is the learned router's own dispatch; it is correct when the probe names that pod.
        if runtime.reflex_counters().hits() > hits_before {
            let routed = sampler
                .routed
                .lock()
                .map_err(|_| "routed-pod record poisoned".to_owned())?
                .get(&probe.request_id)
                .cloned();
            if routed.is_some() && routed == probe.label {
                hits_correct += 1;
            }
        }
    }

    let counters = runtime.reflex_counters();
    let misses = counters.misses();
    let failovers = counters.failovers();
    let miss_reasons: BTreeMap<&str, u64> = counters
        .miss_reasons()
        .iter()
        .map(|(reason, count)| (reason.code(), *count))
        .collect();
    let uncovered_reasons: BTreeMap<&str, u64> = counters
        .uncovered_reasons()
        .iter()
        .map(|(reason, count)| (reason.code(), *count))
        .collect();
    Ok(json!({
        "default_pod": default_pod.to_string(),
        "max_attempts": REFLEX_MAX_ATTEMPTS,
        "requests": probes.len(),
        "dispatch_hits": counters.hits(),
        "misses": misses,
        "failovers": failovers,
        "rejected": counters.rejected(),
        "uncovered": counters.uncovered(),
        "inconsistent": counters.inconsistent(),
        "miss_reasons": miss_reasons,
        "uncovered_reasons": uncovered_reasons,
        // The mechanism claim: the share of misses the default pod caught, permille rounded
        // down. With no miss the rate is 0 and `misses` fails the coverage rule: a failover
        // path that was never exercised is not a pass.
        "failover_coverage_permille": Tally::permille(failovers, misses),
        "all_misses_covered": counters.all_misses_covered(),
        // The benefit claim: of the dispatches the learned router makes itself (the hits), the
        // share that went to the pod the probe names, against the same share for description
        // cosine, which dispatches every probe. Signed permille of the difference, rounded down.
        // With no hit the precision is 0 and `dispatch_hits` fails the coverage rule: a router
        // that never dispatched is not a pass.
        "dispatch_hits_correct": hits_correct,
        "dispatch_precision_permille": Tally::permille(hits_correct, counters.hits()),
        "baseline_precision_permille": Tally::permille(baseline.correct, baseline.dispatches),
        "dispatch_precision_gap_permille": baseline.precision_gap_permille(
            hits_correct,
            counters.hits(),
        ),
        // Reported, not judged: what sending every labelled probe to the default would get right.
        "default_only_correct": default_only_correct,
        "latency_samples": counters.latency().len(),
        "latency_failed": counters.latency().failed(),
    }))
}

fn route<R: Router + ?Sized>(
    router: &R,
    request: &RouteRequest<'_>,
    key: &SamplingKey,
    nonce: &DecisionNonce,
) -> Result<Outcome, String> {
    match route_checked(router, request, key, nonce) {
        Ok(Some(choice)) => Ok(Outcome::Routed(choice.route().pod_id.clone())),
        Ok(None) => Ok(Outcome::Abstained),
        Err(error) => match error.error() {
            RouteError::Abstained { .. } => Ok(Outcome::Abstained),
            other => Err(other.to_string()),
        },
    }
}

/// Run the harness for one seed and print its JSON object; exit status 2 when it cannot run.
pub fn run(seed: u64, smoke: bool) {
    match run_seed(seed, smoke) {
        Ok(result) => println!("{result}"),
        Err(message) => {
            eprintln!("ptr-bench dragonfly-addressing: {message}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(command: &Command) -> Vec<String> {
        command
            .get_args()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn a_confirmatory_run_goes_through_the_locked_environment() {
        let command = training_command(
            2017,
            Path::new("/scratch/out"),
            false,
            Path::new("/scratch/venv"),
        );
        assert_eq!(command.get_program(), "uv");
        let args = words(&command);
        assert_eq!(&args[..2], ["run", "--locked"]);
        assert!(args.windows(2).any(|pair| pair == ["--extra", "dragonfly"]));
        assert!(args.contains(&"ptr_training.dragonfly.experiment".to_owned()));
        assert!(!args.contains(&"--smoke".to_owned()));
        let envs: Vec<_> = command.get_envs().collect();
        assert!(envs.iter().any(|(key, value)| {
            *key == "UV_PROJECT_ENVIRONMENT"
                && value.map(|v| v.to_string_lossy().into_owned()) == Some("/scratch/venv".into())
        }));
    }

    #[test]
    fn a_smoke_run_uses_the_plain_interpreter_and_marks_itself() {
        let command = training_command(
            1001,
            Path::new("/scratch/out"),
            true,
            Path::new("/scratch/venv"),
        );
        assert_eq!(command.get_program(), "python3");
        let args = words(&command);
        assert_eq!(args.last().map(String::as_str), Some("--smoke"));
        assert!(!args.contains(&"uv".to_owned()) && !args.contains(&"--locked".to_owned()));
    }

    #[test]
    fn gaps_round_towards_minus_infinity() {
        assert_eq!(gap_permille(1, 0, 3), 333);
        assert_eq!(gap_permille(0, 1, 3), -334);
        assert_eq!(gap_permille(5, 5, 7), 0);
        assert_eq!(gap_permille(1, 0, 0), 0);
    }

    #[test]
    fn the_compiled_thresholds_restate_config_toml() {
        let config = include_str!(
            "../../../../experiments/runtime/R015-dragonfly-pod-addressing/config.toml"
        );
        let list = |values: Vec<u32>| {
            format!(
                "[{}]",
                values
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let line = |key: &str, value: String| format!("{key} = {value}");
        for expected in [
            line("abstain_below_permille", ABSTAIN_BELOW_PERMILLE.to_string()),
            line("abstain_margin_permille", MIN_MARGIN_PERMILLE.to_string()),
            line("reflex_max_attempts", REFLEX_MAX_ATTEMPTS.to_string()),
            line(
                "reflex_default_pod_rule",
                "\"lexically-smallest-pod-id\"".into(),
            ),
            line(
                "sweep_abstain_below_permille",
                list(SWEEP.iter().map(|pair| pair.0).collect()),
            ),
            line(
                "sweep_min_margin_permille",
                list(SWEEP.iter().map(|pair| pair.1).collect()),
            ),
        ] {
            assert!(config.contains(&expected), "config.toml lacks `{expected}`");
        }
        let default = LearnedRouterConfig::default();
        assert_eq!(
            default.abstain_below,
            f64::from(ABSTAIN_BELOW_PERMILLE) / 1000.0
        );
        assert_eq!(default.min_margin, f64::from(MIN_MARGIN_PERMILLE) / 1000.0);
    }

    #[test]
    fn the_dispatch_precision_gap_rounds_towards_minus_infinity_and_needs_dispatches() {
        let baseline = DispatchBaseline {
            correct: 1,
            dispatches: 3,
        };
        // 2/2 against 1/3: 666.67 permille, rounded down.
        assert_eq!(baseline.precision_gap_permille(2, 2), 666);
        // 0/3 against 1/3: -333.33 rounds to -334.
        assert_eq!(baseline.precision_gap_permille(0, 3), -334);
        assert_eq!(baseline.precision_gap_permille(1, 3), 0);
        // No dispatch of the learned router: no precision, so no gap (the coverage rule fails).
        assert_eq!(baseline.precision_gap_permille(0, 0), 0);
    }

    #[test]
    fn every_reflex_reference_of_the_criteria_is_a_key_the_harness_prints() {
        let criteria = include_str!(
            "../../../../experiments/runtime/R015-dragonfly-pod-addressing/criteria.toml"
        );
        let source = include_str!("r015.rs");
        let keys: Vec<&str> = criteria
            .lines()
            .filter(|line| !line.trim_start().starts_with('#'))
            .flat_map(|line| line.split('"'))
            .filter_map(|token| {
                token
                    .split_whitespace()
                    .find_map(|t| t.strip_prefix("reflex."))
            })
            .collect();
        assert!(!keys.is_empty(), "the criteria name no reflex key");
        for key in keys {
            assert!(
                source.contains(&format!("\"{key}\":")),
                "criteria.toml reads reflex.{key} but the harness does not print it"
            );
        }
    }

    #[test]
    fn a_tally_counts_a_decline_on_a_distractor_as_correct_abstention_not_a_false_route() {
        let mut tally = Tally::default();
        tally.record(None, &Outcome::Abstained);
        tally.record(None, &Outcome::Routed(PodId::from("a")));
        tally.record(Some(&PodId::from("a")), &Outcome::Routed(PodId::from("a")));
        assert_eq!((tally.probes, tally.labelled), (3, 1));
        assert_eq!(
            (tally.abstained, tally.false_routes, tally.correct),
            (1, 1, 1)
        );
    }
}
