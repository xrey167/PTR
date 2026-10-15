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
    route_checked, CheckedRouter, DecisionNonce, DefaultPodPlan, EncoderRef, LearnedPodRouter,
    LearnedRouterConfig, NonceSequence, PodRepresentation, PodRepresentationParts, PodRouter,
    RepresentationStore, RouteError, RouteFeatures, RouteRequest, RoutedChoice, RoutedError,
    Router, SamplingKey,
};
use ptr_runtime::{
    pod_lifecycle_target, PodSampler, PtrRuntime, RouteQuery, RuntimeError, SetFallbackPlan,
};
use ptr_types::{
    CapabilityId, CapsuleId, Effect, NamespaceId, PodAddress, PodId, PodRevisionAddress,
    Probability, ProjectId, RequestId, TypeId, VerificationLevel,
};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::Mutex;

/// The preregistered thresholds of the `learned` arm, in permille. They restate
/// `LearnedRouterConfig::default()` and `config.toml`'s `abstain_below_permille` and
/// `abstain_margin_permille`.
const ABSTAIN_BELOW_PERMILLE: u32 = 500;
const MIN_MARGIN_PERMILLE: u32 = 150;

/// The exact interpreter of a confirmatory run: passed to `uv run --python`, and checked by the
/// training step against `sys.version_info`. Restates `config.toml`'s `python_version`.
const PYTHON_VERSION: &str = "3.13.16";

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
    let mut command = training_step(smoke, venv);
    command
        .arg("--seed")
        .arg(seed.to_string())
        .arg("--out")
        .arg(out)
        // The harness prints exactly one JSON object on stdout; the training step's own report is
        // not part of it (its world is read from the scratch directory).
        .stdout(std::process::Stdio::null());
    if smoke {
        command.arg("--smoke");
    }
    command
}

/// The training module launched in the confirmatory environment (`uv run --locked` with the
/// pinned interpreter, virtual environment in `venv`) or, for a smoke run, with plain `python3`.
fn training_step(smoke: bool, venv: &Path) -> Command {
    let training = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../training");
    let module_args = ["python", "-B", "-m", "ptr_training.dragonfly.experiment"];
    let mut command;
    if smoke {
        command = Command::new("python3");
        command.args(&module_args[1..]);
    } else {
        command = Command::new("uv");
        command
            .args(["run", "--locked", "--python", PYTHON_VERSION, "--project"])
            .arg(&training)
            .args(["--extra", "dragonfly"])
            .args(module_args)
            .env("UV_PROJECT_ENVIRONMENT", venv);
    }
    command
        .env("PYTHONPATH", training.join("src"))
        .env("PYTHONDONTWRITEBYTECODE", "1");
    command
}

/// The preflight of a confirmatory run: the training step in the confirmatory environment with
/// `--preflight`, which checks the pinned encoder (it loads and encodes), the interpreter and the
/// locked packages and builds no world. The runner runs it before it reserves a seed.
fn preflight_command(venv: &Path) -> Command {
    let mut command = training_step(false, venv);
    command.arg("--preflight");
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
        namespace: representation.namespace().clone(),
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
    evaluate_world(seed, smoke, dir)
}

/// Routes the probes of the world in `dir` with every arm and returns the result object.
fn evaluate_world(seed: u64, smoke: bool, dir: &Path) -> Result<Value, String> {
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
    // The pods are registered as the implementations of the world's lifecycle namespace at
    // their world generation: a learned address scores only the implementation of its own
    // namespace and generation.
    let namespace = NamespaceId::from(world["namespace"].as_str().ok_or("namespace missing")?);
    let mut registry = PodRegistry::default();
    let mut representations: Vec<PodRepresentation> = Vec::new();
    let mut descriptions: Vec<(PodId, Vec<f32>)> = Vec::new();
    // Artifacts that are valid but learned for another project, namespace, pod or generation than
    // the world entry that names them. They count as refused, so the seed is INVALID through the
    // coverage rule `representations_refused <= 0` instead of being judged as a routing result.
    let mut unbound: u64 = 0;
    for pod in world["pods"].as_array().ok_or("pods missing")? {
        let pod_id = PodId::from(pod["pod_id"].as_str().ok_or("pod_id missing")?);
        let generation =
            ptr_types::Generation(pod["generation"].as_u64().ok_or("pod generation missing")?);
        registry.register_at(
            namespace.clone(),
            generation,
            Arc::new(ProbePod(PodManifest {
                project: project.clone(),
                id: pod_id.clone(),
                capabilities: vec![CapabilityId::from("answer")],
                accepts: vec![TypeId::from("Text")],
                produces: vec![TypeId::from("Text")],
                effects: vec![Effect::Pure],
                protocol_version: 1,
            })),
        );
        descriptions.push((
            pod_id.clone(),
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
        let representation = PodRepresentation::from_artifact(&manifest, &weights)
            .map_err(|error| format!("{}: {error}", manifest_path.display()))?;
        if representation_is_bound(&representation, &project, &namespace, &pod_id, generation) {
            representations.push(representation);
        } else {
            unbound += 1;
        }
    }
    let loaded = learned_router(&encoder, &representations, 500, 150)?
        .store()
        .len() as u64;
    let refused = unbound + (representations.len() as u64 - loaded);
    let registered: BTreeSet<PodId> = descriptions.iter().map(|(pod, _)| pod.clone()).collect();

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
    // Labels that name no registered pod. A malformed world is INVALID through the coverage rule
    // `unregistered_labels <= 0`; the oracle would otherwise route to the unknown label and count
    // it as correct.
    let mut unregistered_labels: u64 = 0;
    for (index, probe) in world["probes"]
        .as_array()
        .ok_or("probes missing")?
        .iter()
        .enumerate()
    {
        let label = probe["label"].as_str().map(PodId::from);
        if label.as_ref().is_some_and(|pod| !registered.contains(pod)) {
            unregistered_labels += 1;
        }
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
        "unregistered_labels": unregistered_labels,
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

/// What one dispatch changed in the reflex counters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CounterDelta {
    hits: u64,
    misses: u64,
    failovers: u64,
}

/// Decides whether one dispatch is accounted for. A served request is exactly one hit, or one miss
/// caught by the default; a refused request is tolerated only when it is a miss nothing covered
/// (no pod, a routing decline, a refused failover) and exactly one miss was counted. Any other
/// error (invalid features, the resume limit, a failing pod) or a counter change that fits
/// neither is a harness failure: the experiment must not read counters off a run that broke.
fn accounted_dispatch(
    probe: &str,
    outcome: &Result<(), RuntimeError>,
    delta: CounterDelta,
) -> Result<(), String> {
    let tolerated = match outcome {
        Ok(()) => matches!(
            (delta.hits, delta.misses, delta.failovers),
            (1, 0, 0) | (0, 1, 1)
        ),
        Err(error) => {
            let uncovered_miss = match error {
                RuntimeError::PodUnavailable { .. } | RuntimeError::FailoverRefused { .. } => true,
                RuntimeError::RouteFailed { reason } => matches!(
                    *reason,
                    "routing abstained" | "routing has no learned representation"
                ),
                _ => false,
            };
            uncovered_miss && (delta.hits, delta.misses, delta.failovers) == (0, 1, 0)
        }
    };
    if tolerated {
        Ok(())
    } else {
        Err(format!(
            "reflex dispatch of {probe} is not accounted for ({outcome:?}, hits +{}, misses +{}, failovers +{})",
            delta.hits, delta.misses, delta.failovers
        ))
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
    // The default is bound to the implementation the registry holds: its namespace and
    // generation are the registry's, and the lifecycle holds exactly that generation live.
    let namespace = registry
        .namespace(project, &default_pod)
        .cloned()
        .ok_or("the default pod is registered without a namespace")?;
    let generation = registry
        .generation(project, &default_pod)
        .ok_or("the default pod is registered without a generation")?;
    let revision = PodRevisionAddress {
        address: PodAddress::new(project.clone(), namespace, default_pod.clone())
            .map_err(|error| format!("default pod address: {error:?}"))?,
        semantic_revision: [1; 32],
        generation,
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
            plan: DefaultPodPlan::new(Some(default_pod.clone()), REFLEX_MAX_ATTEMPTS)
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

    let (mut hits_correct, mut default_only_correct, mut dispatched) = (0u64, 0u64, 0u64);
    for probe in probes {
        let before = runtime.reflex_counters();
        let outcome = runtime
            .run_resumable_with_pods_using_router(
                RequestId::from(probe.request_id.as_str()),
                project,
                "probe",
                &AskOnce,
                registry,
                &sampler,
                &AcceptAll,
                2,
            )
            .map(|_| ());
        let after = runtime.reflex_counters();
        let delta = CounterDelta {
            hits: after.hits().saturating_sub(before.hits()),
            misses: after.misses().saturating_sub(before.misses()),
            failovers: after.failovers().saturating_sub(before.failovers()),
        };
        // Only an uncovered miss is tolerated; anything else is a harness failure.
        accounted_dispatch(&probe.request_id, &outcome, delta)?;
        dispatched += 1;
        if probe.label.as_ref() == Some(&default_pod) {
            default_only_correct += 1;
        }
        // A hit is the learned router's own dispatch; it is correct when the probe names that pod.
        if delta.hits == 1 {
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
        // Requests whose outcome the counters accounted for, not the number of probes.
        "requests": dispatched,
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

/// Whether `representation` was learned for exactly the world entry that names it.
fn representation_is_bound(
    representation: &PodRepresentation,
    project: &ProjectId,
    namespace: &NamespaceId,
    pod: &PodId,
    generation: ptr_types::Generation,
) -> bool {
    representation.project() == project
        && representation.namespace() == namespace
        && representation.pod() == pod
        && representation.generation() == generation
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
            // A learned router with no activated representation at all (every gate failed)
            // declines like one with partial coverage; the seed is then INVALID through the
            // coverage rule `representations_refused <= 0`, not a harness failure.
            RouteError::Abstained { .. } | RouteError::NoRepresentation { .. } => {
                Ok(Outcome::Abstained)
            }
            other => Err(other.to_string()),
        },
    }
}

/// `dragonfly-addressing --preflight`: check that a confirmatory run can start here (uv, the
/// pinned interpreter, the locked packages, the pinned encoder) without reserving or spending a
/// seed. Exit status 0 when it can, 2 when it cannot.
pub fn preflight() {
    let outcome = Scratch::create(0).and_then(|scratch| {
        let mut command = preflight_command(&scratch.0.join("venv"));
        let program = command.get_program().to_string_lossy().into_owned();
        let status = command
            .status()
            .map_err(|error| format!("cannot start the training step ({program}): {error}"))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!(
                "the preflight failed ({status}); a confirmatory run needs uv, Python \
                 {PYTHON_VERSION}, the packages training/uv.lock names and the encoder the \
                 preregistration pins"
            ))
        }
    });
    if let Err(message) = outcome {
        eprintln!("ptr-bench dragonfly-addressing --preflight: {message}");
        std::process::exit(2);
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
    fn the_preflight_runs_the_training_step_in_the_confirmatory_environment_without_a_seed() {
        let command = preflight_command(Path::new("/scratch/venv"));
        assert_eq!(command.get_program(), "uv");
        let args = words(&command);
        assert_eq!(
            &args[..5],
            ["run", "--locked", "--python", PYTHON_VERSION, "--project"]
        );
        assert_eq!(args.last().map(String::as_str), Some("--preflight"));
        for absent in ["--seed", "--out", "--smoke"] {
            assert!(!args.contains(&absent.to_owned()), "{absent} in {args:?}");
        }
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
    fn the_confirmatory_interpreter_is_pinned_exactly() {
        let command = training_command(2017, Path::new("/o"), false, Path::new("/v"));
        let args = words(&command);
        assert!(args
            .windows(2)
            .any(|pair| pair == ["--python", PYTHON_VERSION]));
        assert!(
            PYTHON_VERSION
                .split('.')
                .all(|part| part.parse::<u32>().is_ok())
                && PYTHON_VERSION.split('.').count() == 3
        );
        // A smoke run takes whatever interpreter is there and only records it.
        let smoke = training_command(1001, Path::new("/o"), true, Path::new("/v"));
        assert!(!words(&smoke).contains(&"--python".to_owned()));
    }

    /// A world in which every representation failed its activation gate.
    fn world_with_failed_gates(dir: &Path) {
        write_world(dir, false);
    }

    /// A two-pod world (`alpha`, `beta`, namespace `supply`, generation 1) whose representations
    /// passed (`activated`) or failed their activation gate.
    fn write_world(dir: &Path, activated: bool) {
        let encoder = EncoderRef {
            id: "test-encoder".into(),
            revision: "r1".into(),
        };
        let gate = ptr_router::GateReport {
            alias_threshold_permille: 800,
            // Above the alias threshold only when the gate passed.
            alias_min_permille: if activated { 850 } else { 700 },
            negative_threshold_permille: 900,
            negative_reject_permille: 960,
            alias_count: 4,
            negative_count: 20,
        };
        let mut pods = Vec::new();
        for (index, name) in ["alpha", "beta"].into_iter().enumerate() {
            let mut z = vec![0.0f32; 3];
            z[index] = 1.0;
            let representation = PodRepresentation::new(PodRepresentationParts {
                project: ProjectId::from("demo"),
                namespace: ptr_types::NamespaceId::from("supply"),
                pod: PodId::from(name),
                identity: ptr_types::PodIdentity::from(format!("pod:{name}").as_str()),
                cluster_key: ptr_types::ClusterKey([1; 32]),
                encoder: encoder.clone(),
                generation: ptr_types::Generation(1),
                z,
                bias: 10.0,
                weights: ptr_router::ScoreWeights::default(),
                gate,
                activated,
            })
            .unwrap();
            let (manifest, weights) = representation.to_artifact();
            let mut manifest_json: Value = serde_json::from_str(&manifest).unwrap();
            manifest_json["weights_file"] = json!(format!("{name}.bin"));
            std::fs::write(dir.join(format!("{name}.bin")), weights).unwrap();
            std::fs::write(dir.join(format!("{name}.json")), manifest_json.to_string()).unwrap();
            pods.push(json!({
                "pod_id": name,
                "generation": 1,
                "manifest": format!("{name}.json"),
                "status": if activated { "activated" } else { "not-activated" },
                "description_vector": [1.0, 0.0, 0.0],
            }));
        }
        let probes: Vec<Value> = (0..4)
            .map(|index| {
                json!({
                    "text": "question",
                    "label": if index < 2 { json!("alpha") } else { Value::Null },
                    "kind": "held-out",
                    "vector": [1.0, 0.0, 0.0],
                })
            })
            .collect();
        let world = json!({
            "seed": 1,
            "encoder": {"id": "test-encoder", "revision": "r1"},
            "dims": 3,
            "project": "demo",
            "namespace": "supply",
            "smoke": true,
            "evidence": "none (stand-in encoder)",
            "environment": {"python": "3.13.16"},
            "pods": pods,
            "probes": probes,
        });
        std::fs::write(dir.join("world.json"), world.to_string()).unwrap();
    }

    #[test]
    fn a_world_where_every_gate_fails_still_yields_a_result_that_the_coverage_rule_invalidates() {
        let scratch = Scratch::create(u64::MAX - 1).unwrap();
        world_with_failed_gates(&scratch.0);
        let result = evaluate_world(1, true, &scratch.0).unwrap();
        assert_eq!(result["representations_loaded"], 0);
        assert_eq!(result["representations_refused"], 2);
        // The learned arm declines every probe instead of failing the harness.
        let learned = &result["arms"]["learned"];
        assert_eq!(learned["abstained"], 4);
        assert_eq!(learned["correct"], 0);
        assert_eq!(learned["false_routes"], 0);
        // criteria.toml marks such a seed INVALID: its coverage list holds this rule.
        let criteria = include_str!(
            "../../../../experiments/runtime/R015-dragonfly-pod-addressing/criteria.toml"
        );
        assert!(criteria.contains("\"representations_refused <= 0\""));
        assert!(result["representations_refused"].as_i64().unwrap() > 0);
    }

    fn edit_world(dir: &Path, edit: impl FnOnce(&mut Value)) {
        let path = dir.join("world.json");
        let mut world = read_json(&path).unwrap();
        edit(&mut world);
        std::fs::write(path, world.to_string()).unwrap();
    }

    #[test]
    fn an_artifact_learned_for_another_world_entry_counts_as_refused() {
        let scratch = Scratch::create(u64::MAX - 2).unwrap();
        write_world(&scratch.0, true);
        let result = evaluate_world(1, true, &scratch.0).unwrap();
        assert_eq!(result["representations_loaded"], 2);
        assert_eq!(result["representations_refused"], 0);

        // The world registers beta at generation 2, its artifact was learned for generation 1.
        edit_world(&scratch.0, |world| {
            world["pods"][1]["generation"] = json!(2)
        });
        let result = evaluate_world(1, true, &scratch.0).unwrap();
        assert_eq!(result["representations_loaded"], 1);
        assert_eq!(result["representations_refused"], 1);

        // Both artifacts were learned for namespace supply, the world names another one.
        write_world(&scratch.0, true);
        edit_world(&scratch.0, |world| world["namespace"] = json!("other"));
        let result = evaluate_world(1, true, &scratch.0).unwrap();
        assert_eq!(result["representations_loaded"], 0);
        assert_eq!(result["representations_refused"], 2);

        // An artifact under another pod's entry.
        write_world(&scratch.0, true);
        edit_world(&scratch.0, |world| {
            world["pods"][1]["manifest"] = json!("alpha.json");
        });
        let result = evaluate_world(1, true, &scratch.0).unwrap();
        assert_eq!(result["representations_refused"], 1);
    }

    #[test]
    fn a_label_that_names_no_registered_pod_invalidates_the_seed() {
        let scratch = Scratch::create(u64::MAX - 3).unwrap();
        write_world(&scratch.0, true);
        let result = evaluate_world(1, true, &scratch.0).unwrap();
        assert_eq!(result["unregistered_labels"], 0);

        edit_world(&scratch.0, |world| {
            world["probes"][0]["label"] = json!("gamma")
        });
        let result = evaluate_world(1, true, &scratch.0).unwrap();
        assert_eq!(result["unregistered_labels"], 1);
        // The oracle routes to the unknown label and counts it as correct, so only the coverage
        // rule keeps such a world from being judged.
        assert_eq!(result["arms"]["oracle"]["correct"], 2);
        let criteria = include_str!(
            "../../../../experiments/runtime/R015-dragonfly-pod-addressing/criteria.toml"
        );
        assert!(criteria.contains("\"unregistered_labels <= 0\""));
    }

    #[test]
    fn a_world_where_every_gate_fails_runs_the_reflex_pass_and_counts_every_decline_as_a_miss() {
        // No representation is activated, so the learned router has nothing to score on any probe:
        // every dispatch is a NoRepresentation miss caught by the default pod. The pass must
        // still produce its block (no harness failure) and count what it accounted for.
        let scratch = Scratch::create(u64::MAX - 2).unwrap();
        world_with_failed_gates(&scratch.0);
        let result = evaluate_world(1, true, &scratch.0).unwrap();
        let reflex = &result["reflex"];
        assert_eq!(reflex["requests"], result["probes"]);
        assert_eq!(reflex["dispatch_hits"], 0);
        assert_eq!(reflex["misses"], 4);
        assert_eq!(reflex["failovers"], 4);
        assert_eq!(reflex["miss_reasons"]["no_representation"], 4);
        assert_eq!(reflex["inconsistent"], 0);
        // The seed is INVALID for the routing claim (refused representations) and for the
        // dispatch claim (no hit), whatever the failover claim says.
        let rules = criteria_rules();
        assert!(!rules_hold(&rules.routing_coverage, &result));
        assert!(!rules_hold(&rules.dispatch_coverage, &result));
    }

    fn delta(hits: u64, misses: u64, failovers: u64) -> CounterDelta {
        CounterDelta {
            hits,
            misses,
            failovers,
        }
    }

    #[test]
    fn the_reflex_pass_tolerates_only_accounted_outcomes() {
        let unavailable = RuntimeError::PodUnavailable {
            capability: "answer".into(),
            input_type: "Text".into(),
        };
        let declined = RuntimeError::RouteFailed {
            reason: "routing abstained",
        };
        // Served: one hit, or one miss caught by the default.
        assert!(accounted_dispatch("p", &Ok(()), delta(1, 0, 0)).is_ok());
        assert!(accounted_dispatch("p", &Ok(()), delta(0, 1, 1)).is_ok());
        // A miss nothing covered: tolerated, with exactly one miss counted.
        assert!(accounted_dispatch("p", &Err(unavailable.clone()), delta(0, 1, 0)).is_ok());
        assert!(accounted_dispatch("p", &Err(declined), delta(0, 1, 0)).is_ok());
        // Served without any counter change, or with two, is not accounted for.
        assert!(accounted_dispatch("p", &Ok(()), delta(0, 0, 0)).is_err());
        assert!(accounted_dispatch("p", &Ok(()), delta(1, 1, 1)).is_err());
        // An uncovered miss that was not counted is not accounted for either.
        assert!(accounted_dispatch("p", &Err(unavailable), delta(0, 0, 0)).is_err());
    }

    #[test]
    fn the_reflex_pass_fails_the_harness_on_an_unexpected_runtime_error() {
        for error in [
            RuntimeError::ModelResumeLimit { max_rounds: 2 },
            RuntimeError::Pod("pod failed".into()),
            RuntimeError::RouteFailed {
                reason: "routing features invalid",
            },
            RuntimeError::InvalidConfig("x".into()),
        ] {
            // Even when a miss was counted, an error that is no classified decline is a failure.
            let outcome = accounted_dispatch("probe-3", &Err(error), delta(0, 1, 0));
            let message = outcome.expect_err("an unexpected error must stop the harness");
            assert!(message.contains("probe-3"), "{message}");
        }
    }

    /// The coverage rules of `criteria.toml`: the routing claim's (top level) and the dispatch
    /// and failover claims'.
    struct CriteriaRules {
        routing_coverage: Vec<String>,
        failover_coverage: Vec<String>,
        dispatch_coverage: Vec<String>,
    }

    fn criteria_rules() -> CriteriaRules {
        let criteria: toml::Table = include_str!(
            "../../../../experiments/runtime/R015-dragonfly-pod-addressing/criteria.toml"
        )
        .parse()
        .expect("criteria.toml parses");
        let list = |table: &toml::Table| -> Vec<String> {
            table["coverage"]
                .as_array()
                .expect("coverage is a list")
                .iter()
                .map(|rule| rule.as_str().expect("a rule is a string").to_owned())
                .collect()
        };
        CriteriaRules {
            routing_coverage: list(&criteria),
            failover_coverage: list(
                criteria["claims"]["reflex_failover_coverage"]
                    .as_table()
                    .unwrap(),
            ),
            dispatch_coverage: list(
                criteria["claims"]["reflex_dispatch_hits"]
                    .as_table()
                    .unwrap(),
            ),
        }
    }

    /// The rule grammar of `criteria.toml`: `<left> <op> <right>`, a token an integer literal or a
    /// path of object keys; a reference that is missing, null or not an integer makes the rule
    /// not hold (the seed is INVALID).
    fn rule_holds(rule: &str, result: &Value) -> bool {
        let tokens: Vec<&str> = rule.split(' ').collect();
        assert_eq!(tokens.len(), 3, "{rule}");
        let value = |token: &str| -> Option<i64> {
            token.parse::<i64>().ok().or_else(|| {
                token
                    .split('.')
                    .try_fold(result, |node, key| node.get(key))
                    .and_then(Value::as_i64)
            })
        };
        let (Some(left), Some(right)) = (value(tokens[0]), value(tokens[2])) else {
            return false;
        };
        match tokens[1] {
            ">=" => left >= right,
            "<=" => left <= right,
            "==" => left == right,
            other => panic!("unknown operator {other}"),
        }
    }

    fn rules_hold(rules: &[String], result: &Value) -> bool {
        rules.iter().all(|rule| rule_holds(rule, result))
    }

    #[test]
    fn the_coverage_rules_fix_the_design_counts_exactly() {
        let rules = criteria_rules().routing_coverage;
        for exact in ["probes == 43", "labelled_probes == 36", "pods == 6"] {
            assert!(rules.iter().any(|rule| rule == exact), "{exact} missing");
        }
        // No count of the design may be a lower bound: more probes or pods than designed is a
        // different experiment, not extra evidence.
        for key in ["probes", "labelled_probes", "pods"] {
            assert!(
                !rules
                    .iter()
                    .any(|rule| rule.starts_with(&format!("{key} >="))),
                "{key} is only bounded from below"
            );
        }
        let design = json!({"probes": 43, "labelled_probes": 36, "pods": 6});
        for exact in ["probes == 43", "labelled_probes == 36", "pods == 6"] {
            assert!(rule_holds(exact, &design));
        }
        // One probe, one labelled probe or one pod more fails the rule.
        for (key, rule) in [
            ("probes", "probes == 43"),
            ("labelled_probes", "labelled_probes == 36"),
            ("pods", "pods == 6"),
        ] {
            let mut more = design.clone();
            more[key] = json!(more[key].as_i64().unwrap() + 1);
            assert!(!rule_holds(rule, &more), "{rule} holds for {more}");
        }
        // A missing reference is not a pass.
        assert!(!rule_holds("probes == 43", &json!({})));
    }

    #[test]
    fn the_failover_claim_is_invalid_without_a_miss_and_the_dispatch_claim_without_a_hit() {
        let rules = criteria_rules();
        let reflex = |misses: i64, hits: i64| {
            json!({"probes": 43, "reflex": {
                "requests": 43, "misses": misses, "dispatch_hits": hits, "inconsistent": 0}})
        };
        assert!(rules_hold(&rules.failover_coverage, &reflex(5, 38)));
        assert!(!rules_hold(&rules.failover_coverage, &reflex(0, 43)));
        assert!(rules_hold(&rules.dispatch_coverage, &reflex(5, 38)));
        assert!(!rules_hold(&rules.dispatch_coverage, &reflex(43, 0)));
        // A run that accounted for fewer requests than probes is not a result.
        let mut short = reflex(5, 38);
        short["reflex"]["requests"] = json!(42);
        assert!(!rules_hold(&rules.failover_coverage, &short));
        assert!(!rules_hold(&rules.dispatch_coverage, &short));
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
            line("python_version", format!("\"{PYTHON_VERSION}\"")),
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
