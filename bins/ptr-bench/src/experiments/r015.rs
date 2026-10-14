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

use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_router::{
    route_checked, DecisionNonce, EncoderRef, LearnedPodRouter, LearnedRouterConfig,
    PodRepresentation, PodRepresentationParts, PodRouter, RepresentationStore, RouteError,
    RouteFeatures, RouteRequest, Router, SamplingKey,
};
use ptr_types::{CapabilityId, Effect, PodId, ProjectId, RequestId, TypeId};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

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
        if denominator == 0 {
            0
        } else {
            (numerator * 1000 / denominator) as i64
        }
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
    if probes == 0 {
        0
    } else {
        ((left as i128 - right as i128) * 1000).div_euclid(probes as i128) as i64
    }
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

/// Runs the training step for `seed` into `out`; a non-zero exit is a harness failure.
fn generate_world(seed: u64, out: &Path, smoke: bool) -> Result<(), String> {
    let training_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../training/src");
    let mut command = Command::new("python3");
    command
        .args(["-B", "-m", "ptr_training.dragonfly.experiment", "--seed"])
        .arg(seed.to_string())
        .arg("--out")
        .arg(out)
        .env("PYTHONPATH", &training_source)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        // The harness prints exactly one JSON object on stdout; the training step's own report is
        // not part of it (its world is read from the scratch directory).
        .stdout(std::process::Stdio::null());
    if smoke {
        command.arg("--smoke");
    }
    let status = command
        .status()
        .map_err(|error| format!("cannot start the training step (python3): {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "the training step failed ({status}); a confirmatory run needs the encoder the \
             preregistration pins"
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
    generate_world(seed, dir, smoke)?;
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
