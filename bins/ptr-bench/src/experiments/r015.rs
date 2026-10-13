//! Experiment R015, Dragonfly pod addressing, as a harness: `ptr-bench dragonfly-addressing
//! <seed>` routes the held-out questions of one seed's synthetic world with three routers and
//! prints exactly one JSON object on stdout.
//!
//! The world is the output of the training step (`python3 -m ptr_training.dragonfly.experiment
//! --seed <seed>`, default directory `target/dragonfly-addressing/<seed>`): one exported address
//! per pod, the pods' description vectors and the probe questions with their labels. The arms:
//!
//! - `learned`: [`LearnedPodRouter`] over the activated addresses, through `route_checked`. It may
//!   abstain, and an abstention is not a routed answer.
//! - `cosine_only`: the pod whose description vector is closest to the query, always answering.
//! - `constant_score`: the registry's default [`PodRouter`] (every candidate scores 1.0).
//!
//! Metrics per arm, over all probes: `routing_accuracy` (labelled probes routed to their pod, of
//! all labelled probes), `abstain_rate` (probes the arm declined) and `false_route_rate` (probes
//! routed to a wrong pod, or to any pod when the probe names none). The harness only measures; it
//! judges nothing, and the experiment stays `planned` until its preregistration is frozen.

use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_router::{
    route_checked, DecisionNonce, EncoderRef, LearnedPodRouter, LearnedRouterConfig,
    PodRepresentation, PodRouter, RepresentationStore, RouteError, RouteFeatures, RouteRequest,
    Router, SamplingKey,
};
use ptr_types::{CapabilityId, Effect, PodId, ProjectId, RequestId, TypeId};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

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

#[derive(Default)]
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

    fn metrics(&self) -> Value {
        json!({
            "routing_accuracy": Self::rate(self.correct, self.labelled),
            "abstain_rate": Self::rate(self.abstained, self.probes),
            "false_route_rate": Self::rate(self.false_routes, self.probes),
            "correct": self.correct,
            "abstained": self.abstained,
            "false_routes": self.false_routes,
        })
    }
}

fn world_dir(seed: u64) -> PathBuf {
    Path::new("target/dragonfly-addressing").join(seed.to_string())
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        format!(
            "cannot read {}: {error}; run the training step first: \
             python3 -m ptr_training.dragonfly.experiment --seed <seed>",
            path.display()
        )
    })?;
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

fn run_seed(seed: u64) -> Result<Value, String> {
    let dir = world_dir(seed);
    let world = read_json(&dir.join("world.json"))?;
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

    let project = ProjectId::from("demo");
    let mut registry = PodRegistry::default();
    let mut store = RepresentationStore::new(encoder);
    let mut descriptions: Vec<(PodId, Vec<f32>)> = Vec::new();
    let (mut loaded, mut refused) = (0u64, 0u64);
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
        let representation = PodRepresentation::from_artifact(&manifest, &weights)
            .map_err(|error| format!("{}: {error}", manifest_path.display()))?;
        match store.insert(representation) {
            Ok(_) => loaded += 1,
            // A representation that failed its gate is a result, not an error.
            Err(RouteError::InactiveRepresentation { .. }) => refused += 1,
            Err(error) => return Err(error.to_string()),
        }
    }
    let learned = LearnedPodRouter::new(store, LearnedRouterConfig::default())
        .map_err(|error| error.to_string())?;

    let key = SamplingKey::from_bytes([0x15; 32]);
    let capability = CapabilityId::from("answer");
    let input_type = TypeId::from("Text");
    let (mut learned_tally, mut cosine_tally, mut constant_tally) =
        (Tally::default(), Tally::default(), Tally::default());
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
            request_digest: [index as u8; 32],
            features: &features,
        };
        let nonce = DecisionNonce::from_bytes([index as u8; 16]);

        learned_tally.record(label.as_ref(), &route(&learned, &request, &key, &nonce)?);
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
    }

    let accuracy = |tally: &Tally| Tally::rate(tally.correct, tally.labelled);
    let false_rate = |tally: &Tally| Tally::rate(tally.false_routes, tally.probes);
    Ok(json!({
        "experiment": "R015",
        "seed": seed,
        "encoder": world["encoder"],
        "pods": world["pods"].as_array().map_or(0, Vec::len),
        "representations_loaded": loaded,
        "representations_refused": refused,
        "probes": learned_tally.probes,
        "labelled_probes": learned_tally.labelled,
        "arms": {
            "learned": learned_tally.metrics(),
            "cosine_only": cosine_tally.metrics(),
            "constant_score": constant_tally.metrics(),
        },
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
pub fn run(seed: u64) {
    match run_seed(seed) {
        Ok(result) => println!("{result}"),
        Err(message) => {
            eprintln!("ptr-bench dragonfly-addressing: {message}");
            std::process::exit(2);
        }
    }
}
