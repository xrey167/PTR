//! The Python exporter (`training/src/ptr_training/dragonfly`) and the Rust loader agree.
//!
//! `tests/fixtures` holds artifacts the exporter wrote (regenerate with
//! `PYTHONPATH=training/src python3 -m ptr_training.dragonfly.fixture crates/ptr-router/tests/fixtures`)
//! and `dragonfly-expected.json`, which lists what the Python side computed: pod identities and
//! cluster keys, gate status, and the score of every probe query for every pod.
mod common;

use common::{key, nonce, registry, request, route};
use ptr_router::{
    EncoderRef, LearnedPodRouter, LearnedRouterConfig, PodRepresentation, RepresentationError,
    RepresentationStore, RouteError, RouteFeatures,
};
use ptr_types::{ClusterKey, PodId, PodIdentityDescriptor, ProjectId};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn expected() -> Value {
    let text = fs::read_to_string(fixtures().join("dragonfly-expected.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn load(manifest: &str) -> Result<PodRepresentation, RepresentationError> {
    let manifest_text = fs::read_to_string(fixtures().join(manifest)).unwrap();
    let value: Value = serde_json::from_str(&manifest_text).unwrap();
    let weights = fs::read(fixtures().join(value["weights_file"].as_str().unwrap())).unwrap();
    PodRepresentation::from_artifact(&manifest_text, &weights)
}

fn pod(entry: &Value) -> (&str, &str) {
    (
        entry["pod_id"].as_str().unwrap(),
        entry["manifest"].as_str().unwrap(),
    )
}

fn descriptor(value: &Value) -> PodIdentityDescriptor {
    PodIdentityDescriptor {
        schema_version: value["schema_version"].as_u64().unwrap() as u32,
        domain: value["domain"].as_str().unwrap().to_owned(),
        kind: value["kind"].as_str().unwrap().to_owned(),
        predicate: value["predicate"].as_str().map(str::to_owned),
        semantic_role: value["semantic_role"].as_str().unwrap().to_owned(),
        aliases: value["aliases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|alias| alias.as_str().unwrap().to_owned())
            .collect(),
    }
}

fn activated_store() -> RepresentationStore {
    let expected = expected();
    let encoder = EncoderRef {
        id: expected["encoder"]["id"].as_str().unwrap().to_owned(),
        revision: expected["encoder"]["revision"].as_str().unwrap().to_owned(),
    };
    let mut store = RepresentationStore::new(encoder);
    for entry in expected["pods"].as_array().unwrap() {
        if entry["status"] == "activated" {
            store.insert(load(pod(entry).1).unwrap()).unwrap();
        }
    }
    store
}

#[test]
fn identities_and_cluster_keys_match_the_python_computation() {
    for vector in expected()["identity_vectors"].as_array().unwrap() {
        let descriptor = descriptor(&vector["descriptor"]);
        descriptor.validate().unwrap();
        assert_eq!(
            descriptor.identity().0,
            vector["identity"].as_str().unwrap()
        );
        assert_eq!(
            descriptor.cluster_key().to_hex(),
            vector["cluster_key"].as_str().unwrap()
        );
    }
}

#[test]
fn every_exported_pod_loads_with_the_fields_the_exporter_wrote() {
    let expected = expected();
    for entry in expected["pods"].as_array().unwrap() {
        let (pod_id, manifest) = pod(entry);
        let representation = load(manifest).unwrap();
        assert_eq!(representation.pod(), &PodId::from(pod_id));
        assert_eq!(
            representation.identity().0,
            entry["pod_identity"].as_str().unwrap()
        );
        assert_eq!(
            representation.cluster_key(),
            ClusterKey::from_hex(entry["cluster_key"].as_str().unwrap()).unwrap()
        );
        assert_eq!(
            representation.generation().0,
            entry["generation"].as_u64().unwrap()
        );
        assert_eq!(
            representation.activated(),
            entry["status"] == "activated",
            "{pod_id}"
        );
        assert_eq!(
            representation.dims() as u64,
            expected["dims"].as_u64().unwrap()
        );
    }
}

#[test]
fn the_weights_survive_the_round_trip_bit_for_bit() {
    for entry in expected()["pods"].as_array().unwrap() {
        let (_, manifest) = pod(entry);
        let representation = load(manifest).unwrap();
        let on_disk = fs::read(fixtures().join(entry["weights"].as_str().unwrap())).unwrap();
        assert_eq!(representation.weights_bytes(), on_disk);
    }
}

#[test]
fn scores_match_the_python_scores_for_every_probe() {
    let expected = expected();
    let router = LearnedPodRouter::new(activated_store(), LearnedRouterConfig::default()).unwrap();
    for probe in expected["probes"].as_array().unwrap() {
        let query: Vec<f32> = probe["vector"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect();
        for (pod_id, score) in probe["scores"].as_object().unwrap() {
            let python = score.as_f64().unwrap();
            match router.score(&PodId::from(pod_id.as_str()), &query).unwrap() {
                Some(rust) => assert!(
                    (rust - python).abs() < 1e-9,
                    "{} / {pod_id}: rust {rust} python {python}",
                    probe["text"]
                ),
                // The not-activated pod is not loaded, so it has no score to compare.
                None => assert_eq!(pod_id, "supplier-kowalski"),
            }
        }
    }
}

#[test]
fn a_not_activated_artifact_loads_but_is_never_admitted_to_the_store() {
    let expected = expected();
    let entry = expected["pods"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["status"] == "not-activated")
        .unwrap();
    let representation = load(pod(entry).1).unwrap();
    assert!(!representation.activated());
    let encoder = EncoderRef {
        id: expected["encoder"]["id"].as_str().unwrap().to_owned(),
        revision: expected["encoder"]["revision"].as_str().unwrap().to_owned(),
    };
    let mut store = RepresentationStore::new(encoder);
    assert!(matches!(
        store.insert(representation),
        Err(RouteError::InactiveRepresentation { .. })
    ));
    assert!(store.is_empty());
}

#[test]
fn the_exported_representations_route_the_probe_questions_to_their_pods() {
    let project = ProjectId::from("demo");
    let registry = registry(&project, &["supplier-mueller", "supplier-schmidt"]);
    let router = LearnedPodRouter::new(activated_store(), LearnedRouterConfig::default()).unwrap();
    let capability = "read".into();
    let input_type = "Text".into();
    let request_id = "request".into();
    let expected = expected();
    let probes = expected["probes"].as_array().unwrap();
    for (index, expected_pod) in [(0, "supplier-mueller"), (1, "supplier-schmidt")] {
        let query: Vec<f32> = probes[index]["vector"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect();
        let features = RouteFeatures::new().with_vector(query).unwrap();
        let request = request(
            &registry,
            &project,
            &capability,
            &input_type,
            &request_id,
            &features,
        );
        let choice = route(&router, &request).unwrap().unwrap();
        assert_eq!(choice.route().pod_id, PodId::from(expected_pod));
    }
    // Keep the shared helpers used even when a case above is edited.
    let _ = (key(), nonce());
}

fn tampered(edit: impl Fn(&mut Value)) -> Result<PodRepresentation, RepresentationError> {
    let expected = expected();
    let (_, manifest) = pod(&expected["pods"][0]);
    let mut value: Value =
        serde_json::from_str(&fs::read_to_string(fixtures().join(manifest)).unwrap()).unwrap();
    let weights = fs::read(fixtures().join(value["weights_file"].as_str().unwrap())).unwrap();
    edit(&mut value);
    PodRepresentation::from_artifact(&value.to_string(), &weights)
}

#[test]
fn any_changed_manifest_field_is_refused_by_the_digest() {
    assert_eq!(
        tampered(|manifest| manifest["generation"] = Value::from(99)),
        Err(RepresentationError::DigestMismatch)
    );
    assert_eq!(
        tampered(|manifest| manifest["pod_id"] = Value::from("supplier-other")),
        Err(RepresentationError::DigestMismatch)
    );
    assert_eq!(
        tampered(|manifest| manifest["encoder"]["revision"] = Value::from("2")),
        Err(RepresentationError::DigestMismatch)
    );
    assert_eq!(
        tampered(|manifest| manifest["gate"]["alias_count"] = Value::from(7)),
        Err(RepresentationError::DigestMismatch)
    );
}

#[test]
fn a_flipped_gate_status_cannot_activate_a_representation() {
    // The not-activated artifact says so; rewriting the status contradicts its measured gate.
    let expected = expected();
    let entry = expected["pods"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["status"] == "not-activated")
        .unwrap();
    let manifest = pod(entry).1;
    let mut value: Value =
        serde_json::from_str(&fs::read_to_string(fixtures().join(manifest)).unwrap()).unwrap();
    let weights = fs::read(fixtures().join(value["weights_file"].as_str().unwrap())).unwrap();
    value["gate"]["status"] = Value::from("activated");
    assert_eq!(
        PodRepresentation::from_artifact(&value.to_string(), &weights),
        Err(RepresentationError::GateInconsistent)
    );
}

#[test]
fn changed_or_truncated_weights_are_refused() {
    let expected = expected();
    let (_, manifest) = pod(&expected["pods"][0]);
    let manifest_text = fs::read_to_string(fixtures().join(manifest)).unwrap();
    let value: Value = serde_json::from_str(&manifest_text).unwrap();
    let mut weights = fs::read(fixtures().join(value["weights_file"].as_str().unwrap())).unwrap();
    weights[0] ^= 0x01;
    assert_eq!(
        PodRepresentation::from_artifact(&manifest_text, &weights),
        Err(RepresentationError::WeightsDigestMismatch)
    );
    let truncated = &weights[..weights.len() - 4];
    assert!(matches!(
        PodRepresentation::from_artifact(&manifest_text, truncated),
        Err(RepresentationError::WeightsLength { .. })
    ));
}

#[test]
fn malformed_manifests_are_refused_with_the_failing_field() {
    assert_eq!(
        tampered(|manifest| manifest["schema"] = Value::from(2)),
        Err(RepresentationError::UnsupportedSchema {
            expected: 1,
            actual: 2
        })
    );
    assert_eq!(
        tampered(|manifest| manifest["kind"] = Value::from("other")),
        Err(RepresentationError::Malformed { field: "kind" })
    );
    assert_eq!(
        tampered(|manifest| {
            manifest.as_object_mut().unwrap().remove("pod_identity");
        }),
        Err(RepresentationError::Malformed {
            field: "pod_identity"
        })
    );
    assert_eq!(
        tampered(|manifest| manifest["scoring"]["cosine_weight"] = Value::from(12.3)),
        Err(RepresentationError::Malformed {
            field: "scoring.cosine_weight"
        })
    );
    assert_eq!(
        tampered(|manifest| manifest["cluster_key"] = Value::from("nothex")),
        Err(RepresentationError::InvalidClusterKey)
    );
    assert_eq!(
        PodRepresentation::from_artifact("not json", &[]),
        Err(RepresentationError::Malformed { field: "manifest" })
    );
}
