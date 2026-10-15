//! Admission of learned pod representations (Dragonfly) as neural state.
//!
//! A [`PodRepresentation`] is opaque model state like any other: it is only usable while the
//! committed facts it was produced under still hold. It is bound through the same
//! [`StateDeclaration`] / [`PtrRuntime::bind_state`] path as KV caches and checkpoints, under one
//! lifecycle target per pod ([`pod_lifecycle_target`]) at that pod's exact generation. Revoking
//! that pod generation therefore denies exactly that pod's representation
//! ([`Denial::RevokedGeneration`]); the representations of other pods carry their own targets and
//! stay admitted (Invariant 17). A new pod generation denies the old representation
//! ([`Denial::StaleGeneration`]) until a representation trained for it is bound.
//!
//! The binding does not cover its payload, so the digest of the representation is recorded in the
//! binding's provenance and compared on every admission: a payload swapped under a valid binding
//! is refused.

use crate::neural::{Denial, NeuralError, NeuralState, NeuralStateCache, StateDeclaration};
use crate::pod_head::pod_lifecycle_target;
use crate::{PtrRuntime, RuntimeError};
use ptr_router::{
    AdmittedRepresentation, EncoderRef, PodRepresentation, RepresentationError,
    RepresentationStore, RouteError,
};
use ptr_types::{CodebookVersion, EvidenceId, PodRevisionAddress, ProvenanceRef};

use std::collections::BTreeSet;

const PROVENANCE_PREFIX: &str = "dragonfly:";

/// What binds a representation to the pod revision it was trained for.
pub struct BindPodRepresentation<'a> {
    pub representation: &'a PodRepresentation,
    /// The pod revision the representation is for. Its generation must be the live generation of
    /// the pod's lifecycle target and the representation's own.
    pub pod: &'a PodRevisionAddress,
    /// The codebook version the neural-state binding is recorded under. A representation holds no
    /// codes; the binding needs a known version.
    pub codebook: CodebookVersion,
}

/// Why a representation could not be bound.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PodRepresentationError {
    /// The representation belongs to another project than `pod`.
    ProjectMismatch {
        representation: String,
        address: String,
    },
    /// The representation is for the same pod id in another lifecycle namespace than `pod`.
    NamespaceMismatch {
        representation: String,
        address: String,
    },
    /// The representation was trained for another pod than `pod`.
    PodMismatch {
        representation: String,
        address: String,
    },
    /// The representation was trained for another generation than `pod`.
    GenerationMismatch { representation: u64, address: u64 },
}

/// Why a stored representation is not admitted now.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RepresentationDenial {
    /// The neural-state admission refused the binding (revoked or stale generation, fenced, ...).
    State(Denial),
    /// The payload does not decode to a valid representation.
    Payload(RepresentationError),
    /// The payload is not the representation the binding was made for.
    BindingMismatch,
}

/// The representations that passed admission, as a router store, and the ones that did not.
#[derive(Debug)]
pub struct AdmittedRepresentations {
    pub store: RepresentationStore,
    /// Cache key and reason, in key order.
    pub refused: Vec<(String, RepresentationRefusal)>,
}

#[derive(Debug)]
pub enum RepresentationRefusal {
    /// Not admitted by the runtime.
    Admission(RepresentationDenial),
    /// Admitted, but the router store refused it (not activated, other encoder or dimension, not
    /// newer than the loaded one).
    Store(RouteError),
}

impl PtrRuntime {
    /// Binds `request.representation` to the committed facts of its pod, producing the opaque
    /// state to keep in a [`NeuralStateCache`].
    ///
    /// # Errors
    /// [`PodRepresentationError`] when the representation is not the one of `request.pod`, and the
    /// binding errors of [`PtrRuntime::bind_state`] (an unknown pod target, or a pod generation
    /// that is not the representation's).
    pub fn bind_pod_representation(
        &self,
        request: BindPodRepresentation<'_>,
    ) -> Result<NeuralState, RuntimeError> {
        let BindPodRepresentation {
            representation,
            pod,
            codebook,
        } = request;
        if representation.project() != &pod.address.project {
            return Err(RuntimeError::PodRepresentation(
                PodRepresentationError::ProjectMismatch {
                    representation: representation.project().to_string(),
                    address: pod.address.project.to_string(),
                },
            ));
        }
        if representation.namespace() != &pod.address.namespace {
            return Err(RuntimeError::PodRepresentation(
                PodRepresentationError::NamespaceMismatch {
                    representation: representation.namespace().to_string(),
                    address: pod.address.namespace.to_string(),
                },
            ));
        }
        if representation.pod() != &pod.address.pod_id {
            return Err(RuntimeError::PodRepresentation(
                PodRepresentationError::PodMismatch {
                    representation: representation.pod().to_string(),
                    address: pod.address.pod_id.to_string(),
                },
            ));
        }
        if representation.generation() != pod.generation {
            return Err(RuntimeError::PodRepresentation(
                PodRepresentationError::GenerationMismatch {
                    representation: representation.generation().0,
                    address: pod.generation.0,
                },
            ));
        }
        let target = pod_lifecycle_target(pod);
        let declaration = StateDeclaration::at(codebook)
            .under(target.clone())
            .produced_by(ProvenanceRef {
                source: EvidenceId(format!(
                    "{PROVENANCE_PREFIX}{}",
                    digest_hex(representation.digest())
                )),
                note: Some(representation.pod().to_string()),
            });
        let binding = self.bind_state(&declaration)?;
        // `bind_state` binds the live generation. A live generation that is not the
        // representation's means the representation describes another pod revision.
        match binding.generations.get(&target) {
            Some(bound) if *bound == representation.generation() => {}
            Some(bound) => {
                return Err(RuntimeError::Neural(NeuralError::Unbindable(
                    Denial::StaleGeneration {
                        target,
                        bound: representation.generation(),
                        current: *bound,
                    },
                )))
            }
            None => {
                return Err(RuntimeError::Neural(NeuralError::Unbindable(
                    Denial::UnknownTarget { target },
                )))
            }
        }
        Ok(NeuralState::new(binding, representation.to_payload()))
    }

    /// Decides now whether a stored representation may be used, and returns it if so.
    ///
    /// Re-decided on every call: a revocation that arrived after the state was bound denies it.
    pub fn admit_pod_representation(
        &self,
        state: &NeuralState,
    ) -> Result<PodRepresentation, RepresentationDenial> {
        let admitted = self.admit(state).map_err(RepresentationDenial::State)?;
        let representation = PodRepresentation::from_payload(admitted.payload())
            .map_err(RepresentationDenial::Payload)?;
        let recorded = admitted.binding().provenance.iter().any(|entry| {
            entry.source.0 == format!("{PROVENANCE_PREFIX}{}", digest_hex(representation.digest()))
        });
        let bound_to_its_pod = admitted
            .binding()
            .generations
            .iter()
            .all(|(target, generation)| {
                *generation == representation.generation()
                    && target_names(target, &representation)
                    // The lifecycle target must have been committed by the representation's own
                    // project.
                    && self.capsule_project(target) == Some(representation.project().0.as_str())
            })
            && admitted.binding().generations.len() == 1;
        if !recorded || !bound_to_its_pod {
            return Err(RepresentationDenial::BindingMismatch);
        }
        Ok(representation)
    }

    /// The representations of `cache` that are admitted now, named by pod and digest: the set a
    /// learned router may score with for one request. It is recomputed from the committed
    /// lifecycle on every call, so a revocation or a new pod generation that was committed since
    /// the state was bound removes the representation from the very next request.
    pub fn admitted_representation_set(
        &self,
        cache: &NeuralStateCache,
    ) -> BTreeSet<AdmittedRepresentation> {
        cache
            .keys()
            .filter_map(|key| self.admit_cached_representation(cache, key).ok())
            .map(|representation| AdmittedRepresentation {
                project: representation.project().clone(),
                pod: representation.pod().clone(),
                digest: *representation.digest(),
            })
            .collect()
    }

    /// Builds the router store from every cached representation that is admitted now. A refused
    /// entry is reported with its reason and costs no other pod its representation.
    pub fn admitted_representations(
        &self,
        encoder: EncoderRef,
        cache: &NeuralStateCache,
    ) -> AdmittedRepresentations {
        let mut store = RepresentationStore::new(encoder);
        let mut refused = Vec::new();
        for key in cache.keys() {
            let outcome = match cache.admit(self, key) {
                Err(denial) => Err(RepresentationRefusal::Admission(
                    RepresentationDenial::State(denial),
                )),
                Ok(_) => self
                    .admit_cached_representation(cache, key)
                    .map_err(RepresentationRefusal::Admission)
                    .and_then(|representation| {
                        store
                            .insert(representation)
                            .map(|_| ())
                            .map_err(RepresentationRefusal::Store)
                    }),
            };
            if let Err(reason) = outcome {
                refused.push((key.to_owned(), reason));
            }
        }
        AdmittedRepresentations { store, refused }
    }

    fn admit_cached_representation(
        &self,
        cache: &NeuralStateCache,
        key: &str,
    ) -> Result<PodRepresentation, RepresentationDenial> {
        let admitted = cache
            .admit(self, key)
            .map_err(RepresentationDenial::State)?;
        let state = NeuralState::new(admitted.binding().clone(), admitted.payload().to_vec());
        self.admit_pod_representation(&state)
    }
}

/// Whether the lifecycle target `pod:<project>/<namespace>/<pod>` names the representation's own
/// project, namespace and pod. A representation never moves to the lifecycle target of the same
/// pod id in another namespace, so revoking its own target denies it everywhere.
fn target_names(target: &str, representation: &PodRepresentation) -> bool {
    let Some(rest) = target.strip_prefix("pod:") else {
        return false;
    };
    let parts: Vec<&str> = rest.split('/').collect();
    parts.len() == 3
        && parts[0] == representation.project().0
        && parts[1] == representation.namespace().0
        && parts[2] == representation.pod().0
}

fn digest_hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
