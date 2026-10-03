use ptr_ledger::{CommittedEvent, LedgerEvent};
use ptr_state::projection_entries;
use ptr_types::{Generation, ProjectId};

/// What a committed event does to the lifecycle catalog, keyed exactly as the
/// runtime keys its live generations and tombstones.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LifecycleChange {
    /// `target` is now live at `generation`. `project` is set for capsule
    /// commits only.
    SetLive {
        target: String,
        generation: Generation,
        project: Option<ProjectId>,
    },
    /// `(subject, generation)` joins the revocation set. The live generation of
    /// the subject is unchanged, as in the runtime; a generation is live only
    /// when it is current *and* not in this set.
    Tombstone {
        subject: String,
        generation: Generation,
    },
    /// A semantic revision was published.
    Revision { base: u64, revision: u64 },
    /// Nothing in the lifecycle catalog changes.
    None,
}

/// The lifecycle effect of one committed event, mirroring the runtime's own
/// application of it.
pub fn lifecycle_change(committed: &CommittedEvent) -> LifecycleChange {
    match &committed.event {
        LedgerEvent::CapsuleCommitted {
            project,
            capsule,
            generation,
        } => LifecycleChange::SetLive {
            target: capsule.to_string(),
            generation: *generation,
            project: Some(project.clone()),
        },
        LedgerEvent::CapsuleSuperseded { capsule, new, .. } => LifecycleChange::SetLive {
            target: capsule.to_string(),
            generation: *new,
            project: None,
        },
        LedgerEvent::HardConstraintCommitted { key, generation } => LifecycleChange::SetLive {
            target: format!("constraint:{key}"),
            generation: *generation,
            project: None,
        },
        LedgerEvent::ProcedurePromoted { id, generation } => LifecycleChange::SetLive {
            target: format!("procedure:{id}"),
            generation: *generation,
            project: None,
        },
        LedgerEvent::Revoked {
            subject,
            generation,
        } => LifecycleChange::Tombstone {
            subject: subject.clone(),
            generation: *generation,
        },
        LedgerEvent::ProcedureRevoked { id, generation } => LifecycleChange::Tombstone {
            subject: format!("procedure:{id}"),
            generation: *generation,
        },
        LedgerEvent::SemanticDeltaCommitted {
            base_revision,
            revision,
            ..
        } => LifecycleChange::Revision {
            base: base_revision.0,
            revision: revision.0,
        },
        LedgerEvent::VerifierAttested { .. }
        | LedgerEvent::SnapshotCommitted { .. }
        | LedgerEvent::EffectAttempted { .. }
        | LedgerEvent::EffectSettled { .. }
        | LedgerEvent::EffectReconciled { .. }
        | LedgerEvent::ScopeLifecycle(_)
        | LedgerEvent::ProtectedStateCommitted { .. }
        | LedgerEvent::MeshTunnelLifecycle(_)
        | LedgerEvent::ExecutionManifestAdmitted { .. }
        | LedgerEvent::ExecutionManifestRevoked { .. }
        | LedgerEvent::PodEvidenceCommitted { .. }
        | LedgerEvent::PodOutputAdmitted { .. }
        | LedgerEvent::PodHypothesisCommitted { .. }
        | LedgerEvent::PolicyBundleActivated { .. }
        | LedgerEvent::PolicyBundleRevoked { .. }
        | LedgerEvent::SessionRevoked { .. }
        | LedgerEvent::TierBackendLifecycle { .. }
        | LedgerEvent::TierObjectCommitted { .. }
        | LedgerEvent::TierReplicaLifecycle { .. } => LifecycleChange::None,
    }
}

/// Stable topic name of an event on the projection event log.
pub fn event_topic(event: &LedgerEvent) -> &'static str {
    match event {
        LedgerEvent::SemanticDeltaCommitted { .. } => "semantic.delta_committed",
        LedgerEvent::CapsuleCommitted { .. } => "capsule.committed",
        LedgerEvent::CapsuleSuperseded { .. } => "capsule.superseded",
        LedgerEvent::Revoked { .. } => "lifecycle.revoked",
        LedgerEvent::HardConstraintCommitted { .. } => "constraint.committed",
        LedgerEvent::VerifierAttested { .. } => "verifier.attested",
        LedgerEvent::ProcedurePromoted { .. } => "procedure.promoted",
        LedgerEvent::ProcedureRevoked { .. } => "procedure.revoked",
        LedgerEvent::SnapshotCommitted { .. } => "snapshot.committed",
        LedgerEvent::EffectAttempted { .. } => "effect.attempted",
        LedgerEvent::EffectSettled { .. } => "effect.settled",
        LedgerEvent::EffectReconciled { .. } => "effect.reconciled",
        LedgerEvent::ScopeLifecycle(_) => "scope.lifecycle",
        LedgerEvent::ProtectedStateCommitted { .. } => "protected_state.committed",
        LedgerEvent::MeshTunnelLifecycle(_) => "mesh.tunnel_lifecycle",
        LedgerEvent::ExecutionManifestAdmitted { .. } => "execution_manifest.admitted",
        LedgerEvent::ExecutionManifestRevoked { .. } => "execution_manifest.revoked",
        LedgerEvent::PodEvidenceCommitted { .. } => "pod.evidence_committed",
        LedgerEvent::PodOutputAdmitted { .. } => "pod.output_admitted",
        LedgerEvent::PodHypothesisCommitted { .. } => "pod.hypothesis_committed",
        LedgerEvent::PolicyBundleActivated { .. } => "policy.bundle_activated",
        LedgerEvent::PolicyBundleRevoked { .. } => "policy.bundle_revoked",
        LedgerEvent::SessionRevoked { .. } => "identity.session_revoked",
        LedgerEvent::TierBackendLifecycle { .. } => "storage.tier_backend_lifecycle",
        LedgerEvent::TierObjectCommitted { .. } => "storage.tier_object_committed",
        LedgerEvent::TierReplicaLifecycle { .. } => "storage.tier_replica_lifecycle",
    }
}

/// The subject a consumer partitions or deduplicates by.
pub fn event_subject(committed: &CommittedEvent) -> String {
    match &committed.event {
        LedgerEvent::CapsuleCommitted { capsule, .. }
        | LedgerEvent::CapsuleSuperseded { capsule, .. } => capsule.to_string(),
        LedgerEvent::Revoked { subject, .. } | LedgerEvent::VerifierAttested { subject, .. } => {
            subject.clone()
        }
        LedgerEvent::HardConstraintCommitted { key, .. } => format!("constraint:{key}"),
        LedgerEvent::ProcedurePromoted { id, .. } | LedgerEvent::ProcedureRevoked { id, .. } => {
            format!("procedure:{id}")
        }
        LedgerEvent::SemanticDeltaCommitted { revision, .. } => format!("revision:{}", revision.0),
        LedgerEvent::SnapshotCommitted { revision, .. } => format!("snapshot:{revision}"),
        LedgerEvent::EffectAttempted { .. } => format!("effect:{}", committed.index.0),
        LedgerEvent::EffectSettled { attempt, .. }
        | LedgerEvent::EffectReconciled { attempt, .. } => {
            format!("effect:{}", attempt.0)
        }
        LedgerEvent::ScopeLifecycle(event) => format!("scope:{}", event.scope_id),
        LedgerEvent::ProtectedStateCommitted { logical_id, .. } => {
            format!("protected:{logical_id}")
        }
        LedgerEvent::MeshTunnelLifecycle(event) => {
            format!("mesh:{}:{}", event.network_id, event.peer_id)
        }
        LedgerEvent::ExecutionManifestAdmitted {
            manifest_digest, ..
        }
        | LedgerEvent::ExecutionManifestRevoked {
            manifest_digest, ..
        } => {
            format!("execution-manifest:{}", hex_digest(manifest_digest))
        }
        LedgerEvent::PodEvidenceCommitted {
            session_id,
            trace_id,
            ..
        } => format!("pod-evidence:{}:{}", session_id, trace_id),
        LedgerEvent::PodOutputAdmitted {
            request_id,
            pod_id,
            output_digest,
            ..
        } => format!(
            "pod-output:{}:{}:{}",
            request_id,
            pod_id,
            hex_digest(output_digest)
        ),
        LedgerEvent::PodHypothesisCommitted { branch_id, .. } => {
            format!("pod-hypothesis:{branch_id}")
        }
        LedgerEvent::PolicyBundleActivated { revision, .. }
        | LedgerEvent::PolicyBundleRevoked { revision, .. } => {
            format!("policy:{}", revision.0)
        }
        LedgerEvent::SessionRevoked { session_id, .. } => format!("session:{session_id}"),
        LedgerEvent::TierBackendLifecycle { backend_id, .. } => {
            format!("tier-backend:{backend_id}")
        }
        LedgerEvent::TierObjectCommitted { root_digest, .. } => {
            format!("tier-object:{}", hex_digest(root_digest))
        }
        LedgerEvent::TierReplicaLifecycle {
            root_digest,
            backend_id,
            ..
        } => format!("tier-replica:{}:{backend_id}", hex_digest(root_digest)),
    }
}

fn hex_digest(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The projection event payload: the commit index and exactly the key/value
/// entries the event projects to, as a JSON object. A consumer sees what the
/// projection applied, not a second rendering of the event that could drift
/// from it.
pub fn event_payload(committed: &CommittedEvent) -> serde_json::Value {
    let entries: serde_json::Map<String, serde_json::Value> = projection_entries(committed)
        .into_iter()
        .map(|(key, value)| (key, serde_json::Value::String(value)))
        .collect();
    serde_json::json!({
        "commit_index": committed.index.0,
        "entries": entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_types::{CapsuleId, CommitIndex, Revision};

    fn committed(index: u64, event: LedgerEvent) -> CommittedEvent {
        CommittedEvent {
            index: CommitIndex(index),
            event,
        }
    }

    #[test]
    fn capsule_constraint_and_procedure_targets_are_keyed_as_the_runtime_keys_them() {
        let capsule = committed(
            3,
            LedgerEvent::CapsuleCommitted {
                project: ProjectId::from("p"),
                capsule: CapsuleId::from("c1"),
                generation: Generation(2),
            },
        );
        assert_eq!(
            lifecycle_change(&capsule),
            LifecycleChange::SetLive {
                target: "c1".into(),
                generation: Generation(2),
                project: Some(ProjectId::from("p")),
            }
        );
        let constraint = committed(
            4,
            LedgerEvent::HardConstraintCommitted {
                key: "budget".into(),
                generation: Generation(1),
            },
        );
        assert!(matches!(
            lifecycle_change(&constraint),
            LifecycleChange::SetLive { ref target, .. } if target == "constraint:budget"
        ));
        let payload = event_payload(&capsule);
        assert_eq!(payload["commit_index"], 3);
        assert_eq!(payload["entries"]["capsule:c1:generation"], "2");
        assert_eq!(event_subject(&capsule), "c1");
    }

    #[test]
    fn a_revocation_is_a_tombstone_and_never_a_live_generation_change() {
        let revoked = committed(
            5,
            LedgerEvent::ProcedureRevoked {
                id: "deploy".into(),
                generation: Generation(1),
            },
        );
        assert_eq!(
            lifecycle_change(&revoked),
            LifecycleChange::Tombstone {
                subject: "procedure:deploy".into(),
                generation: Generation(1)
            }
        );
    }

    #[test]
    fn a_semantic_delta_publishes_a_revision_whatever_its_origin() {
        for origin in [
            ptr_ledger::SemanticOrigin::Legacy,
            ptr_ledger::SemanticOrigin::Request {
                request: "request-1".into(),
            },
        ] {
            let delta = committed(
                6,
                LedgerEvent::SemanticDeltaCommitted {
                    base_revision: Revision(2),
                    revision: Revision(3),
                    encoded_delta: vec![],
                    origin,
                },
            );
            assert_eq!(
                lifecycle_change(&delta),
                LifecycleChange::Revision {
                    base: 2,
                    revision: 3
                }
            );
            assert_eq!(event_topic(&delta.event), "semantic.delta_committed");
            assert_eq!(event_subject(&delta), "revision:3");
        }
    }
}
