//! Integration tests against a real PostgreSQL server.
//!
//! `PTR_PG_TEST_DSN` must name a loopback PostgreSQL 16+ server where the
//! connecting role may create schemas and where pgvector 0.8+ is installed or
//! installable. Every test works in its own schema prefix and drops it at the
//! end, so tests run in parallel against one database.

use std::sync::atomic::{AtomicU32, Ordering};

use ptr_analytics::{Grouping, Metric, MetricRow, MetricSpec};
use ptr_branch::{
    BranchId, BranchOp, RangeDigest, SealedBranch, TriageDecision, TriageOutcome, ValueDigest,
};
use ptr_fastmem::{Decay, FastMemory, FastMemoryConfig, SourceRef, WriteRequest};
use ptr_ledger::integrity::{chain_anchors, LogAnchor};
use ptr_ledger::{CommittedEvent, LedgerEvent};
use ptr_pg::{
    BranchOutcome, EmbeddingSpace, FastMemoryRecord, HybridQuery, Identifier, PgError, PgSubstrate,
    SchemaSet, SearchDocument, LEXICAL_BACKEND, VECTOR_BACKEND,
};
use ptr_search::EvidenceStage;
use ptr_semdb::{SemanticPayload, SemanticValue};
use ptr_state::{ApplyOutcome, MaterializedState};
use ptr_types::{CapsuleId, CommitIndex, Generation, PrincipalId, ProjectId, Revision, TypeId};

static NEXT_PREFIX: AtomicU32 = AtomicU32::new(0);

fn dsn() -> String {
    match std::env::var("PTR_PG_TEST_DSN") {
        Ok(dsn) if !dsn.trim().is_empty() => dsn,
        _ => panic!(
            "PTR_PG_TEST_DSN is unset: the postgres test target needs a loopback PostgreSQL 16+ \
             server with pgvector, e.g. PTR_PG_TEST_DSN=\"host=127.0.0.1 port=5432 user=postgres\""
        ),
    }
}

/// The test server, with every session defaulting to repeatable read. The
/// substrate must pin the isolation its lock ordering needs rather than
/// inherit an operator's default. The connection-string parser removes one
/// level of backslashes and the server's option parser the next, which leaves
/// the escaped space inside the option value.
fn repeatable_read_dsn() -> String {
    format!(
        "{} options='-c default_transaction_isolation=repeatable\\\\ read'",
        dsn()
    )
}

/// A raw client for setup and for adversarial statements the substrate API
/// deliberately does not offer.
async fn raw_client() -> tokio_postgres::Client {
    raw_client_at(&dsn()).await
}

async fn raw_client_at(dsn: &str) -> tokio_postgres::Client {
    let (client, connection) = tokio_postgres::connect(dsn, tokio_postgres::NoTls)
        .await
        .expect("connect to PTR_PG_TEST_DSN");
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
}

/// A migrated substrate in a fresh schema prefix.
async fn substrate() -> PgSubstrate {
    substrate_at(&dsn()).await
}

async fn substrate_at(dsn: &str) -> PgSubstrate {
    let raw = raw_client().await;
    // Parallel tests race on CREATE EXTENSION; serialize it. The lock must be
    // held until the extension is committed, so it is a transaction lock: a
    // session lock released inside the batch's implicit transaction would let
    // the next test in before the extension is visible to it.
    raw.batch_execute(
        "BEGIN; \
         SELECT pg_advisory_xact_lock(7402301); \
         CREATE EXTENSION IF NOT EXISTS vector; \
         COMMIT;",
    )
    .await
    .expect("pgvector must be installed or installable");
    let prefix = format!(
        "ptrt_{}_{}",
        std::process::id(),
        NEXT_PREFIX.fetch_add(1, Ordering::SeqCst)
    );
    let schemas = SchemaSet::with_prefix(&prefix).unwrap();
    let mut substrate = PgSubstrate::connect_with(dsn, schemas).await.unwrap();
    // A previous run with the same process id may have left schemas behind.
    raw.batch_execute(&format!(
        "DROP SCHEMA IF EXISTS {prefix}_derived CASCADE; \
         DROP SCHEMA IF EXISTS {prefix}_projection CASCADE; \
         DROP SCHEMA IF EXISTS {prefix}_work CASCADE;"
    ))
    .await
    .unwrap();
    substrate.migrate().await.unwrap();
    substrate
}

fn committed(index: u64, event: LedgerEvent) -> CommittedEvent {
    CommittedEvent {
        index: CommitIndex(index),
        event,
    }
}

fn capsule(index: u64, id: &str, generation: u64) -> CommittedEvent {
    committed(
        index,
        LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("atlas"),
            capsule: CapsuleId::from(id),
            generation: Generation(generation),
        },
    )
}

fn supersede(index: u64, id: &str, old: u64, new: u64) -> CommittedEvent {
    committed(
        index,
        LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from(id),
            old: Generation(old),
            new: Generation(new),
        },
    )
}

fn revoke(index: u64, subject: &str, generation: u64) -> CommittedEvent {
    committed(
        index,
        LedgerEvent::Revoked {
            subject: subject.into(),
            generation: Generation(generation),
        },
    )
}

/// Anchors of `log` from the empty log, computed with the ledger's encoder.
fn anchors(log: &[CommittedEvent]) -> Vec<LogAnchor> {
    chain_anchors(log, LogAnchor::empty()).unwrap()
}

/// A log that exercises every lifecycle change.
fn mixed_log() -> Vec<CommittedEvent> {
    vec![
        capsule(1, "c1", 1),
        committed(
            2,
            LedgerEvent::HardConstraintCommitted {
                key: "budget".into(),
                generation: Generation(1),
            },
        ),
        committed(
            3,
            LedgerEvent::SemanticDeltaCommitted {
                base_revision: Revision(0),
                revision: Revision(1),
                encoded_delta: vec![1, 2, 3],
            },
        ),
        supersede(4, "c1", 1, 2),
        committed(
            5,
            LedgerEvent::ProcedurePromoted {
                id: "deploy".into(),
                generation: Generation(1),
            },
        ),
        committed(
            6,
            LedgerEvent::ProcedureRevoked {
                id: "deploy".into(),
                generation: Generation(1),
            },
        ),
        committed(
            7,
            LedgerEvent::VerifierAttested {
                subject: "c1".into(),
                passed: true,
            },
        ),
        revoke(8, "c1", 2),
    ]
}

#[tokio::test]
async fn migrations_apply_once_and_record_their_checksums() {
    let mut substrate = substrate().await;
    let again = substrate.migrate().await.unwrap();
    assert!(again.projection.is_empty() && again.derived.is_empty() && again.work.is_empty());
    let raw = raw_client().await;
    let work = substrate.schemas().work.clone();
    let count: i64 = raw
        .query_one(
            &format!("SELECT count(*) FROM {work}.schema_migration"),
            &[],
        )
        .await
        .unwrap()
        .get(0);
    assert_eq!(count, ptr_pg::WORK_MIGRATIONS.len() as i64);
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn an_edited_migration_is_refused_as_drift() {
    let mut substrate = substrate().await;
    let raw = raw_client().await;
    let work = substrate.schemas().work.clone();
    raw.execute(
        &format!("UPDATE {work}.schema_migration SET checksum = $1 WHERE version = 2"),
        &[&vec![0u8; 32]],
    )
    .await
    .unwrap();
    assert_eq!(
        substrate.migrate().await.unwrap_err(),
        PgError::MigrationDrift { version: 2 }
    );
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn replay_projects_state_and_lifecycle_exactly_as_the_reference_does() {
    let mut substrate = substrate().await;
    let log = mixed_log();
    assert_eq!(substrate.replay(&log).await.unwrap(), log.len() as u64);

    let mut reference = MaterializedState::default();
    for record in &log {
        assert_eq!(reference.try_apply(record), ApplyOutcome::Applied);
    }
    let fence = CommitIndex(log.len() as u64);
    for (key, value) in &reference.values {
        assert_eq!(
            substrate.state_value(key, fence).await.unwrap().as_ref(),
            Some(value),
            "{key}"
        );
    }
    // Every entry at once: no key the reference lacks, none missing.
    assert_eq!(
        substrate.state_entries(fence).await.unwrap(),
        reference.values
    );

    // Superseding moves the live generation; revoking never does, it only
    // makes the live generation inadmissible.
    assert_eq!(
        substrate.live_generation("c1", fence).await.unwrap(),
        Some(Generation(2))
    );
    assert!(!substrate
        .is_admissible("c1", Generation(2), fence)
        .await
        .unwrap());
    assert!(!substrate
        .is_admissible("c1", Generation(1), fence)
        .await
        .unwrap());
    assert!(substrate
        .is_admissible("constraint:budget", Generation(1), fence)
        .await
        .unwrap());
    assert!(!substrate
        .is_admissible("procedure:deploy", Generation(1), fence)
        .await
        .unwrap());
    assert_eq!(
        substrate.revision_commit(Revision(1)).await.unwrap(),
        Some(CommitIndex(3))
    );
    assert_eq!(
        substrate.watermark().await.unwrap(),
        *anchors(&log).last().unwrap()
    );

    let events = substrate.events_after(CommitIndex(0), 100).await.unwrap();
    let topics: Vec<&str> = events.iter().map(|event| event.topic.as_str()).collect();
    assert_eq!(
        topics,
        [
            "capsule.committed",
            "constraint.committed",
            "semantic.delta_committed",
            "capsule.superseded",
            "procedure.promoted",
            "procedure.revoked",
            "verifier.attested",
            "lifecycle.revoked",
        ]
    );
    assert_eq!(events[0].payload["entries"]["capsule:c1:project"], "atlas");
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn a_redelivered_record_is_a_duplicate_and_a_different_one_is_foreign_history() {
    let mut substrate = substrate().await;
    let log = vec![capsule(1, "c1", 1), capsule(2, "c2", 1)];
    let chain = anchors(&log);
    substrate.replay(&log).await.unwrap();

    let again = substrate.apply_committed(&log[1], chain[1]).await.unwrap();
    assert_eq!(again.outcome, ApplyOutcome::Duplicate);
    let older = substrate.apply_committed(&log[0], chain[0]).await.unwrap();
    assert_eq!(older.outcome, ApplyOutcome::OutOfOrder);

    let other = vec![capsule(1, "c1", 1), capsule(2, "other", 1)];
    let other_chain = anchors(&other);
    assert_eq!(
        substrate
            .apply_committed(&other[1], other_chain[1])
            .await
            .unwrap_err(),
        PgError::ForeignHistory { index: 2 }
    );
    // The true anchor does not make a different record a duplicate, at the
    // head or behind it; the true record with a foreign anchor is refused too.
    for (record, anchor) in [
        (&other[1], chain[1]),
        (&capsule(1, "other", 1), chain[0]),
        (&log[1], other_chain[1]),
    ] {
        assert_eq!(
            substrate.apply_committed(record, anchor).await.unwrap_err(),
            PgError::ForeignHistory {
                index: record.index.0
            }
        );
    }
    let gap = substrate
        .apply_committed(
            &capsule(9, "c9", 1),
            LogAnchor {
                index: CommitIndex(9),
                digest: [0; 32],
            },
        )
        .await
        .unwrap();
    assert_eq!(gap.outcome, ApplyOutcome::Gap);
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn a_record_that_does_not_chain_from_the_stored_anchor_is_refused() {
    let mut substrate = substrate().await;
    let ours = vec![capsule(1, "c1", 1)];
    substrate.replay(&ours).await.unwrap();
    // The ledger's history differs at index 1, so its record 2 chains from an
    // anchor this projection never stored.
    let theirs = vec![capsule(1, "c1", 7), capsule(2, "c2", 1)];
    let chain = anchors(&theirs);
    assert_eq!(
        substrate
            .apply_committed(&theirs[1], chain[1])
            .await
            .unwrap_err(),
        PgError::ForeignHistory { index: 2 }
    );
    // A record handed over with another index's anchor is malformed.
    assert!(matches!(
        substrate.apply_committed(&theirs[1], chain[0]).await,
        Err(PgError::InvalidRecord { index: 2, .. })
    ));
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn a_projection_ahead_of_or_beside_the_ledger_is_refused() {
    let mut substrate = substrate().await;
    let log = vec![
        capsule(1, "c1", 1),
        capsule(2, "c2", 1),
        capsule(3, "c3", 1),
    ];
    let chain = anchors(&log);
    substrate.replay(&log[..2]).await.unwrap();

    assert_eq!(substrate.check_against_ledger(chain[2]).await.unwrap(), 1);
    assert_eq!(substrate.check_against_ledger(chain[1]).await.unwrap(), 0);
    assert_eq!(
        substrate.check_against_ledger(chain[0]).await.unwrap_err(),
        PgError::ProjectionAhead {
            projection: 2,
            ledger: 1
        }
    );
    let beside = LogAnchor {
        index: CommitIndex(2),
        digest: [9; 32],
    };
    assert_eq!(
        substrate.check_against_ledger(beside).await.unwrap_err(),
        PgError::ForeignHistory { index: 2 }
    );
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn a_read_fenced_beyond_the_watermark_is_refused() {
    let mut substrate = substrate().await;
    substrate.replay(&[capsule(1, "c1", 1)]).await.unwrap();
    assert_eq!(
        substrate
            .state_value("capsule:c1:generation", CommitIndex(2))
            .await
            .unwrap_err(),
        PgError::ProjectionBehind {
            watermark: 1,
            fence: 2
        }
    );
    assert_eq!(
        substrate
            .state_value("capsule:c1:generation", CommitIndex(1))
            .await
            .unwrap(),
        Some("1".into())
    );
    assert_eq!(
        substrate.state_entries(CommitIndex(2)).await.unwrap_err(),
        PgError::ProjectionBehind {
            watermark: 1,
            fence: 2
        }
    );
    assert_eq!(
        substrate
            .state_entries(CommitIndex(1))
            .await
            .unwrap()
            .get("capsule:c1:generation"),
        Some(&"1".to_owned())
    );
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn append_only_tables_refuse_rewrites() {
    let mut substrate = substrate().await;
    substrate
        .replay(&[capsule(1, "c1", 1), revoke(2, "c1", 1)])
        .await
        .unwrap();
    let raw = raw_client().await;
    let projection = substrate.schemas().projection.clone();
    for statement in [
        format!("UPDATE {projection}.projection_event SET topic = 'x'"),
        format!("DELETE FROM {projection}.applied_commit"),
        format!("DELETE FROM {projection}.tombstone"),
    ] {
        let error = raw.execute(&statement, &[]).await.unwrap_err();
        assert_eq!(
            error.as_db_error().unwrap().code().code(),
            "23000",
            "{statement}"
        );
    }
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn consumer_offsets_only_move_forward_and_never_past_the_watermark() {
    let mut substrate = substrate().await;
    substrate
        .replay(&[capsule(1, "c1", 1), capsule(2, "c2", 1)])
        .await
        .unwrap();
    assert_eq!(
        substrate.consumer_offset("indexer").await.unwrap(),
        CommitIndex(0)
    );
    substrate
        .commit_consumer("indexer", CommitIndex(2))
        .await
        .unwrap();
    substrate
        .commit_consumer("indexer", CommitIndex(1))
        .await
        .unwrap();
    assert_eq!(
        substrate.consumer_offset("indexer").await.unwrap(),
        CommitIndex(2)
    );
    assert_eq!(
        substrate
            .commit_consumer("indexer", CommitIndex(3))
            .await
            .unwrap_err(),
        PgError::ProjectionBehind {
            watermark: 2,
            fence: 3
        }
    );
    let rest = substrate.events_after(CommitIndex(1), 10).await.unwrap();
    assert_eq!(rest.len(), 1);
    assert_eq!(rest[0].commit_index, CommitIndex(2));
    substrate.drop_all().await.unwrap();
}

fn space() -> EmbeddingSpace {
    EmbeddingSpace {
        id: Identifier::new("toy4").unwrap(),
        model: "toy-encoder".into(),
        revision: "2026-09".into(),
        dims: 4,
    }
}

fn document(capsule: &str, generation: u64, body: &str, embedding: &[f32]) -> SearchDocument {
    SearchDocument {
        capsule: CapsuleId::from(capsule),
        generation: Generation(generation),
        content_digest: [generation as u8; 32],
        body: body.into(),
        embedding: Some((space().id, embedding.to_vec())),
    }
}

#[tokio::test]
async fn only_live_generations_are_indexed_and_superseding_drops_the_old_document() {
    let mut substrate = substrate().await;
    substrate.register_space(&space()).await.unwrap();
    // Registering the same definition again is a no-op; a different one is not.
    substrate.register_space(&space()).await.unwrap();
    let mut changed = space();
    changed.dims = 8;
    assert_eq!(
        substrate.register_space(&changed).await.unwrap_err(),
        PgError::SpaceConflict {
            space: "toy4".into()
        }
    );

    let log = vec![capsule(1, "c1", 1), supersede(2, "c1", 1, 2)];
    let chain = anchors(&log);
    substrate.apply_committed(&log[0], chain[0]).await.unwrap();
    substrate
        .upsert_document(&document(
            "c1",
            1,
            "first generation",
            &[1.0, 0.0, 0.0, 0.0],
        ))
        .await
        .unwrap();
    assert_eq!(
        substrate
            .upsert_document(&document("c9", 1, "never committed", &[1.0, 0.0, 0.0, 0.0]))
            .await
            .unwrap_err(),
        PgError::NotLive {
            target: "c9".into(),
            generation: 1
        }
    );
    let report = substrate.apply_committed(&log[1], chain[1]).await.unwrap();
    assert_eq!(report.dropped_documents, 1);
    assert!(matches!(
        substrate
            .upsert_document(&document("c1", 1, "stale", &[1.0, 0.0, 0.0, 0.0]))
            .await,
        Err(PgError::NotLive { .. })
    ));
    assert_eq!(
        substrate
            .upsert_document(&document("c1", 2, "second", &[1.0, 0.0, 0.0]))
            .await
            .unwrap_err(),
        PgError::DimensionMismatch {
            expected: 4,
            actual: 3
        }
    );
    assert!(matches!(
        substrate
            .upsert_document(&document("c1", 2, "second", &[f32::NAN, 0.0, 0.0, 0.0]))
            .await,
        Err(PgError::InvalidEmbedding { .. })
    ));
    // Nonzero in f32 but zero once stored in half precision.
    assert!(matches!(
        substrate
            .upsert_document(&document("c1", 2, "second", &[1e-8, 1e-8, 1e-8, 1e-8]))
            .await,
        Err(PgError::InvalidEmbedding { .. })
    ));
    substrate
        .upsert_document(&document(
            "c1",
            2,
            "second generation",
            &[0.0, 1.0, 0.0, 0.0],
        ))
        .await
        .unwrap();
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn hybrid_search_returns_live_candidates_fused_by_capsule_and_generation() {
    let mut substrate = substrate().await;
    substrate.register_space(&space()).await.unwrap();
    substrate
        .replay(&[
            capsule(1, "c1", 1),
            capsule(2, "c2", 1),
            capsule(3, "c3", 1),
        ])
        .await
        .unwrap();
    for doc in [
        document(
            "c1",
            1,
            "projection watermark anchors",
            &[1.0, 0.0, 0.0, 0.0],
        ),
        document(
            "c2",
            1,
            "fast weight memory revocation",
            &[0.0, 1.0, 0.0, 0.0],
        ),
        document("c3", 1, "revocation of adapters", &[0.0, 0.0, 1.0, 0.0]),
    ] {
        substrate.upsert_document(&doc).await.unwrap();
    }
    let query = HybridQuery {
        project: Some(ProjectId::from("atlas")),
        text: Some("revocation".into()),
        embedding: Some((space().id, vec![0.1, 0.9, 0.0, 0.0])),
        limit: 10,
        lexical_weight: 1.0,
        vector_weight: 1.0,
        rank_constant: 60.0,
    };
    let results = substrate.search(&query).await.unwrap();
    assert_eq!(results.watermark, 3);
    let lexical: Vec<&str> = results
        .lexical
        .iter()
        .map(|hit| hit.capsule.0.as_str())
        .collect();
    assert_eq!(lexical.len(), 2);
    assert!(lexical.contains(&"c2") && lexical.contains(&"c3"));
    assert_eq!(results.vector[0].capsule, CapsuleId::from("c2"));
    assert_eq!(results.vector.len(), 3);
    assert_eq!(results.fused[0].capsule, CapsuleId::from("c2"));
    assert_eq!(
        results.fused[0].backends,
        vec![LEXICAL_BACKEND.to_owned(), VECTOR_BACKEND.to_owned()]
    );
    assert!(results
        .lexical
        .iter()
        .chain(&results.vector)
        .all(|hit| hit.stage() == EvidenceStage::SearchCandidate));

    // A revoked generation disappears from both modes in the same commit.
    let chain = anchors(&[
        capsule(1, "c1", 1),
        capsule(2, "c2", 1),
        capsule(3, "c3", 1),
        revoke(4, "c2", 1),
    ]);
    let report = substrate
        .apply_committed(&revoke(4, "c2", 1), chain[3])
        .await
        .unwrap();
    assert_eq!(report.dropped_documents, 1);
    let after = substrate.search(&query).await.unwrap();
    assert!(after
        .fused
        .iter()
        .all(|hit| hit.capsule != CapsuleId::from("c2")));
    substrate.drop_all().await.unwrap();
}

fn memory_config() -> FastMemoryConfig {
    FastMemoryConfig {
        heads: 1,
        key_dim: 4,
        value_dim: 4,
        checkpoint_interval: 2,
        max_writes: 64,
    }
}

fn write_request(source: &str, key: [f32; 4], value: [f32; 4]) -> WriteRequest {
    WriteRequest {
        source: SourceRef {
            key: source.into(),
            generation: Generation(1),
            input_digest: [source.len() as u8; 32],
        },
        key: key.to_vec(),
        value: value.to_vec(),
        beta: 0.75,
        decay: Decay::Scalar(0.9),
    }
}

fn bits(cells: &[f32]) -> Vec<u32> {
    cells.iter().map(|cell| cell.to_bits()).collect()
}

#[tokio::test]
async fn a_revocation_deletes_exactly_the_revoked_writes_and_the_checkpoints_that_folded_them() {
    let mut substrate = substrate().await;
    let log = vec![
        capsule(1, "keep", 1),
        capsule(2, "gone", 1),
        revoke(3, "gone", 1),
    ];
    let chain = anchors(&log);
    substrate.apply_committed(&log[0], chain[0]).await.unwrap();
    substrate.apply_committed(&log[1], chain[1]).await.unwrap();

    let config = memory_config();
    substrate
        .create_memory(&FastMemoryRecord {
            id: "m1".into(),
            principal: PrincipalId::from("agent-7"),
            thread: "t1".into(),
            config,
            projection_digest: [3; 32],
            codebook_seed: u64::MAX - 5,
        })
        .await
        .unwrap();
    let loaded = substrate.load_memory("m1").await.unwrap().unwrap();
    assert_eq!(loaded.codebook_seed, u64::MAX - 5);
    assert_eq!(loaded.config, config);

    let requests = [
        write_request("keep", [1.0, 0.0, 0.0, 0.0], [0.5, 0.1, 0.0, 0.0]),
        write_request("keep", [0.0, 2.0, 0.0, 0.0], [0.0, 0.3, 0.2, 0.0]),
        write_request("gone", [0.3, 0.3, 0.9, 0.0], [0.9, 0.9, 0.9, 0.9]),
        write_request("keep", [0.1, 0.0, 0.0, 1.0], [0.0, 0.0, 0.4, 0.7]),
    ];
    let mut memory = FastMemory::new(config).unwrap();
    for (position, request) in requests.iter().enumerate() {
        let receipt = memory.write(request.clone()).unwrap();
        substrate
            .append_write("m1", receipt.seq, request)
            .await
            .unwrap();
        if position % 2 == 1 {
            substrate
                .put_checkpoint("m1", memory.binding_digest(), memory.state())
                .await
                .unwrap();
        }
    }
    // A stored journal refolds to the in-process state, bit for bit.
    let journal = substrate.load_journal("m1").await.unwrap();
    let restored = FastMemory::restore(config, journal).unwrap();
    assert_eq!(bits(restored.state().cells()), bits(memory.state().cells()));

    let report = substrate.apply_committed(&log[2], chain[2]).await.unwrap();
    assert_eq!(report.removed_writes, 1);
    // Checkpoints at 2 and 4; only the one at 4 folded write 3.
    assert_eq!(report.dropped_checkpoints, 1);

    memory.revoke(|source| source.key == "gone");
    let journal = substrate.load_journal("m1").await.unwrap();
    assert_eq!(journal.len(), 3);
    let refolded = FastMemory::restore(config, journal).unwrap();
    assert_eq!(bits(refolded.state().cells()), bits(memory.state().cells()));
    // ...and equals a memory that never saw the revoked write.
    let mut never = FastMemory::new(config).unwrap();
    for request in requests
        .iter()
        .filter(|request| request.source.key == "keep")
    {
        never.write(request.clone()).unwrap();
    }
    assert_eq!(bits(refolded.state().cells()), bits(never.state().cells()));

    let checkpoint = substrate.latest_checkpoint("m1").await.unwrap().unwrap();
    assert_eq!(checkpoint.applied.0, 2);
    let mut prefix = FastMemory::new(config).unwrap();
    for request in &requests[..2] {
        prefix.write(request.clone()).unwrap();
    }
    assert_eq!(checkpoint.binding_digest, prefix.binding_digest());
    assert_eq!(bits(checkpoint.state.cells()), bits(prefix.state().cells()));
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn a_write_from_an_inadmissible_source_or_out_of_sequence_is_refused() {
    let mut substrate = substrate().await;
    substrate
        .replay(&[
            capsule(1, "live", 1),
            capsule(2, "revoked", 1),
            revoke(3, "revoked", 1),
        ])
        .await
        .unwrap();
    let config = memory_config();
    substrate
        .create_memory(&FastMemoryRecord {
            id: "m1".into(),
            principal: PrincipalId::from("agent-7"),
            thread: "t1".into(),
            config,
            projection_digest: [3; 32],
            codebook_seed: 11,
        })
        .await
        .unwrap();
    let revoked = write_request("revoked", [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(
        substrate
            .append_write("m1", ptr_fastmem::WriteSeq(1), &revoked)
            .await
            .unwrap_err(),
        PgError::NotLive {
            target: "revoked".into(),
            generation: 1
        }
    );
    let mut stale = write_request("live", [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
    stale.source.generation = Generation(0);
    assert!(matches!(
        substrate
            .append_write("m1", ptr_fastmem::WriteSeq(1), &stale)
            .await,
        Err(PgError::NotLive { .. })
    ));
    let live = write_request("live", [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
    substrate
        .append_write("m1", ptr_fastmem::WriteSeq(5), &live)
        .await
        .unwrap();
    assert!(matches!(
        substrate
            .append_write("m1", ptr_fastmem::WriteSeq(5), &live)
            .await,
        Err(PgError::InvalidWrite { .. })
    ));
    assert!(matches!(
        substrate
            .append_write("nope", ptr_fastmem::WriteSeq(6), &live)
            .await,
        Err(PgError::InvalidWrite { .. })
    ));
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn malformed_writes_never_enter_the_journal() {
    let mut substrate = substrate().await;
    substrate.replay(&[capsule(1, "live", 1)]).await.unwrap();
    substrate
        .create_memory(&FastMemoryRecord {
            id: "m1".into(),
            principal: PrincipalId::from("agent"),
            thread: "thread".into(),
            config: memory_config(),
            projection_digest: [3; 32],
            codebook_seed: 11,
        })
        .await
        .unwrap();
    let valid = write_request("live", [2.0, 0.0, 0.0, 0.0], [1.0; 4]);
    let mut invalid = Vec::new();
    let mut request = valid.clone();
    request.key.pop();
    invalid.push(request);
    let mut request = valid.clone();
    request.value.pop();
    invalid.push(request);
    for value in [0.0, f32::NAN, f32::INFINITY, f32::MAX] {
        let mut request = valid.clone();
        request.key.fill(value);
        invalid.push(request);
    }
    for value in [f32::NAN, f32::INFINITY] {
        let mut request = valid.clone();
        request.value[0] = value;
        invalid.push(request);
    }
    for value in [0.0, -0.1, 1.1, f32::NAN, f32::INFINITY] {
        let mut request = valid.clone();
        request.beta = value;
        invalid.push(request);
        let mut request = valid.clone();
        request.decay = Decay::Scalar(value);
        invalid.push(request);
        let mut request = valid.clone();
        request.decay = Decay::PerChannel(vec![value; 4]);
        invalid.push(request);
    }
    let mut request = valid.clone();
    request.decay = Decay::PerChannel(vec![1.0; 3]);
    invalid.push(request);
    for request in invalid {
        assert!(ptr_fastmem::validate_write(&memory_config(), &request).is_err());
        assert!(matches!(
            substrate
                .append_write("m1", ptr_fastmem::WriteSeq(1), &request)
                .await,
            Err(PgError::InvalidWrite { .. })
        ));
        assert!(substrate.load_journal("m1").await.unwrap().is_empty());
    }
    substrate
        .append_write("m1", ptr_fastmem::WriteSeq(1), &valid)
        .await
        .unwrap();
    let journal = substrate.load_journal("m1").await.unwrap();
    assert_eq!(journal[0].1, valid);
    FastMemory::restore(memory_config(), journal).unwrap();
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn a_memory_whose_shape_fast_memory_refuses_is_never_registered() {
    let substrate = substrate().await;
    let record = |id: &str, config: FastMemoryConfig| FastMemoryRecord {
        id: id.into(),
        principal: PrincipalId::from("agent"),
        thread: id.into(),
        config,
        projection_digest: [3; 32],
        codebook_seed: 11,
    };
    // Each dimension is within its column's bounds; only their product,
    // 64 Mi cells, exceeds the state bound.
    let oversized = FastMemoryConfig {
        heads: 64,
        key_dim: 1024,
        value_dim: 1024,
        ..memory_config()
    };
    let no_heads = FastMemoryConfig {
        heads: 0,
        ..memory_config()
    };
    for (id, config) in [("oversized", oversized), ("no-heads", no_heads)] {
        assert!(ptr_fastmem::check_config(&config).is_err());
        assert_eq!(
            substrate.create_memory(&record(id, config)).await,
            Err(PgError::InvalidMemory {
                memory: id.into(),
                reason: "the configuration is outside the supported ranges",
            })
        );
        assert_eq!(substrate.load_memory(id).await.unwrap(), None);
    }
    // A state of exactly `MAX_STATE_CELLS` is supported and registers.
    let largest = FastMemoryConfig {
        heads: 16,
        key_dim: 1024,
        value_dim: 1024,
        ..memory_config()
    };
    assert_eq!(largest.state_cells(), ptr_fastmem::MAX_STATE_CELLS);
    substrate
        .create_memory(&record("largest", largest))
        .await
        .unwrap();
    assert_eq!(
        substrate
            .load_memory("largest")
            .await
            .unwrap()
            .unwrap()
            .config,
        largest
    );
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn invalid_search_parameters_are_refused_before_sql() {
    let substrate = substrate().await;
    let schemas = substrate.schemas().clone();
    substrate.drop_all().await.unwrap();
    // Missing schemas ensure a valid query would fail if SQL were reached.
    let mut substrate = PgSubstrate::connect_with(&dsn(), schemas).await.unwrap();
    let query = HybridQuery {
        project: None,
        text: None,
        embedding: None,
        limit: 1,
        rank_constant: 0.0,
        lexical_weight: 0.0,
        vector_weight: 0.0,
    };
    let mut invalid = query.clone();
    invalid.limit = 0;
    assert_eq!(
        substrate.search(&invalid).await.unwrap_err(),
        PgError::OutOfRange {
            field: "query.limit"
        }
    );
    for value in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for field in [
            "query.rank_constant",
            "query.lexical_weight",
            "query.vector_weight",
        ] {
            let mut invalid = query.clone();
            match field {
                "query.rank_constant" => invalid.rank_constant = value,
                "query.lexical_weight" => invalid.lexical_weight = value,
                _ => invalid.vector_weight = value,
            }
            assert_eq!(
                substrate.search(&invalid).await.unwrap_err(),
                PgError::OutOfRange { field }
            );
        }
    }
}

#[tokio::test]
async fn ddl_failures_leave_the_connection_outside_a_failed_transaction() {
    let substrate = substrate().await;
    let schemas = substrate.schemas().clone();
    let read_only = format!("{} options='-c default_transaction_read_only=on'", dsn());
    let mut reader = PgSubstrate::connect_with(&read_only, schemas)
        .await
        .unwrap();
    for _ in 0..2 {
        for error in [
            reader.migrate().await.unwrap_err(),
            reader.rebuild_projection().await.unwrap_err(),
        ] {
            assert!(
                matches!(error, PgError::Database { ref sqlstate, .. }
                if sqlstate == "25006"),
                "{error:?}"
            );
        }
    }
    substrate.drop_all().await.unwrap();
}

fn sealed_branch(id: &str, author: &str) -> SealedBranch {
    let payload = SemanticValue::Payload(SemanticPayload {
        type_id: TypeId::from("ptr.test.bytes"),
        source: "unit-test".into(),
        bytes: vec![0, 1, 2, 255],
    });
    SealedBranch {
        id: BranchId::from(id),
        author: PrincipalId::from(author),
        base_revision: Revision(4),
        reads: [
            (
                "order:1".to_owned(),
                ValueDigest::of("order:1", Some(&SemanticValue::from("open"))).unwrap(),
            ),
            (
                "order:2".to_owned(),
                ValueDigest::of("order:2", None).unwrap(),
            ),
        ]
        .into_iter()
        .collect(),
        scans: [("order:".to_owned(), RangeDigest::from_bytes([5; 32]))]
            .into_iter()
            .collect(),
        relied: [("constraint:budget".to_owned(), Generation(3))]
            .into_iter()
            .collect(),
        touched_base: [(
            "order:1".to_owned(),
            ValueDigest::of("order:1", Some(&SemanticValue::from("open"))).unwrap(),
        )]
        .into_iter()
        .collect(),
        ops: vec![
            BranchOp::Put {
                key: "order:1".into(),
                value: SemanticValue::from("shipped"),
            },
            BranchOp::Put {
                key: "order:blob".into(),
                value: payload,
            },
            BranchOp::Remove {
                key: "order:2".into(),
            },
            BranchOp::Add {
                key: "stock:widget".into(),
                amount: -3,
            },
            BranchOp::SetInsert {
                key: "tags:1".into(),
                member: "urgent".into(),
            },
            BranchOp::SetRemove {
                key: "tags:1".into(),
                member: "draft".into(),
            },
        ],
    }
}

#[tokio::test]
async fn a_sealed_branch_round_trips_with_every_dependency_and_op() {
    let mut substrate = substrate().await;
    let branch = sealed_branch("b1", "agent-7");
    substrate.store_branch(&branch).await.unwrap();
    assert_eq!(
        substrate.load_branch(&branch.id).await.unwrap(),
        Some(branch.clone())
    );
    assert_eq!(
        substrate
            .load_branch(&BranchId::from("none"))
            .await
            .unwrap(),
        None
    );
    // A branch id is stored once.
    assert!(matches!(
        substrate.store_branch(&branch).await,
        Err(PgError::Database { ref sqlstate, .. }) if sqlstate == "23505"
    ));
    substrate.drop_all().await.unwrap();
}

fn triage(decision: TriageDecision, eligible: bool, calibration_slice: bool) -> TriageOutcome {
    TriageOutcome {
        decision,
        eligible,
        calibration_slice,
        score: 0.8,
        auto_propensity: if eligible { 0.9 } else { 0.0 },
    }
}

#[tokio::test]
async fn triage_logs_and_outcomes_feed_the_platform_metrics() {
    let mut substrate = substrate().await;
    let plan = [
        (
            "b1",
            "agent-a",
            triage(TriageDecision::AutoPropose, true, false),
        ),
        (
            "b2",
            "agent-a",
            triage(TriageDecision::Escalate, true, true),
        ),
        (
            "b3",
            "agent-b",
            triage(TriageDecision::Escalate, true, true),
        ),
        (
            "b4",
            "agent-b",
            triage(TriageDecision::Discard, false, false),
        ),
    ];
    for (id, author, outcome) in &plan {
        substrate
            .store_branch(&sealed_branch(id, author))
            .await
            .unwrap();
        substrate
            .record_triage(&BranchId::from(*id), outcome, "policy-1")
            .await
            .unwrap();
    }
    substrate
        .record_outcome(&BranchId::from("b1"), BranchOutcome::Merged(CommitIndex(9)))
        .await
        .unwrap();
    substrate
        .record_outcome(&BranchId::from("b2"), BranchOutcome::AdjudicatedHarmful)
        .await
        .unwrap();
    substrate
        .record_outcome(&BranchId::from("b3"), BranchOutcome::AdjudicatedHarmless)
        .await
        .unwrap();
    substrate
        .record_outcome(&BranchId::from("b4"), BranchOutcome::Conflicted)
        .await
        .unwrap();

    let overall = |metric| MetricSpec {
        metric,
        grouping: Grouping::Overall,
    };
    let row = |numerator, denominator| MetricRow {
        group: String::new(),
        numerator,
        denominator,
    };
    assert_eq!(
        substrate
            .metric(overall(Metric::AutoProposeShare))
            .await
            .unwrap(),
        vec![row(1, 3)]
    );
    assert_eq!(
        substrate
            .metric(overall(Metric::EscalationShare))
            .await
            .unwrap(),
        vec![row(2, 4)]
    );
    assert_eq!(
        substrate
            .metric(overall(Metric::ConflictRate))
            .await
            .unwrap(),
        vec![row(1, 4)]
    );
    assert_eq!(
        substrate
            .metric(overall(Metric::AdjudicatedHarmRate))
            .await
            .unwrap(),
        vec![row(1, 2)]
    );
    let by_principal = substrate
        .metric(MetricSpec {
            metric: Metric::EscalationShare,
            grouping: Grouping::ByPrincipal,
        })
        .await
        .unwrap();
    assert_eq!(
        by_principal,
        vec![
            MetricRow {
                group: "agent-a".into(),
                numerator: 1,
                denominator: 2
            },
            MetricRow {
                group: "agent-b".into(),
                numerator: 1,
                denominator: 2
            },
        ]
    );

    // A calibration-slice branch is by definition escalated, and an outcome
    // is never rewritten.
    substrate
        .store_branch(&sealed_branch("b5", "agent-a"))
        .await
        .unwrap();
    assert!(matches!(
        substrate
            .record_triage(
                &BranchId::from("b5"),
                &triage(TriageDecision::AutoPropose, true, true),
                "policy-1"
            )
            .await,
        Err(PgError::Database { ref sqlstate, .. }) if sqlstate == "23514"
    ));
    let raw = raw_client().await;
    let work = substrate.schemas().work.clone();
    let error = raw
        .execute(
            &format!("UPDATE {work}.branch_outcome SET outcome = 'discarded' WHERE branch = 'b4'"),
            &[],
        )
        .await
        .unwrap_err();
    assert_eq!(error.as_db_error().unwrap().code().code(), "23000");

    // Neither is an outcome removed on its own, nor a logged triage changed,
    // nor a branch adjudicated twice.
    for statement in [
        format!("DELETE FROM {work}.branch_outcome WHERE branch = 'b4'"),
        format!("UPDATE {work}.branch_triage SET score = 0.1 WHERE branch = 'b1'"),
        format!("DELETE FROM {work}.branch_triage WHERE branch = 'b1'"),
    ] {
        let error = raw.execute(&statement, &[]).await.unwrap_err();
        assert_eq!(
            error.as_db_error().unwrap().code().code(),
            "23000",
            "{statement}"
        );
    }
    assert!(matches!(
        substrate
            .record_outcome(&BranchId::from("b2"), BranchOutcome::AdjudicatedHarmless)
            .await,
        Err(PgError::Database { ref sqlstate, .. }) if sqlstate == "23505"
    ));
    // Erasing a whole branch removes its triage and outcomes with it.
    raw.execute(&format!("DELETE FROM {work}.branch WHERE id = 'b4'"), &[])
        .await
        .unwrap();
    let left: i64 = raw
        .query_one(
            &format!("SELECT count(*) FROM {work}.branch_outcome WHERE branch = 'b4'"),
            &[],
        )
        .await
        .unwrap()
        .get(0);
    assert_eq!(left, 0);
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn rebuild_drops_projection_and_derived_caches_but_keeps_working_state() {
    let mut substrate = substrate().await;
    substrate.register_space(&space()).await.unwrap();
    let log = vec![capsule(1, "c1", 1), capsule(2, "c2", 1)];
    substrate.replay(&log).await.unwrap();
    substrate
        .upsert_document(&document("c1", 1, "indexed", &[1.0, 0.0, 0.0, 0.0]))
        .await
        .unwrap();
    let branch = sealed_branch("b1", "agent-7");
    substrate.store_branch(&branch).await.unwrap();
    let before = substrate.watermark().await.unwrap();

    let report = substrate.rebuild_projection().await.unwrap();
    assert_eq!(report.projection, vec![1]);
    assert_eq!(report.derived, vec![1]);
    assert!(report.work.is_empty());
    assert_eq!(substrate.watermark().await.unwrap(), LogAnchor::empty());

    assert_eq!(substrate.replay(&log).await.unwrap(), 2);
    assert_eq!(substrate.watermark().await.unwrap(), before);
    assert_eq!(
        substrate.load_branch(&branch.id).await.unwrap(),
        Some(branch)
    );
    // Derived caches are recomputed, not restored: the space must be
    // registered again before anything is indexed.
    let query = HybridQuery {
        project: None,
        text: Some("indexed".into()),
        embedding: None,
        limit: 10,
        lexical_weight: 1.0,
        vector_weight: 1.0,
        rank_constant: 60.0,
    };
    assert!(substrate.search(&query).await.unwrap().lexical.is_empty());
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn working_state_constraints_hold_in_the_database() {
    let substrate = substrate().await;
    let raw = raw_client().await;
    let work = substrate.schemas().work.clone();
    for (statement, sqlstate) in [
        // A held-out sample can never enter the replay pool.
        (
            format!(
                "INSERT INTO {work}.replay_sample \
                 (id, stratum, split, stability, difficulty, last_probe_model_time) \
                 VALUES ('s1', 'x', 'heldout', 1, 5, 0)"
            ),
            "23514",
        ),
        // A consolidated adapter has sources, not a parent.
        (
            format!(
                "INSERT INTO {work}.adapter \
                 (id, domain, base_model, base_revision, origin, parent, rank, artifact, \
                  artifact_sha256, data_fingerprint, status) \
                 VALUES ('a1', 'd', 'm', 'r', 'consolidated', 'a0', 8, 'x', \
                         decode(repeat('00', 32), 'hex'), decode(repeat('00', 32), 'hex'), \
                         'candidate')"
            ),
            "23514",
        ),
        // A label schema needs at least two classes.
        (
            format!("INSERT INTO {work}.label_schema (id, classes) VALUES ('s', ARRAY['only'])"),
            "23514",
        ),
    ] {
        let error = raw.execute(&statement, &[]).await.unwrap_err();
        assert_eq!(
            error.as_db_error().unwrap().code().code(),
            sqlstate,
            "{statement}"
        );
    }
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn a_hostaddr_that_is_not_loopback_is_refused_whatever_the_host() {
    // With hostaddr set the driver connects there and uses host only as a name.
    for dsn in [
        "hostaddr=192.0.2.2 user=ptr",
        "host=localhost hostaddr=192.0.2.2 user=ptr",
        "host=/var/run/postgresql hostaddr=192.0.2.2 user=ptr",
    ] {
        let schemas = SchemaSet::with_prefix("ptr_unused").unwrap();
        let error = PgSubstrate::connect_with(dsn, schemas)
            .await
            .err()
            .expect("a remote hostaddr must be refused");
        assert_eq!(
            error,
            PgError::TlsRequired {
                host: "192.0.2.2".into()
            },
            "{dsn}"
        );
    }
}

#[tokio::test]
async fn a_non_loopback_host_is_refused_without_a_tls_connector() {
    let schemas = SchemaSet::with_prefix("ptr_unused").unwrap();
    let error = PgSubstrate::connect_with("host=db.example.com user=ptr", schemas)
        .await
        .err()
        .expect("a remote host must be refused");
    assert_eq!(
        error,
        PgError::TlsRequired {
            host: "db.example.com".into()
        }
    );
}

#[tokio::test]
async fn a_derived_write_holding_the_lifecycle_row_is_ordered_before_the_supersede() {
    supersede_waits_for_the_derived_writer(substrate().await).await;
}

#[tokio::test]
async fn the_lock_ordering_holds_when_sessions_default_to_repeatable_read() {
    let isolation: String = raw_client_at(&repeatable_read_dsn())
        .await
        .query_one("SHOW default_transaction_isolation", &[])
        .await
        .unwrap()
        .get(0);
    assert_eq!(isolation, "repeatable read");
    supersede_waits_for_the_derived_writer(substrate_at(&repeatable_read_dsn()).await).await;
}

async fn supersede_waits_for_the_derived_writer(mut substrate: PgSubstrate) {
    let log = vec![capsule(1, "c1", 1), supersede(2, "c1", 1, 2)];
    let chain = anchors(&log);
    substrate.apply_committed(&log[0], chain[0]).await.unwrap();
    let (projection, derived) = (
        substrate.schemas().projection.clone(),
        substrate.schemas().derived.clone(),
    );

    // A cache writer takes the row lock exactly as upsert_document does and
    // indexes generation 1 while the supersede is in flight.
    let mut writer = raw_client().await;
    let transaction = writer.transaction().await.unwrap();
    transaction
        .query_one(
            &format!(
                "SELECT generation FROM {projection}.live_generation \
                 WHERE target = 'c1' FOR SHARE"
            ),
            &[],
        )
        .await
        .unwrap();
    transaction
        .execute(
            &format!(
                "INSERT INTO {derived}.search_document \
                 (capsule, generation, project, content_digest, body, indexed_at_commit) \
                 VALUES ('c1', 1, 'atlas', decode(repeat('00', 32), 'hex'), 'late', 1)"
            ),
            &[],
        )
        .await
        .unwrap();

    let supersede = log[1].clone();
    let projector = tokio::spawn(async move {
        let report = substrate
            .apply_committed(&supersede, chain[1])
            .await
            .unwrap();
        (substrate, report)
    });
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    assert!(
        !projector.is_finished(),
        "the projector must wait for the row lock"
    );
    transaction.commit().await.unwrap();

    let (substrate, report) = projector.await.unwrap();
    assert_eq!(report.dropped_documents, 1);
    let remaining: i64 = writer
        .query_one(
            &format!("SELECT count(*) FROM {derived}.search_document"),
            &[],
        )
        .await
        .unwrap()
        .get(0);
    assert_eq!(remaining, 0);
    substrate.drop_all().await.unwrap();
}

async fn memory_with(substrate: &PgSubstrate, id: &str, max_writes: u32) -> FastMemoryConfig {
    let config = FastMemoryConfig {
        max_writes,
        ..memory_config()
    };
    substrate
        .create_memory(&FastMemoryRecord {
            id: id.into(),
            principal: PrincipalId::from("agent-7"),
            thread: id.into(),
            config,
            projection_digest: [3; 32],
            codebook_seed: 11,
        })
        .await
        .unwrap();
    config
}

/// Journal a write from another connection while holding the memory row,
/// the way a concurrent `append_write` does, and let `append` run meanwhile.
async fn race_an_append(
    mut substrate: PgSubstrate,
    memory: &str,
    held_seq: i64,
    racing_seq: u64,
) -> (PgSubstrate, Result<(), PgError>) {
    let work = substrate.schemas().work.clone();
    let mut holder = raw_client().await;
    let transaction = holder.transaction().await.unwrap();
    transaction
        .query_one(
            &format!("SELECT max_writes FROM {work}.fastmem_memory WHERE id = $1 FOR UPDATE"),
            &[&memory],
        )
        .await
        .unwrap();
    transaction
        .execute(
            &format!(
                "INSERT INTO {work}.fastmem_write \
                 (memory, seq, source_key, source_generation, input_digest, key_cells, \
                  value_cells, beta, decay_kind) \
                 VALUES ($1, $2, 'live', 1, decode(repeat('04', 32), 'hex'), \
                         decode(repeat('00', 16), 'hex'), decode(repeat('00', 16), 'hex'), \
                         0.5, 'none')"
            ),
            &[&memory, &held_seq],
        )
        .await
        .unwrap();
    let request = write_request("live", [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
    let memory_id = memory.to_owned();
    let appender = tokio::spawn(async move {
        let result = substrate
            .append_write(&memory_id, ptr_fastmem::WriteSeq(racing_seq), &request)
            .await;
        (substrate, result)
    });
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    assert!(
        !appender.is_finished(),
        "the append must wait for the memory row"
    );
    transaction.commit().await.unwrap();
    appender.await.unwrap()
}

#[tokio::test]
async fn an_append_that_waited_for_another_sees_its_write() {
    let mut substrate = substrate().await;
    substrate.replay(&[capsule(1, "live", 1)]).await.unwrap();
    // Out of order: seq 3 commits while seq 2 waits.
    memory_with(&substrate, "ordered", 64).await;
    let first = write_request("live", [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
    substrate
        .append_write("ordered", ptr_fastmem::WriteSeq(1), &first)
        .await
        .unwrap();
    let (mut substrate, result) = race_an_append(substrate, "ordered", 3, 2).await;
    assert_eq!(
        result,
        Err(PgError::InvalidWrite {
            memory: "ordered".into(),
            reason: "the sequence number does not follow the journal"
        })
    );
    // Over capacity: the journal fills while the append waits.
    memory_with(&substrate, "bounded", 2).await;
    substrate
        .append_write("bounded", ptr_fastmem::WriteSeq(1), &first)
        .await
        .unwrap();
    let (substrate, result) = race_an_append(substrate, "bounded", 2, 3).await;
    assert_eq!(
        result,
        Err(PgError::InvalidWrite {
            memory: "bounded".into(),
            reason: "the journal is full"
        })
    );
    assert_eq!(substrate.load_journal("bounded").await.unwrap().len(), 2);
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn superseding_a_source_removes_its_fast_memory_writes() {
    let mut substrate = substrate().await;
    let log = vec![capsule(1, "c1", 1), supersede(2, "c1", 1, 2)];
    let chain = anchors(&log);
    substrate.apply_committed(&log[0], chain[0]).await.unwrap();
    let config = memory_with(&substrate, "m1", 64).await;
    let request = write_request("c1", [1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]);
    let mut memory = FastMemory::new(config).unwrap();
    let receipt = memory.write(request.clone()).unwrap();
    substrate
        .append_write("m1", receipt.seq, &request)
        .await
        .unwrap();
    substrate
        .put_checkpoint("m1", memory.binding_digest(), memory.state())
        .await
        .unwrap();
    let report = substrate.apply_committed(&log[1], chain[1]).await.unwrap();
    assert_eq!(report.removed_writes, 1);
    assert_eq!(report.dropped_checkpoints, 1);
    assert!(substrate.load_journal("m1").await.unwrap().is_empty());
    assert_eq!(substrate.latest_checkpoint("m1").await.unwrap(), None);
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn a_checkpoint_that_does_not_fold_the_stored_journal_is_refused_or_skipped() {
    let mut substrate = substrate().await;
    let log = vec![
        capsule(1, "keep", 1),
        capsule(2, "gone", 1),
        revoke(3, "gone", 1),
    ];
    let chain = anchors(&log);
    substrate.apply_committed(&log[0], chain[0]).await.unwrap();
    substrate.apply_committed(&log[1], chain[1]).await.unwrap();
    let config = memory_with(&substrate, "m1", 64).await;
    let keep = write_request("keep", [1.0, 0.0, 0.0, 0.0], [0.5, 0.1, 0.0, 0.0]);
    let gone = write_request("gone", [0.0, 1.0, 0.0, 0.0], [0.9, 0.9, 0.9, 0.9]);
    let mut memory = FastMemory::new(config).unwrap();
    for request in [&keep, &gone] {
        let receipt = memory.write(request.clone()).unwrap();
        substrate
            .append_write("m1", receipt.seq, request)
            .await
            .unwrap();
    }
    // A checkpointer folded both writes; the revocation commits before it stores.
    let stale_digest = memory.binding_digest();
    let stale_state = memory.state().clone();
    substrate.apply_committed(&log[2], chain[2]).await.unwrap();
    assert!(matches!(
        substrate
            .put_checkpoint("m1", stale_digest, &stale_state)
            .await,
        Err(PgError::InvalidCheckpoint { .. })
    ));

    // A fold of the surviving prefix is accepted; a wrong binding is not.
    let mut clean = FastMemory::new(config).unwrap();
    clean.write(keep.clone()).unwrap();
    assert!(matches!(
        substrate.put_checkpoint("m1", [0; 32], clean.state()).await,
        Err(PgError::InvalidCheckpoint { .. })
    ));
    substrate
        .put_checkpoint("m1", clean.binding_digest(), clean.state())
        .await
        .unwrap();

    // A stale row written behind the substrate's back is never handed out.
    let raw = raw_client().await;
    let work = substrate.schemas().work.clone();
    raw.execute(
        &format!(
            "INSERT INTO {work}.fastmem_checkpoint (memory, applied_seq, binding_digest, state) \
             VALUES ('m1', 2, $1, $2)"
        ),
        &[
            &stale_digest.to_vec(),
            &ptr_fastmem::encode_state(&stale_state),
        ],
    )
    .await
    .unwrap();
    let checkpoint = substrate.latest_checkpoint("m1").await.unwrap().unwrap();
    assert_eq!(checkpoint.applied.0, 1);
    assert_eq!(checkpoint.binding_digest, clean.binding_digest());
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn vector_search_is_not_truncated_at_the_default_candidate_list() {
    let mut substrate = substrate().await;
    substrate.register_space(&space()).await.unwrap();
    let raw = raw_client().await;
    let (projection, derived) = (
        substrate.schemas().projection.clone(),
        substrate.schemas().derived.clone(),
    );
    raw.batch_execute(&format!(
        "INSERT INTO {projection}.live_generation (target, generation, project, commit_index) \
             SELECT 'c' || i, 1, 'atlas', 1 FROM generate_series(1, 2000) i; \
         INSERT INTO {derived}.search_document \
             (capsule, generation, project, content_digest, body, space, embedding, \
              indexed_at_commit) \
             SELECT 'c' || i, 1, 'atlas', decode(repeat('00', 32), 'hex'), 'doc', 'toy4', \
                    ARRAY[1 + random(), random(), random(), random()]::real[]::halfvec(4), 0 \
             FROM generate_series(1, 2000) i; \
         ANALYZE {derived}.search_document;"
    ))
    .await
    .unwrap();
    for project in [None, Some(ProjectId::from("atlas"))] {
        let query = HybridQuery {
            project,
            text: None,
            embedding: Some((space().id, vec![1.0, 0.5, 0.5, 0.5])),
            limit: 100,
            lexical_weight: 1.0,
            vector_weight: 1.0,
            rank_constant: 60.0,
        };
        let results = substrate.search(&query).await.unwrap();
        assert_eq!(results.vector.len(), 100, "{:?}", query.project);
    }
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn strings_postgresql_text_cannot_hold_are_refused_before_anything_is_written() {
    let mut substrate = substrate().await;
    let mut branch = sealed_branch("b1", "agent-7");
    branch.ops.push(BranchOp::Put {
        key: "order:1".into(),
        value: SemanticValue::from("a\0b"),
    });
    assert_eq!(
        substrate.store_branch(&branch).await.unwrap_err(),
        PgError::InvalidText {
            field: "branch_op.value_text"
        }
    );
    assert_eq!(substrate.load_branch(&branch.id).await.unwrap(), None);

    let record = capsule(1, "c\0", 1);
    let chain = anchors(std::slice::from_ref(&record));
    assert!(matches!(
        substrate.apply_committed(&record, chain[0]).await,
        Err(PgError::InvalidRecord { index: 1, .. })
    ));
    assert_eq!(substrate.watermark().await.unwrap(), LogAnchor::empty());
    substrate.drop_all().await.unwrap();
}
