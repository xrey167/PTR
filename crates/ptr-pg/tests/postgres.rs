//! Integration tests against a real PostgreSQL server.
//!
//! `PTR_PG_TEST_DSN` must name a loopback PostgreSQL 16+ server where the
//! connecting role may create schemas and where pgvector 0.8+ is installed or
//! installable. Every test works in its own schema prefix and drops it at the
//! end, so tests run in parallel against one database.

use std::future::Future;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use ptr_analytics::{Grouping, Metric, MetricRow, MetricSpec, Window};
use ptr_branch::{
    AutoThreshold, BranchId, BranchOp, CalibrationSample, InputsDigest, PolicyRecord, RangeDigest,
    SealedBranch, ThresholdRule, TriageDecision, TriageOutcome, TriagePolicy, ValueDigest,
};
use ptr_fastmem::{Decay, FastMemory, FastMemoryConfig, SourceRef, WriteRequest};
use ptr_ledger::integrity::{chain_anchors, LogAnchor};
use ptr_ledger::{CommittedEvent, LedgerEvent};
use ptr_lineage::{AdapterId, InterferenceReport, LayerInterference};
use ptr_pg::{
    BranchOutcome, EmbeddingSpace, FastMemoryRecord, HybridQuery, Identifier, PgError, PgSubstrate,
    SchemaSet, SearchDocument, LEXICAL_BACKEND, VECTOR_BACKEND, WORK_MIGRATIONS,
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
    let mut substrate = unmigrated_substrate_at(dsn).await;
    substrate.migrate().await.unwrap();
    substrate
}

/// A substrate in a fresh schema prefix with nothing migrated yet, on a
/// server where pgvector is installed.
async fn unmigrated_substrate_at(dsn: &str) -> PgSubstrate {
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
    let substrate = PgSubstrate::connect_with(dsn, schemas).await.unwrap();
    // A previous run with the same process id may have left schemas behind.
    raw.batch_execute(&format!(
        "DROP SCHEMA IF EXISTS {prefix}_derived CASCADE; \
         DROP SCHEMA IF EXISTS {prefix}_projection CASCADE; \
         DROP SCHEMA IF EXISTS {prefix}_work CASCADE;"
    ))
    .await
    .unwrap();
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
async fn a_work_schema_holding_triage_rows_upgrades_and_keeps_their_unrecorded_policies() {
    let mut substrate = unmigrated_substrate_at(&dsn()).await;
    let raw = raw_client().await;
    let work = substrate.schemas().work.clone();
    // A work schema as a build before recorded policies left it: migrations 1
    // to 4 applied, and a triage row citing a policy no table recorded.
    raw.batch_execute(&format!(
        "CREATE SCHEMA {work}; \
         CREATE TABLE {work}.schema_migration ( \
             version integer PRIMARY KEY, \
             name text NOT NULL, \
             checksum bytea NOT NULL, \
             applied_at timestamptz NOT NULL DEFAULT now())"
    ))
    .await
    .unwrap();
    for migration in &WORK_MIGRATIONS[..4] {
        raw.batch_execute(&migration.render(substrate.schemas()))
            .await
            .unwrap();
        raw.execute(
            &format!(
                "INSERT INTO {work}.schema_migration (version, name, checksum) \
                 VALUES ($1, $2, $3)"
            ),
            &[
                &(migration.version as i32),
                &migration.name,
                &migration.checksum().to_vec(),
            ],
        )
        .await
        .unwrap();
    }
    raw.batch_execute(&format!(
        "INSERT INTO {work}.branch (id, author, base_revision) VALUES ('legacy', 'agent-a', 0); \
         INSERT INTO {work}.branch_triage \
             (branch, decision, eligible, calibration_slice, score, auto_propensity, \
              policy_version) \
         VALUES ('legacy', 'escalate', true, false, 0.5, 0.0, 'policy-0')"
    ))
    .await
    .unwrap();

    let report = substrate.migrate().await.unwrap();
    assert_eq!(
        report.work,
        (5..=WORK_MIGRATIONS.len() as u32).collect::<Vec<_>>()
    );
    // The legacy row keeps citing the version it was logged under.
    let cited: String = raw
        .query_one(
            &format!("SELECT policy_version FROM {work}.branch_triage WHERE branch = 'legacy'"),
            &[],
        )
        .await
        .unwrap()
        .get(0);
    assert_eq!(cited, "policy-0");
    // A triage logged from now on must cite a recorded policy.
    substrate
        .store_branch(&sealed_branch("new", "agent-a"))
        .await
        .unwrap();
    assert!(matches!(
        substrate
            .record_triage(
                &BranchId::from("new"),
                &triage(TriageDecision::Escalate, true, false),
                "policy-0"
            )
            .await,
        Err(PgError::Database { ref sqlstate, .. }) if sqlstate == "23503"
    ));
    substrate.drop_all().await.unwrap();
}

/// Whether any session of the test database holds the migration lock of
/// `schemas`. `pg_locks` shows a bigint advisory key as its high half
/// (`classid`) and its low half (`objid`), with `objsubid` 1.
async fn migration_lock_held(raw: &tokio_postgres::Client, schemas: &SchemaSet) -> bool {
    raw.query_one(
        "SELECT EXISTS (SELECT 1 FROM pg_locks \
         WHERE locktype = 'advisory' AND granted AND objsubid = 1 \
           AND database = (SELECT oid FROM pg_database WHERE datname = current_database()) \
           AND ((classid::bigint << 32) | objid::bigint) = hashtext($1)::bigint)",
        &[&format!("ptr-pg:{}", schemas.projection)],
    )
    .await
    .unwrap()
    .get(0)
}

/// A third session holding the projection's migration table, so a schema
/// change that has taken the migration lock blocks at its first statement on
/// that table. `ROLLBACK` on the returned session releases it.
async fn block_schema_changes(schemas: &SchemaSet) -> tokio_postgres::Client {
    let blocker = raw_client().await;
    blocker
        .batch_execute(&format!(
            "BEGIN; LOCK TABLE {}.schema_migration IN ACCESS EXCLUSIVE MODE",
            schemas.projection
        ))
        .await
        .unwrap();
    blocker
}

/// Drive `change` until it holds the migration lock of `schemas` (it cannot
/// finish while [`block_schema_changes`] holds the table), then drop it where
/// it stands, as a caller's timeout or an aborted task does.
async fn cancel_once_locked(
    change: impl Future,
    raw: &tokio_postgres::Client,
    schemas: &SchemaSet,
) {
    let mut change = std::pin::pin!(change);
    for _ in 0..500 {
        let finished = tokio::time::timeout(Duration::from_millis(20), change.as_mut()).await;
        assert!(finished.is_err(), "a blocked schema change finished");
        if migration_lock_held(raw, schemas).await {
            return;
        }
    }
    panic!("the schema change never took the migration lock");
}

#[tokio::test]
async fn a_migration_cancelled_while_holding_the_lock_does_not_block_the_next_migrator() {
    let mut first = substrate().await;
    let schemas = first.schemas().clone();
    let raw = raw_client().await;
    let blocker = block_schema_changes(&schemas).await;
    cancel_once_locked(first.migrate(), &raw, &schemas).await;
    blocker.batch_execute("ROLLBACK").await.unwrap();

    // The first substrate's session stays open; the lock went with the
    // cancelled migration's own session.
    let mut second = PgSubstrate::connect_with(&dsn(), schemas.clone())
        .await
        .unwrap();
    let report = tokio::time::timeout(Duration::from_secs(30), second.migrate())
        .await
        .expect("a cancelled migration must not keep the migration lock")
        .unwrap();
    assert!(report.projection.is_empty() && report.derived.is_empty() && report.work.is_empty());
    assert!(!migration_lock_held(&raw, &schemas).await);
    first.capabilities().await.unwrap();
    drop(second);
    first.drop_all().await.unwrap();
}

#[tokio::test]
async fn a_migration_retried_after_a_cancellation_completes_and_leaves_no_lock_held() {
    let mut substrate = substrate().await;
    let schemas = substrate.schemas().clone();
    let raw = raw_client().await;
    let blocker = block_schema_changes(&schemas).await;
    cancel_once_locked(substrate.migrate(), &raw, &schemas).await;
    blocker.batch_execute("ROLLBACK").await.unwrap();

    // Every attempt takes the lock afresh, so the retry never re-enters one
    // the cancelled attempt still holds and one unlock releases it.
    let report = tokio::time::timeout(Duration::from_secs(30), substrate.migrate())
        .await
        .expect("a retry must not wait for the cancelled attempt")
        .unwrap();
    assert!(report.projection.is_empty() && report.derived.is_empty() && report.work.is_empty());
    assert!(!migration_lock_held(&raw, &schemas).await);
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn a_rebuild_cancelled_while_holding_the_lock_releases_it_and_keeps_the_schemas() {
    let mut first = substrate().await;
    let schemas = first.schemas().clone();
    let raw = raw_client().await;
    let blocker = block_schema_changes(&schemas).await;
    cancel_once_locked(first.rebuild_projection(), &raw, &schemas).await;
    blocker.batch_execute("ROLLBACK").await.unwrap();

    // Another migrator gets the lock, and finds nothing to apply: the
    // cancelled rebuild's drop was rolled back, not left half done.
    let mut second = PgSubstrate::connect_with(&dsn(), schemas.clone())
        .await
        .unwrap();
    let report = tokio::time::timeout(Duration::from_secs(30), second.migrate())
        .await
        .expect("a cancelled rebuild must not keep the migration lock")
        .unwrap();
    assert!(report.projection.is_empty() && report.derived.is_empty() && report.work.is_empty());
    drop(second);

    // A retry on the same substrate rebuilds and leaves no lock behind.
    let report = tokio::time::timeout(Duration::from_secs(30), first.rebuild_projection())
        .await
        .expect("a retried rebuild must not wait for the cancelled attempt")
        .unwrap();
    assert_eq!(report.projection, vec![1]);
    assert_eq!(report.derived, vec![1]);
    assert!(!migration_lock_held(&raw, &schemas).await);
    first.drop_all().await.unwrap();
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
        touched_base: [
            (
                "order:1".to_owned(),
                ValueDigest::of("order:1", Some(&SemanticValue::from("open"))).unwrap(),
            ),
            (
                "stock:widget".to_owned(),
                ValueDigest::of("stock:widget", None).unwrap(),
            ),
        ]
        .into_iter()
        .collect(),
        touched_inputs: [
            (
                "order:1".to_owned(),
                InputsDigest::of("order:1", ["order:2"]),
            ),
            (
                "stock:widget".to_owned(),
                InputsDigest::of("stock:widget", []),
            ),
        ]
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

#[tokio::test]
async fn touched_input_sets_survive_a_round_trip_and_a_branch_sealed_before_them_is_refused() {
    let mut substrate = substrate().await;
    let branch = sealed_branch("b1", "agent-7");
    substrate.store_branch(&branch).await.unwrap();
    let loaded = substrate.load_branch(&branch.id).await.unwrap().unwrap();
    assert_eq!(loaded.touched_inputs, branch.touched_inputs);
    // Every touched row carries its digest, the empty input set's included.
    let raw = raw_client().await;
    let work = substrate.schemas().work.clone();
    let stored: Vec<(String, Vec<u8>)> = raw
        .query(
            &format!(
                "SELECT key, inputs_digest FROM {work}.branch_touched \
                 WHERE branch = 'b1' ORDER BY key"
            ),
            &[],
        )
        .await
        .unwrap()
        .into_iter()
        .map(|row| (row.get(0), row.get(1)))
        .collect();
    let expected: Vec<(String, Vec<u8>)> = branch
        .touched_inputs
        .iter()
        .map(|(key, digest)| (key.clone(), digest.as_bytes().to_vec()))
        .collect();
    assert_eq!(stored, expected);

    // A touched key without an input-set digest, or an input-set digest for
    // a key without a base digest, is refused before any row is written.
    let mut partial = sealed_branch("b2", "agent-7");
    partial.touched_inputs.remove("stock:widget");
    let mut extra = sealed_branch("b3", "agent-7");
    extra
        .touched_inputs
        .insert("tags:1".into(), InputsDigest::of("tags:1", []));
    for refused in [partial, extra] {
        let error = substrate.store_branch(&refused).await.unwrap_err();
        assert!(
            matches!(error, PgError::InvalidBranch { ref branch, .. } if *branch == refused.id.0),
            "{error:?}"
        );
        assert_eq!(error.code(), "PTR_PG_INVALID_BRANCH");
        assert_eq!(substrate.load_branch(&refused.id).await.unwrap(), None);
    }

    // A branch stored before input sets were recorded, as the earlier schema
    // wrote it, cannot be certified: loading it says so rather than returning
    // a branch certification would have to trust.
    raw.batch_execute(&format!(
        "INSERT INTO {work}.branch (id, author, base_revision) VALUES ('legacy', 'agent-7', 4); \
         INSERT INTO {work}.branch_touched (branch, key, base_digest) \
         VALUES ('legacy', 'order:1', decode(repeat('00', 32), 'hex'));"
    ))
    .await
    .unwrap();
    let error = substrate
        .load_branch(&BranchId::from("legacy"))
        .await
        .unwrap_err();
    assert_eq!(
        error,
        PgError::BranchWithoutInputSets {
            branch: "legacy".into(),
            key: "order:1".into(),
        }
    );
    assert_eq!(error.code(), "PTR_PG_BRANCH_WITHOUT_INPUT_SETS");
    assert!(error.to_string().contains("must be re-run"), "{error}");

    // The column holds a whole digest or nothing.
    let error = raw
        .execute(
            &format!(
                "INSERT INTO {work}.branch_touched (branch, key, base_digest, inputs_digest) \
                 VALUES ('legacy', 'order:2', decode(repeat('00', 32), 'hex'), \
                         decode(repeat('00', 31), 'hex'))"
            ),
            &[],
        )
        .await
        .unwrap_err();
    assert_eq!(error.as_db_error().unwrap().code().code(), "23514");
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

/// Record the manual policy a test's triage rows cite.
async fn record_manual_policy(substrate: &mut PgSubstrate, version: &str) {
    let policy = TriagePolicy::new(AutoThreshold::Never, 0.1).unwrap();
    substrate
        .record_policy(&PolicyRecord::manual(version, policy).unwrap())
        .await
        .unwrap();
}

#[tokio::test]
async fn triage_logs_and_outcomes_feed_the_platform_metrics() {
    let mut substrate = substrate().await;
    record_manual_policy(&mut substrate, "policy-1").await;
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
        window: Window::All,
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
            window: Window::All,
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

#[tokio::test]
async fn revert_share_counts_merged_branches_later_reverted_within_a_window() {
    let mut substrate = substrate().await;
    record_manual_policy(&mut substrate, "policy-r").await;
    for (id, author) in [
        ("r1", "agent-a"),
        ("r2", "agent-a"),
        ("r3", "agent-b"),
        ("r5", "agent-b"),
        ("r6", "agent-a"),
    ] {
        substrate
            .store_branch(&sealed_branch(id, author))
            .await
            .unwrap();
        substrate
            .record_triage(
                &BranchId::from(id),
                &triage(TriageDecision::AutoPropose, true, false),
                "policy-r",
            )
            .await
            .unwrap();
    }
    for (id, outcome) in [
        ("r1", BranchOutcome::Merged(CommitIndex(10))),
        ("r1", BranchOutcome::Reverted(CommitIndex(12))),
        ("r2", BranchOutcome::Merged(CommitIndex(11))),
        ("r5", BranchOutcome::Merged(CommitIndex(14))),
        ("r6", BranchOutcome::Merged(CommitIndex(15))),
    ] {
        substrate
            .record_outcome(&BranchId::from(id), outcome)
            .await
            .unwrap();
    }
    // A merge and a triage from a month ago, as the store's clock recorded
    // them; the month-old merge is reverted today.
    let raw = raw_client().await;
    let work = substrate.schemas().work.clone();
    raw.execute(
        &format!(
            "INSERT INTO {work}.branch_outcome (branch, outcome, commit_index, observed_at) \
             VALUES ('r3', 'merged', 5, now() - interval '30 days')"
        ),
        &[],
    )
    .await
    .unwrap();
    substrate
        .record_outcome(
            &BranchId::from("r3"),
            BranchOutcome::Reverted(CommitIndex(13)),
        )
        .await
        .unwrap();
    substrate
        .store_branch(&sealed_branch("r4", "agent-b"))
        .await
        .unwrap();
    raw.execute(
        &format!(
            "INSERT INTO {work}.branch_triage (branch, decision, eligible, calibration_slice, \
             score, auto_propensity, policy_version, decided_at) \
             VALUES ('r4', 'escalate', true, false, 0.2, 0.9, 'policy-r', \
                     now() - interval '30 days')"
        ),
        &[],
    )
    .await
    .unwrap();
    // Merges inside the window whose reverts carry stamps outside it: nothing
    // orders the stamps of two records, and a merge counts as reverted
    // whenever its revert was stamped.
    raw.execute(
        &format!(
            "INSERT INTO {work}.branch_outcome (branch, outcome, commit_index, observed_at) \
             VALUES ('r5', 'reverted', 16, now() - interval '10 days'), \
                    ('r6', 'reverted', 17, now() - interval '10 days')"
        ),
        &[],
    )
    .await
    .unwrap();

    let spec = |metric, grouping, window| MetricSpec {
        metric,
        grouping,
        window,
    };
    let row = |group: &str, numerator, denominator| MetricRow {
        group: group.into(),
        numerator,
        denominator,
    };
    let revert = |grouping, window| spec(Metric::RevertShare, grouping, window);
    assert_eq!(
        substrate
            .metric(revert(Grouping::Overall, Window::All))
            .await
            .unwrap(),
        vec![row("", 4, 5)]
    );
    // Only merges within the window enter the denominator, and each of them
    // counts as reverted wherever its revert's stamp falls: the month-old
    // merge reverted today is outside, the reverts stamped before the window
    // of their merges count.
    assert_eq!(
        substrate
            .metric(revert(Grouping::Overall, Window::LastDays(7)))
            .await
            .unwrap(),
        vec![row("", 3, 4)]
    );
    assert_eq!(
        substrate
            .metric(revert(Grouping::ByPrincipal, Window::All))
            .await
            .unwrap(),
        vec![row("agent-a", 2, 3), row("agent-b", 2, 2)]
    );
    assert_eq!(
        substrate
            .metric(revert(Grouping::ByPolicyVersion, Window::LastDays(7)))
            .await
            .unwrap(),
        vec![row("policy-r", 3, 4)]
    );
    // A triage-based metric is windowed on the triage.
    let auto = |window| spec(Metric::AutoProposeShare, Grouping::Overall, window);
    assert_eq!(
        substrate.metric(auto(Window::All)).await.unwrap(),
        vec![row("", 5, 6)]
    );
    assert_eq!(
        substrate.metric(auto(Window::LastDays(7))).await.unwrap(),
        vec![row("", 5, 5)]
    );
    // A zero-day window counts nothing.
    assert_eq!(
        substrate
            .metric(revert(Grouping::Overall, Window::LastDays(0)))
            .await
            .unwrap(),
        vec![]
    );
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn every_metric_windows_a_branch_once_on_the_record_that_enters_its_denominator() {
    let mut substrate = substrate().await;
    record_manual_policy(&mut substrate, "policy-w").await;
    let raw = raw_client().await;
    let work = substrate.schemas().work.clone();
    // (branch, decision, eligible, calibration slice, days since the triage)
    let triages = [
        ("m1", "auto_propose", true, false, 3),
        ("m2", "auto_propose", true, false, 31),
        ("m3", "auto_propose", true, false, 4),
        ("m4", "auto_propose", true, false, 41),
        ("c1", "escalate", true, false, 11),
        ("c2", "discard", false, false, 2),
        ("s1", "escalate", true, true, 30),
        ("s2", "escalate", true, true, 30),
        ("s3", "escalate", true, true, 3),
        ("e1", "escalate", true, false, 2),
    ];
    for (id, decision, eligible, slice, days) in triages {
        substrate
            .store_branch(&sealed_branch(id, "agent-w"))
            .await
            .unwrap();
        let propensity: f64 = if eligible { 0.5 } else { 0.0 };
        raw.execute(
            &format!(
                "INSERT INTO {work}.branch_triage (branch, decision, eligible, \
                 calibration_slice, score, auto_propensity, policy_version, decided_at) \
                 VALUES ($1, $2, $3, $4, 0.5, $5, 'policy-w', \
                         now() - make_interval(days => $6))"
            ),
            &[&id, &decision, &eligible, &slice, &propensity, &days],
        )
        .await
        .unwrap();
    }
    // (branch, outcome, commit index, days since it was observed)
    let outcomes: [(&str, &str, Option<i64>, i32); 13] = [
        // Merged in the window and reverted since.
        ("m1", "merged", Some(1), 2),
        ("m1", "reverted", Some(5), 1),
        // Merged a month ago and reverted in the window.
        ("m2", "merged", Some(2), 30),
        ("m2", "reverted", Some(6), 1),
        ("m3", "merged", Some(3), 3),
        ("m4", "merged", Some(4), 40),
        // Conflicted before the window and discarded in it: the branch
        // entered the record before the window.
        ("c1", "conflicted", None, 10),
        ("c1", "discarded", None, 1),
        ("c2", "conflicted", None, 1),
        // Triaged a month ago and adjudicated in the window.
        ("s1", "adjudicated_harmful", None, 1),
        ("s2", "adjudicated_harmful", None, 20),
        ("s3", "adjudicated_harmless", None, 1),
        // Adjudicated in the window, but outside the calibration slice.
        ("e1", "adjudicated_harmful", None, 1),
    ];
    for (id, outcome, commit, days) in outcomes {
        raw.execute(
            &format!(
                "INSERT INTO {work}.branch_outcome (branch, outcome, commit_index, observed_at) \
                 VALUES ($1, $2, $3, now() - make_interval(days => $4))"
            ),
            &[&id, &outcome, &commit, &days],
        )
        .await
        .unwrap();
    }

    let counts = |metric: Metric, window: Window| {
        let substrate = &substrate;
        async move {
            let rows = substrate
                .metric(MetricSpec {
                    metric,
                    grouping: Grouping::Overall,
                    window,
                })
                .await
                .unwrap();
            assert_eq!(rows.len(), 1, "{metric:?} {window:?}: {rows:?}");
            (rows[0].numerator, rows[0].denominator)
        }
    };
    let week = Window::LastDays(7);
    // On the triage: auto-proposed m1..m4 over the eligible (all but c2),
    // and within the week m1 and m3 over m1, m3, s3 and e1.
    assert_eq!(counts(Metric::AutoProposeShare, Window::All).await, (4, 9));
    assert_eq!(counts(Metric::AutoProposeShare, week).await, (2, 4));
    // On the triage: escalated c1, s1, s2, s3 and e1 over all ten, and within
    // the week s3 and e1 over m1, m3, c2, s3 and e1.
    assert_eq!(counts(Metric::EscalationShare, Window::All).await, (5, 10));
    assert_eq!(counts(Metric::EscalationShare, week).await, (2, 5));
    // On the branch's first outcome, once per branch: c1 and c2 conflicted
    // over all ten; within the week c2 over m1, m3, c2, s1, s3 and e1. c1
    // and m2, whose first outcomes precede the week, are in neither count.
    assert_eq!(counts(Metric::ConflictRate, Window::All).await, (2, 10));
    assert_eq!(counts(Metric::ConflictRate, week).await, (1, 6));
    // On the adjudication, over the calibration slice only: s1 and s2
    // harmful over s1, s2 and s3; within the week s1 over s1 and s3.
    assert_eq!(
        counts(Metric::AdjudicatedHarmRate, Window::All).await,
        (2, 3)
    );
    assert_eq!(counts(Metric::AdjudicatedHarmRate, week).await, (1, 2));
    // On the merge: m1 and m2 reverted over the four merges; within the week
    // m1 over m1 and m3, while m2's revert in the week does not bring back
    // its month-old merge.
    assert_eq!(counts(Metric::RevertShare, Window::All).await, (2, 4));
    assert_eq!(counts(Metric::RevertShare, week).await, (1, 2));
    substrate.drop_all().await.unwrap();
}

/// Store a branch, log it as a calibration-slice escalation under `policy`
/// with `score`, and optionally adjudicate it.
async fn calibration_branch(
    substrate: &mut PgSubstrate,
    id: &str,
    score: f32,
    policy: &str,
    harmful: Option<bool>,
) {
    substrate
        .store_branch(&sealed_branch(id, "agent-c"))
        .await
        .unwrap();
    let outcome = TriageOutcome {
        decision: TriageDecision::Escalate,
        eligible: true,
        calibration_slice: true,
        score,
        auto_propensity: 0.0,
    };
    substrate
        .record_triage(&BranchId::from(id), &outcome, policy)
        .await
        .unwrap();
    if let Some(harmful) = harmful {
        let verdict = if harmful {
            BranchOutcome::AdjudicatedHarmful
        } else {
            BranchOutcome::AdjudicatedHarmless
        };
        substrate
            .record_outcome(&BranchId::from(id), verdict)
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn triage_policies_record_their_calibration_and_hold_out_everything_else() {
    let mut substrate = substrate().await;
    // A triage row may only cite a recorded policy.
    substrate
        .store_branch(&sealed_branch("x", "agent-c"))
        .await
        .unwrap();
    assert!(matches!(
        substrate
            .record_triage(
                &BranchId::from("x"),
                &triage(TriageDecision::Discard, false, false),
                "unrecorded"
            )
            .await,
        Err(PgError::Database { ref sqlstate, .. }) if sqlstate == "23503"
    ));

    let bootstrap = PolicyRecord::manual(
        "bootstrap",
        TriagePolicy::new(AutoThreshold::Never, 0.999).unwrap(),
    )
    .unwrap();
    substrate.record_policy(&bootstrap).await.unwrap();
    assert_eq!(
        substrate.load_policy("bootstrap").await.unwrap(),
        Some(bootstrap)
    );
    assert_eq!(substrate.load_policy("none").await.unwrap(), None);

    // Forty adjudicated calibration-slice branches under the bootstrap policy.
    for i in 0..40 {
        let score = i as f32 / 40.0;
        calibration_branch(
            &mut substrate,
            &format!("c{i:02}"),
            score,
            "bootstrap",
            Some(score < 0.3),
        )
        .await;
    }
    let samples = substrate.adjudicated_samples().await.unwrap();
    assert_eq!(samples.len(), 40);
    let rule = ThresholdRule::LearnThenTest {
        alpha: 0.2,
        delta: 0.1,
    };
    let calibrated = PolicyRecord::calibrate("policy-2", rule, 0.05, &samples).unwrap();
    substrate.record_policy(&calibrated).await.unwrap();
    assert_eq!(
        substrate.load_policy("policy-2").await.unwrap(),
        Some(calibrated.clone())
    );

    // Adjudications after the calibration are the policy's held-out set.
    for i in 0..5 {
        calibration_branch(
            &mut substrate,
            &format!("h{i}"),
            0.9,
            "policy-2",
            Some(false),
        )
        .await;
    }
    let everything = substrate.adjudicated_samples().await.unwrap();
    assert_eq!(everything.len(), 45);
    let held_out = calibrated.held_out(&everything);
    assert_eq!(held_out.len(), 5);
    assert!(held_out.iter().all(|(branch, _)| branch.0.starts_with('h')));

    // A policy calibrated on a branch nobody adjudicated is refused whole.
    calibration_branch(&mut substrate, "pending", 0.5, "policy-2", None).await;
    let mut with_pending = samples.clone();
    let pending_sample = with_pending[0].1;
    with_pending.push((BranchId::from("pending"), pending_sample));
    let dishonest = PolicyRecord::calibrate("policy-3", rule, 0.05, &with_pending).unwrap();
    assert!(matches!(
        substrate.record_policy(&dishonest).await,
        Err(PgError::InvalidPolicy { ref version, .. }) if version == "policy-3"
    ));
    assert_eq!(substrate.load_policy("policy-3").await.unwrap(), None);

    // A recorded policy and its calibration set are never rewritten, not even
    // a policy nothing cites, and a branch a policy was calibrated on cannot
    // be deleted.
    record_manual_policy(&mut substrate, "unused").await;
    let raw = raw_client().await;
    let work = substrate.schemas().work.clone();
    for statement in [
        format!(
            "UPDATE {work}.triage_policy SET calibration_rate = 0.5 WHERE version = 'policy-2'"
        ),
        format!("DELETE FROM {work}.triage_policy WHERE version = 'unused'"),
        format!("UPDATE {work}.triage_policy_sample SET branch = 'h0' WHERE branch = 'c00'"),
        format!("DELETE FROM {work}.triage_policy_sample WHERE branch = 'c00'"),
    ] {
        let error = raw.execute(&statement, &[]).await.unwrap_err();
        assert_eq!(
            error.as_db_error().unwrap().code().code(),
            "23000",
            "{statement}"
        );
    }
    let error = raw
        .execute(&format!("DELETE FROM {work}.branch WHERE id = 'c00'"), &[])
        .await
        .unwrap_err();
    assert_eq!(error.as_db_error().unwrap().code().code(), "23503");
    substrate.drop_all().await.unwrap();
}

/// The refusal of a calibration branch that is not an adjudicated
/// calibration-slice branch.
const NOT_ADJUDICATED: &str = "a calibration branch is not an adjudicated calibration-slice branch";

/// The refusal of a record whose rule, rerun on the stored adjudications of
/// its calibration branches, chooses another threshold.
const NOT_REPRODUCED: &str =
    "the rule on the stored adjudications of the calibration branches chooses another threshold";

/// A calibration sample as the adjudication of an eligible triage at `score`.
fn sample(score: f32, harmful: bool) -> CalibrationSample {
    TriageOutcome {
        decision: TriageDecision::Escalate,
        eligible: true,
        calibration_slice: true,
        score,
        auto_propensity: 0.0,
    }
    .adjudicate(harmful)
    .unwrap()
}

#[tokio::test]
async fn a_policy_is_recorded_only_when_its_rule_on_the_stored_adjudications_chooses_it() {
    let mut substrate = substrate().await;
    record_manual_policy(&mut substrate, "bootstrap").await;
    for i in 0..40 {
        let score = i as f32 / 40.0;
        calibration_branch(
            &mut substrate,
            &format!("c{i:02}"),
            score,
            "bootstrap",
            Some(score < 0.3),
        )
        .await;
    }
    let samples = substrate.adjudicated_samples().await.unwrap();
    let rule = ThresholdRule::LearnThenTest {
        alpha: 0.2,
        delta: 0.1,
    };
    let honest = PolicyRecord::calibrate("honest", rule, 0.05, &samples).unwrap();

    // A threshold nobody computed from these adjudications, the stored
    // branches with their verdicts flipped, and one half's samples under the
    // other half's branches each claim a guarantee the stored evidence does
    // not give.
    let invented = PolicyRecord::from_parts(
        "invented",
        AutoThreshold::AtLeast(0.0),
        0.05,
        rule,
        honest.calibrated_on().to_vec(),
    )
    .unwrap();
    let flipped: Vec<(BranchId, CalibrationSample)> = (0..40)
        .map(|i| {
            let score = i as f32 / 40.0;
            (BranchId(format!("c{i:02}")), sample(score, score >= 0.3))
        })
        .collect();
    let flipped = PolicyRecord::calibrate("flipped", rule, 0.05, &flipped).unwrap();
    let misattributed: Vec<(BranchId, CalibrationSample)> = samples[..20]
        .iter()
        .zip(&samples[20..])
        .map(|((branch, _), (_, other))| (branch.clone(), *other))
        .collect();
    let misattributed =
        PolicyRecord::calibrate("misattributed", rule, 0.05, &misattributed).unwrap();
    for forged in [&invented, &flipped, &misattributed] {
        let stored: Vec<(BranchId, CalibrationSample)> = samples
            .iter()
            .filter(|(branch, _)| forged.calibrated_on().contains(branch))
            .cloned()
            .collect();
        let chosen = PolicyRecord::calibrate(forged.version(), rule, 0.05, &stored).unwrap();
        assert_ne!(forged.policy().threshold(), chosen.policy().threshold());
        assert_eq!(
            substrate.record_policy(forged).await,
            Err(PgError::InvalidPolicy {
                version: forged.version().to_owned(),
                reason: NOT_REPRODUCED,
            })
        );
        assert_eq!(substrate.load_policy(forged.version()).await.unwrap(), None);
    }

    // An adjudicated branch outside the calibration slice, and a slice branch
    // whose only outcome is a merge, are no calibration evidence, even paired
    // with the very sample their stored row would give.
    substrate
        .store_branch(&sealed_branch("outside", "agent-c"))
        .await
        .unwrap();
    substrate
        .record_triage(
            &BranchId::from("outside"),
            &triage(TriageDecision::Escalate, true, false),
            "bootstrap",
        )
        .await
        .unwrap();
    substrate
        .record_outcome(
            &BranchId::from("outside"),
            BranchOutcome::AdjudicatedHarmless,
        )
        .await
        .unwrap();
    calibration_branch(&mut substrate, "merged", 0.5, "bootstrap", None).await;
    substrate
        .record_outcome(
            &BranchId::from("merged"),
            BranchOutcome::Merged(CommitIndex(3)),
        )
        .await
        .unwrap();
    for (branch, score) in [("outside", 0.8), ("merged", 0.5)] {
        let mut with_branch = samples.clone();
        with_branch.push((BranchId::from(branch), sample(score, false)));
        let version = format!("with-{branch}");
        let record = PolicyRecord::calibrate(version.as_str(), rule, 0.05, &with_branch).unwrap();
        assert_eq!(
            substrate.record_policy(&record).await,
            Err(PgError::InvalidPolicy {
                version: version.clone(),
                reason: NOT_ADJUDICATED,
            })
        );
        assert_eq!(substrate.load_policy(&version).await.unwrap(), None);
    }

    substrate.record_policy(&honest).await.unwrap();
    assert_eq!(substrate.load_policy("honest").await.unwrap(), Some(honest));
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn a_calibration_set_is_complete_when_its_policy_commits_and_never_grows() {
    let mut substrate = substrate().await;
    record_manual_policy(&mut substrate, "bootstrap").await;
    for i in 0..12 {
        calibration_branch(
            &mut substrate,
            &format!("c{i:02}"),
            i as f32 / 12.0,
            "bootstrap",
            Some(i < 3),
        )
        .await;
    }
    let samples = substrate.adjudicated_samples().await.unwrap();
    let rule = ThresholdRule::ConformalRiskControl { alpha: 0.3 };
    let record = PolicyRecord::calibrate("policy-2", rule, 0.05, &samples[..10]).unwrap();
    substrate.record_policy(&record).await.unwrap();

    // A sample appended in a later transaction is refused, for a calibrated
    // and for a manual policy alike.
    let raw = raw_client().await;
    let work = substrate.schemas().work.clone();
    for (policy, branch) in [("policy-2", "c10"), ("bootstrap", "c11")] {
        let error = raw
            .execute(
                &format!(
                    "INSERT INTO {work}.triage_policy_sample (policy_version, branch) \
                     VALUES ($1, $2)"
                ),
                &[&policy, &branch],
            )
            .await
            .unwrap_err();
        assert_eq!(
            error.as_db_error().unwrap().code().code(),
            "23000",
            "{policy}"
        );
    }
    assert_eq!(
        substrate.load_policy("policy-2").await.unwrap(),
        Some(record.clone())
    );

    // A policy whose sample rows fall short of or exceed its calibration size
    // cannot commit: a shortfall is refused at COMMIT, an excess by the
    // statement that adds it, which leaves the transaction to roll back.
    for (size, branches) in [(2, &["c10"][..]), (1, &["c10", "c11"][..])] {
        let samples: String = branches
            .iter()
            .map(|branch| {
                format!(
                    "INSERT INTO {work}.triage_policy_sample (policy_version, branch) \
                     VALUES ('sized', '{branch}');"
                )
            })
            .collect();
        let error = raw
            .batch_execute(&format!(
                "BEGIN; \
                 INSERT INTO {work}.triage_policy \
                     (version, rule, calibration_rate, alpha, calibration_size) \
                 VALUES ('sized', 'conformal_risk_control', 0.05, 0.3, {size}); \
                 {samples} \
                 COMMIT;"
            ))
            .await
            .unwrap_err();
        assert_eq!(
            error.as_db_error().unwrap().code().code(),
            "23000",
            "size {size}"
        );
        raw.batch_execute("ROLLBACK").await.unwrap();
        assert_eq!(substrate.load_policy("sized").await.unwrap(), None);
    }

    // A calibration set that no longer has its recorded size (here past a
    // disabled check) is refused on load rather than returned.
    raw.batch_execute(&format!(
        "ALTER TABLE {work}.triage_policy_sample \
             DISABLE TRIGGER triage_policy_sample_calibration_size; \
         INSERT INTO {work}.triage_policy_sample (policy_version, branch) \
             VALUES ('policy-2', 'c10'); \
         ALTER TABLE {work}.triage_policy_sample \
             ENABLE TRIGGER triage_policy_sample_calibration_size;"
    ))
    .await
    .unwrap();
    assert!(matches!(
        substrate.load_policy("policy-2").await,
        Err(PgError::CorruptRow {
            table: "triage_policy",
            ..
        })
    ));
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn policy_rows_that_break_a_rule_level_or_size_constraint_are_refused() {
    let substrate = substrate().await;
    let raw = raw_client().await;
    let work = substrate.schemas().work.clone();
    for values in [
        // An empty version and an unknown rule.
        "('', 'manual', NULL, 0.1, NULL, NULL, 0)",
        "('p', 'bayes', NULL, 0.1, 0.2, NULL, 3)",
        // A manual threshold has no risk level; a calibrated one has one.
        "('p', 'manual', NULL, 0.1, 0.2, NULL, 0)",
        "('p', 'conformal_risk_control', NULL, 0.1, NULL, NULL, 3)",
        // Only learn-then-test has a confidence level.
        "('p', 'conformal_risk_control', NULL, 0.1, 0.2, 0.1, 3)",
        "('p', 'learn_then_test', NULL, 0.1, 0.2, NULL, 3)",
        // A threshold lies in [0, 1], a calibration rate in [0, 1), a risk or
        // confidence level in (0, 1).
        "('p', 'manual', 1.5, 0.1, NULL, NULL, 0)",
        "('p', 'manual', -0.1, 0.1, NULL, NULL, 0)",
        "('p', 'manual', NULL, 1.0, NULL, NULL, 0)",
        "('p', 'conformal_risk_control', NULL, 0.1, 1.0, NULL, 3)",
        "('p', 'learn_then_test', NULL, 0.1, 0.2, 0.0, 3)",
        // A manual threshold was calibrated on nothing, a calibrated one on
        // something, and no set has a negative size.
        "('p', 'manual', NULL, 0.1, NULL, NULL, 2)",
        "('p', 'conformal_risk_control', NULL, 0.1, 0.2, NULL, 0)",
        "('p', 'conformal_risk_control', NULL, 0.1, 0.2, NULL, -1)",
    ] {
        let error = raw
            .execute(
                &format!(
                    "INSERT INTO {work}.triage_policy \
                     (version, rule, threshold, calibration_rate, alpha, delta, \
                      calibration_size) \
                     VALUES {values}"
                ),
                &[],
            )
            .await
            .unwrap_err();
        assert_eq!(
            error.as_db_error().unwrap().code().code(),
            "23514",
            "{values}"
        );
    }
    substrate.drop_all().await.unwrap();
}

/// Rows of `table` the current transaction's scans have read so far, whatever
/// plan read them: sequential-scan tuples plus index-scan heap fetches.
async fn rows_read_in_transaction(raw: &tokio_postgres::Client, table: &str) -> i64 {
    raw.query_one(
        &format!(
            "SELECT seq_tup_read + idx_tup_fetch FROM pg_stat_xact_user_tables \
             WHERE relid = '{table}'::regclass"
        ),
        &[],
    )
    .await
    .unwrap()
    .get(0)
}

#[tokio::test]
async fn a_calibration_set_and_an_interference_report_are_counted_once_not_once_per_row() {
    const ROWS: i64 = 2000;
    let substrate = substrate().await;
    let raw = raw_client().await;
    let work = substrate.schemas().work.clone();
    raw.execute(
        &format!(
            "INSERT INTO {work}.branch (id, author, base_revision) \
             SELECT 'cal' || g, 'agent-a', 0 FROM generate_series(1, {ROWS}) AS g"
        ),
        &[],
    )
    .await
    .unwrap();
    insert_adapter(&raw, &work, "wide").await;
    // Each set is written in one statement, as record_policy and
    // record_interference write theirs, and SET CONSTRAINTS runs the checks
    // deferred to commit inside the transaction, where its scans are counted.
    // One count per set reads each row a bounded number of times; a count
    // per row, as the checks used to make, reads ROWS * (ROWS + 1) rows.
    for (header, rows, table) in [
        (
            format!(
                "INSERT INTO {work}.triage_policy \
                 (version, rule, calibration_rate, alpha, calibration_size) \
                 VALUES ('large', 'conformal_risk_control', 0.05, 0.3, {ROWS})"
            ),
            format!(
                "INSERT INTO {work}.triage_policy_sample (policy_version, branch) \
                 SELECT 'large', 'cal' || g FROM generate_series(1, {ROWS}) AS g"
            ),
            format!("{work}.triage_policy_sample"),
        ),
        (
            format!(
                "INSERT INTO {work}.adapter_interference_report (adapter, layer_count) \
                 VALUES ('wide', {ROWS})"
            ),
            format!(
                "INSERT INTO {work}.adapter_interference \
                 (adapter, layer, output_overlap, input_overlap, output_chance, input_chance) \
                 SELECT 'wide', 'layer' || g, 0.1, 0.1, 0.1, 0.1 \
                 FROM generate_series(1, {ROWS}) AS g"
            ),
            format!("{work}.adapter_interference"),
        ),
    ] {
        raw.batch_execute(&format!(
            "BEGIN; {header}; {rows}; SET CONSTRAINTS ALL IMMEDIATE;"
        ))
        .await
        .unwrap();
        let read = rows_read_in_transaction(&raw, &table).await;
        raw.batch_execute("COMMIT").await.unwrap();
        assert!(
            read <= 4 * ROWS,
            "{table}: {read} rows read to check {ROWS}"
        );
        let stored: i64 = raw
            .query_one(&format!("SELECT count(*) FROM {table}"), &[])
            .await
            .unwrap()
            .get(0);
        assert_eq!(stored, ROWS, "{table}");
    }
    substrate.drop_all().await.unwrap();
}

/// Insert a trained adapter into the lineage catalog.
async fn insert_adapter(raw: &tokio_postgres::Client, work: &Identifier, id: &str) {
    raw.execute(
        &format!(
            "INSERT INTO {work}.adapter (id, domain, base_model, base_revision, origin, rank, \
             artifact, artifact_sha256, data_fingerprint, status) \
             VALUES ($1, 'support', 'base', 'r1', 'trained', 8, $2, $3, $3, 'candidate')"
        ),
        &[&id, &format!("s3://adapters/{id}"), &vec![7u8; 32]],
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn a_labeling_function_names_an_adapter_only_as_a_model() {
    let substrate = substrate().await;
    let raw = raw_client().await;
    let work = substrate.schemas().work.clone();
    insert_adapter(&raw, &work, "ranker-v2").await;
    raw.execute(
        &format!(
            "INSERT INTO {work}.labeling_function (name, kind, adapter) \
             VALUES ('ranker', 'model', 'ranker-v2'), ('rule', 'heuristic', NULL)"
        ),
        &[],
    )
    .await
    .unwrap();
    for (statement, code) in [
        (
            format!(
                "INSERT INTO {work}.labeling_function (name, kind, adapter) \
                 VALUES ('keyword', 'heuristic', 'ranker-v2')"
            ),
            "23514",
        ),
        (
            format!(
                "INSERT INTO {work}.labeling_function (name, kind, adapter) \
                 VALUES ('ghost', 'model', 'no-such-adapter')"
            ),
            "23503",
        ),
    ] {
        let error = raw.execute(&statement, &[]).await.unwrap_err();
        assert_eq!(
            error.as_db_error().unwrap().code().code(),
            code,
            "{statement}"
        );
    }
    substrate.drop_all().await.unwrap();
}

fn layer(name: &str, overlap: f64, worst: Option<&str>) -> LayerInterference {
    LayerInterference {
        layer: name.into(),
        output_overlap: overlap,
        input_overlap: overlap / 2.0,
        output_chance: 0.0625,
        input_chance: 0.03125,
        worst: worst.map(AdapterId::from),
    }
}

#[tokio::test]
async fn interference_reports_are_stored_once_as_measured() {
    let mut substrate = substrate().await;
    let raw = raw_client().await;
    let work = substrate.schemas().work.clone();
    for id in ["a1", "a2", "a3"] {
        insert_adapter(&raw, &work, id).await;
    }
    // Layers come back in name order, exactly as measured.
    let report = InterferenceReport {
        layers: vec![layer("q", 0.4, Some("a1")), layer("k", 0.1, None)],
    };
    let stored = InterferenceReport {
        layers: vec![layer("k", 0.1, None), layer("q", 0.4, Some("a1"))],
    };
    let adapter = AdapterId::from("a2");
    substrate
        .record_interference(&adapter, &report)
        .await
        .unwrap();
    assert_eq!(
        substrate.load_interference(&adapter).await.unwrap(),
        Some(stored.clone())
    );
    assert_eq!(
        substrate
            .load_interference(&AdapterId::from("a1"))
            .await
            .unwrap(),
        None
    );
    // Stored once: a second report for the adapter is refused whether it
    // repeats, overlaps or avoids the stored layers, and never merges into
    // the first.
    for second in [
        report.clone(),
        InterferenceReport {
            layers: vec![layer("k", 0.2, None), layer("v", 0.3, None)],
        },
        InterferenceReport {
            layers: vec![layer("v", 0.3, None)],
        },
    ] {
        assert!(matches!(
            substrate.record_interference(&adapter, &second).await,
            Err(PgError::Database { ref sqlstate, .. }) if sqlstate == "23505"
        ));
        assert_eq!(
            substrate.load_interference(&adapter).await.unwrap(),
            Some(stored.clone())
        );
    }
    // Every overlap and chance level lies in [0, 1]; one layer outside it
    // refuses the whole report.
    let outside = |edit: fn(&mut LayerInterference)| {
        let mut refused = layer("v", 0.2, None);
        edit(&mut refused);
        InterferenceReport {
            layers: vec![layer("k", 0.2, None), refused],
        }
    };
    for impossible in [
        outside(|layer| layer.output_overlap = 1.5),
        outside(|layer| layer.input_overlap = -0.25),
        outside(|layer| layer.output_chance = f64::NAN),
        outside(|layer| layer.input_chance = 1.5),
    ] {
        assert!(matches!(
            substrate
                .record_interference(&AdapterId::from("a3"), &impossible)
                .await,
            Err(PgError::Database { ref sqlstate, .. }) if sqlstate == "23514"
        ));
        assert_eq!(
            substrate
                .load_interference(&AdapterId::from("a3"))
                .await
                .unwrap(),
            None
        );
    }
    // The worst overlap names another catalogued adapter.
    for (worst, code) in [("a3", "23514"), ("no-such-adapter", "23503")] {
        let naming = InterferenceReport {
            layers: vec![layer("q", 0.3, Some(worst))],
        };
        assert!(matches!(
            substrate
                .record_interference(&AdapterId::from("a3"), &naming)
                .await,
            Err(PgError::Database { ref sqlstate, .. }) if sqlstate == code
        ));
    }
    // A layer appended to a stored report in a later transaction is refused,
    // and a report whose layer rows do not match its layer count cannot
    // commit.
    let error = raw
        .execute(
            &format!(
                "INSERT INTO {work}.adapter_interference \
                 (adapter, layer, output_overlap, input_overlap, output_chance, input_chance) \
                 VALUES ('a2', 'v', 0.1, 0.1, 0.1, 0.1)"
            ),
            &[],
        )
        .await
        .unwrap_err();
    assert_eq!(error.as_db_error().unwrap().code().code(), "23000");
    let error = raw
        .batch_execute(&format!(
            "BEGIN; \
             INSERT INTO {work}.adapter_interference_report (adapter, layer_count) \
             VALUES ('a3', 2); \
             INSERT INTO {work}.adapter_interference \
             (adapter, layer, output_overlap, input_overlap, output_chance, input_chance) \
             VALUES ('a3', 'k', 0.1, 0.1, 0.1, 0.1); \
             COMMIT;"
        ))
        .await
        .unwrap_err();
    assert_eq!(error.as_db_error().unwrap().code().code(), "23000");
    assert_eq!(
        substrate
            .load_interference(&AdapterId::from("a3"))
            .await
            .unwrap(),
        None
    );
    // Neither a report nor any of its layers is ever updated or deleted.
    for statement in [
        format!("UPDATE {work}.adapter_interference SET output_overlap = 0 WHERE adapter = 'a2'"),
        format!("DELETE FROM {work}.adapter_interference WHERE adapter = 'a2'"),
        format!(
            "UPDATE {work}.adapter_interference_report SET recorded_at = now() \
             WHERE adapter = 'a2'"
        ),
        format!("DELETE FROM {work}.adapter_interference_report WHERE adapter = 'a2'"),
    ] {
        let error = raw.execute(&statement, &[]).await.unwrap_err();
        assert_eq!(
            error.as_db_error().unwrap().code().code(),
            "23000",
            "{statement}"
        );
    }
    // A report that no longer has its recorded layer count (here past a
    // disabled check) is refused on load rather than returned.
    raw.batch_execute(&format!(
        "ALTER TABLE {work}.adapter_interference \
             DISABLE TRIGGER adapter_interference_layer_count; \
         INSERT INTO {work}.adapter_interference \
             (adapter, layer, output_overlap, input_overlap, output_chance, input_chance) \
             VALUES ('a2', 'v', 0.1, 0.1, 0.1, 0.1); \
         ALTER TABLE {work}.adapter_interference \
             ENABLE TRIGGER adapter_interference_layer_count;"
    ))
    .await
    .unwrap();
    assert!(matches!(
        substrate.load_interference(&adapter).await,
        Err(PgError::CorruptRow {
            table: "adapter_interference",
            ..
        })
    ));
    substrate.drop_all().await.unwrap();
}

#[tokio::test]
async fn an_empty_interference_report_or_one_for_an_unknown_adapter_stores_nothing() {
    let mut substrate = substrate().await;
    let raw = raw_client().await;
    let work = substrate.schemas().work.clone();
    insert_adapter(&raw, &work, "a1").await;
    let adapter = AdapterId::from("a1");
    // An empty report is refused before anything is written, so it neither
    // claims the adapter's one report nor loads back as evidence.
    let error = substrate
        .record_interference(&adapter, &InterferenceReport { layers: vec![] })
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        PgError::InvalidInterference { adapter: ref refused, .. } if refused == "a1"
    ));
    assert_eq!(error.code(), "PTR_PG_INVALID_INTERFERENCE");
    let headers: i64 = raw
        .query_one(
            &format!("SELECT count(*) FROM {work}.adapter_interference_report"),
            &[],
        )
        .await
        .unwrap()
        .get(0);
    assert_eq!(headers, 0);
    assert_eq!(substrate.load_interference(&adapter).await.unwrap(), None);
    let report = InterferenceReport {
        layers: vec![layer("k", 0.1, None)],
    };
    substrate
        .record_interference(&adapter, &report)
        .await
        .unwrap();
    assert_eq!(
        substrate.load_interference(&adapter).await.unwrap(),
        Some(report.clone())
    );
    // An adapter the catalog does not know gets no report.
    let ghost = AdapterId::from("ghost");
    assert!(matches!(
        substrate.record_interference(&ghost, &report).await,
        Err(PgError::Database { ref sqlstate, .. }) if sqlstate == "23503"
    ));
    assert_eq!(substrate.load_interference(&ghost).await.unwrap(), None);
    substrate.drop_all().await.unwrap();
}
