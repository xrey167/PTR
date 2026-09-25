//! PostgreSQL substrate for PTR: one database, three kinds of schema, no second
//! authority.
//!
//! * **Projection** — a function of the committed ledger prefix: the key/value
//!   state entries, the lifecycle catalog (live generations and the tombstone
//!   set, keyed exactly as the runtime keys them), the revision map and an
//!   append-only projection event log. The projector is its only writer; it
//!   applies one commit at a time behind a watermark and verifies that every
//!   record chains from the anchor it last stored, so a discarded tail, a
//!   restored backup or a foreign log is refused instead of silently kept.
//!   Rebuild = drop and replay.
//! * **Derived** — caches computed outside the projection (search documents
//!   and their embeddings). Only live generations are indexed; the projector
//!   deletes a generation's row in the transaction that supersedes or revokes
//!   it. Every hit is still a search candidate that must be validated against
//!   the lifecycle authority before use.
//! * **Work** — non-authoritative working state: sealed agent branches and
//!   their triage log, fast-memory journals and checkpoints, the adapter
//!   lineage catalog and replay pool, and the weak-supervision store. Not
//!   derived from the ledger, never dropped by a rebuild, never read as
//!   authority.
//!
//! Without the `postgres-backend` feature this crate is the schemas, their
//! checksummed migration catalogs, capability parsing, the lifecycle mapping
//! and the metric compiler; with it, the async adapters on `PgSubstrate`.

mod capability;
mod config;
mod error;
mod event;
mod metric;
mod migrate;

#[cfg(feature = "postgres-backend")]
mod adapters;

pub use capability::{
    parse_version, Capabilities, LexicalBackend, MINIMUM_PGVECTOR, MINIMUM_SERVER,
};
pub use config::{check_identifier, Identifier, PgConfig, SchemaSet};
pub use error::PgError;
pub use event::{event_payload, event_subject, event_topic, lifecycle_change, LifecycleChange};
pub use metric::metric_sql;
pub use migrate::{
    check_catalog, plan_migrations, Migration, SchemaClass, DERIVED_MIGRATIONS,
    PROJECTION_MIGRATIONS, WORK_MIGRATIONS,
};

#[cfg(feature = "postgres-backend")]
pub use adapters::{
    BranchOutcome, EmbeddingSpace, FastMemoryCheckpoint, FastMemoryRecord, HybridQuery,
    MigrationReport, PgSubstrate, ProjectionApply, ProjectionEventRow, SearchDocument,
    SearchResults, LEXICAL_BACKEND, VECTOR_BACKEND,
};
