//! The PostgreSQL arm: the product (`ptr-pg`'s projector, `upsert_document` and `search`) driven
//! through its public API, and the raw SQL the harness owns (planting stale rows, the no-live-join
//! control, row checks, the index rebuild).
//!
//! Every call into the product runs under `catch_unwind`: a panic is a counted product panic
//! ([`Call::Panicked`]), the substrate is reconnected, and the run goes on. Setup that is not the
//! claim (migrating, applying the world's own events, planting) fails the run with exit status 2.

use std::future::Future;
use std::panic::{catch_unwind, AssertUnwindSafe};

use ptr_ledger::integrity::{chain_anchors, sha256, LogAnchor};
use ptr_ledger::{CommittedEvent, LedgerEvent};
use ptr_pg::{
    EmbeddingSpace, HybridQuery, Identifier, PgError, PgSubstrate, SearchDocument, SearchResults,
};
use ptr_types::{CapsuleId, CommitIndex, Generation, ProjectId};
use tokio::runtime::Runtime;
use tokio_postgres::Client;

use super::params;
use super::workload::{capsule_id, Event, EventKind};
use crate::experiments::pg::{self, Instance};

pub const PROJECT: &str = "q003";
pub const SPACE: &str = "minilm_q003";

/// (capsule, generation) pairs of a result list.
pub type Pairs = Vec<(String, u64)>;

/// What a call into the product did.
pub enum Call<T> {
    Done(T),
    Failed(PgError),
    Panicked,
}

/// Run `future` to completion on `runtime`, catching a panic of the product inside it.
fn guarded<T>(runtime: &Runtime, future: impl Future<Output = T>) -> Option<T> {
    catch_unwind(AssertUnwindSafe(|| runtime.block_on(future))).ok()
}

fn die(message: impl std::fmt::Display) -> ! {
    eprintln!("{}: {message}", params::BENCHMARK);
    std::process::exit(2);
}

pub struct PgArm {
    runtime: Runtime,
    raw: Client,
    instance: Instance,
    substrate: Option<PgSubstrate>,
    space: Identifier,
    next_index: u64,
    head: LogAnchor,
    pub server: String,
}

pub fn search_document(
    capsule: &str,
    generation: u64,
    body: &str,
    embedding: &[f32],
) -> SearchDocument {
    SearchDocument {
        capsule: CapsuleId::from(capsule),
        generation: Generation(generation),
        content_digest: sha256(body.as_bytes()),
        body: body.to_string(),
        embedding: Some((
            Identifier::new(SPACE).expect("the space id is an identifier"),
            embedding.to_vec(),
        )),
    }
}

impl PgArm {
    /// Create a fresh instance (schemas dropped first if a previous run left them) and register
    /// the embedding space. Returns the arm and the nanoseconds `register_space` took.
    pub fn open(seed: u64) -> (Self, u64) {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap_or_else(|error| die(format!("tokio runtime: {error}")));
        let (raw, server) = runtime.block_on(async {
            let raw = pg::raw_client().await;
            let server = pg::server_version(&raw).await;
            (raw, server)
        });
        // One schema set per run in this process, so runs side by side (the unit tests) never share one.
        static RUNS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let case = RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let instance = Instance::new("q003", seed, case);
        let mut substrate = runtime.block_on(instance.create(&raw));
        let space = Identifier::new(SPACE).expect("the space id is an identifier");
        let definition = EmbeddingSpace {
            id: space.clone(),
            model: params::EMBEDDING_MODEL.to_string(),
            revision: params::EMBEDDING_REVISION.to_string(),
            dims: params::EMBEDDING_DIMENSIONS as u16,
        };
        let started = std::time::Instant::now();
        if let Err(error) = runtime.block_on(substrate.register_space(&definition)) {
            die(format!("register_space failed: {error}"));
        }
        let register_ns = started.elapsed().as_nanos() as u64;
        (
            Self {
                runtime,
                raw,
                instance,
                substrate: Some(substrate),
                space,
                next_index: 1,
                head: LogAnchor::empty(),
                server,
            },
            register_ns,
        )
    }

    fn derived(&self) -> String {
        self.instance.schemas.derived.as_str().to_string()
    }

    fn reconnect(&mut self) {
        match guarded(&self.runtime, self.instance.connect()) {
            Some(substrate) => self.substrate = Some(substrate),
            None => die("cannot reconnect the substrate after a product panic"),
        }
    }

    /// Apply the world's own event to the projection with the anchor the ledger computes. The
    /// projector is part of the product: a refusal, a record it does not apply, or a panic is the
    /// caller's to count, and the chain does not advance on it.
    pub fn apply(&mut self, event: &Event) -> Call<()> {
        let capsule = capsule_id(event.cue);
        let ledger_event = match event.kind {
            EventKind::Commit => LedgerEvent::CapsuleCommitted {
                project: ProjectId::from(PROJECT),
                capsule: CapsuleId::from(capsule.as_str()),
                generation: Generation(1),
            },
            EventKind::Supersede { old, new } => LedgerEvent::CapsuleSuperseded {
                capsule: CapsuleId::from(capsule.as_str()),
                old: Generation(old),
                new: Generation(new),
            },
            EventKind::Revoke { generation } => LedgerEvent::Revoked {
                subject: capsule,
                generation: Generation(generation),
            },
        };
        let record = CommittedEvent {
            index: CommitIndex(self.next_index),
            event: ledger_event,
        };
        let anchor = chain_anchors(std::slice::from_ref(&record), self.head)
            .unwrap_or_else(|error| die(format!("cannot anchor a record: {error}")))
            .pop()
            .unwrap_or_else(|| die("the anchor encoder returned nothing"));
        let outcome = guarded(
            &self.runtime,
            self.substrate
                .as_mut()
                .expect("the substrate is connected")
                .apply_committed(&record, anchor),
        );
        match outcome {
            Some(Ok(report)) if report.outcome == ptr_state::ApplyOutcome::Applied => {
                self.head = anchor;
                self.next_index += 1;
                Call::Done(())
            }
            Some(Ok(report)) => Call::Failed(PgError::InvalidRecord {
                index: self.next_index,
                reason: format!(
                    "the projector did not apply the record: {:?}",
                    report.outcome
                ),
            }),
            Some(Err(error)) => Call::Failed(error),
            None => {
                self.reconnect();
                Call::Panicked
            }
        }
    }

    pub fn upsert(&mut self, document: &SearchDocument) -> Call<()> {
        let outcome = guarded(
            &self.runtime,
            self.substrate
                .as_mut()
                .expect("the substrate is connected")
                .upsert_document(document),
        );
        match outcome {
            Some(Ok(())) => Call::Done(()),
            Some(Err(error)) => Call::Failed(error),
            None => {
                self.reconnect();
                Call::Panicked
            }
        }
    }

    pub fn search(&mut self, text: &str, embedding: &[f32]) -> Call<SearchResults> {
        let query = HybridQuery {
            project: Some(ProjectId::from(PROJECT)),
            text: Some(text.to_string()),
            embedding: Some((self.space.clone(), embedding.to_vec())),
            limit: params::SEARCH_LIMIT,
            lexical_weight: params::LEXICAL_WEIGHT_PERMILLE as f32 / 1000.0,
            vector_weight: params::VECTOR_WEIGHT_PERMILLE as f32 / 1000.0,
            rank_constant: params::RANK_CONSTANT as f32,
        };
        let outcome = guarded(
            &self.runtime,
            self.substrate
                .as_mut()
                .expect("the substrate is connected")
                .search(&query),
        );
        match outcome {
            Some(Ok(results)) => Call::Done(results),
            Some(Err(error)) => Call::Failed(error),
            None => {
                self.reconnect();
                Call::Panicked
            }
        }
    }

    /// The product's own answer to "is this generation live and unrevoked", as of the head.
    pub fn admissible(&mut self, capsule: &str, generation: u64) -> Call<bool> {
        let fence = CommitIndex(self.head.index.0);
        let outcome = guarded(
            &self.runtime,
            self.substrate
                .as_mut()
                .expect("the substrate is connected")
                .is_admissible(capsule, Generation(generation), fence),
        );
        match outcome {
            Some(Ok(answer)) => Call::Done(answer),
            Some(Err(error)) => Call::Failed(error),
            None => {
                self.reconnect();
                Call::Panicked
            }
        }
    }

    /// Plant a derived row by raw SQL, bypassing `upsert_document`: the state an index that lags
    /// behind the ledger, or a failed delete, leaves behind.
    pub fn plant(&self, capsule: &str, generation: u64, body: &str, embedding: &[f32]) {
        let derived = self.derived();
        let digest = sha256(body.as_bytes()).to_vec();
        let generation = generation as i64;
        let watermark = self.head.index.0 as i64;
        let embedding = embedding.to_vec();
        let result = self.runtime.block_on(self.raw.execute(
            &format!(
                "INSERT INTO {derived}.search_document \
                 (capsule, generation, project, content_digest, body, space, embedding, \
                  indexed_at_commit) \
                 VALUES ($1, $2, $3, $4, $5, $6, $7::real[]::halfvec({}), $8) \
                 ON CONFLICT (capsule, generation) DO UPDATE \
                 SET project = EXCLUDED.project, content_digest = EXCLUDED.content_digest, \
                     body = EXCLUDED.body, space = EXCLUDED.space, \
                     embedding = EXCLUDED.embedding, \
                     indexed_at_commit = EXCLUDED.indexed_at_commit",
                params::EMBEDDING_DIMENSIONS
            ),
            &[
                &capsule,
                &generation,
                &PROJECT,
                &digest,
                &body,
                &SPACE,
                &embedding,
                &watermark,
            ],
        ));
        match result {
            Ok(1) => {}
            Ok(other) => die(format!(
                "planting {capsule}@{generation} wrote {other} rows"
            )),
            Err(error) => die(format!("planting {capsule}@{generation} failed: {error}")),
        }
    }

    /// Every (capsule, generation) the derived table holds.
    pub fn all_rows(&self) -> Pairs {
        let derived = self.derived();
        match self.runtime.block_on(self.raw.query(
            &format!("SELECT capsule, generation FROM {derived}.search_document"),
            &[],
        )) {
            Ok(rows) => rows
                .into_iter()
                .map(|row| (row.get::<_, String>(0), row.get::<_, i64>(1) as u64))
                .collect(),
            Err(error) => die(format!("listing the documents failed: {error}")),
        }
    }

    pub fn row_exists(&self, capsule: &str, generation: u64) -> bool {
        let derived = self.derived();
        let generation = generation as i64;
        let row = self.runtime.block_on(self.raw.query_one(
            &format!(
                "SELECT count(*) FROM {derived}.search_document WHERE capsule = $1 AND generation = $2"
            ),
            &[&capsule, &generation],
        ));
        match row {
            Ok(row) => row.get::<_, i64>(0) > 0,
            Err(error) => die(format!("counting rows failed: {error}")),
        }
    }

    /// The same two queries the product runs, without the lifecycle join: the negative control. It
    /// is the harness's SQL (exact full-text match, and the nearest rows by the index), so it
    /// returns whatever the derived table holds, planted rows included.
    pub fn search_without_live_join(&self, text: &str, embedding: &[f32]) -> (Pairs, Pairs) {
        let derived = self.derived();
        let limit = i64::from(params::SEARCH_LIMIT);
        let dims = params::EMBEDDING_DIMENSIONS;
        let embedding = embedding.to_vec();
        self.runtime.block_on(async {
            let lexical = self
                .raw
                .query(
                    &format!(
                        "SELECT d.capsule, d.generation FROM {derived}.search_document d \
                         CROSS JOIN plainto_tsquery('simple'::regconfig, $1) q \
                         WHERE d.lexeme @@ q AND d.project = $2 \
                         ORDER BY ts_rank_cd(d.lexeme, q) DESC, d.capsule, d.generation LIMIT $3"
                    ),
                    &[&text, &PROJECT, &limit],
                )
                .await
                .unwrap_or_else(|error| die(format!("control lexical query failed: {error}")));
            let vector = self
                .raw
                .query(
                    &format!(
                        "SELECT d.capsule, d.generation FROM {derived}.search_document d \
                         WHERE d.space = '{SPACE}' AND d.project = $2 \
                         ORDER BY d.embedding::halfvec({dims}) <=> $1::real[]::halfvec({dims}) \
                         LIMIT $3"
                    ),
                    &[&embedding, &PROJECT, &limit],
                )
                .await
                .unwrap_or_else(|error| die(format!("control vector query failed: {error}")));
            let pairs = |rows: Vec<tokio_postgres::Row>| {
                rows.into_iter()
                    .map(|row| (row.get::<_, String>(0), row.get::<_, i64>(1) as u64))
                    .collect::<Vec<_>>()
            };
            (pairs(lexical), pairs(vector))
        })
    }

    /// Drop the space's HNSW index and have the product build it again over the full table: the
    /// index build time on the corpus as it stands. Run after every query, so it changes no result.
    pub fn rebuild_index(&mut self) -> Call<u64> {
        let derived = self.derived();
        let drop = self.runtime.block_on(
            self.raw
                .batch_execute(&format!("DROP INDEX IF EXISTS {derived}.sd_hnsw_{SPACE}")),
        );
        if let Err(error) = drop {
            die(format!("dropping the index failed: {error}"));
        }
        let definition = EmbeddingSpace {
            id: self.space.clone(),
            model: params::EMBEDDING_MODEL.to_string(),
            revision: params::EMBEDDING_REVISION.to_string(),
            dims: params::EMBEDDING_DIMENSIONS as u16,
        };
        let started = std::time::Instant::now();
        let outcome = guarded(
            &self.runtime,
            self.substrate
                .as_mut()
                .expect("the substrate is connected")
                .register_space(&definition),
        );
        match outcome {
            Some(Ok(())) => Call::Done(started.elapsed().as_nanos() as u64),
            Some(Err(error)) => Call::Failed(error),
            None => {
                self.reconnect();
                Call::Panicked
            }
        }
    }

    /// Drop the instance's schemas.
    pub fn finish(mut self) {
        if let Some(substrate) = self.substrate.take() {
            let _ = self.runtime.block_on(substrate.drop_all());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_space_and_project_names_are_valid_identifiers() {
        assert!(Identifier::new(SPACE).is_ok());
        let document = search_document("cap-0001", 2, "body", &[0.5; 4]);
        assert_eq!(document.generation, Generation(2));
        assert_eq!(document.content_digest, sha256(b"body"));
    }

    #[test]
    fn a_panic_inside_a_guarded_call_is_caught() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        assert_eq!(guarded(&runtime, async { 7 }), Some(7));
        let caught: Option<()> = guarded(&runtime, async { panic!("planted") });
        assert!(caught.is_none());
        // The runtime still works afterwards.
        assert_eq!(guarded(&runtime, async { 8 }), Some(8));
    }
}
