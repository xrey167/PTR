//! One seed of Q003: the world is driven through the product, queried at two checkpoints, and held
//! against the harness's own live table and the two baselines.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use ptr_pg::{PgError, SearchResults};

use super::bridge::{Bridge, DocRef};
use super::data::{repo_root, Data};
use super::metrics;
use super::oracle::{
    discordance_bound_permille, discordant_queries, paired_se_permille, quantile, rank_within,
    ArmRanks, LiveTable,
};
use super::params;
use super::pgarm::{search_document, Call, PgArm};
use super::resume::{self, Resume};
use super::workload::{capsule_id, doc_id, EventKind, Phase, Query, Workload, PHASES};
use crate::experiments::layered_kv::emit::{Json, Tally};

/// What the strong arm runs as.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StrongMode {
    /// BM25 + bge-m3 dense + RRF + bge-reranker-v2-m3: the baseline.
    Full,
    /// The same without the cross-encoder (fused list only): used by the unit tests to drive the strong
    /// path cheaply. Never evidence.
    #[cfg_attr(not(test), allow(dead_code))]
    NoRerank,
    /// No strong arm: a development run. Never evidence.
    Off,
}

impl StrongMode {
    pub fn name(self) -> &'static str {
        match self {
            StrongMode::Full => "full",
            StrongMode::NoRerank => "no-rerank",
            StrongMode::Off => "off",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Options {
    pub strong: StrongMode,
    pub strong_python: Option<String>,
    pub models_root: Option<PathBuf>,
    pub strong_threads: Option<u32>,
    /// Where the strong arm keeps its per-query answers so that a failed run resumes.
    pub resume_dir: Option<PathBuf>,
}

/// The longest the harness waits for one reply of a driver (`params::REQUEST_TIMEOUT_SECONDS`).
const REPLY_TIMEOUT: Duration = Duration::from_secs(params::REQUEST_TIMEOUT_SECONDS);

#[derive(Default)]
struct Ranks {
    pg: ArmRanks,
    /// The product's ranks on the queries the strong arm answered (answerable ones), in order.
    pg_on_strong: ArmRanks,
    reference: ArmRanks,
    strong: ArmRanks,
    lexical: ArmRanks,
    vector: ArmRanks,
    bm25: ArmRanks,
    /// The harness's exact dense search over the live set (descriptive).
    exact: ArmRanks,
}

/// Everything a run owns while it goes.
pub struct Run<'a> {
    workload: &'a Workload,
    data: &'a Data,
    options: &'a Options,
    pub tally: Tally,
    oracle: LiveTable,
    arm: PgArm,
    reference: Bridge,
    bm25: Bridge,
    strong: Option<Bridge>,
    strong_resume: Option<Resume>,
    /// (document id, pool row) of every live document at the current checkpoint.
    live_rows: Vec<(String, usize)>,
    projector_ok: bool,
    planted: HashSet<(String, u64)>,
    ranks: Ranks,
    pg_latency: Vec<u64>,
    reference_latency: Vec<u64>,
    pub ingest_ns: u64,
    pub register_ns: u64,
    pub strong_ns: u64,
    pub index_rebuild_ns: u64,
    pub hellos: Vec<(String, Json)>,
    pub errors_shown: usize,
}

fn python_bridge(
    name: &str,
    program: &str,
    args: &[String],
    root: &std::path::Path,
) -> Result<Bridge, String> {
    Bridge::spawn(name, program, args, root, REPLY_TIMEOUT).map_err(String::from)
}

/// Start the reference driver exactly as a seed does (`python3` from `PATH`, same arguments).
pub(super) fn spawn_reference(root: &std::path::Path) -> Result<Bridge, String> {
    python_bridge(
        "reference",
        "python3",
        &["research/baselines/rag_reference/q003_driver.py".to_string()],
        root,
    )
}

/// Start the BM25 driver exactly as a seed does (`python3` from `PATH`, same arguments).
pub(super) fn spawn_bm25(root: &std::path::Path) -> Result<Bridge, String> {
    python_bridge(
        "bm25",
        "python3",
        &["experiments/retrieval/Q003-v3-postgres-hybrid/bm25_driver.py".to_string()],
        root,
    )
}

impl<'a> Run<'a> {
    pub fn open(
        seed: u64,
        workload: &'a Workload,
        workload_digest: &str,
        data: &'a Data,
        options: &'a Options,
        tally: Tally,
    ) -> Result<Self, String> {
        let root = repo_root();
        let mut hellos = Vec::new();
        let mut reference = spawn_reference(&root)?;
        let hello = reference.hello()?;
        if crate::experiments::layered_kv::emit::resolve(&hello, "embedding_model")
            != Some(&Json::str(params::EMBEDDING_MODEL))
        {
            return Err(format!(
                "rag_reference pins another embedding model than Q003: {}",
                hello.render()
            ));
        }
        hellos.push(("reference".to_string(), hello));
        let mut bm25 = spawn_bm25(&root)?;
        hellos.push(("bm25".to_string(), bm25.hello()?));
        let strong = match options.strong {
            StrongMode::Off => None,
            mode => {
                let (program, mut args) = match mode {
                    StrongMode::Full => {
                        let python = options.strong_python.clone().ok_or(
                            "the strong arm needs an interpreter with torch and transformers: pass --strong-python or set PTR_Q003_STRONG_PYTHON (or --no-strong for a development run)",
                        )?;
                        (python, vec!["--reranker".to_string(), "bge".to_string()])
                    }
                    _ => (
                        "python3".to_string(),
                        vec!["--reranker".to_string(), "none".to_string()],
                    ),
                };
                if mode == StrongMode::Full {
                    let models = options.models_root.clone().ok_or(
                        "the strong arm needs the pinned models: pass --models-root or set PTR_Q003_MODELS_ROOT",
                    )?;
                    args.push("--models-root".to_string());
                    args.push(models.display().to_string());
                }
                if let Some(threads) = options.strong_threads {
                    args.push("--threads".to_string());
                    args.push(threads.to_string());
                }
                let mut all = vec!["research/baselines/strong_retrieval/driver.py".to_string()];
                all.extend(args);
                let mut bridge = python_bridge("strong", &program, &all, &root)?;
                let hello = bridge.hello()?;
                let active =
                    crate::experiments::layered_kv::emit::resolve(&hello, "reranker.active");
                if active != Some(&Json::Bool(mode == StrongMode::Full)) {
                    return Err(format!(
                        "the strong driver's reranker state is not what mode {} needs: {}",
                        mode.name(),
                        hello.render()
                    ));
                }
                hellos.push(("strong".to_string(), hello));
                Some(bridge)
            }
        };
        let strong_resume = match (&strong, &options.resume_dir) {
            (Some(_), Some(dir)) => {
                let hello = &hellos.last().expect("the strong hello").1;
                let vectors = data
                    .strong_vectors_sha256
                    .as_deref()
                    .ok_or("the strong vectors are not loaded")?;
                let (commit, dirty) = super::data::checkout_state(&root);
                let provenance = vec![
                    ("harness_commit".to_string(), commit),
                    ("checkout".to_string(), dirty),
                    (
                        "strong_retrieval_fingerprint".to_string(),
                        super::data::directory_fingerprint(
                            &root,
                            "research/baselines/strong_retrieval",
                        )?,
                    ),
                    (
                        "rag_reference_fingerprint".to_string(),
                        super::data::directory_fingerprint(
                            &root,
                            "research/baselines/rag_reference",
                        )?,
                    ),
                ];
                let head = resume::header(
                    seed,
                    workload_digest,
                    vectors,
                    hello,
                    options.strong.name(),
                    &provenance,
                )?;
                let allowed: HashSet<usize> = (0..workload.strong.len())
                    .filter(|index| workload.strong[*index])
                    .collect();
                Some(Resume::open(dir, seed, workload_digest, &head, &allowed)?)
            }
            _ => None,
        };
        let mut tally = tally;
        for name in [
            "queries",
            "answerable_queries",
            "unanswerable_queries",
            "probe_queries",
            "superseded_targets_queried",
            "revoked_targets_queried",
            "planted_stale_rows",
            "planted_stale_returned",
            "stale_hits",
            "late_write_attempts",
            "late_write_accepted",
            "late_write_wrong_refusal",
            "product_errors",
            "lifecycle_projection_mismatches",
            "product_panics",
            "control_planted_returned",
            "reference_queries",
            "projector_residual_rows",
            "control_vector_returned",
            "main_query_lexical_hits",
            "strong_results_resumed",
            "late_write_precondition_rows",
            "nondeterministic_searches",
            "live_probe_lexical_self_hits",
            "live_probe_queries",
            "strong_non_live_ids",
        ] {
            tally.add(name, 0);
        }
        if options.strong != StrongMode::Off {
            tally.add("strong_queries", 0);
        }
        tally.seed_group("stale_hits_by_list", &["lexical", "vector", "fused"]);
        tally.seed_group("retrieval_hits", &["lexical", "vector", "fused"]);
        tally.seed_group(
            "late_writes_by_kind",
            &["superseded", "revoked", "future", "ghost"],
        );
        tally.seed_group("planted_by_kind", &["superseded", "revoked"]);
        let (arm, register_ns) = PgArm::open(seed);
        Ok(Self {
            workload,
            data,
            options,
            tally,
            oracle: LiveTable::default(),
            arm,
            reference,
            bm25,
            strong,
            strong_resume,
            live_rows: Vec::new(),
            projector_ok: true,
            planted: HashSet::new(),
            ranks: Ranks::default(),
            pg_latency: Vec::new(),
            reference_latency: Vec::new(),
            ingest_ns: 0,
            register_ns,
            strong_ns: 0,
            index_rebuild_ns: 0,
            hellos,
            errors_shown: 0,
        })
    }

    pub fn server(&self) -> String {
        self.arm.server.clone()
    }

    /// End the drivers and the product arm. A driver that failed (it exited with a status other
    /// than 0, or died before it answered the quit) is the harness's: the first such stop is the
    /// result, after every driver and the arm were ended.
    pub fn close(self) -> Result<(), String> {
        let mut first: Option<String> = None;
        let mut note = |ended: Result<(), super::watch::Stop>| {
            if let Err(stop) = ended {
                first.get_or_insert(stop.into());
            }
        };
        note(self.reference.close());
        note(self.bm25.close());
        if let Some(strong) = self.strong {
            note(strong.close());
        }
        self.arm.finish();
        first.map_or(Ok(()), Err)
    }

    fn note_error(&mut self, what: &str, error: &PgError) {
        if self.errors_shown < 5 {
            eprintln!("{}: {what}: {error}", params::BENCHMARK);
        }
        self.errors_shown += 1;
    }

    /// Count the outcome of a product call that must succeed.
    fn expect_done<T>(&mut self, what: &str, call: Call<T>) -> Option<T> {
        match call {
            Call::Done(value) => Some(value),
            Call::Failed(error) => {
                self.note_error(what, &error);
                self.tally.add("product_errors", 1);
                None
            }
            Call::Panicked => {
                self.tally.add("product_panics", 1);
                None
            }
        }
    }

    /// The whole run.
    pub fn go(&mut self) -> Result<(), String> {
        super::watch::monitor().phase("ingest");
        self.ingest(Phase::Commits);
        self.ingest(Phase::FirstSupersessions);
        self.ingest(Phase::SecondSupersessions);
        self.checkpoint(1)?;
        self.ingest(Phase::Revocations);
        self.checkpoint(2)?;
        // After every query: rebuild the index over the corpus as it stands.
        let rebuilt = self.arm.rebuild_index();
        if let Some(nanos) = self.expect_done("index rebuild", rebuilt) {
            self.index_rebuild_ns = nanos;
        }
        self.finish_metrics();
        Ok(())
    }

    fn ingest(&mut self, phase: Phase) {
        debug_assert!(PHASES.contains(&phase));
        let started = Instant::now();
        for event in self.workload.events(phase) {
            self.oracle.apply(&event);
            if self.projector_ok {
                let call = self.arm.apply(&event);
                if self
                    .expect_done("the projector failed on the world's own record", call)
                    .is_none()
                {
                    // The chain cannot continue past a record the projector did not take; every
                    // later write is then refused and counted.
                    self.projector_ok = false;
                }
            }
            let generation = match event.kind {
                EventKind::Commit => 1,
                EventKind::Supersede { new, .. } => new,
                EventKind::Revoke { .. } => continue,
            };
            let capsule = capsule_id(event.cue);
            let row = self.data.fact(event.cue, generation);
            let document = search_document(
                &capsule,
                generation,
                self.data.text(row),
                self.data.embedding(row),
            );
            let call = self.arm.upsert(&document);
            if self.expect_done("a live write was refused", call).is_some() {
                self.tally.add("live_writes_accepted", 1);
            }
        }
        self.ingest_ns += started.elapsed().as_nanos() as u64;
    }

    fn checkpoint(&mut self, checkpoint: u8) -> Result<(), String> {
        super::watch::monitor().phase(format!("checkpoint {checkpoint}: lifecycle checks"));
        self.cross_check_lifecycle();
        self.residual_rows();
        if checkpoint == 2 {
            self.late_writes()?;
        }
        self.plant(checkpoint);
        self.rebuild_baselines(checkpoint)?;
        if checkpoint == 2 {
            self.probes()?;
        }
        self.queries(checkpoint)?;
        if checkpoint == 2 {
            self.repeat_queries()?;
        }
        Ok(())
    }

    /// The harness's live table against the product's own projection: for every capsule and every
    /// generation, the product's `is_admissible` must agree with the table.
    fn cross_check_lifecycle(&mut self) {
        for cue in 0..self.workload.shape.capsules {
            let capsule = capsule_id(cue);
            for generation in 1..=3u64 {
                let call = self.arm.admissible(&capsule, generation);
                if let Some(answer) = self.expect_done("is_admissible", call) {
                    self.tally.add("lifecycle_checks", 1);
                    if answer != self.oracle.is_live(&capsule, generation) {
                        self.tally.add("lifecycle_projection_mismatches", 1);
                    }
                }
            }
        }
    }

    /// Rows of the derived table that are not live, not planted by the harness and not an earlier
    /// late write: what a projector leaves behind when it fails to delete a superseded or revoked
    /// document. A search that joins the lifecycle hides them, so only this check sees them.
    fn residual_rows(&mut self) {
        for (capsule, generation) in self.arm.all_rows() {
            if !self.oracle.is_live(&capsule, generation)
                && !self.planted.contains(&(capsule, generation))
            {
                self.tally.add("projector_residual_rows", 1);
            }
        }
    }

    fn late_writes(&mut self) -> Result<(), String> {
        for late in self.workload.late_writes.clone() {
            let capsule = late.capsule();
            let text_generation = late.generation.min(3);
            let row = self.data.fact(late.cue, text_generation);
            let document = search_document(
                &capsule,
                late.generation,
                self.data.text(row),
                self.data.embedding(row),
            );
            if self.arm.row_exists(&capsule, late.generation) {
                // A row the projector left (counted as projector_residual_rows): the attempt
                // would prove nothing about the write, so it is skipped.
                self.tally.add("late_write_precondition_rows", 1);
                continue;
            }
            self.tally.add("late_write_attempts", 1);
            self.tally
                .add(&format!("late_writes_by_kind.{}", late.kind.name()), 1);
            let call = self.arm.upsert(&document);
            let stored = self.arm.row_exists(&capsule, late.generation);
            match call {
                Call::Done(()) => self.tally.add("late_write_accepted", 1),
                Call::Failed(PgError::NotLive { .. }) => {
                    if stored {
                        self.tally.add("late_write_accepted", 1);
                    } else {
                        self.tally.add("late_writes_refused_not_live", 1);
                    }
                }
                Call::Failed(error) => {
                    self.note_error("a late write got another refusal", &error);
                    self.tally.add("late_write_wrong_refusal", 1);
                    if stored {
                        self.tally.add("late_write_accepted", 1);
                    }
                }
                Call::Panicked => self.tally.add("product_panics", 1),
            }
        }
        Ok(())
    }

    fn plant(&mut self, checkpoint: u8) {
        for (_, plant) in self.workload.plants_at(checkpoint) {
            let capsule = capsule_id(plant.cue);
            let row = self.data.fact(plant.cue, plant.generation);
            self.arm.plant(
                &capsule,
                plant.generation,
                self.data.text(row),
                self.data.embedding(row),
            );
            self.planted.insert((capsule, plant.generation));
            self.tally.add("planted_stale_rows", 1);
            self.tally.add(
                if plant.revoked {
                    "planted_by_kind.revoked"
                } else {
                    "planted_by_kind.superseded"
                },
                1,
            );
        }
    }

    /// Rebuild the three baseline indexes from the harness's live table.
    fn rebuild_baselines(&mut self, checkpoint: u8) -> Result<(), String> {
        super::watch::monitor().phase(format!("checkpoint {checkpoint}: loading the baselines"));
        let pairs = self.oracle.live_pairs();
        let ids: Vec<String> = pairs
            .iter()
            .map(|(capsule, generation)| doc_id(capsule, *generation))
            .collect();
        let rows: Vec<usize> = pairs
            .iter()
            .map(|(capsule, generation)| {
                let cue: usize = capsule["cap-".len()..].parse().expect("a capsule id");
                self.data.fact(cue, *generation)
            })
            .collect();
        self.live_rows = ids.iter().cloned().zip(rows.iter().copied()).collect();
        let minilm: Vec<DocRef<'_>> = ids
            .iter()
            .zip(&rows)
            .map(|(id, row)| DocRef {
                id,
                text: self.data.text(*row),
                embedding: self.data.embedding(*row),
            })
            .collect();
        self.reference.load(&minilm)?;
        if self.reference.size()? != pairs.len() as i64 {
            return Err("the reference index does not hold the live set".to_string());
        }
        let bare: Vec<DocRef<'_>> = ids
            .iter()
            .zip(&rows)
            .map(|(id, row)| DocRef {
                id,
                text: self.data.text(*row),
                embedding: &[],
            })
            .collect();
        self.bm25.load(&bare)?;
        if self.bm25.size()? != pairs.len() as i64 {
            return Err("the BM25 index does not hold the live set".to_string());
        }
        // The strong index is loaded only if some query of this checkpoint still needs the driver
        // (a resumed run may have every answer already).
        let needed = self
            .workload
            .queries_at(checkpoint)
            .filter(|(index, _)| self.workload.strong[*index])
            .any(|(index, _)| {
                self.strong_resume
                    .as_ref()
                    .is_none_or(|resume| resume.get(index).is_none())
            });
        if let (Some(strong), true) = (self.strong.as_mut(), needed) {
            let started = Instant::now();
            let docs: Vec<DocRef<'_>> = ids
                .iter()
                .zip(&rows)
                .map(|(id, row)| DocRef {
                    id,
                    text: self.data.text(*row),
                    embedding: self.data.strong_embedding(*row).expect("strong vectors"),
                })
                .collect();
            strong.load(&docs)?;
            if strong.size()? != pairs.len() as i64 {
                return Err("the strong index does not hold the live set".to_string());
            }
            self.strong_ns += started.elapsed().as_nanos() as u64;
        }
        Ok(())
    }

    /// Hold every list of one product answer against the live table and the planted set.
    fn audit(&mut self, results: &SearchResults) {
        let lists: [(&str, Vec<(String, u64)>); 3] = [
            (
                "lexical",
                results
                    .lexical
                    .iter()
                    .map(|hit| (hit.capsule().0.clone(), hit.generation().0))
                    .collect(),
            ),
            (
                "vector",
                results
                    .vector
                    .iter()
                    .map(|hit| (hit.capsule().0.clone(), hit.generation().0))
                    .collect(),
            ),
            (
                "fused",
                results
                    .fused
                    .iter()
                    .map(|hit| (hit.capsule.0.clone(), hit.generation.0))
                    .collect(),
            ),
        ];
        for (name, list) in lists {
            self.tally
                .add(&format!("retrieval_hits.{name}"), list.len() as i64);
            for (capsule, generation) in list {
                if !self.oracle.is_live(&capsule, generation) {
                    self.tally.add(&format!("stale_hits_by_list.{name}"), 1);
                    self.tally.add("stale_hits", 1);
                }
                if self.planted.contains(&(capsule, generation)) {
                    self.tally.add("planted_stale_returned", 1);
                }
            }
        }
    }

    fn timed_search(
        &mut self,
        text: &str,
        embedding: &[f32],
        measure: bool,
    ) -> Option<SearchResults> {
        let started = Instant::now();
        let call = self.arm.search(text, embedding);
        if measure {
            self.pg_latency.push(started.elapsed().as_nanos() as u64);
        }
        let results = self.expect_done("search", call)?;
        self.audit(&results);
        Some(results)
    }

    /// The probes: queries whose text is a planted row's own body (the lexical mode matches it),
    /// with the no-live-join control run beside each, and queries whose text is a live body.
    fn probes(&mut self) -> Result<(), String> {
        let mut control_lexical_missed = 0;
        let stale: Vec<(usize, u64)> = self
            .workload
            .stale_probes
            .iter()
            .map(|index| {
                let plant = self.workload.plants[*index];
                (plant.cue, plant.generation)
            })
            .collect();
        for (cue, generation) in stale {
            let row = self.data.fact(cue, generation);
            let (text, embedding) = (self.data.text(row), self.data.embedding(row));
            self.tally.add("probe_queries", 1);
            self.tally.add("stale_probe_queries", 1);
            self.timed_search(text, embedding, false);
            let planted = (capsule_id(cue), generation);
            let (lexical, vector) = self.arm.search_without_live_join(text, embedding);
            if lexical.contains(&planted) {
                self.tally.add("control_planted_returned", 1);
                self.tally.add("control_lexical_returned", 1);
            } else {
                control_lexical_missed += 1;
            }
            if vector.contains(&planted) {
                self.tally.add("control_planted_returned", 1);
                self.tally.add("control_vector_returned", 1);
            }
        }
        if control_lexical_missed > 0 {
            return Err(format!(
                "the no-live-join control missed the planted row of {control_lexical_missed} stale probes in the exact lexical mode: the control does not show the defect it exists to show"
            ));
        }
        for cue in self.workload.live_probes.clone() {
            let fate = self.workload.fates[cue];
            let row = self.data.fact(cue, fate.history());
            let (text, embedding) = (self.data.text(row), self.data.embedding(row));
            self.tally.add("probe_queries", 1);
            self.tally.add("live_probe_queries", 1);
            let wanted = (capsule_id(cue), fate.history());
            if let Some(results) = self.timed_search(text, embedding, false) {
                if results
                    .lexical
                    .iter()
                    .any(|hit| (hit.capsule().0.clone(), hit.generation().0) == wanted)
                {
                    self.tally.add("live_probe_lexical_self_hits", 1);
                }
                if results
                    .vector
                    .first()
                    .is_some_and(|hit| (hit.capsule().0.clone(), hit.generation().0) == wanted)
                {
                    self.tally.add("live_probe_vector_self_top1", 1);
                }
            }
        }
        Ok(())
    }

    fn strong_search(
        &mut self,
        index: usize,
        text: &str,
        row: usize,
    ) -> Result<Option<Vec<String>>, String> {
        let mode = self.options.strong;
        let Some(strong) = self.strong.as_mut() else {
            return Ok(None);
        };
        self.tally.add("strong_queries", 1);
        if let Some(ids) = self
            .strong_resume
            .as_ref()
            .and_then(|resume| resume.get(index))
        {
            // A resumed answer is not trusted blindly: every id must be a live document of this
            // checkpoint, else the file is not this run's answer.
            if let Some(stale) = ids
                .iter()
                .find(|id| !self.live_rows.iter().any(|(live, _)| live == *id))
            {
                return Err(format!(
                    "the resume file holds {stale} for query {index}, which is not live at this checkpoint"
                ));
            }
            self.tally.add("strong_results_resumed", 1);
            return Ok(Some(ids.clone()));
        }
        let started = Instant::now();
        let reply = strong.search(
            text,
            self.data.strong_embedding(row).expect("strong vectors"),
            params::K,
        )?;
        self.strong_ns += started.elapsed().as_nanos() as u64;
        if reply.reranked != Some(mode == StrongMode::Full) {
            return Err(format!(
                "the strong arm answered reranked={:?} in mode {}",
                reply.reranked,
                mode.name()
            ));
        }
        let non_live = reply
            .hits
            .iter()
            .filter(|id| !self.live_rows.iter().any(|(live, _)| live == *id))
            .count();
        self.tally.add("strong_non_live_ids", non_live as i64);
        if let Some(resume) = self.strong_resume.as_mut() {
            resume.record(index, &reply.hits)?;
        }
        Ok(Some(reply.hits))
    }

    /// The 1-based rank of `wanted_id` in the harness's own exact cosine search over the live
    /// documents (ties by document id, as the reference breaks them), if within the first k.
    fn exact_rank(&self, query: &[f32], wanted_id: &str) -> Option<usize> {
        let cosine = |row: usize| -> f64 {
            let vector = self.data.embedding(row);
            let dot: f64 = query
                .iter()
                .zip(vector)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            let norms = query
                .iter()
                .map(|a| f64::from(*a).powi(2))
                .sum::<f64>()
                .sqrt()
                * vector
                    .iter()
                    .map(|b| f64::from(*b).powi(2))
                    .sum::<f64>()
                    .sqrt();
            dot / norms
        };
        let wanted = self.live_rows.iter().find(|(id, _)| id == wanted_id)?;
        let target = cosine(wanted.1);
        let ahead = self
            .live_rows
            .iter()
            .filter(|(id, row)| {
                let score = cosine(*row);
                score > target || (score == target && id.as_str() < wanted_id)
            })
            .count();
        (ahead < params::K).then_some(ahead + 1)
    }

    fn queries(&mut self, checkpoint: u8) -> Result<(), String> {
        let queries: Vec<(usize, Query)> = self
            .workload
            .queries_at(checkpoint)
            .map(|(index, query)| (index, *query))
            .collect();
        let total = queries.len();
        for (done, (index, query)) in queries.into_iter().enumerate() {
            super::watch::monitor().phase(format!("checkpoint {checkpoint}: query {done}/{total}"));
            if done % 50 == 0 {
                eprintln!(
                    "{}: checkpoint {checkpoint}: query {done}/{total}, strong arm {} s so far",
                    params::BENCHMARK,
                    self.strong_ns / 1_000_000_000
                );
            }
            let capsule = capsule_id(query.cue);
            let row = self.data.query(query.cue, query.paraphrase);
            let (text, embedding) = (self.data.text(row), self.data.embedding(row));
            self.tally.add("queries", 1);
            if query.revoked_target {
                self.tally.add("unanswerable_queries", 1);
                self.tally.add("revoked_targets_queried", 1);
            } else {
                self.tally.add("answerable_queries", 1);
                if query.generation >= 2 {
                    self.tally.add("superseded_targets_queried", 1);
                }
            }
            let results = self.timed_search(text, embedding, true);
            if let Some(results) = &results {
                self.tally
                    .add("main_query_lexical_hits", results.lexical.len() as i64);
            }
            let started = Instant::now();
            let reference = self.reference.search(text, embedding, params::K)?;
            self.reference_latency
                .push(started.elapsed().as_nanos() as u64);
            self.tally.add("reference_queries", 1);
            let bm25 = self.bm25.search(text, &[], params::K)?;
            let in_strong = self.workload.strong[index];
            let strong = if in_strong {
                self.strong_search(index, text, row)?
            } else {
                None
            };
            if query.revoked_target {
                continue;
            }
            let wanted = (capsule.clone(), query.generation);
            let wanted_id = doc_id(&capsule, query.generation);
            let rank_in = |pairs: &mut dyn Iterator<Item = (String, u64)>| {
                pairs
                    .take(params::K)
                    .position(|pair| pair == wanted)
                    .map(|at| at + 1)
            };
            let (fused, lexical, vector) = match &results {
                Some(results) => (
                    rank_in(
                        &mut results
                            .fused
                            .iter()
                            .map(|hit| (hit.capsule.0.clone(), hit.generation.0)),
                    ),
                    rank_in(
                        &mut results
                            .lexical
                            .iter()
                            .map(|hit| (hit.capsule().0.clone(), hit.generation().0)),
                    ),
                    rank_in(
                        &mut results
                            .vector
                            .iter()
                            .map(|hit| (hit.capsule().0.clone(), hit.generation().0)),
                    ),
                ),
                None => (None, None, None),
            };
            self.ranks.pg.push(fused);
            let exact = self.exact_rank(embedding, &wanted_id);
            self.ranks.exact.push(exact);
            if in_strong {
                self.ranks.pg_on_strong.push(fused);
            }
            self.ranks.lexical.push(lexical);
            self.ranks.vector.push(vector);
            self.ranks
                .reference
                .push(rank_within(&reference.hits, &wanted_id, params::K));
            self.ranks
                .bm25
                .push(rank_within(&bm25.hits, &wanted_id, params::K));
            if let Some(hits) = strong {
                self.ranks
                    .strong
                    .push(rank_within(&hits, &wanted_id, params::K));
            }
        }
        Ok(())
    }

    /// A search on the unchanged snapshot must answer as it did; one that does not is the
    /// product's (`nondeterministic_searches`, a falsifier), not the harness's.
    fn repeat_queries(&mut self) -> Result<(), String> {
        let picks: Vec<Query> = self
            .workload
            .repeats
            .iter()
            .map(|index| self.workload.queries[*index])
            .filter(|query| query.checkpoint == 2)
            .collect();
        for query in picks {
            let row = self.data.query(query.cue, query.paraphrase);
            let (text, embedding) = (self.data.text(row), self.data.embedding(row));
            let first = self.arm.search(text, embedding);
            let second = self.arm.search(text, embedding);
            if let (Call::Done(first), Call::Done(second)) = (first, second) {
                self.tally.add("repeat_queries", 1);
                let key = |results: &SearchResults| {
                    (
                        results
                            .lexical
                            .iter()
                            .map(|hit| (hit.capsule().0.clone(), hit.generation().0))
                            .collect::<Vec<_>>(),
                        results
                            .vector
                            .iter()
                            .map(|hit| (hit.capsule().0.clone(), hit.generation().0))
                            .collect::<Vec<_>>(),
                        results
                            .fused
                            .iter()
                            .map(|hit| (hit.capsule.0.clone(), hit.generation.0))
                            .collect::<Vec<_>>(),
                    )
                };
                if key(&first) != key(&second) {
                    self.tally.add("nondeterministic_searches", 1);
                    eprintln!(
                        "{}: a search of {text:?} answered differently twice on an unchanged snapshot",
                        params::BENCHMARK
                    );
                }
            }
        }
        Ok(())
    }

    fn finish_metrics(&mut self) {
        let strong_ran = self.strong.is_some();
        self.tally
            .set("recall10_pg_permille", self.ranks.pg.recall_permille());
        self.tally
            .set("ndcg10_pg_permille", self.ranks.pg.ndcg_permille());
        self.tally.set(
            "recall10_reference_permille",
            self.ranks.reference.recall_permille(),
        );
        self.tally.set(
            "ndcg10_reference_permille",
            self.ranks.reference.ndcg_permille(),
        );
        self.tally.set(
            "recall10_pg_lexical_only_permille",
            self.ranks.lexical.recall_permille(),
        );
        self.tally.set(
            "recall10_pg_vector_only_permille",
            self.ranks.vector.recall_permille(),
        );
        self.tally.set(
            "recall10_exact_cosine_permille",
            self.ranks.exact.recall_permille(),
        );
        self.tally.set(
            "recall10_bm25_only_permille",
            self.ranks.bm25.recall_permille(),
        );
        self.tally.set(
            "recall_paired_se_vs_reference_permille",
            paired_se_permille(&self.ranks.pg, &self.ranks.reference),
        );
        if strong_ran {
            self.tally.set(
                "recall10_strong_permille",
                self.ranks.strong.recall_permille(),
            );
            self.tally
                .set("ndcg10_strong_permille", self.ranks.strong.ndcg_permille());
            self.tally.set(
                "recall10_pg_on_strong_subset_permille",
                self.ranks.pg_on_strong.recall_permille(),
            );
            self.tally.set(
                "ndcg10_pg_on_strong_subset_permille",
                self.ranks.pg_on_strong.ndcg_permille(),
            );
            // Not a standard error: at the ceiling both arms hit every query and any SE is 0. The
            // discordant queries (one arm hits, the other misses) and a 95 percent upper bound on
            // their share say how much disagreement the sample can still hide.
            let discordant = discordant_queries(&self.ranks.pg_on_strong, &self.ranks.strong);
            self.tally.set("strong_discordant_queries", discordant);
            self.tally.set(
                "strong_discordance_bound_permille",
                discordance_bound_permille(discordant, self.ranks.strong.ranks.len() as i64),
            );
        } else {
            for name in metrics::STRONG_METRICS {
                self.tally
                    .unsupported(name, "the strong arm did not run in this development run");
            }
        }
        metrics::fill_derived(&mut self.tally, strong_ran);
        for (name, values) in [
            ("pg_search", &mut self.pg_latency),
            ("reference_search", &mut self.reference_latency),
        ] {
            let (p50, p99) = (quantile(values, 500), quantile(values, 990));
            self.tally
                .set(&format!("latency_ns.{name}_p50"), p50 as i64);
            self.tally
                .set(&format!("latency_ns.{name}_p99"), p99 as i64);
        }
        self.tally
            .set("index_build_ns", self.index_rebuild_ns as i64);
    }
}
