//! Candidates, critiques and the generate-verify-repair loop, with candidate clustering,
//! Pareto/tournament selection, trajectory persistence and text-model adaptors.
//!
//! The crate owns one acceptance rule, [`Candidate::passes`], and uses it everywhere a candidate
//! is accepted: [`RepairLoop`] returns a candidate only through it and
//! [`Population::best_verified`] selects only through it.

mod candidate;
mod cluster;
mod cycle;
mod json;
mod log;
mod prompted;
mod record;
mod redact;
mod repair;
mod select;
mod text;

pub use candidate::{Candidate, Critique, Population};
pub use cycle::{BudgetExhausted, CycleOutcomeKind, CycleRecord, ImprovementCycleBudget};
pub use repair::{
    Decision, ExhaustionReason, Generator, Origin, Outcome, RepairBudget, RepairError, RepairLoop,
    RepairPolicy, RepairRun, Repairer, RewriteReason, Step, Trajectory, Usage,
};

pub use cluster::{
    cluster_candidates, dedup_population, population_diversity, Cluster, ClusterError,
    ClusterOptions, Fingerprint, SimilarityThreshold, MAX_CLUSTER_CANDIDATES,
    MAX_FINGERPRINT_TEXT_BYTES, MAX_MINHASH_FUNCTIONS,
};
pub use log::{
    read_log, rotated_path, AppendReport, LogConfig, LogError, ReadMode, ReadOutcome,
    TrajectoryLog, MAX_KEEP, MIN_ROTATION_BYTES,
};
pub use prompted::{
    generation_prompt, repair_prompt, PromptError, PromptOptions, PromptedGenerator,
    PromptedRepairer, PromptedStats, ScriptedTextModel, TextModel, TextModelError,
};
pub use record::{
    OutcomeRecord, RecordError, StepRecord, TrajectoryRecord, UsageRecord, MAX_ID_CHARS,
    MAX_ISSUES, MAX_LINE_BYTES, MAX_STEPS, MAX_TEXT_CHARS, SCHEMA_VERSION,
};
pub use redact::{redact, REDACTED};
pub use select::{
    crowding_distance, dominates, gate_of, pareto_front, pareto_fronts, rank_points,
    standard_points, standard_space, Direction, Gate, ObjectiveSpace, Point, Ranked,
    SelectionError, Tournament, MAX_OBJECTIVES, MAX_SELECTION_POINTS, MAX_TOURNAMENT_WINNERS,
};
pub use text::{CandidateText, ModelOutput};
