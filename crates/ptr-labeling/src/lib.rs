//! Weak supervision with verifier precedence: labeling-function votes, a
//! Dawid-Skene label model, active-learning acquisition and gold labels that
//! cannot be confused with predictions.
//!
//! Verifier-backed functions are asymmetric, as verifier precedence is: they
//! can rule a class out and are never outvoted, but they do not assert a class
//! by themselves. A class is determined only by eliminating every other one;
//! if every class is ruled out the item is `Disputed` and goes to a person. An
//! item nothing decides with enough probability is `Unknown`, which is a valid
//! outcome. Calibration and agreement come from `ptr-analytics`.

mod acquisition;
mod error;
mod gold;
mod model;
mod votes;

pub use acquisition::{rank_for_annotation, Acquisition};
pub use error::LabelingError;
pub use gold::{
    evaluate, function_accuracy, Calibration, EvaluationReport, EvaluationSet, FunctionAccuracy,
    GoldLabel, GoldSampling, GoldSource,
};
pub use model::{
    fit_label_model, resolve, DawidSkeneParams, LabelModel, LabelOutcome, ModelWarning,
};
pub use votes::{FunctionKind, LabelSchema, LabelingFunction, Vote, VoteMatrix};
