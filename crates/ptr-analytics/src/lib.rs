//! Statistics kernel and metric definitions shared by the platform crates.
//!
//! Every interval, calibration score and agreement coefficient the branch,
//! labeling and lineage crates report is computed here, once. Metrics are
//! defined in PTR terms; a storage backend compiles them to its own query
//! language, so no SQL lives in this crate.

mod agreement;
mod binomial;
mod calibration;
mod error;
mod metric;
mod running;
mod weighted;

pub use agreement::krippendorff_alpha_nominal;
pub use binomial::{binomial_cdf, clopper_pearson_upper, wilson_interval, RateEstimate};
pub use calibration::{brier_score, expected_calibration_error, Binning};
pub use error::StatsError;
pub use metric::{Grouping, Metric, MetricRow, MetricSpec, Window};
pub use running::RunningMoments;
pub use weighted::{weighted_rate, WeightedOutcome, WeightedRate};
