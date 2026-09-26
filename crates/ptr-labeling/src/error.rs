use std::fmt;

/// Refusals from building label matrices, fitting the label model and scoring.
#[derive(Clone, Debug, PartialEq)]
pub enum LabelingError {
    /// A schema needs at least two classes, all distinct.
    InvalidSchema { message: &'static str },
    /// A vote names a class index outside the schema.
    UnknownClass { class: usize, classes: usize },
    /// A row has a different number of votes than there are functions.
    RaggedVotes {
        item: usize,
        expected: usize,
        actual: usize,
    },
    /// A verifier cast a class vote, or a probabilistic function cast a veto.
    VoteKind { item: usize, function: String },
    /// A computation in the statistics kernel refused its input.
    Statistics { code: &'static str },
    /// Two inputs that must align have different lengths.
    LengthMismatch { expected: usize, actual: usize },
    /// A parameter is outside its valid range.
    InvalidParameter {
        field: &'static str,
        message: &'static str,
    },
    /// There is nothing to compute over.
    Empty { field: &'static str },
    /// A function that is not a model names an adapter, or names an empty one.
    AdapterAttribution { function: String },
}

impl LabelingError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidSchema { .. } => "PTR_LABELING_INVALID_SCHEMA",
            Self::UnknownClass { .. } => "PTR_LABELING_UNKNOWN_CLASS",
            Self::RaggedVotes { .. } => "PTR_LABELING_RAGGED_VOTES",
            Self::VoteKind { .. } => "PTR_LABELING_VOTE_KIND",
            Self::Statistics { .. } => "PTR_LABELING_STATISTICS",
            Self::LengthMismatch { .. } => "PTR_LABELING_LENGTH_MISMATCH",
            Self::InvalidParameter { .. } => "PTR_LABELING_INVALID_PARAMETER",
            Self::Empty { .. } => "PTR_LABELING_EMPTY",
            Self::AdapterAttribution { .. } => "PTR_LABELING_ADAPTER_ATTRIBUTION",
        }
    }
}

impl fmt::Display for LabelingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchema { message } => write!(formatter, "invalid label schema: {message}"),
            Self::UnknownClass { class, classes } => {
                write!(formatter, "class {class} is outside a schema of {classes}")
            }
            Self::RaggedVotes {
                item,
                expected,
                actual,
            } => write!(
                formatter,
                "item {item} has {actual} votes, expected {expected}"
            ),
            Self::VoteKind { item, function } => write!(
                formatter,
                "item {item}: {function:?} cast a vote its kind may not cast"
            ),
            Self::Statistics { code } => write!(formatter, "statistics refused input: {code}"),
            Self::LengthMismatch { expected, actual } => {
                write!(formatter, "length {actual} does not match {expected}")
            }
            Self::InvalidParameter { field, message } => write!(formatter, "{field}: {message}"),
            Self::Empty { field } => write!(formatter, "{field} is empty"),
            Self::AdapterAttribution { function } => write!(
                formatter,
                "{function:?} names an adapter but is not a model, or names an empty one"
            ),
        }
    }
}

impl std::error::Error for LabelingError {}

impl From<ptr_analytics::StatsError> for LabelingError {
    fn from(error: ptr_analytics::StatsError) -> Self {
        Self::Statistics { code: error.code() }
    }
}
