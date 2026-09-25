use std::collections::BTreeSet;

use crate::error::LabelingError;

/// The closed set of classes a labeling task decides between.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LabelSchema {
    classes: Vec<String>,
}

impl LabelSchema {
    /// At least two distinct, non-empty class names.
    pub fn new<I, S>(classes: I) -> Result<Self, LabelingError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let classes: Vec<String> = classes.into_iter().map(Into::into).collect();
        if classes.len() < 2 {
            return Err(LabelingError::InvalidSchema {
                message: "a schema needs at least two classes",
            });
        }
        if classes.iter().any(String::is_empty) {
            return Err(LabelingError::InvalidSchema {
                message: "class names may not be empty",
            });
        }
        if classes.iter().collect::<BTreeSet<_>>().len() != classes.len() {
            return Err(LabelingError::InvalidSchema {
                message: "class names must be distinct",
            });
        }
        Ok(Self { classes })
    }

    pub fn len(&self) -> usize {
        self.classes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    pub fn class_name(&self, class: usize) -> Option<&str> {
        self.classes.get(class).map(String::as_str)
    }
}

/// One labeling function's output for one item.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Vote {
    /// An estimate that the item belongs to a class.
    Class(usize),
    /// A verifier's finding that the item cannot belong to a class. Only
    /// [`FunctionKind::Verifier`] functions may veto.
    Veto(usize),
    /// No opinion. Abstaining is always allowed and is never read as evidence
    /// for or against any class.
    Abstain,
}

/// What a labeling function's output is worth.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FunctionKind {
    /// A verifier-backed check (an executed test, a schema constraint). Its
    /// output is asymmetric, as verifier precedence is: it can rule a class out
    /// (a deterministic contradiction beats any learned estimate) but it never
    /// asserts one by itself. It is excluded from the probabilistic model.
    /// A rule that is merely reproducible — a regular expression, a keyword
    /// list — is a [`FunctionKind::Heuristic`], not a verifier.
    Verifier,
    /// A heuristic, pattern or weak rule.
    Heuristic,
    /// A trained model.
    Model,
    /// An agent's judgement.
    Agent,
}

/// A named labeling function.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LabelingFunction {
    pub name: String,
    pub kind: FunctionKind,
}

/// Votes of every function on every item: `votes[item][function]`.
#[derive(Clone, Debug, PartialEq)]
pub struct VoteMatrix {
    schema: LabelSchema,
    functions: Vec<LabelingFunction>,
    votes: Vec<Vec<Vote>>,
}

impl VoteMatrix {
    /// Validate votes arranged as `votes[item][function]`. An empty item list
    /// is allowed; each row must contain one vote per function.
    ///
    /// # Errors
    /// Rejects an empty function list, ragged rows, out-of-range class indices,
    /// class votes from verifiers, and vetoes from nonverifiers.
    pub fn new(
        schema: LabelSchema,
        functions: Vec<LabelingFunction>,
        votes: Vec<Vec<Vote>>,
    ) -> Result<Self, LabelingError> {
        if functions.is_empty() {
            return Err(LabelingError::Empty {
                field: "labeling functions",
            });
        }
        for (item, row) in votes.iter().enumerate() {
            if row.len() != functions.len() {
                return Err(LabelingError::RaggedVotes {
                    item,
                    expected: functions.len(),
                    actual: row.len(),
                });
            }
            for (vote, function) in row.iter().zip(&functions) {
                let class = match vote {
                    Vote::Class(class) | Vote::Veto(class) => *class,
                    Vote::Abstain => continue,
                };
                if class >= schema.len() {
                    return Err(LabelingError::UnknownClass {
                        class,
                        classes: schema.len(),
                    });
                }
                let allowed = matches!(
                    (vote, function.kind),
                    (Vote::Veto(_), FunctionKind::Verifier)
                        | (
                            Vote::Class(_),
                            FunctionKind::Heuristic | FunctionKind::Model | FunctionKind::Agent
                        )
                );
                if !allowed {
                    return Err(LabelingError::VoteKind {
                        item,
                        function: function.name.clone(),
                    });
                }
            }
        }
        Ok(Self {
            schema,
            functions,
            votes,
        })
    }

    pub fn schema(&self) -> &LabelSchema {
        &self.schema
    }

    pub fn functions(&self) -> &[LabelingFunction] {
        &self.functions
    }

    pub fn items(&self) -> usize {
        self.votes.len()
    }

    /// Votes for the zero-based item index, in function order.
    ///
    /// # Panics
    /// Panics if `item` is outside the matrix.
    pub fn row(&self, item: usize) -> &[Vote] {
        &self.votes[item]
    }

    /// Fraction of items on which at least one function voted.
    pub fn coverage(&self) -> f64 {
        if self.votes.is_empty() {
            return 0.0;
        }
        let covered = self
            .votes
            .iter()
            .filter(|row| row.iter().any(|vote| *vote != Vote::Abstain))
            .count();
        covered as f64 / self.votes.len() as f64
    }
}
