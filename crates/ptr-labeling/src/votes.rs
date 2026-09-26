use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

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
    /// For a [`FunctionKind::Model`], the adapter that produced its votes, by
    /// its id in the adapter lineage catalog, so labeling quality can be
    /// attributed to an adapter ([`crate::function_accuracy`]). Only a model
    /// function may name one; [`VoteMatrix::new`] refuses it on any other kind.
    pub adapter: Option<String>,
}

impl LabelingFunction {
    /// A function with no adapter attribution.
    pub fn new(name: impl Into<String>, kind: FunctionKind) -> Self {
        Self {
            name: name.into(),
            kind,
            adapter: None,
        }
    }

    /// A trained model's function, attributed to the adapter that produced
    /// its votes.
    pub fn model(name: impl Into<String>, adapter: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: FunctionKind::Model,
            adapter: Some(adapter.into()),
        }
    }
}

/// Votes of every function on every item: `votes[item][function]`.
#[derive(Clone, Debug, PartialEq)]
pub struct VoteMatrix {
    schema: LabelSchema,
    functions: Vec<LabelingFunction>,
    votes: Vec<Vec<Vote>>,
    digest: [u8; 32],
}

impl VoteMatrix {
    /// Validate votes arranged as `votes[item][function]`. An empty item list
    /// is allowed; each row must contain one vote per function.
    ///
    /// Functions are identified by name, so every name is non-empty and
    /// distinct: a function listed twice would have its votes counted twice
    /// as independent evidence by the label model.
    ///
    /// # Errors
    /// Rejects an empty function list, a function with an empty name
    /// ([`LabelingError::Empty`]) or a name an earlier function has
    /// ([`LabelingError::DuplicateFunction`]), an adapter named by a function
    /// that is not a model or an empty adapter id, ragged rows, out-of-range
    /// class indices, class votes from verifiers, and vetoes from
    /// nonverifiers.
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
        let mut names = BTreeSet::new();
        for function in &functions {
            if function.name.is_empty() {
                return Err(LabelingError::Empty {
                    field: "labeling function name",
                });
            }
            if !names.insert(function.name.as_str()) {
                return Err(LabelingError::DuplicateFunction {
                    function: function.name.clone(),
                });
            }
        }
        for function in &functions {
            let attributed = match &function.adapter {
                None => true,
                Some(adapter) => function.kind == FunctionKind::Model && !adapter.is_empty(),
            };
            if !attributed {
                return Err(LabelingError::AdapterAttribution {
                    function: function.name.clone(),
                });
            }
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
        let digest = matrix_digest(&schema, &functions, &votes);
        Ok(Self {
            schema,
            functions,
            votes,
            digest,
        })
    }

    /// SHA-256 of the schema's class names, every function's name, kind and
    /// adapter, and every vote, in order, under an unambiguous length-prefixed
    /// encoding. Two matrices have equal digests exactly when they are equal
    /// (up to a SHA-256 collision); a fitted [`crate::LabelModel`] carries the
    /// digest of the matrix it was fitted on.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
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

/// The encoding behind [`VoteMatrix::digest`]. Every string is prefixed with
/// its length and every list with its count; rows need none, since each holds
/// exactly one vote per function.
fn matrix_digest(
    schema: &LabelSchema,
    functions: &[LabelingFunction],
    votes: &[Vec<Vote>],
) -> [u8; 32] {
    fn text(hasher: &mut Sha256, value: &str) {
        hasher.update((value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    let mut hasher = Sha256::new();
    hasher.update(b"ptr-labeling/vote-matrix/v1");
    hasher.update((schema.classes.len() as u64).to_le_bytes());
    for class in &schema.classes {
        text(&mut hasher, class);
    }
    hasher.update((functions.len() as u64).to_le_bytes());
    for function in functions {
        text(&mut hasher, &function.name);
        hasher.update([match function.kind {
            FunctionKind::Verifier => 0,
            FunctionKind::Heuristic => 1,
            FunctionKind::Model => 2,
            FunctionKind::Agent => 3,
        }]);
        match &function.adapter {
            None => hasher.update([0]),
            Some(adapter) => {
                hasher.update([1]);
                text(&mut hasher, adapter);
            }
        }
    }
    hasher.update((votes.len() as u64).to_le_bytes());
    for vote in votes.iter().flatten() {
        match vote {
            Vote::Abstain => hasher.update([0]),
            Vote::Class(class) => {
                hasher.update([1]);
                hasher.update((*class as u64).to_le_bytes());
            }
            Vote::Veto(class) => {
                hasher.update([2]);
                hasher.update((*class as u64).to_le_bytes());
            }
        }
    }
    hasher.finalize().into()
}
