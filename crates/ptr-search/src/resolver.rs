//! Resolving the name a query uses to the canonical name a document is filed under.
//!
//! Minimal on purpose: a closed table of aliases, each a name the operator registered, matched
//! against whole words of the query, the longest match winning. There is no fuzzy matching, no
//! stemming and no model: a name that is not in the table is [`ResolveError::Unknown`], and two
//! different canonical names found in one query are [`ResolveError::Ambiguous`]. Refusing is a
//! result, not a failure to try harder: a soft guess cannot establish an identity, so the
//! caller is told to ask which one was meant.
//!
//! Matching works on lower-cased alphanumeric words, so punctuation and case do not matter and
//! a match never starts or ends inside a word.
use std::collections::BTreeMap;
use std::fmt;

/// The most aliases a table holds.
pub const MAX_ALIASES: usize = 4096;

/// The most words a query may have.
pub const MAX_QUERY_WORDS: usize = 256;

/// The most words in one alias.
pub const MAX_ALIAS_WORDS: usize = 8;

/// Why an alias was refused when the table was built.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AliasError {
    /// An alias or a canonical name with no word in it.
    Empty,
    /// An alias of more than [`MAX_ALIAS_WORDS`] words.
    TooLong { limit: usize, actual: usize },
    /// The table holds [`MAX_ALIASES`] already.
    TableFull { limit: usize },
    /// The alias already names another canonical name.
    Conflict { alias: String, existing: String },
}

/// Why a query was not resolved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResolveError {
    /// The query names nothing the table knows.
    Unknown,
    /// The query names more than one canonical name; they are listed, sorted.
    Ambiguous { candidates: Vec<String> },
    /// A query over [`MAX_QUERY_WORDS`] words.
    QueryTooLong { limit: usize, actual: usize },
}

impl ResolveError {
    /// A stable code for each refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Unknown => "PTR_SEARCH_RESOLVE_UNKNOWN",
            Self::Ambiguous { .. } => "PTR_SEARCH_RESOLVE_AMBIGUOUS",
            Self::QueryTooLong { .. } => "PTR_SEARCH_RESOLVE_QUERY_TOO_LONG",
        }
    }
}

impl fmt::Display for ResolveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown => formatter.write_str("the query names nothing the table knows"),
            Self::Ambiguous { candidates } => {
                write!(
                    formatter,
                    "the query names several: {}",
                    candidates.join(", ")
                )
            }
            Self::QueryTooLong { limit, actual } => {
                write!(
                    formatter,
                    "the query has {actual} words, over the limit of {limit}"
                )
            }
        }
    }
}

impl std::error::Error for ResolveError {}

/// What a query resolved to.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Resolved {
    /// The canonical name the matched alias belongs to.
    pub canonical: String,
    /// The alias that matched, in its normalised form.
    pub alias: String,
}

fn words(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// A closed table of aliases.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AliasTable {
    aliases: BTreeMap<Vec<String>, String>,
}

impl AliasTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.aliases.len()
    }

    pub fn is_empty(&self) -> bool {
        self.aliases.is_empty()
    }

    /// Register `alias` for `canonical`. Registering the same pair again is a no-op; the same
    /// alias for another canonical name is a conflict, because the table is a function.
    pub fn add(&mut self, alias: &str, canonical: &str) -> Result<(), AliasError> {
        let key = words(alias);
        if key.is_empty() || words(canonical).is_empty() {
            return Err(AliasError::Empty);
        }
        if key.len() > MAX_ALIAS_WORDS {
            return Err(AliasError::TooLong {
                limit: MAX_ALIAS_WORDS,
                actual: key.len(),
            });
        }
        if let Some(existing) = self.aliases.get(&key) {
            return if existing == canonical {
                Ok(())
            } else {
                Err(AliasError::Conflict {
                    alias: key.join(" "),
                    existing: existing.clone(),
                })
            };
        }
        if self.aliases.len() >= MAX_ALIASES {
            return Err(AliasError::TableFull { limit: MAX_ALIASES });
        }
        self.aliases.insert(key, canonical.to_owned());
        Ok(())
    }

    /// Resolve the one canonical name `query` names.
    ///
    /// Every alias is looked for as a run of whole words. A match inside a longer match is
    /// dropped, so a longer known name settles its shorter prefix (`Müller Werke` over
    /// `Müller`). The canonical names that remain decide: none is [`ResolveError::Unknown`],
    /// more than one is [`ResolveError::Ambiguous`].
    pub fn resolve(&self, query: &str) -> Result<Resolved, ResolveError> {
        let query_words = words(query);
        if query_words.len() > MAX_QUERY_WORDS {
            return Err(ResolveError::QueryTooLong {
                limit: MAX_QUERY_WORDS,
                actual: query_words.len(),
            });
        }
        // (start, end, alias key) over word positions.
        let mut matches: Vec<(usize, usize, &Vec<String>)> = Vec::new();
        for (alias, _) in &self.aliases {
            let length = alias.len();
            if length > query_words.len() {
                continue;
            }
            for start in 0..=query_words.len() - length {
                if query_words[start..start + length] == alias[..] {
                    matches.push((start, start + length, alias));
                }
            }
        }
        let kept: Vec<_> = matches
            .iter()
            .filter(|(start, end, _)| {
                !matches.iter().any(|(other_start, other_end, _)| {
                    other_start <= start
                        && other_end >= end
                        && (other_end - other_start) > (end - start)
                })
            })
            .collect();
        let mut canonicals: Vec<&String> = kept
            .iter()
            .filter_map(|(_, _, alias)| self.aliases.get(*alias))
            .collect();
        canonicals.sort();
        canonicals.dedup();
        match canonicals.as_slice() {
            [] => Err(ResolveError::Unknown),
            [canonical] => {
                // The longest alias for that canonical name is the one reported.
                let alias = kept
                    .iter()
                    .filter(|(_, _, alias)| self.aliases.get(*alias) == Some(*canonical))
                    .max_by_key(|(start, end, _)| (end - start, usize::MAX - start))
                    .map(|(_, _, alias)| alias.join(" "))
                    .unwrap_or_default();
                Ok(Resolved {
                    canonical: (*canonical).clone(),
                    alias,
                })
            }
            several => Err(ResolveError::Ambiguous {
                candidates: several.iter().map(|name| (*name).clone()).collect(),
            }),
        }
    }
}
