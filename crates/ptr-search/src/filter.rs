//! Who may see a document, when it holds, and which of its metadata a query asks for.
//!
//! These types are backend-neutral. A backend turns a [`SearchFilter`] into its own predicate
//! and applies it **before** it limits: a filter applied after `LIMIT` leaves a result short
//! of hits that exist, and an approximate index that stops after a fixed number of visited
//! entries returns fewer still. [`SearchFilter::admits`] is the reference definition of the
//! predicate, in plain Rust, against which a backend's translation is held.
//!
//! **Access.** A document lists the principals and groups that may see it in
//! [`SearchAttributes::principals`]; an empty list means the whole project may. A query's
//! audience is its principal plus its groups ([`SearchFilter::audience`]), and a document with
//! a non-empty list is admitted only when the two lists share a member. A filter that names no
//! principal and no group therefore sees only project-wide documents: a missing identity
//! narrows what is seen, never widens it. Principals and groups share one namespace of
//! strings, so a group named like a principal admits that principal's holders; callers keep
//! group names distinguishable.
//!
//! **Time.** Validity is half-open (`valid_from <= now < valid_until`, a missing bound open).
//! A filter with no `now` applies no time rule at all, which is for an audit or administrative
//! view and says so by the field being absent. Whoever relies on a retrieved hit asks again at
//! the moment it relies on it ([`crate::SearchHit::observe_at`],
//! [`crate::SearchHit::promote_verified_at`]): a document can expire between retrieval and use.
use ptr_types::{KnowledgeValidity, PrincipalId, Timestamp};
use std::collections::BTreeSet;
use std::fmt;

/// The most principals or groups a document or a query names.
pub const MAX_FILTER_PRINCIPALS: usize = 64;

/// The most tags a document carries or a query asks for.
pub const MAX_FILTER_TAGS: usize = 32;

/// The longest principal, group, tag, role or predicate, in bytes.
pub const MAX_FILTER_TEXT_BYTES: usize = 128;

/// Why a filter or a document's attributes were refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FilterError {
    /// A name that is empty.
    Empty { field: &'static str },
    /// A name over [`MAX_FILTER_TEXT_BYTES`].
    TooLong {
        field: &'static str,
        limit: usize,
        actual: usize,
    },
    /// More members than the bound allows.
    TooMany {
        field: &'static str,
        limit: usize,
        actual: usize,
    },
    /// A name with a control character, which no backend predicate should have to quote.
    ControlCharacter { field: &'static str },
    /// A validity whose end does not come after its start holds at no instant.
    EmptyValidity,
}

impl FilterError {
    /// A stable code for each refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Empty { .. } => "PTR_SEARCH_FILTER_EMPTY",
            Self::TooLong { .. } => "PTR_SEARCH_FILTER_TOO_LONG",
            Self::TooMany { .. } => "PTR_SEARCH_FILTER_TOO_MANY",
            Self::ControlCharacter { .. } => "PTR_SEARCH_FILTER_CONTROL_CHARACTER",
            Self::EmptyValidity => "PTR_SEARCH_FILTER_EMPTY_VALIDITY",
        }
    }
}

impl fmt::Display for FilterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { field } => write!(formatter, "{field} is empty"),
            Self::TooLong {
                field,
                limit,
                actual,
            } => write!(
                formatter,
                "{field} is {actual} bytes, over the limit of {limit}"
            ),
            Self::TooMany {
                field,
                limit,
                actual,
            } => write!(
                formatter,
                "{field} has {actual} members, over the limit of {limit}"
            ),
            Self::ControlCharacter { field } => {
                write!(formatter, "{field} holds a control character")
            }
            Self::EmptyValidity => formatter.write_str("the validity ends no later than it starts"),
        }
    }
}

impl std::error::Error for FilterError {}

fn check_text(field: &'static str, value: &str) -> Result<(), FilterError> {
    if value.is_empty() {
        return Err(FilterError::Empty { field });
    }
    if value.len() > MAX_FILTER_TEXT_BYTES {
        return Err(FilterError::TooLong {
            field,
            limit: MAX_FILTER_TEXT_BYTES,
            actual: value.len(),
        });
    }
    if value.chars().any(char::is_control) {
        return Err(FilterError::ControlCharacter { field });
    }
    Ok(())
}

fn check_set(
    field: &'static str,
    limit: usize,
    members: &BTreeSet<String>,
) -> Result<(), FilterError> {
    if members.len() > limit {
        return Err(FilterError::TooMany {
            field,
            limit,
            actual: members.len(),
        });
    }
    members
        .iter()
        .try_for_each(|member| check_text(field, member))
}

fn check_optional(field: &'static str, value: &Option<String>) -> Result<(), FilterError> {
    value
        .as_deref()
        .map_or(Ok(()), |value| check_text(field, value))
}

/// What a document declares about itself for retrieval. The default is a project-wide
/// document that always holds and carries no role, predicate or tag.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SearchAttributes {
    /// The principals and groups that may see the document; empty means the whole project.
    pub principals: BTreeSet<String>,
    /// When the document holds.
    pub validity: KnowledgeValidity,
    /// The semantic role of what the document states (for example a fact or a rule).
    pub semantic_role: Option<String>,
    /// The predicate it states, when it states one.
    pub predicate: Option<String>,
    /// Free labels a query can ask for.
    pub tags: BTreeSet<String>,
}

impl SearchAttributes {
    /// Refuse attributes a backend should not have to store.
    pub fn validate(&self) -> Result<(), FilterError> {
        check_set("principals", MAX_FILTER_PRINCIPALS, &self.principals)?;
        check_set("tags", MAX_FILTER_TAGS, &self.tags)?;
        check_optional("semantic_role", &self.semantic_role)?;
        check_optional("predicate", &self.predicate)?;
        if !self.validity.is_well_formed() {
            return Err(FilterError::EmptyValidity);
        }
        Ok(())
    }
}

/// What a query asks of the documents it may return.
///
/// There is no `Default`: a filter is built for an instant ([`SearchFilter::at`],
/// [`SearchFilter::for_principal`]) so the time rule cannot be forgotten, or deliberately
/// without one ([`SearchFilter::audit_without_time`]). Either way a filter that names nobody
/// sees project-wide documents only.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchFilter {
    /// Who is asking, as the runtime admitted them.
    pub principal: Option<PrincipalId>,
    /// The groups the asker belongs to.
    pub groups: BTreeSet<String>,
    /// The instant validity is judged at, in milliseconds since the Unix epoch. `None` applies
    /// no time rule and is only for an audit view ([`SearchFilter::audit_without_time`]).
    pub now: Option<Timestamp>,
    /// Only documents of this role.
    pub semantic_role: Option<String>,
    /// Only documents stating this predicate.
    pub predicate: Option<String>,
    /// Only documents carrying at least one of these tags; empty asks for none in particular.
    pub tags_any: BTreeSet<String>,
}

impl SearchFilter {
    /// A filter that names nobody and judges validity at `now`: it sees the project-wide
    /// documents that hold at `now`.
    pub fn at(now: Timestamp) -> Self {
        Self {
            principal: None,
            groups: BTreeSet::new(),
            now: Some(now),
            semantic_role: None,
            predicate: None,
            tags_any: BTreeSet::new(),
        }
    }

    /// A filter with no time rule: expired and not-yet-valid documents are visible. For an
    /// audit or an administrative view that must see everything the cache holds for its
    /// audience; never the filter of a retrieval that feeds an answer.
    pub fn audit_without_time() -> Self {
        Self {
            now: None,
            ..Self::at(Timestamp(0))
        }
    }

    /// A filter for an admitted principal and its groups, judging validity at `now`.
    pub fn for_principal(
        principal: PrincipalId,
        groups: impl IntoIterator<Item = String>,
        now: Timestamp,
    ) -> Self {
        Self {
            principal: Some(principal),
            groups: groups.into_iter().collect(),
            ..Self::at(now)
        }
    }

    /// Refuse a filter a backend should not have to translate.
    pub fn validate(&self) -> Result<(), FilterError> {
        if let Some(principal) = &self.principal {
            check_text("principal", &principal.0)?;
        }
        check_set("groups", MAX_FILTER_PRINCIPALS, &self.groups)?;
        check_set("tags_any", MAX_FILTER_TAGS, &self.tags_any)?;
        check_optional("semantic_role", &self.semantic_role)?;
        check_optional("predicate", &self.predicate)?;
        // The audience is the principal plus the groups, and is what the document bound counts.
        if self.audience().len() > MAX_FILTER_PRINCIPALS {
            return Err(FilterError::TooMany {
                field: "audience",
                limit: MAX_FILTER_PRINCIPALS,
                actual: self.audience().len(),
            });
        }
        Ok(())
    }

    /// The principal and groups a document's access list is compared with.
    pub fn audience(&self) -> BTreeSet<String> {
        let mut audience = self.groups.clone();
        if let Some(principal) = &self.principal {
            audience.insert(principal.0.clone());
        }
        audience
    }

    /// The reference definition of the predicate: whether a document with `attributes` is
    /// visible to this filter. A backend's translation must admit exactly these documents.
    pub fn admits(&self, attributes: &SearchAttributes) -> bool {
        let access = attributes.principals.is_empty()
            || attributes
                .principals
                .iter()
                .any(|member| self.is_in_audience(member));
        let time = self
            .now
            .is_none_or(|now| attributes.validity.is_valid_at(now));
        let role = self
            .semantic_role
            .as_ref()
            .is_none_or(|role| attributes.semantic_role.as_ref() == Some(role));
        let predicate = self
            .predicate
            .as_ref()
            .is_none_or(|predicate| attributes.predicate.as_ref() == Some(predicate));
        let tags = self.tags_any.is_empty() || !self.tags_any.is_disjoint(&attributes.tags);
        access && time && role && predicate && tags
    }

    fn is_in_audience(&self, member: &str) -> bool {
        self.groups.contains(member)
            || self
                .principal
                .as_ref()
                .is_some_and(|principal| principal.0 == member)
    }
}
