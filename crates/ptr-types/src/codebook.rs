//! A versioned assignment of stable integer identities to the cognitive type
//! kernel.
//!
//! A model that embeds `SemanticRole` needs an integer per role, and the obvious
//! way to get one — `role as u16` — is a trap. Rust enum discriminants follow
//! declaration order, so inserting a variant or sorting the list alphabetically
//! silently renumbers every role that comes after it. Weights trained against the
//! old numbering keep loading, keep producing plausible outputs, and now mean
//! something else. Nothing fails; the model is just wrong about what it is looking
//! at.
//!
//! So the assignment lives here as explicit data. A code is the member's position
//! in a versioned table, never its discriminant, and the tests assert exact
//! numeric values so a reordered enum breaks a build rather than a checkpoint.
//!
//! Codes are dense — `0..cardinality` with no gaps or duplicates — because they
//! index embedding tables directly, and a version fixes both the assignment and
//! the table sizes. That pair is what a checkpoint has to record: a code without
//! its codebook version does not identify anything.
use crate::{EpistemicState, ReasoningOperator, SemanticRole, UncertaintyKind, Validity};
use std::fmt;

/// Version of a whole codebook assignment.
///
/// Every stored code, dataset row and checkpoint is only interpretable against the
/// version that produced it, so this travels with them rather than being implied.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CodebookVersion(pub u32);

impl CodebookVersion {
    /// Initial frozen assignment shared by the current type kernel.
    pub const V1: Self = Self(1);
}

impl fmt::Display for CodebookVersion {
    /// Render the version in manifest-friendly form.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// Stable identity of one member within its family, for use as a tensor index.
///
/// Deliberately not convertible from an arbitrary integer: a code only means
/// anything as the output of [`Codebook::code_of`] or the input to
/// [`Codebook::member_of`], both of which name a version.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TypeCode(u16);

impl TypeCode {
    /// The raw index, for indexing an embedding table sized by
    /// [`Codebook::cardinality_of`].
    pub fn index(self) -> u16 {
        self.0
    }
}

impl fmt::Display for TypeCode {
    /// Render the raw table index.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// One contiguous code space of the type kernel.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CodeFamily {
    SemanticRole,
    EpistemicState,
    UncertaintyKind,
    ReasoningOperator,
    Validity,
}

impl CodeFamily {
    /// Stable name, used in canonical bytes and in run/dataset manifests.
    pub fn name(self) -> &'static str {
        match self {
            Self::SemanticRole => "semantic_role",
            Self::EpistemicState => "epistemic_state",
            Self::UncertaintyKind => "uncertainty_kind",
            Self::ReasoningOperator => "reasoning_operator",
            Self::Validity => "validity",
        }
    }

    /// The family a stable name denotes.
    ///
    /// An explicit table in both directions, never a position in [`Self::ALL`]: a
    /// name read from an artifact must map to the same family it mapped to when
    /// the artifact was written, and an index would silently follow a reordering.
    /// An unknown name is refused rather than approximated, because guessing the
    /// family is guessing what every code in that artifact means.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "semantic_role" => Some(Self::SemanticRole),
            "epistemic_state" => Some(Self::EpistemicState),
            "uncertainty_kind" => Some(Self::UncertaintyKind),
            "reasoning_operator" => Some(Self::ReasoningOperator),
            "validity" => Some(Self::Validity),
            _ => None,
        }
    }

    /// Every family, in canonical order.
    pub const ALL: [Self; 5] = [
        Self::SemanticRole,
        Self::EpistemicState,
        Self::UncertaintyKind,
        Self::ReasoningOperator,
        Self::Validity,
    ];
}

/// Why a codebook lookup was refused.
///
/// Every variant is a refusal. Nothing here falls back to a default, because a
/// defaulted code is how a stale or foreign identity gets admitted without anyone
/// noticing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodebookError {
    /// The version has no table for this family.
    UnknownVersion {
        family: CodeFamily,
        version: CodebookVersion,
    },
    /// The code is not assigned in this version.
    UnassignedCode {
        family: CodeFamily,
        version: CodebookVersion,
        code: u16,
    },
    /// The member exists in this build but has no code in this version, which is
    /// what a variant added after the version was frozen looks like.
    UnassignedMember {
        family: CodeFamily,
        version: CodebookVersion,
    },
}

impl CodebookError {
    /// Stable diagnostic code for this lookup refusal.
    pub fn code(self) -> &'static str {
        match self {
            Self::UnknownVersion { .. } => "PTR_CODEBOOK_UNKNOWN_VERSION",
            Self::UnassignedCode { .. } => "PTR_CODEBOOK_UNASSIGNED_CODE",
            Self::UnassignedMember { .. } => "PTR_CODEBOOK_UNASSIGNED_MEMBER",
        }
    }
}

impl fmt::Display for CodebookError {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for CodebookError {}

/// A member of the cognitive type kernel that carries a stable coded identity.
///
/// `table` is the assignment: the member at index *n* has code *n*. Implementations
/// list members explicitly rather than deriving order from the enum, which is the
/// whole point of this module.
pub trait CognitiveType: Copy + PartialEq + Sized + 'static {
    const FAMILY: CodeFamily;

    /// Members of this family in code order for `version`, or `None` when the
    /// version predates or postdates this family.
    fn table(version: CodebookVersion) -> Option<&'static [Self]>;

    /// Stable name, independent of Rust identifiers and of code order.
    fn name(self) -> &'static str;
}

/// A frozen assignment of codes to the cognitive type kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Codebook {
    version: CodebookVersion,
}

impl Codebook {
    /// The first frozen assignment.
    ///
    /// Its tables must never be reordered or have entries removed. A change goes
    /// into a new version so that artifacts recorded against this one keep meaning
    /// what they meant.
    pub const V1: Self = Self {
        version: CodebookVersion::V1,
    };

    /// Adopt a version that this build knows.
    ///
    /// An unknown version is refused rather than approximated: interpreting codes
    /// under the wrong assignment is exactly the silent remapping this module
    /// exists to prevent.
    pub fn at(version: CodebookVersion) -> Result<Self, CodebookError> {
        let book = Self { version };
        for family in CodeFamily::ALL {
            if book.cardinality(family) == 0 {
                return Err(CodebookError::UnknownVersion { family, version });
            }
        }
        Ok(book)
    }

    /// Version that fixes every assignment in this codebook.
    pub fn version(&self) -> CodebookVersion {
        self.version
    }

    /// Size of a family's embedding table in this version.
    pub fn cardinality_of<T: CognitiveType>(&self) -> u16 {
        T::table(self.version).map_or(0, |table| table.len() as u16)
    }

    /// Stable code for a member.
    pub fn code_of<T: CognitiveType>(&self, member: T) -> Result<TypeCode, CodebookError> {
        let table = T::table(self.version).ok_or(CodebookError::UnknownVersion {
            family: T::FAMILY,
            version: self.version,
        })?;
        table
            .iter()
            .position(|candidate| *candidate == member)
            .map(|index| TypeCode(index as u16))
            .ok_or(CodebookError::UnassignedMember {
                family: T::FAMILY,
                version: self.version,
            })
    }

    /// The member a code denotes, or a refusal.
    pub fn member_of<T: CognitiveType>(&self, code: TypeCode) -> Result<T, CodebookError> {
        let table = T::table(self.version).ok_or(CodebookError::UnknownVersion {
            family: T::FAMILY,
            version: self.version,
        })?;
        table
            .get(usize::from(code.index()))
            .copied()
            .ok_or(CodebookError::UnassignedCode {
                family: T::FAMILY,
                version: self.version,
                code: code.index(),
            })
    }

    /// Every member of a family with its code, in code order.
    pub fn assignment_of<T: CognitiveType>(&self) -> Result<Vec<(TypeCode, T)>, CodebookError> {
        let table = T::table(self.version).ok_or(CodebookError::UnknownVersion {
            family: T::FAMILY,
            version: self.version,
        })?;
        Ok(table
            .iter()
            .enumerate()
            .map(|(index, member)| (TypeCode(index as u16), *member))
            .collect())
    }

    /// Cardinality by family, without naming the Rust type.
    pub fn cardinality(&self, family: CodeFamily) -> u16 {
        match family {
            CodeFamily::SemanticRole => self.cardinality_of::<SemanticRole>(),
            CodeFamily::EpistemicState => self.cardinality_of::<EpistemicState>(),
            CodeFamily::UncertaintyKind => self.cardinality_of::<UncertaintyKind>(),
            CodeFamily::ReasoningOperator => self.cardinality_of::<ReasoningOperator>(),
            CodeFamily::Validity => self.cardinality_of::<Validity>(),
        }
    }

    /// The complete assignment as deterministic bytes.
    ///
    /// Hash these to bind a checkpoint, dataset or run to the assignment it was
    /// produced under; any reordering, addition or removal changes them. This is a
    /// commitment, **not** authentication: it detects a mismatch, and cannot detect
    /// an attacker who recomputes the hash. Hashing lives in the caller because
    /// this crate deliberately has no dependencies.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"PTRCODEBOOK\x00");
        out.extend_from_slice(&self.version.0.to_le_bytes());
        out.extend_from_slice(&(CodeFamily::ALL.len() as u16).to_le_bytes());
        for family in CodeFamily::ALL {
            put_str(&mut out, family.name());
            out.extend_from_slice(&self.cardinality(family).to_le_bytes());
            for name in self.member_names(family) {
                put_str(&mut out, name);
            }
        }
        out
    }

    /// Stable member names for one family in code order.
    ///
    /// Public because a shared codebook has to be readable from outside Rust: the
    /// alternative is every other language retyping the table, which is the
    /// duplication this type exists to prevent.
    pub fn member_names(&self, family: CodeFamily) -> Vec<&'static str> {
        /// Collect names from one typed assignment table.
        fn names<T: CognitiveType>(book: &Codebook) -> Vec<&'static str> {
            T::table(book.version)
                .unwrap_or(&[])
                .iter()
                .map(|member| member.name())
                .collect()
        }
        match family {
            CodeFamily::SemanticRole => names::<SemanticRole>(self),
            CodeFamily::EpistemicState => names::<EpistemicState>(self),
            CodeFamily::UncertaintyKind => names::<UncertaintyKind>(self),
            CodeFamily::ReasoningOperator => names::<ReasoningOperator>(self),
            CodeFamily::Validity => names::<Validity>(self),
        }
    }
}

/// Append a length-prefixed string to the canonical assignment bytes.
fn put_str(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u16).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

// The V1 tables. Order here *is* the code assignment, so entries are appended,
// never inserted or sorted. Removing one requires a new version.

const V1_SEMANTIC_ROLES: [SemanticRole; 9] = [
    SemanticRole::Goal,
    SemanticRole::Constraint,
    SemanticRole::Claim,
    SemanticRole::Evidence,
    SemanticRole::Resource,
    SemanticRole::Capability,
    SemanticRole::Relation,
    SemanticRole::Procedure,
    SemanticRole::Action,
];

const V1_EPISTEMIC_STATES: [EpistemicState; 6] = [
    EpistemicState::Unknown,
    EpistemicState::Assumed,
    EpistemicState::Hypothesis,
    EpistemicState::Observed,
    EpistemicState::Inferred,
    EpistemicState::Verified,
];

const V1_UNCERTAINTY_KINDS: [UncertaintyKind; 3] = [
    UncertaintyKind::Point,
    UncertaintyKind::Interval,
    UncertaintyKind::Distribution,
];

const V1_REASONING_OPERATORS: [ReasoningOperator; 11] = [
    ReasoningOperator::Semantic,
    ReasoningOperator::Deductive,
    ReasoningOperator::Probabilistic,
    ReasoningOperator::Statistical,
    ReasoningOperator::Temporal,
    ReasoningOperator::Causal,
    ReasoningOperator::Search,
    ReasoningOperator::Optimization,
    ReasoningOperator::Simulation,
    ReasoningOperator::Symbolic,
    ReasoningOperator::ExternalPod,
];

const V1_VALIDITIES: [Validity; 4] = [
    Validity::Live,
    Validity::Superseded,
    Validity::Revoked,
    Validity::Disputed,
];

impl CognitiveType for SemanticRole {
    const FAMILY: CodeFamily = CodeFamily::SemanticRole;
    fn table(version: CodebookVersion) -> Option<&'static [Self]> {
        (version == CodebookVersion::V1).then_some(&V1_SEMANTIC_ROLES)
    }
    fn name(self) -> &'static str {
        match self {
            Self::Goal => "goal",
            Self::Constraint => "constraint",
            Self::Claim => "claim",
            Self::Evidence => "evidence",
            Self::Resource => "resource",
            Self::Capability => "capability",
            Self::Relation => "relation",
            Self::Procedure => "procedure",
            Self::Action => "action",
        }
    }
}

impl CognitiveType for EpistemicState {
    const FAMILY: CodeFamily = CodeFamily::EpistemicState;
    fn table(version: CodebookVersion) -> Option<&'static [Self]> {
        (version == CodebookVersion::V1).then_some(&V1_EPISTEMIC_STATES)
    }
    fn name(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Assumed => "assumed",
            Self::Hypothesis => "hypothesis",
            Self::Observed => "observed",
            Self::Inferred => "inferred",
            Self::Verified => "verified",
        }
    }
}

impl CognitiveType for UncertaintyKind {
    const FAMILY: CodeFamily = CodeFamily::UncertaintyKind;
    fn table(version: CodebookVersion) -> Option<&'static [Self]> {
        (version == CodebookVersion::V1).then_some(&V1_UNCERTAINTY_KINDS)
    }
    fn name(self) -> &'static str {
        match self {
            Self::Point => "point",
            Self::Interval => "interval",
            Self::Distribution => "distribution",
        }
    }
}

impl CognitiveType for ReasoningOperator {
    const FAMILY: CodeFamily = CodeFamily::ReasoningOperator;
    fn table(version: CodebookVersion) -> Option<&'static [Self]> {
        (version == CodebookVersion::V1).then_some(&V1_REASONING_OPERATORS)
    }
    fn name(self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::Deductive => "deductive",
            Self::Probabilistic => "probabilistic",
            Self::Statistical => "statistical",
            Self::Temporal => "temporal",
            Self::Causal => "causal",
            Self::Search => "search",
            Self::Optimization => "optimization",
            Self::Simulation => "simulation",
            Self::Symbolic => "symbolic",
            Self::ExternalPod => "external_pod",
        }
    }
}

impl CognitiveType for Validity {
    const FAMILY: CodeFamily = CodeFamily::Validity;
    fn table(version: CodebookVersion) -> Option<&'static [Self]> {
        (version == CodebookVersion::V1).then_some(&V1_VALIDITIES)
    }
    fn name(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::Superseded => "superseded",
            Self::Revoked => "revoked",
            Self::Disputed => "disputed",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_versions_are_refused_rather_than_approximated() {
        assert_eq!(Codebook::at(CodebookVersion::V1), Ok(Codebook::V1));
        for version in [
            CodebookVersion(0),
            CodebookVersion(2),
            CodebookVersion(u32::MAX),
        ] {
            let error = Codebook::at(version).expect_err("unknown version");
            assert_eq!(error.code(), "PTR_CODEBOOK_UNKNOWN_VERSION");
        }
    }

    #[test]
    fn a_code_is_never_a_discriminant() {
        // These exact numbers are the contract. Reordering a kernel enum must break
        // this test rather than silently renumber a trained embedding table.
        let book = Codebook::V1;
        for (member, expected) in [
            (SemanticRole::Goal, 0),
            (SemanticRole::Constraint, 1),
            (SemanticRole::Claim, 2),
            (SemanticRole::Evidence, 3),
            (SemanticRole::Resource, 4),
            (SemanticRole::Capability, 5),
            (SemanticRole::Relation, 6),
            (SemanticRole::Procedure, 7),
            (SemanticRole::Action, 8),
        ] {
            assert_eq!(book.code_of(member).unwrap().index(), expected);
        }
        for (member, expected) in [
            (EpistemicState::Unknown, 0),
            (EpistemicState::Assumed, 1),
            (EpistemicState::Hypothesis, 2),
            (EpistemicState::Observed, 3),
            (EpistemicState::Inferred, 4),
            (EpistemicState::Verified, 5),
        ] {
            assert_eq!(book.code_of(member).unwrap().index(), expected);
        }
        for (member, expected) in [
            (UncertaintyKind::Point, 0),
            (UncertaintyKind::Interval, 1),
            (UncertaintyKind::Distribution, 2),
        ] {
            assert_eq!(book.code_of(member).unwrap().index(), expected);
        }
        for (member, expected) in [
            (ReasoningOperator::Semantic, 0),
            (ReasoningOperator::Deductive, 1),
            (ReasoningOperator::Probabilistic, 2),
            (ReasoningOperator::Statistical, 3),
            (ReasoningOperator::Temporal, 4),
            (ReasoningOperator::Causal, 5),
            (ReasoningOperator::Search, 6),
            (ReasoningOperator::Optimization, 7),
            (ReasoningOperator::Simulation, 8),
            (ReasoningOperator::Symbolic, 9),
            (ReasoningOperator::ExternalPod, 10),
        ] {
            assert_eq!(book.code_of(member).unwrap().index(), expected);
        }
        for (member, expected) in [
            (Validity::Live, 0),
            (Validity::Superseded, 1),
            (Validity::Revoked, 2),
            (Validity::Disputed, 3),
        ] {
            assert_eq!(book.code_of(member).unwrap().index(), expected);
        }
    }

    /// Exhaustive matches: adding a kernel variant fails to compile here until the
    /// V1 table question ("new version, or append?") has been answered explicitly.
    #[test]
    fn every_member_of_every_family_is_assigned() {
        let book = Codebook::V1;
        fn covered<T: CognitiveType + std::fmt::Debug>(book: &Codebook, member: T) {
            let code = book.code_of(member).expect("member is assigned");
            assert_eq!(book.member_of::<T>(code).unwrap(), member);
        }
        for role in V1_SEMANTIC_ROLES {
            match role {
                SemanticRole::Goal
                | SemanticRole::Constraint
                | SemanticRole::Claim
                | SemanticRole::Evidence
                | SemanticRole::Resource
                | SemanticRole::Capability
                | SemanticRole::Relation
                | SemanticRole::Procedure
                | SemanticRole::Action => covered(&book, role),
            }
        }
        for state in V1_EPISTEMIC_STATES {
            match state {
                EpistemicState::Unknown
                | EpistemicState::Assumed
                | EpistemicState::Hypothesis
                | EpistemicState::Observed
                | EpistemicState::Inferred
                | EpistemicState::Verified => covered(&book, state),
            }
        }
        for kind in V1_UNCERTAINTY_KINDS {
            match kind {
                UncertaintyKind::Point
                | UncertaintyKind::Interval
                | UncertaintyKind::Distribution => covered(&book, kind),
            }
        }
        for operator in V1_REASONING_OPERATORS {
            match operator {
                ReasoningOperator::Semantic
                | ReasoningOperator::Deductive
                | ReasoningOperator::Probabilistic
                | ReasoningOperator::Statistical
                | ReasoningOperator::Temporal
                | ReasoningOperator::Causal
                | ReasoningOperator::Search
                | ReasoningOperator::Optimization
                | ReasoningOperator::Simulation
                | ReasoningOperator::Symbolic
                | ReasoningOperator::ExternalPod => covered(&book, operator),
            }
        }
        for validity in V1_VALIDITIES {
            match validity {
                Validity::Live | Validity::Superseded | Validity::Revoked | Validity::Disputed => {
                    covered(&book, validity)
                }
            }
        }
    }

    #[test]
    fn codes_are_dense_and_unique_so_they_can_index_a_table() {
        let book = Codebook::V1;
        let expected = [
            (CodeFamily::SemanticRole, 9u16),
            (CodeFamily::EpistemicState, 6),
            (CodeFamily::UncertaintyKind, 3),
            (CodeFamily::ReasoningOperator, 11),
            (CodeFamily::Validity, 4),
        ];
        for (family, cardinality) in expected {
            assert_eq!(book.cardinality(family), cardinality, "{}", family.name());
        }
        let roles = book.assignment_of::<SemanticRole>().unwrap();
        let codes: Vec<u16> = roles.iter().map(|(code, _)| code.index()).collect();
        assert_eq!(codes, (0..9).collect::<Vec<u16>>());
        let names: Vec<&str> = roles.iter().map(|(_, role)| role.name()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "member names must be distinct");
    }

    #[test]
    fn a_code_outside_the_assignment_is_refused() {
        let book = Codebook::V1;
        for raw in [4u16, 9, 10, u16::MAX] {
            let code = book
                .code_of(SemanticRole::Goal)
                .map(|_| TypeCode(raw))
                .unwrap();
            let outcome = book.member_of::<SemanticRole>(code);
            if raw < book.cardinality(CodeFamily::SemanticRole) {
                assert!(outcome.is_ok());
            } else {
                assert_eq!(outcome.unwrap_err().code(), "PTR_CODEBOOK_UNASSIGNED_CODE");
            }
        }
    }

    #[test]
    fn canonical_bytes_commit_to_the_whole_assignment() {
        let bytes = Codebook::V1.canonical_bytes();
        assert!(bytes.starts_with(b"PTRCODEBOOK\x00"));
        assert_eq!(bytes[12..16], 1u32.to_le_bytes());
        // Deterministic, and every family and member name is committed.
        assert_eq!(bytes, Codebook::V1.canonical_bytes());
        let text = String::from_utf8_lossy(&bytes).to_string();
        for family in CodeFamily::ALL {
            assert!(text.contains(family.name()), "{}", family.name());
        }
        for name in ["goal", "action", "hypothesis", "external_pod", "revoked"] {
            assert!(text.contains(name), "{name}");
        }
    }
}
