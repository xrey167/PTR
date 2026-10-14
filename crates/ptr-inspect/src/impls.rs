//! [`Inspectable`] for std types and for the public domain types of `ptr-types`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use ptr_types::{
    ActionIr, AdapterVersion, ArtifactId, CandidateId, CapabilityId, CapsuleId, CommitIndex,
    ConfidenceEstimate, ConfidenceTarget, DeviceId, Effect, EntityId, Epistemic, EpistemicState,
    EventRole, EvidenceId, ExecutorIdentity, FencingToken, Generation, InvitationId, KeyId,
    KnowledgeObjectId, ModelVersion, NamespaceId, NetworkId, NodeId, PeerId, PodId, PodIdentity,
    PolicyVersion, PrincipalId, Probability, ProjectId, ProvenanceRef, RawEventId,
    ReasoningOperator, RegionId, RequestId, RetrievalKey, Revision, ScopeId, SemanticRole,
    SessionId, StateId, Timestamp, TraceId, TypeId, TypedValue, UncertaintyKind, Validity,
    VerificationLevel, ZoneId,
};

use crate::node::{fingerprint_label, fnv1a64, InspectNode, Inspectable};

impl Inspectable for bool {
    fn inspect(&self) -> InspectNode {
        InspectNode::Bool(*self)
    }
}

macro_rules! inspect_signed {
    ($($ty:ty),*) => {$(
        impl Inspectable for $ty {
            fn inspect(&self) -> InspectNode {
                InspectNode::Int(i128::from(*self))
            }
        }
    )*};
}

macro_rules! inspect_unsigned {
    ($($ty:ty),*) => {$(
        impl Inspectable for $ty {
            fn inspect(&self) -> InspectNode {
                InspectNode::UInt(u128::from(*self))
            }
        }
    )*};
}

inspect_signed!(i8, i16, i32, i64, i128);
inspect_unsigned!(u8, u16, u32, u64, u128);

impl Inspectable for isize {
    fn inspect(&self) -> InspectNode {
        InspectNode::Int(*self as i128)
    }
}

impl Inspectable for usize {
    fn inspect(&self) -> InspectNode {
        InspectNode::UInt(*self as u128)
    }
}

impl Inspectable for f32 {
    fn inspect(&self) -> InspectNode {
        InspectNode::Number(f64::from(*self))
    }
}

impl Inspectable for f64 {
    fn inspect(&self) -> InspectNode {
        InspectNode::Number(*self)
    }
}

impl Inspectable for char {
    fn inspect(&self) -> InspectNode {
        InspectNode::Text(self.to_string())
    }
}

impl Inspectable for str {
    fn inspect(&self) -> InspectNode {
        InspectNode::Text(self.to_owned())
    }
}

impl Inspectable for String {
    fn inspect(&self) -> InspectNode {
        InspectNode::Text(self.clone())
    }
}

impl Inspectable for () {
    fn inspect(&self) -> InspectNode {
        InspectNode::Null
    }
}

impl<T: Inspectable + ?Sized> Inspectable for &T {
    fn inspect(&self) -> InspectNode {
        (**self).inspect()
    }
}

impl<T: Inspectable + ?Sized> Inspectable for Box<T> {
    fn inspect(&self) -> InspectNode {
        (**self).inspect()
    }
}

impl<T: Inspectable + ?Sized> Inspectable for Rc<T> {
    fn inspect(&self) -> InspectNode {
        (**self).inspect()
    }
}

impl<T: Inspectable + ?Sized> Inspectable for Arc<T> {
    fn inspect(&self) -> InspectNode {
        (**self).inspect()
    }
}

/// `None` is `Null` and `Some(x)` is `x`, so `Some(None)` and `None` look alike. Call it as
/// `Inspectable::inspect(&option)`: `Option::inspect` and `Result::inspect` are inherent std
/// methods that take precedence over the trait method.
impl<T: Inspectable> Inspectable for Option<T> {
    fn inspect(&self) -> InspectNode {
        self.as_ref()
            .map_or(InspectNode::Null, Inspectable::inspect)
    }
}

impl<T: Inspectable, E: Inspectable> Inspectable for Result<T, E> {
    fn inspect(&self) -> InspectNode {
        match self {
            Ok(value) => InspectNode::variant("Result", "Ok", vec![("0", value.inspect())]),
            Err(error) => InspectNode::variant("Result", "Err", vec![("0", error.inspect())]),
        }
    }
}

impl<T: Inspectable> Inspectable for [T] {
    fn inspect(&self) -> InspectNode {
        InspectNode::List(self.iter().map(Inspectable::inspect).collect())
    }
}

impl<T: Inspectable> Inspectable for Vec<T> {
    fn inspect(&self) -> InspectNode {
        self.as_slice().inspect()
    }
}

impl<T: Inspectable, const N: usize> Inspectable for [T; N] {
    fn inspect(&self) -> InspectNode {
        self.as_slice().inspect()
    }
}

impl<T: Inspectable> Inspectable for BTreeSet<T> {
    fn inspect(&self) -> InspectNode {
        InspectNode::List(self.iter().map(Inspectable::inspect).collect())
    }
}

/// Elements are ordered by their rendered form so the output is deterministic; equal bounded
/// labels (for example long strings sharing a prefix) are ordered by their full JSON rendering,
/// as in [`InspectNode::map_sorted`].
impl<T: Inspectable> Inspectable for HashSet<T> {
    fn inspect(&self) -> InspectNode {
        let mut items: Vec<InspectNode> = self.iter().map(Inspectable::inspect).collect();
        items.sort_by_cached_key(|item| {
            (
                crate::render::key_label(item),
                crate::render::render_json(item),
            )
        });
        InspectNode::List(items)
    }
}

impl<K: Inspectable, V: Inspectable> Inspectable for BTreeMap<K, V> {
    fn inspect(&self) -> InspectNode {
        InspectNode::Map(
            self.iter()
                .map(|(k, v)| (k.inspect(), v.inspect()))
                .collect(),
        )
    }
}

/// Entries are ordered by the rendered key so the output is deterministic.
impl<K: Inspectable, V: Inspectable, S> Inspectable for HashMap<K, V, S> {
    fn inspect(&self) -> InspectNode {
        InspectNode::map_sorted(
            self.iter()
                .map(|(k, v)| (k.inspect(), v.inspect()))
                .collect(),
        )
    }
}

macro_rules! inspect_tuple {
    ($($name:ident : $idx:tt),+) => {
        impl<$($name: Inspectable),+> Inspectable for ($($name,)+) {
            fn inspect(&self) -> InspectNode {
                InspectNode::List(vec![$(self.$idx.inspect()),+])
            }
        }
    };
}

inspect_tuple!(A: 0);
inspect_tuple!(A: 0, B: 1);
inspect_tuple!(A: 0, B: 1, C: 2);
inspect_tuple!(A: 0, B: 1, C: 2, D: 3);

/// Byte-string view: renders length, fingerprint and a bounded prefix instead of a list of
/// numbers.
#[derive(Clone, Copy, Debug)]
pub struct InspectBytes<'a>(pub &'a [u8]);

impl Inspectable for InspectBytes<'_> {
    fn inspect(&self) -> InspectNode {
        InspectNode::bytes(self.0)
    }
}

/// Implement [`Inspectable`] for a struct by listing the fields to show.
///
/// ```
/// use ptr_inspect::{inspect_struct, InspectNode, Inspectable, Secret};
///
/// struct Login {
///     user: String,
///     password: Secret<String>,
/// }
/// inspect_struct!(Login { user, password });
///
/// let node = Login { user: "ann".into(), password: Secret::new("pw".into()) }.inspect();
/// assert_eq!(
///     node,
///     InspectNode::new_struct(
///         "Login",
///         vec![("user", InspectNode::text("ann")), ("password", InspectNode::Redacted)],
///     )
/// );
/// ```
#[macro_export]
macro_rules! inspect_struct {
    ($ty:ident { $($field:ident),* $(,)? }) => {
        impl $crate::Inspectable for $ty {
            fn inspect(&self) -> $crate::InspectNode {
                $crate::InspectNode::new_struct(
                    ::core::stringify!($ty),
                    ::std::vec![$((
                        ::core::stringify!($field),
                        $crate::Inspectable::inspect(&self.$field),
                    )),*],
                )
            }
        }
    };
}

macro_rules! inspect_newtype {
    ($($ty:ident => $inner:tt),* $(,)?) => {$(
        impl Inspectable for $ty {
            fn inspect(&self) -> InspectNode {
                InspectNode::new_struct(stringify!($ty), vec![("value", self.$inner.inspect())])
            }
        }
    )*};
}

inspect_newtype!(
    Revision => 0, Generation => 0, CommitIndex => 0, FencingToken => 0, Timestamp => 0,
    ProjectId => 0, CapsuleId => 0, ArtifactId => 0, CapabilityId => 0, TypeId => 0,
    PodId => 0, PodIdentity => 0, CandidateId => 0, NodeId => 0,
    EvidenceId => 0, KnowledgeObjectId => 0, NamespaceId => 0, RawEventId => 0, StateId => 0,
    ScopeId => 0, ModelVersion => 0, AdapterVersion => 0, PolicyVersion => 0,
    EntityId => 0, DeviceId => 0, ExecutorIdentity => 0, KeyId => 0,
    RegionId => 0, ZoneId => 0, PeerId => 0, NetworkId => 0, PrincipalId => 0,
);

/// Digest text for a client-controlled string: `<label>:fnv1a64:<16 hex>`. Deterministic, so the
/// same value correlates across renderings, but the content is not shown. FNV-1a 64 is not a
/// cryptographic digest (`ptr-inspect` has no hash dependency): it hides a high-entropy
/// credential, not a low-entropy value an attacker can enumerate.
fn client_digest(label: &str, value: &str) -> InspectNode {
    InspectNode::text(format!(
        "{label}:{}",
        fingerprint_label(fnv1a64(value.as_bytes()))
    ))
}

/// Identifiers whose content a client chooses or carries (a session id can be a bearer
/// credential, a request or trace id arrives in headers, an invitation id is a join token, a
/// retrieval key is derived from user text), so they render as a digest, never verbatim.
/// System-assigned identifiers stay readable in `inspect_newtype!` above.
macro_rules! inspect_client_digest {
    ($($ty:ident => $label:literal),* $(,)?) => {$(
        impl Inspectable for $ty {
            fn inspect(&self) -> InspectNode {
                InspectNode::new_struct(stringify!($ty), vec![("value", client_digest($label, &self.0))])
            }
        }
    )*};
}

inspect_client_digest!(
    SessionId => "session",
    RequestId => "request",
    TraceId => "trace",
    InvitationId => "invitation",
    RetrievalKey => "retrieval",
);

macro_rules! inspect_unit_enum {
    ($($ty:ident),* $(,)?) => {$(
        impl Inspectable for $ty {
            fn inspect(&self) -> InspectNode {
                InspectNode::unit_variant(stringify!($ty), &format!("{self:?}"))
            }
        }
    )*};
}

inspect_unit_enum!(
    SemanticRole,
    EpistemicState,
    UncertaintyKind,
    ReasoningOperator,
    Effect,
    Validity,
    VerificationLevel,
    EventRole,
);

impl Inspectable for Probability {
    fn inspect(&self) -> InspectNode {
        InspectNode::new_struct("Probability", vec![("value", self.get().inspect())])
    }
}

impl Inspectable for ConfidenceTarget {
    fn inspect(&self) -> InspectNode {
        let (variant, fields) = match self {
            Self::SemanticRole(role) => ("SemanticRole", vec![("0", role.inspect())]),
            Self::ValueType(type_id) => ("ValueType", vec![("0", type_id.inspect())]),
            Self::Proposition => ("Proposition", Vec::new()),
            Self::Operator(operator) => ("Operator", vec![("0", operator.inspect())]),
        };
        InspectNode::variant("ConfidenceTarget", variant, fields)
    }
}

impl Inspectable for ConfidenceEstimate {
    fn inspect(&self) -> InspectNode {
        InspectNode::new_struct(
            "ConfidenceEstimate",
            vec![
                ("target", self.target().inspect()),
                ("probability", self.probability().inspect()),
            ],
        )
    }
}

impl Inspectable for ProvenanceRef {
    fn inspect(&self) -> InspectNode {
        InspectNode::new_struct(
            "ProvenanceRef",
            vec![
                ("source", self.source.inspect()),
                (
                    "note",
                    self.note
                        .as_deref()
                        .map_or(InspectNode::Null, |note| client_digest("note", note)),
                ),
            ],
        )
    }
}

impl<T: Inspectable> Inspectable for TypedValue<T> {
    fn inspect(&self) -> InspectNode {
        InspectNode::new_struct(
            "TypedValue",
            vec![
                ("value", self.value.inspect()),
                ("generation", self.generation.inspect()),
                ("validity", self.validity.inspect()),
                ("provenance", self.provenance.inspect()),
            ],
        )
    }
}

impl<T: Inspectable> Inspectable for Epistemic<T> {
    fn inspect(&self) -> InspectNode {
        let (variant, fields) = match self {
            Self::Unknown => ("Unknown", Vec::new()),
            Self::Assumed(value) => ("Assumed", vec![("0", value.inspect())]),
            Self::Hypothesis { value, confidence } => (
                "Hypothesis",
                vec![
                    ("value", value.inspect()),
                    ("confidence", confidence.inspect()),
                ],
            ),
            Self::Distribution(weighted) => (
                "Distribution",
                vec![(
                    "0",
                    InspectNode::List(
                        weighted
                            .iter()
                            .map(|(value, p)| InspectNode::List(vec![value.inspect(), p.inspect()]))
                            .collect(),
                    ),
                )],
            ),
            Self::Observed(value) => ("Observed", vec![("0", value.inspect())]),
            Self::Inferred(value) => ("Inferred", vec![("0", value.inspect())]),
            Self::Known(value) => ("Known", vec![("0", value.inspect())]),
        };
        InspectNode::variant("Epistemic", variant, fields)
    }
}

impl Inspectable for ActionIr {
    fn inspect(&self) -> InspectNode {
        InspectNode::new_struct(
            "ActionIr",
            vec![
                ("operation", self.operation.inspect()),
                ("target", self.target.inspect()),
                ("capability", self.capability.inspect()),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_are_lossless() {
        assert_eq!(u64::MAX.inspect(), InspectNode::UInt(u128::from(u64::MAX)));
        assert_eq!(i64::MIN.inspect(), InspectNode::Int(i128::from(i64::MIN)));
        assert_eq!(i128::MIN.inspect(), InspectNode::Int(i128::MIN));
        assert_eq!(u128::MAX.inspect(), InspectNode::UInt(u128::MAX));
    }

    #[test]
    fn hash_collections_are_sorted_deterministically() {
        let map: HashMap<String, u8> = ["z", "a", "m", "b"]
            .iter()
            .enumerate()
            .map(|(i, k)| ((*k).to_owned(), i as u8))
            .collect();
        let InspectNode::Map(entries) = map.inspect() else {
            panic!("expected map");
        };
        let keys: Vec<_> = entries.iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(
            keys,
            vec![
                InspectNode::text("a"),
                InspectNode::text("b"),
                InspectNode::text("m"),
                InspectNode::text("z")
            ]
        );
    }
}
