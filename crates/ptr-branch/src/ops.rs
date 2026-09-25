use std::collections::BTreeSet;

use ptr_semdb::{SemanticPayload, SemanticValue};
use ptr_types::TypeId;

use crate::error::BranchError;

/// Type of a counter value: eight little-endian bytes of an `i64`.
pub const COUNTER_TYPE: &str = "ptr.counter.i64";
/// Type of a set value: a little-endian `u32` member count, then each member
/// as a little-endian `u32` length and its UTF-8 bytes, members sorted and
/// distinct.
pub const SET_TYPE: &str = "ptr.set.utf8";
/// Provenance recorded on counter and set values a merge produces.
pub const OP_SOURCE: &str = "ptr-branch/op";

/// One staged change to a semantic key.
///
/// `Put` and `Remove` overwrite, so they are only accepted for keys the branch
/// read. `Add`, `SetInsert` and `SetRemove` are defined on whatever the key
/// holds at merge time: the merge applies them to the target snapshot's value
/// and publishes the resulting absolute value, so two branches that each add
/// to a counter both keep their additions. A domain invariant such as "stock
/// never below zero" is not preserved by such a rebase; the verification that
/// every merge passes has to check it on the rebased state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BranchOp {
    Put {
        key: String,
        value: SemanticValue,
    },
    Remove {
        key: String,
    },
    /// Add to a counter; an absent counter is zero.
    Add {
        key: String,
        amount: i64,
    },
    /// Insert a member into a set; an absent set is empty.
    SetInsert {
        key: String,
        member: String,
    },
    SetRemove {
        key: String,
        member: String,
    },
}

impl BranchOp {
    pub fn key(&self) -> &str {
        match self {
            Self::Put { key, .. }
            | Self::Remove { key }
            | Self::Add { key, .. }
            | Self::SetInsert { key, .. }
            | Self::SetRemove { key, .. } => key,
        }
    }

    /// Whether the op is defined on the value present at merge time rather than
    /// on the value the branch observed.
    pub fn commutes(&self) -> bool {
        match self {
            Self::Put { .. } | Self::Remove { .. } => false,
            Self::Add { .. } | Self::SetInsert { .. } | Self::SetRemove { .. } => true,
        }
    }

    /// The value `current` becomes under this op. `None` is an absent key.
    pub(crate) fn apply(
        &self,
        current: Option<SemanticValue>,
    ) -> Result<Option<SemanticValue>, BranchError> {
        match self {
            Self::Put { value, .. } => Ok(Some(value.clone())),
            Self::Remove { .. } => Ok(None),
            Self::Add { key, amount } => {
                let now = read_counter(key, current.as_ref())?;
                let next = now
                    .checked_add(*amount)
                    .ok_or_else(|| BranchError::CounterOverflow { key: key.clone() })?;
                Ok(Some(counter_value(next)))
            }
            Self::SetInsert { key, member } => {
                check_member(key, member)?;
                let mut set = read_set(key, current.as_ref())?;
                set.insert(member.clone());
                Ok(Some(set_value(&set)))
            }
            Self::SetRemove { key, member } => {
                check_member(key, member)?;
                let mut set = read_set(key, current.as_ref())?;
                set.remove(member);
                Ok(Some(set_value(&set)))
            }
        }
    }
}

/// A counter value.
pub fn counter_value(count: i64) -> SemanticValue {
    SemanticValue::Payload(SemanticPayload {
        type_id: TypeId::from(COUNTER_TYPE),
        source: OP_SOURCE.to_owned(),
        bytes: count.to_le_bytes().to_vec(),
    })
}

/// A set value in canonical form.
pub fn set_value(members: &BTreeSet<String>) -> SemanticValue {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(members.len() as u32).to_le_bytes());
    for member in members {
        bytes.extend_from_slice(&(member.len() as u32).to_le_bytes());
        bytes.extend_from_slice(member.as_bytes());
    }
    SemanticValue::Payload(SemanticPayload {
        type_id: TypeId::from(SET_TYPE),
        source: OP_SOURCE.to_owned(),
        bytes,
    })
}

/// Read a counter value; `None` for anything that is not one.
pub fn read_counter_value(value: &SemanticValue) -> Option<i64> {
    match value {
        SemanticValue::Payload(payload) if payload.type_id.0 == COUNTER_TYPE => {
            let bytes: [u8; 8] = payload.bytes.as_slice().try_into().ok()?;
            Some(i64::from_le_bytes(bytes))
        }
        SemanticValue::Payload(_) | SemanticValue::Text(_) => None,
    }
}

/// Read a set value; `None` for anything that is not a canonical set.
pub fn read_set_value(value: &SemanticValue) -> Option<BTreeSet<String>> {
    let SemanticValue::Payload(payload) = value else {
        return None;
    };
    if payload.type_id.0 != SET_TYPE {
        return None;
    }
    let bytes = payload.bytes.as_slice();
    let count = u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?) as usize;
    let mut offset = 4;
    let mut members = Vec::with_capacity(count.min(1024));
    for _ in 0..count {
        let len = u32::from_le_bytes(bytes.get(offset..offset + 4)?.try_into().ok()?) as usize;
        offset += 4;
        let member = std::str::from_utf8(bytes.get(offset..offset + len)?).ok()?;
        offset += len;
        members.push(member.to_owned());
    }
    let canonical = offset == bytes.len()
        && members.windows(2).all(|pair| pair[0] < pair[1])
        && members.iter().all(|member| !member.is_empty());
    canonical.then(|| members.into_iter().collect())
}

fn read_counter(key: &str, value: Option<&SemanticValue>) -> Result<i64, BranchError> {
    match value {
        None => Ok(0),
        Some(value) => read_counter_value(value).ok_or_else(|| BranchError::NotACounter {
            key: key.to_owned(),
        }),
    }
}

fn read_set(key: &str, value: Option<&SemanticValue>) -> Result<BTreeSet<String>, BranchError> {
    match value {
        None => Ok(BTreeSet::new()),
        Some(value) => read_set_value(value).ok_or_else(|| BranchError::NotASet {
            key: key.to_owned(),
        }),
    }
}

fn check_member(key: &str, member: &str) -> Result<(), BranchError> {
    if member.is_empty() {
        return Err(BranchError::InvalidMember {
            key: key.to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn additions_commute_in_either_order() {
        let a = BranchOp::Add {
            key: "k".into(),
            amount: 5,
        };
        let b = BranchOp::Add {
            key: "k".into(),
            amount: -2,
        };
        let start = Some(counter_value(10));
        let ab = b.apply(a.apply(start.clone()).unwrap()).unwrap();
        let ba = a.apply(b.apply(start).unwrap()).unwrap();
        assert_eq!(ab, ba);
        assert_eq!(read_counter_value(&ab.unwrap()), Some(13));
    }

    #[test]
    fn set_members_stay_sorted_distinct_and_may_contain_any_text() {
        let insert = |member: &str| BranchOp::SetInsert {
            key: "s".into(),
            member: member.into(),
        };
        let value = insert("b").apply(None).unwrap();
        let value = insert("a\nnewline").apply(value).unwrap();
        let value = insert("b").apply(value).unwrap();
        let members = read_set_value(&value.unwrap()).unwrap();
        assert_eq!(
            members.into_iter().collect::<Vec<_>>(),
            vec!["a\nnewline".to_owned(), "b".to_owned()]
        );
    }

    #[test]
    fn a_text_value_is_neither_a_counter_nor_a_set() {
        let text = Some(SemanticValue::Text("10".into()));
        let add = BranchOp::Add {
            key: "c".into(),
            amount: 1,
        };
        assert_eq!(
            add.apply(text.clone()).unwrap_err(),
            BranchError::NotACounter { key: "c".into() }
        );
        let insert = BranchOp::SetInsert {
            key: "c".into(),
            member: "m".into(),
        };
        assert_eq!(
            insert.apply(text).unwrap_err(),
            BranchError::NotASet { key: "c".into() }
        );
    }

    #[test]
    fn counter_overflow_is_refused() {
        let add = BranchOp::Add {
            key: "c".into(),
            amount: 1,
        };
        assert_eq!(
            add.apply(Some(counter_value(i64::MAX))).unwrap_err(),
            BranchError::CounterOverflow { key: "c".into() }
        );
    }

    #[test]
    fn a_malformed_set_payload_is_refused_rather_than_repaired() {
        let mut bytes = set_value(&BTreeSet::from(["a".to_owned()]));
        if let SemanticValue::Payload(payload) = &mut bytes {
            payload.bytes.push(0);
        }
        assert_eq!(read_set_value(&bytes), None);
    }
}
