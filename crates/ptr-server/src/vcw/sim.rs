//! The candidates of the simulated backend.
//!
//! A simulated expert knows a task's clean answer (the request's `reference`) and plants and
//! clears *defect codes* with configured, seeded probabilities. The candidate it hands the loop
//! is the clean answer corrupted according to the codes it carries, rendered as the text the real
//! verifier judges. The verifier, not the simulation, decides what is wrong with that text: a
//! reference that is itself wrong stays wrong however many defects are cleared, and the request
//! ends `Exhausted`.

use std::collections::{BTreeMap, BTreeSet};

use ptr_config::VcwTaskKind;
use ptr_protocol::TypedPayload;
use ptr_runtime::vcw::experts::{DefectPayload, PayloadCodec};
use ptr_runtime::vcw::VcwPayload;
use ptr_types::{Digest, TypeId};
use sha2::{Digest as _, Sha256};

use super::job::{answer_type, claims_text, SimJob};
use super::wire::VcwClaim;

/// The defect codes the simulation can plant in a candidate of `kind`. Every one is a hard code
/// of the `ptr-verifier` table, so the real verifier can report it and the router can route it.
#[must_use]
pub fn plantable(kind: VcwTaskKind) -> &'static [&'static str] {
    match kind {
        VcwTaskKind::Arithmetic => &[
            "arith.value",
            "arith.format",
            "arith.bounds",
            "arith.oversize",
        ],
        VcwTaskKind::JsonObject => &["json.parse", "json.required", "json.type", "json.extra"],
        VcwTaskKind::Code => &[
            "code.syntax",
            "code.test_failed",
            "code.timeout",
            "code.resource",
        ],
        VcwTaskKind::Fact => &["fact.contradicted", "fact.malformed"],
    }
}

/// A clean answer and the defects it currently carries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ServedDefects {
    job: SimJob,
    defects: BTreeSet<String>,
}

impl ServedDefects {
    fn kind(&self) -> VcwTaskKind {
        self.job.kind.unwrap_or(VcwTaskKind::Arithmetic)
    }

    fn has(&self, code: &str) -> bool {
        self.defects.contains(code)
    }

    /// The text the real verifier judges.
    pub(super) fn render(&self) -> String {
        match self.kind() {
            VcwTaskKind::Arithmetic => self.render_arithmetic(),
            VcwTaskKind::JsonObject => self.render_json(),
            VcwTaskKind::Code => self.render_code(),
            VcwTaskKind::Fact => self.render_fact(),
        }
    }

    fn render_arithmetic(&self) -> String {
        let reference = &self.job.reference;
        if self.has("arith.format") {
            "n/a".to_owned()
        } else if self.has("arith.oversize") {
            "9".repeat(200)
        } else if self.has("arith.bounds") {
            "9".repeat(40)
        } else if self.has("arith.value") {
            reference
                .trim()
                .parse::<i128>()
                .ok()
                .and_then(|n| n.checked_add(1))
                .map_or_else(|| format!("{reference}1"), |n| n.to_string())
        } else {
            reference.clone()
        }
    }

    fn render_json(&self) -> String {
        let Ok(mut value) = serde_json::from_str::<serde_json::Value>(&self.job.reference) else {
            return self.job.reference.clone();
        };
        if let Some(object) = value.as_object_mut() {
            if self.has("json.required") {
                if let Some(first) = object.keys().next().cloned() {
                    object.remove(&first);
                }
            } else if self.has("json.type") {
                if let Some(last) = object.keys().next_back().cloned() {
                    object.insert(last, serde_json::Value::Null);
                }
            }
            if self.has("json.extra") {
                object.insert("zz_unexpected".to_owned(), serde_json::Value::from(0));
            }
        }
        let mut text = serde_json::to_string(&value).unwrap_or_default();
        if self.has("json.parse") {
            text.pop();
        }
        text
    }

    fn render_code(&self) -> String {
        let params: Vec<String> = (0..self.job.arity).map(|i| format!("a{i}")).collect();
        let params = params.join(", ");
        let first = if self.job.arity == 0 { "0" } else { "a0" };
        let entry = &self.job.entry;
        if self.has("code.syntax") {
            format!("fn {entry}({params}) {{ return {first} * {first} }}")
        } else if self.has("code.timeout") {
            format!("fn {entry}({params}) {{ while 1 {{ }} return 0; }}")
        } else if self.has("code.resource") {
            format!("fn {entry}({params}) {{ return {entry}({first} + 1); }}")
        } else if self.has("code.test_failed") {
            format!("fn {entry}({params}) {{ return 1234567891; }}")
        } else {
            self.job.reference.clone()
        }
    }

    fn render_fact(&self) -> String {
        let Ok(mut claims) = serde_json::from_str::<Vec<VcwClaim>>(&self.job.reference) else {
            return self.job.reference.clone();
        };
        if let Some(first) = claims.first_mut() {
            if self.has("fact.malformed") {
                first.subject.clear();
            } else if self.has("fact.contradicted") {
                first.value.push_str("-contradicted");
            }
        }
        claims_text(&claims)
    }
}

impl VcwPayload for ServedDefects {
    fn payload_type(&self) -> TypeId {
        answer_type(self.kind())
    }

    fn payload_digest(&self) -> Digest {
        framed_digest(
            SERVED_DIGEST_DOMAIN,
            self.payload_type().0.as_bytes(),
            self.render().as_bytes(),
        )
    }

    fn confined_to(&self, edited: &Self, named: &[String]) -> bool {
        let rest = |doc: &Self| -> BTreeSet<String> {
            doc.defects
                .iter()
                .filter(|code| !named.contains(code))
                .cloned()
                .collect()
        };
        self.job == edited.job && rest(self) == rest(edited)
    }
}

impl DefectPayload for ServedDefects {
    /// `task` is the JSON of a [`SimJob`].
    fn clean(task: &str) -> Self {
        Self {
            job: serde_json::from_str(task).unwrap_or_default(),
            defects: BTreeSet::new(),
        }
    }

    fn defects(&self) -> BTreeSet<String> {
        self.defects.clone()
    }

    fn with_defect(&self, code: &str) -> Self {
        let mut next = self.clone();
        next.defects.insert(code.to_owned());
        next
    }

    fn without_defect(&self, code: &str) -> Self {
        let mut next = self.clone();
        next.defects.remove(code);
        next
    }
}

/// Remembers the defects of each candidate it encoded, so a repair's previous candidate (which
/// the loop hands back as typed bytes) can be turned back into the document with its defects.
/// One per request: it holds at most one entry per candidate, and a request has at most
/// `max_verifier_calls` of them.
#[derive(Default)]
pub(super) struct SimCodec {
    seen: BTreeMap<Digest, ServedDefects>,
}

fn digest_of(payload: &TypedPayload) -> Digest {
    framed_digest(
        CODEC_DIGEST_DOMAIN,
        payload.type_id.0.as_bytes(),
        &payload.bytes,
    )
}

const SERVED_DIGEST_DOMAIN: &[u8] = b"ptr.server.vcw.sim.served.v2";
const CODEC_DIGEST_DOMAIN: &[u8] = b"ptr.server.vcw.sim.codec.v2";

/// v2 (issue #141): a domain tag, then the type id and the bytes, each length-prefixed. v1
/// joined them with a 0 byte, so a type id holding a 0 byte aliased another split.
fn framed_digest(domain: &[u8], type_id: &[u8], bytes: &[u8]) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for field in [type_id, bytes] {
        hasher.update((field.len() as u64).to_le_bytes());
        hasher.update(field);
    }
    hasher.finalize().into()
}

impl PayloadCodec<ServedDefects> for SimCodec {
    fn encode(&mut self, inner: ServedDefects) -> TypedPayload {
        let payload = TypedPayload {
            type_id: inner.payload_type(),
            bytes: inner.render().into_bytes(),
        };
        self.seen.insert(digest_of(&payload), inner);
        payload
    }

    fn decode(&mut self, outer: &TypedPayload) -> Option<ServedDefects> {
        self.seen.get(&digest_of(outer)).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Issue #141: v1 joined the type id and the bytes with a 0 byte, so the boundary could move
    /// and the codec could hand back another candidate's defects.
    #[test]
    fn a_zero_byte_in_the_type_or_bytes_does_not_alias() {
        let left = TypedPayload {
            type_id: TypeId::from("a\0b"),
            bytes: b"c".to_vec(),
        };
        let right = TypedPayload {
            type_id: TypeId::from("a"),
            bytes: b"b\0c".to_vec(),
        };
        assert_ne!(digest_of(&left), digest_of(&right));
        assert_ne!(
            framed_digest(SERVED_DIGEST_DOMAIN, b"a\0b", b"c"),
            framed_digest(SERVED_DIGEST_DOMAIN, b"a", b"b\0c")
        );
    }

    fn clean(kind: VcwTaskKind, reference: &str, entry: &str, arity: usize) -> ServedDefects {
        let job = SimJob {
            kind: Some(kind),
            reference: reference.to_owned(),
            entry: entry.to_owned(),
            arity,
        };
        ServedDefects::clean(&serde_json::to_string(&job).unwrap())
    }

    #[test]
    fn a_clean_candidate_renders_the_reference_and_each_defect_changes_it() {
        let arithmetic = clean(VcwTaskKind::Arithmetic, "35", "", 0);
        assert_eq!(arithmetic.render(), "35");
        assert_eq!(arithmetic.with_defect("arith.value").render(), "36");
        assert_eq!(arithmetic.with_defect("arith.format").render(), "n/a");
        let both = arithmetic
            .with_defect("arith.value")
            .with_defect("arith.format");
        assert_eq!(both.without_defect("arith.format").render(), "36");
        assert_eq!(
            both.without_defect("arith.format")
                .without_defect("arith.value")
                .render(),
            "35"
        );

        let json = clean(VcwTaskKind::JsonObject, r#"{"a":1,"b":"x"}"#, "", 0);
        assert_eq!(json.render(), r#"{"a":1,"b":"x"}"#);
        assert_eq!(json.with_defect("json.required").render(), r#"{"b":"x"}"#);
        assert_eq!(
            json.with_defect("json.type").render(),
            r#"{"a":1,"b":null}"#
        );
        assert_eq!(json.with_defect("json.parse").render(), r#"{"a":1,"b":"x""#);

        let code = clean(VcwTaskKind::Code, "fn f(a0) { return a0 * a0; }", "f", 1);
        assert_eq!(code.render(), "fn f(a0) { return a0 * a0; }");
        assert_eq!(
            code.with_defect("code.syntax").render(),
            "fn f(a0) { return a0 * a0 }"
        );

        let claims = r#"[{"subject":"paris","attribute":"capital-of","value":"france"}]"#;
        let fact = clean(VcwTaskKind::Fact, claims, "", 0);
        assert_eq!(fact.render(), claims);
        assert!(fact
            .with_defect("fact.contradicted")
            .render()
            .contains("france-contradicted"));
    }

    #[test]
    fn the_codec_restores_the_defects_of_a_candidate_it_encoded() {
        let mut codec = SimCodec::default();
        let doc = clean(VcwTaskKind::Arithmetic, "35", "", 0).with_defect("arith.value");
        let typed = codec.encode(doc.clone());
        assert_eq!(typed.bytes, b"36");
        assert_eq!(codec.decode(&typed), Some(doc));
        let stranger = TypedPayload {
            type_id: answer_type(VcwTaskKind::Arithmetic),
            bytes: b"37".to_vec(),
        };
        assert_eq!(codec.decode(&stranger), None);
    }

    #[test]
    fn every_plantable_code_is_a_hard_code_of_the_table() {
        for kind in VcwTaskKind::ALL {
            for code in plantable(kind) {
                let spec = ptr_verifier::code_spec(code).unwrap_or_else(|| panic!("{code}"));
                assert!(spec.hard, "{code}");
            }
        }
    }
}
