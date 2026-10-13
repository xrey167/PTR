//! Deterministic JSON rendering of control frames and negotiated sessions, for logs and
//! debugging. Rendering only: nothing here parses JSON and no wire format depends on it.
//!
//! Output is compact, keys appear in a fixed order, strings are escaped per RFC 8259 and
//! byte strings are lowercase hex. A payload longer than [`MAX_INLINE_PAYLOAD_BYTES`] is not
//! printed; its length and SHA-256 digest are, so a log line stays small and bounded and a
//! payload can still be matched against a digest.

use std::fmt::Write as _;

use sha2::{Digest as _, Sha256};

use crate::control::ControlFrame;
use crate::negotiation::{FeatureSet, NegotiatedSession, ProtocolVersion, VersionOffer};
use crate::TypedPayload;

/// Payloads up to this many bytes are rendered as hex; longer ones are elided.
pub const MAX_INLINE_PAYLOAD_BYTES: usize = 64;

/// A JSON string literal (with quotes) for `text`.
pub fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn version(v: ProtocolVersion) -> String {
    json_string(&v.to_string())
}

fn features(set: FeatureSet) -> String {
    let names: Vec<String> = set
        .members()
        .into_iter()
        .map(|feature| json_string(feature.name()))
        .collect();
    format!("[{}]", names.join(","))
}

fn payload(p: &TypedPayload) -> String {
    let head = format!(
        "{{\"type_id\":{},\"len\":{}",
        json_string(&p.type_id.0),
        p.bytes.len()
    );
    if p.bytes.len() <= MAX_INLINE_PAYLOAD_BYTES {
        format!("{head},\"bytes\":\"{}\"}}", hex(&p.bytes))
    } else {
        let digest: [u8; 32] = Sha256::digest(&p.bytes).into();
        format!("{head},\"elided\":true,\"sha256\":\"{}\"}}", hex(&digest))
    }
}

fn offer(o: &VersionOffer) -> String {
    let supported: Vec<String> = o.supported.iter().map(|v| version(*v)).collect();
    format!(
        "{{\"supported\":[{}],\"minimum\":{},\"features\":{},\"required\":{}}}",
        supported.join(","),
        version(o.minimum),
        features(o.features),
        features(o.required)
    )
}

/// Renders a negotiated session as `{"version":"1.1.0","features":["cancel"]}`.
pub fn render_negotiated(session: &NegotiatedSession) -> String {
    format!(
        "{{\"version\":{},\"features\":{}}}",
        version(session.version),
        features(session.features)
    )
}

/// Renders a control frame as one JSON object with a `"frame"` discriminator first.
pub fn render_control_frame(frame: &ControlFrame) -> String {
    let kind = json_string(frame.kind().name());
    match frame {
        ControlFrame::Hello {
            node_id,
            offer: o,
            nonce,
        } => format!(
            "{{\"frame\":{kind},\"node_id\":{},\"offer\":{},\"nonce\":{nonce}}}",
            json_string(node_id),
            offer(o)
        ),
        ControlFrame::Decl { capabilities } => {
            let items: Vec<String> = capabilities.iter().map(|c| json_string(&c.0)).collect();
            format!(
                "{{\"frame\":{kind},\"capabilities\":[{}]}}",
                items.join(",")
            )
        }
        ControlFrame::Ack {
            seq,
            idempotency_key,
        } => format!(
            "{{\"frame\":{kind},\"seq\":{seq},\"idempotency_key\":{}}}",
            json_string(idempotency_key.as_str())
        ),
        ControlFrame::Cancel { seq, reason } => format!(
            "{{\"frame\":{kind},\"seq\":{seq},\"reason\":{}}}",
            json_string(reason)
        ),
        ControlFrame::Bye { reason } => {
            format!("{{\"frame\":{kind},\"reason\":{}}}", json_string(reason))
        }
        ControlFrame::Call {
            seq,
            idempotency_key,
            payload: p,
        } => format!(
            "{{\"frame\":{kind},\"seq\":{seq},\"idempotency_key\":{},\"payload\":{}}}",
            json_string(idempotency_key.as_str()),
            payload(p)
        ),
        ControlFrame::Reply { seq, payload: p } => {
            format!(
                "{{\"frame\":{kind},\"seq\":{seq},\"payload\":{}}}",
                payload(p)
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::IdempotencyKey;
    use crate::negotiation::Feature;
    use ptr_types::{CapabilityId, TypeId};

    fn typed(bytes: Vec<u8>) -> TypedPayload {
        TypedPayload {
            type_id: TypeId::from("T\"x"),
            bytes,
        }
    }

    #[test]
    fn strings_are_escaped() {
        assert_eq!(json_string("a\"b\\c\n\t\r"), r#""a\"b\\c\n\t\r""#);
        assert_eq!(json_string("\u{01}\u{08}\u{0c}"), "\"\\u0001\\b\\f\"");
        assert_eq!(json_string("héllo \u{1F600}"), "\"héllo \u{1F600}\"");
    }

    #[test]
    fn small_payload_is_hex_and_large_payload_is_elided_with_a_digest() {
        let frame = ControlFrame::Reply {
            seq: 4,
            payload: typed(vec![0x00, 0xAB, 0xFF]),
        };
        assert_eq!(
            render_control_frame(&frame),
            r#"{"frame":"REPLY","seq":4,"payload":{"type_id":"T\"x","len":3,"bytes":"00abff"}}"#
        );

        let big = vec![7u8; MAX_INLINE_PAYLOAD_BYTES + 1];
        let rendered = render_control_frame(&ControlFrame::Reply {
            seq: 4,
            payload: typed(big.clone()),
        });
        let digest: [u8; 32] = Sha256::digest(&big).into();
        assert!(rendered.contains("\"len\":65,\"elided\":true"));
        assert!(rendered.contains(&format!("\"sha256\":\"{}\"", hex(&digest))));
        assert!(!rendered.contains("\"bytes\""));
        // The bound is inclusive.
        let edge = render_control_frame(&ControlFrame::Reply {
            seq: 1,
            payload: typed(vec![1u8; MAX_INLINE_PAYLOAD_BYTES]),
        });
        assert!(edge.contains("\"bytes\""));
    }

    #[test]
    fn every_frame_renders_deterministically() {
        let key = IdempotencyKey::new("k-1").unwrap();
        let hello = ControlFrame::Hello {
            node_id: "n".into(),
            offer: VersionOffer {
                supported: vec![ProtocolVersion::new(1, 0, 0)],
                minimum: ProtocolVersion::new(1, 0, 0),
                features: FeatureSet::of(&[Feature::Idempotency, Feature::Cancel]),
                required: FeatureSet::of(&[Feature::Idempotency]),
            },
            nonce: 9,
        };
        assert_eq!(
            render_control_frame(&hello),
            "{\"frame\":\"HELLO\",\"node_id\":\"n\",\"offer\":{\"supported\":[\"1.0.0\"],\
             \"minimum\":\"1.0.0\",\"features\":[\"idempotency\",\"cancel\"],\
             \"required\":[\"idempotency\"]},\"nonce\":9}"
        );
        assert_eq!(
            render_control_frame(&ControlFrame::Decl {
                capabilities: vec![CapabilityId::from("p"), CapabilityId::from("q")]
            }),
            r#"{"frame":"DECL","capabilities":["p","q"]}"#
        );
        assert_eq!(
            render_control_frame(&ControlFrame::Ack {
                seq: 2,
                idempotency_key: key.clone()
            }),
            r#"{"frame":"ACK","seq":2,"idempotency_key":"k-1"}"#
        );
        assert_eq!(
            render_control_frame(&ControlFrame::Cancel {
                seq: 2,
                reason: "why\n".into()
            }),
            r#"{"frame":"CANCEL","seq":2,"reason":"why\n"}"#
        );
        assert_eq!(
            render_control_frame(&ControlFrame::Bye { reason: "".into() }),
            r#"{"frame":"BYE","reason":""}"#
        );
        assert_eq!(
            render_control_frame(&ControlFrame::Call {
                seq: 1,
                idempotency_key: key,
                payload: typed(vec![]),
            }),
            r#"{"frame":"CALL","seq":1,"idempotency_key":"k-1","payload":{"type_id":"T\"x","len":0,"bytes":""}}"#
        );
    }

    #[test]
    fn negotiated_session_renders() {
        let session = NegotiatedSession {
            version: ProtocolVersion::new(1, 1, 0),
            features: FeatureSet::all(),
        };
        assert_eq!(
            render_negotiated(&session),
            r#"{"version":"1.1.0","features":["idempotency","cancel","drain"]}"#
        );
    }
}
