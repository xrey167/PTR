//! Carrying an adapter's own candidate type through the loop as a [`TypedPayload`].
//!
//! The runtime's output admission takes a typed payload (`admit_verified_answer`), while the
//! expert adapters work on their own candidate types ([`TextPayload`], a simulation's defect
//! document). [`Recoded`] runs the loop on the typed payload and translates at the boundary of
//! each call. It adds no decision of its own: it is a [`RolePods`] over another [`RolePods`], and
//! the adapters implement `RolePods` only through [`Guarded`](super::Guarded), so the call-time
//! eligibility check stays in force underneath it.

use ptr_protocol::TypedPayload;
use ptr_router::PodRef;

use super::task::TextPayload;
use crate::vcw::{EscalateCall, ProposeCall, RepairCall, RolePods, VcwPayload};

/// A two-way translation between an adapter's candidate type `Q` and the typed payload the loop
/// carries.
pub trait PayloadCodec<Q> {
    /// The typed payload for a candidate the adapter produced.
    fn encode(&mut self, inner: Q) -> TypedPayload;

    /// The adapter's candidate for a typed payload the loop hands back (a repair's previous
    /// candidate). `None` when the payload is not one this codec produced; the call is then
    /// declined.
    fn decode(&mut self, outer: &TypedPayload) -> Option<Q>;
}

/// The codec of [`TextPayload`]: the typed payload's bytes are the candidate text.
#[derive(Clone, Copy, Debug, Default)]
pub struct TextCodec;

impl PayloadCodec<TextPayload> for TextCodec {
    fn encode(&mut self, inner: TextPayload) -> TypedPayload {
        TypedPayload {
            type_id: inner.type_id().clone(),
            bytes: inner.text().as_bytes().to_vec(),
        }
    }

    fn decode(&mut self, outer: &TypedPayload) -> Option<TextPayload> {
        let text = std::str::from_utf8(&outer.bytes).ok()?;
        Some(TextPayload::new(outer.type_id.clone(), text))
    }
}

/// [`RolePods`] over typed payloads, answered by a [`RolePods`] over `Q` through a codec.
pub struct Recoded<'a, Q, C> {
    inner: &'a mut dyn RolePods<Q>,
    codec: C,
}

impl<'a, Q, C> Recoded<'a, Q, C> {
    #[must_use]
    pub fn new(inner: &'a mut dyn RolePods<Q>, codec: C) -> Self {
        Self { inner, codec }
    }

    /// The codec, for reading what it learned.
    #[must_use]
    pub fn codec(&self) -> &C {
        &self.codec
    }
}

impl<Q: VcwPayload, C: PayloadCodec<Q>> RolePods<TypedPayload> for Recoded<'_, Q, C> {
    fn propose(&mut self, call: ProposeCall<'_>) -> Option<TypedPayload> {
        let produced = self.inner.propose(call)?;
        Some(self.codec.encode(produced))
    }

    fn repair(&mut self, call: RepairCall<'_, TypedPayload>) -> Option<TypedPayload> {
        let previous = self.codec.decode(call.previous)?;
        let inner_call = RepairCall {
            pod: call.pod,
            request_id: call.request_id,
            round: call.round,
            code: call.code,
            critique: call.critique,
            previous: &previous,
            snapshot: call.snapshot,
        };
        let produced = self.inner.repair(inner_call)?;
        Some(self.codec.encode(produced))
    }

    fn escalate(&mut self, call: EscalateCall<'_, TypedPayload>) -> Option<(PodRef, TypedPayload)> {
        let previous = match call.previous {
            Some(payload) => Some(self.codec.decode(payload)?),
            None => None,
        };
        let inner_call = EscalateCall {
            request_id: call.request_id,
            round: call.round,
            reason: call.reason,
            previous: previous.as_ref(),
            snapshot: call.snapshot,
        };
        let (pod, produced) = self.inner.escalate(inner_call)?;
        Some((pod, self.codec.encode(produced)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_types::TypeId;

    #[test]
    fn the_text_codec_round_trips_and_refuses_non_utf8() {
        let mut codec = TextCodec;
        let payload = TextPayload::new(TypeId::from("t"), "42");
        let typed = codec.encode(payload.clone());
        assert_eq!(typed.bytes, b"42");
        assert_eq!(codec.decode(&typed), Some(payload));
        let binary = TypedPayload {
            type_id: TypeId::from("t"),
            bytes: vec![0xff],
        };
        assert_eq!(codec.decode(&binary), None);
    }
}
