//! The pinned stub Approximate producer of R012 (owner decision 4: harness-only).
//!
//! It wraps the cold path of [`ReferenceKvReuseBackend`] (`infer_cold`: no lookup, no insert),
//! so the content of every run is a genuine full prefill of the R005 tiny model with the pinned
//! shape. It adds two things the backend does not produce: a deterministic `ActionReady` whose
//! payload is the generated text, and a [`KvReuseReport`] it chooses. For
//! [`Reuse::Approximate`] the report claims [`KvReuseMode::Approximate`] with `AllowApproximate`
//! and a hit of one chunk; this experiment tests the containment of such a labelled result, not
//! any real Approximate producer, and the label is the stub's, not a measurement. The other
//! two modes are the controls: [`Reuse::Cold`] reports no reuse (`kv_report == None`, the only
//! kind of run that may clear taint) and [`Reuse::ExactWarm`] labels the same genuine content
//! Exact with a hit of one chunk (a stand-in for a warm Exact run: R005 shows real Exact reuse
//! is bitwise equal to cold, so the content is the same).
//!
//! Name and version are pinned here and recorded in every run's JSON.

use super::params;
use crate::experiments::layered_kv::corpus::CHUNK_TOKENS;
use crate::experiments::layered_kv::lab::{self, fail, Res};
use ptr_model_api::{
    KvReuseMode, KvReusePolicy, KvReuseReport, ModelEvent, ModelRun, SegmentedModelRequest,
};
use ptr_runtime::{ReferenceKvOutcome, ReferenceKvReuseBackend};
use ptr_types::{ActionIr, CapabilityId, Effect, Generation, Revision, TypeId};

/// The producer identity bound into every recipe: `name/version`.
pub fn identity() -> String {
    format!("{}/{}", params::STUB_NAME, params::STUB_VERSION)
}

/// The generation parameters bound into every recipe.
pub fn parameters() -> &'static [u8] {
    params::GENERATION_PARAMETERS.as_bytes()
}

/// What the stub tells the runtime about KV reuse.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Reuse {
    /// An Approximate report with `AllowApproximate` (the producer under test).
    Approximate,
    /// An Exact report with a hit of one chunk (a warm Exact run stand-in).
    ExactWarm,
    /// No report: a cold run, no reuse at all.
    Cold,
}

/// The target, operation and capability of the action every run carries.
pub const TARGET: &str = "artifact:r012";
pub const OPERATION: &str = "write";
pub const CAPABILITY: &str = "file.write";
pub const INPUT_TYPE: &str = "Bytes";

/// The action a run with generated text `text` asks for. Deterministic: the payload is the
/// text, the revision is 0 (a caller stamps the current one when it needs to authorize).
pub fn action_for(text: &str) -> ActionIr {
    ActionIr {
        operation: OPERATION.into(),
        target: TARGET.into(),
        capability: CapabilityId::from(CAPABILITY),
        effect: Effect::Mutation,
        input_type: TypeId::from(INPUT_TYPE),
        generation: Generation(1),
        revision: Revision(0),
        payload: text.as_bytes().to_vec(),
    }
}

/// One stub run.
#[derive(Debug)]
pub struct StubRun {
    /// What the runtime ingests: the generated text, the action and the report.
    pub run: ModelRun,
    /// The generated text (the backend's `Token` events, concatenated).
    pub text: String,
    pub action: ActionIr,
    /// The genuine cold outcome the content comes from.
    pub outcome: ReferenceKvOutcome,
}

/// The generated text of a run's `Token` events.
pub fn generated_text(events: &[ModelEvent]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            ModelEvent::Token(text) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// Run `request` cold on the backend and wrap the result as the stub's output.
pub fn produce(
    backend: &ReferenceKvReuseBackend,
    request: &SegmentedModelRequest,
    reuse: Reuse,
) -> Res<StubRun> {
    let outcome = backend
        .infer_cold(&lab::scope(), request)
        .map_err(fail("the stub's cold run"))?;
    let text = generated_text(&outcome.run.events);
    if text.is_empty() {
        return Err("the stub's cold run generated no text".to_string());
    }
    let action = action_for(&text);
    let mut events = Vec::new();
    for event in &outcome.run.events {
        if matches!(event, ModelEvent::Finished) {
            events.push(ModelEvent::ActionReady(action.clone()));
        }
        events.push(event.clone());
    }
    let total = request.total_tokens();
    let hit = CHUNK_TOKENS.min(total);
    let kv_report = match reuse {
        Reuse::Cold => None,
        Reuse::ExactWarm => Some(
            KvReuseReport::new(hit, KvReuseMode::Exact, total, KvReusePolicy::ExactOnly)
                .map_err(fail("the Exact report"))?,
        ),
        Reuse::Approximate => Some(
            KvReuseReport::new(
                hit,
                KvReuseMode::Approximate,
                total,
                KvReusePolicy::AllowApproximate,
            )
            .map_err(fail("the Approximate report"))?,
        ),
    };
    Ok(StubRun {
        run: ModelRun { events, kv_report },
        text,
        action,
        outcome,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::corpus::{raw_request, sequence};
    use ptr_model_api::KvReuseTaint;

    fn prompt() -> Vec<u32> {
        sequence(
            params::CORPUS_GENERATOR_SEED,
            3,
            params::PROMPT_TOKENS as usize,
        )
    }

    #[test]
    fn the_stub_is_approximate_and_its_content_is_the_cold_reference_output() {
        let backend = lab::backend().unwrap();
        let request = raw_request("stub-test", &prompt());
        let run = produce(&backend, &request, Reuse::Approximate).unwrap();
        let report = run.run.kv_report.expect("a report");
        assert_eq!(report.mode(), KvReuseMode::Approximate);
        assert_eq!(report.hit_tokens(), CHUNK_TOKENS);
        assert!(run.run.taint().is_approximate());
        // The content is the backend's own cold output for the same prompt, bit for bit.
        let cold = lab::cold(&backend, &request, &ptr_runtime::StackBinding::base()).unwrap();
        assert_eq!(run.outcome.generated, cold.generated);
        assert_eq!(run.outcome.logits_digest, cold.logits_digest);
        assert_eq!(generated_text(&cold.run.events), run.text);
        // The independent stack agrees with the backend (the pin check of `verify_pins`).
        let stack = crate::experiments::layered_kv::pins::reference_stack().unwrap();
        let (generated, logits) =
            crate::experiments::layered_kv::pins::independent_cold_run(&stack, None, &prompt())
                .unwrap();
        let tokens: Vec<u32> = run.outcome.generated.iter().map(|token| token.0).collect();
        assert_eq!(tokens, generated);
        assert_eq!(run.outcome.logits_digest, logits);
    }

    #[test]
    fn the_action_is_deterministic_and_carries_the_generated_text() {
        let backend = lab::backend().unwrap();
        let request = raw_request("stub-test", &prompt());
        let a = produce(&backend, &request, Reuse::Approximate).unwrap();
        let b = produce(&backend, &request, Reuse::Cold).unwrap();
        assert_eq!(a.action, b.action);
        assert_eq!(a.action.payload, a.text.as_bytes());
        let in_events = a
            .run
            .events
            .iter()
            .filter(|event| matches!(event, ModelEvent::ActionReady(action) if *action == a.action))
            .count();
        assert_eq!(in_events, 1);
        assert!(matches!(a.run.events.last(), Some(ModelEvent::Finished)));
    }

    #[test]
    fn the_controls_report_what_they_say() {
        let backend = lab::backend().unwrap();
        let request = raw_request("stub-test", &prompt());
        let cold = produce(&backend, &request, Reuse::Cold).unwrap();
        assert!(cold.run.kv_report.is_none());
        assert_eq!(cold.run.taint(), KvReuseTaint::Exact);
        let warm = produce(&backend, &request, Reuse::ExactWarm).unwrap();
        assert_eq!(
            warm.run.kv_report.map(|report| report.mode()),
            Some(KvReuseMode::Exact)
        );
        assert_eq!(warm.run.taint(), KvReuseTaint::Exact);
        assert_eq!(warm.text, cold.text);
    }

    #[test]
    fn the_identity_is_pinned() {
        assert_eq!(identity(), "r012-stub-approximate-producer/1");
    }
}
