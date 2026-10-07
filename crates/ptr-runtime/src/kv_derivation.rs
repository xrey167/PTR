//! Derivation hops, segment admission and the clearing rule of the approximate-KV taint
//! (doc 36, phase P2a; experiment R012).
//!
//! [`PtrRuntime::admit_segments`] decides which taint a request's prompt segments carry, and
//! [`PtrRuntime::ingest_derived_run`] ingests a model run whose input was admitted that way,
//! so Approximate taint follows model output across hops instead of ending at the first run.
//!
//! # Segment admission
//!
//! * `Raw`: admitted only if its text is byte-identical to the request's own ingress record
//!   (`request:<id>:raw`, a text value). Anything else, including text that is another
//!   request's ingress value, a semantic value or a model output, is refused with
//!   [`RuntimeError::MisclassifiedRawSegment`] (fail closed). A tokenizer other than the byte
//!   tokenizer (`ptr.byte` version `1`) cannot be compared here, so a `Raw` segment under it is
//!   refused too. A Raw segment whose text is a live tainted Content subject (tainted output
//!   resubmitted as the request's own text) is admitted but Approximate: laundering keeps the
//!   taint, and honest identical text is tainted as well (accepted false positive).
//! * `Derived`: must name a derivation this runtime minted, else
//!   [`RuntimeError::UnknownDerivedSegment`]. A tainted derivation is recognised from the
//!   persistent registry (so also after a restart); a clean one only from this process's memory,
//!   so after a restart a clean derived id is refused (fail closed). Its taint is Approximate if
//!   its Derivation subject or its Content subject is live.
//! * `Semantic`, `Memory`: unchanged behaviour (caller-asserted), except that a live tainted
//!   Content subject makes them Approximate.
//!
//! Limits: only exact whole-segment digests are matched (`content_subject` over the segment's
//! text bytes). A paraphrase, substring, re-tokenisation or whitespace edit of tainted text is
//! NOT detected. Content checks need the byte tokenizer; under another one only the identity
//! rules (Derived) apply and `Raw` is refused.
//!
//! # Derived runs and hops
//!
//! A derived run's output taint is the join of its segments' taint and the run's own. When it
//! is Approximate, the Derivation subject (of the recipe), the Content subject of the generated
//! text, the Action subject of the run's action and the Request subject are recorded taint
//! first, before the result is returned. The returned [`DerivedRun`] carries a `Derived`
//! segment (id `kv-derived:<hex of the Derivation subject digest>`, generation 0) holding the
//! generated text, which the next hop feeds back in; its taint is read from the registry.
//! Minted derivations are remembered in memory (bounded like the registry) so a clean one is
//! recognised until a restart. A run that generated no text cannot mint a segment and is
//! refused with [`RuntimeError::DerivedRunUnavailable`] before anything is recorded.
//!
//! # Clearing
//!
//! Only `ingest_derived_run` clears, and only when ALL hold: the run is Exact AND cold
//! (`kv_report` is `None`, no KV reuse at all), every input segment was admitted untainted, and
//! the Derivation subject of the recipe is live with that same recipe digest. The Derivation
//! subject is then cleared; the Content subject of the recomputed text and the Action subject
//! of its action are cleared only if they are live under that recipe, that is, only if the
//! recomputed bytes give exactly the digests recorded. The recompute digest committed with the
//! clear is SHA-256 over `ptr.kv.taint.recompute.v1\0 || recipe || u64le(len) generated text
//! || action identity digest or 32 zero bytes`. The Request subject is never cleared.
//!
//! An exact echo (an Exact pass whose input includes the tainted derived segment) cannot clear:
//! its input taint is Approximate and its recipe differs, since the recipe covers the request's
//! lineage digest. A warm (KV-reusing) Exact recompute never clears.
//!
//! The runtime trusts the backend's report: a backend that labels an Approximate run as cold
//! Exact is outside this guarantee. Production loops do not yet call `ingest_derived_run`.

use crate::kv_taint::{
    content_subject, derivation_subject, put_len_prefixed, request_recipe, request_subject,
    run_action, taint_identity, KvTaintRequest, TaintSubject,
};
use crate::semantic::request_raw_key;
use crate::{PtrRuntime, ResumableRun, RuntimeError};
use ptr_ledger::{integrity, KvTaintSubjectKind};
use ptr_model_api::{
    ByteTokenizer, KvReuseTaint, ModelEvent, ModelRun, PromptSegment, SegmentSource,
    SegmentedModelRequest, Tokenizer,
};
use ptr_types::{ActionIr, Digest, EvidenceId, Generation, RequestId};

const DERIVED_ID_PREFIX: &str = "kv-derived:";
const DERIVED_GENERATION: Generation = Generation(0);
const RECOMPUTE_DOMAIN: &[u8] = b"ptr.kv.taint.recompute.v1\0";

/// The taint of every segment of an admitted request and their join.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmittedSegments {
    /// Join of all segment taints; Approximate if any segment is.
    pub taint: KvReuseTaint,
    /// The taint of each segment, in request order.
    pub segments: Vec<KvReuseTaint>,
}

/// What a derived run is ingested for.
#[derive(Clone, Copy, Debug)]
pub struct DerivedRunRequest<'a> {
    /// The request whose ingress record a `Raw` segment must equal.
    pub request_id: &'a RequestId,
    /// The segmented request the run answered; its segments are admitted first.
    pub request: &'a SegmentedModelRequest,
    /// The producer identity (backend name and version) that ran it.
    pub producer: &'a str,
    /// Canonical bytes of the generation parameters.
    pub generation_parameters: &'a [u8],
}

/// The result of one derivation hop.
#[derive(Clone, Debug, PartialEq)]
pub struct DerivedRun {
    /// The runtime-minted `Derived` segment holding the generated text, for the next hop.
    pub segment: PromptSegment,
    /// Input taint joined with the run's own.
    pub taint: KvReuseTaint,
    /// The recipe digest this derivation was recorded (or cleared) under.
    pub recipe_digest: Digest,
    /// The ingested run, as [`PtrRuntime::ingest_model_run_for`] returns it; its `kv_taint`
    /// is the joined output taint.
    pub resumable: ResumableRun,
}

/// The segment's text bytes under the byte tokenizer; `None` under any other tokenizer or for
/// a token that is no byte.
fn segment_bytes(segment: &PromptSegment) -> Option<Vec<u8>> {
    if segment.tokenizer() != ByteTokenizer::new().identity() {
        return None;
    }
    segment
        .tokens()
        .iter()
        .map(|token| u8::try_from(token.0).ok())
        .collect()
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The Derivation subject digest a derived segment id names.
fn parse_derived_id(id: &str) -> Option<[u8; 32]> {
    let text = id.strip_prefix(DERIVED_ID_PREFIX)?;
    if text.len() != 64 || !text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return None;
    }
    let mut out = [0u8; 32];
    for (slot, pair) in out.iter_mut().zip(text.as_bytes().chunks(2)) {
        *slot = u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?;
    }
    Some(out)
}

fn generated_text(run: &ModelRun) -> String {
    run.events
        .iter()
        .filter_map(|event| match event {
            ModelEvent::Token(text) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

fn recompute_digest(recipe: &Digest, text: &[u8], action: Option<&ActionIr>) -> Digest {
    let mut material = Vec::from(RECOMPUTE_DOMAIN);
    material.extend_from_slice(recipe);
    put_len_prefixed(&mut material, text);
    material.extend_from_slice(&action.map_or([0u8; 32], taint_identity));
    integrity::sha256(&material)
}

impl PtrRuntime {
    /// The taint of a live subject, Exact if it is not live.
    fn subject_kv_taint(&self, subject: &TaintSubject) -> KvReuseTaint {
        self.subject_taint(subject.0, subject.1)
            .map_or(KvReuseTaint::Exact, |entry| KvReuseTaint::Approximate {
                report_digest: entry.report_digest,
            })
    }

    /// The taint of a text by its exact whole-text Content digest.
    pub(crate) fn text_kv_taint(&self, text: &[u8]) -> KvReuseTaint {
        self.subject_kv_taint(&content_subject(text))
    }

    /// Admit the segments of `request`, returning each one's taint and their join.
    ///
    /// See the module documentation for the rules per segment source. Nothing is appended.
    ///
    /// # Errors
    /// [`RuntimeError::MisclassifiedRawSegment`] for a `Raw` segment that is not the request's
    /// own ingress text, [`RuntimeError::UnknownDerivedSegment`] for a `Derived` segment that
    /// names no derivation this runtime minted.
    pub fn admit_segments(
        &self,
        request_id: &RequestId,
        request: &SegmentedModelRequest,
    ) -> Result<AdmittedSegments, RuntimeError> {
        let snapshot = self.snapshot();
        let ingress = snapshot.get(&request_raw_key(request_id));
        let mut segments = Vec::with_capacity(request.segments().len());
        let mut joined = KvReuseTaint::Exact;
        for (index, segment) in request.segments().iter().enumerate() {
            let bytes = segment_bytes(segment);
            let content = bytes
                .as_deref()
                .map_or(KvReuseTaint::Exact, |bytes| self.text_kv_taint(bytes));
            let taint = match segment.source() {
                SegmentSource::Raw => {
                    let misclassified = |reason| RuntimeError::MisclassifiedRawSegment {
                        segment: index,
                        reason,
                    };
                    let bytes = bytes.as_deref().ok_or_else(|| {
                        misclassified("raw segment cannot be compared under this tokenizer")
                    })?;
                    let ingress = ingress
                        .ok_or_else(|| misclassified("request has no text ingress record"))?;
                    if bytes != ingress.as_bytes() {
                        return Err(misclassified(
                            "raw segment is not the request's own ingress text",
                        ));
                    }
                    content
                }
                SegmentSource::Derived { id, generation } => {
                    let derivation = parse_derived_id(&id.0)
                        .ok_or(RuntimeError::UnknownDerivedSegment { segment: index })?;
                    let live = self.subject_kv_taint(&(KvTaintSubjectKind::Derivation, derivation));
                    let minted = *generation == DERIVED_GENERATION
                        && self.taint.minted_segment(&derivation) == Some(segment.content_digest());
                    if !live.is_approximate() && !minted {
                        return Err(RuntimeError::UnknownDerivedSegment { segment: index });
                    }
                    live.join(content)
                }
                SegmentSource::Semantic { .. } | SegmentSource::Memory { .. } => content,
            };
            joined = joined.join(taint);
            segments.push(taint);
        }
        Ok(AdmittedSegments {
            taint: joined,
            segments,
        })
    }

    /// Ingest a model run whose input was `request`, one derivation hop.
    ///
    /// Admits the segments first (a refusal propagates and nothing is appended), joins their
    /// taint with the run's, and mints the `Derived` segment holding the generated text. If the
    /// output is Approximate, its Derivation, Content, Action and Request subjects are recorded
    /// taint first, before this returns; if it is Exact, cold and every input was clean, the
    /// clearing rule of the module documentation may clear a live derivation of the same recipe.
    ///
    /// # Errors
    /// Those of [`PtrRuntime::admit_segments`]; [`RuntimeError::MixedControlEvents`] and
    /// [`RuntimeError::MultipleControlEvents`] as for [`PtrRuntime::ingest_model_run`];
    /// [`RuntimeError::DerivedRunUnavailable`] when the run generated no text;
    /// [`RuntimeError::KvTaintRegistryFull`]; or a ledger error.
    pub fn ingest_derived_run(
        &mut self,
        request: &DerivedRunRequest<'_>,
        run: &ModelRun,
    ) -> Result<DerivedRun, RuntimeError> {
        let admitted = self.admit_segments(request.request_id, request.request)?;
        let action = run_action(run)?;
        let recipe = request_recipe(&KvTaintRequest {
            request: request.request,
            producer: request.producer,
            generation_parameters: request.generation_parameters,
        });
        let taint = admitted.taint.join(run.taint());
        let derivation = derivation_subject(&recipe);
        let text = generated_text(run);
        if text.is_empty() {
            return Err(RuntimeError::DerivedRunUnavailable {
                reason: "the run generated no text",
            });
        }
        let tokenizer = ByteTokenizer::new();
        let segment = PromptSegment::from_text(
            &tokenizer,
            SegmentSource::Derived {
                id: EvidenceId(format!("{DERIVED_ID_PREFIX}{}", hex(&derivation.1))),
                generation: DERIVED_GENERATION,
            },
            &text,
        )
        .map_err(|_| RuntimeError::DerivedRunUnavailable {
            reason: "the generated text is no valid segment",
        })?;
        self.taint.mint_room(&derivation.1)?;
        let content = content_subject(text.as_bytes());
        let action_subject = action
            .as_ref()
            .map(|action| (KvTaintSubjectKind::Action, taint_identity(action)));
        match taint {
            KvReuseTaint::Approximate { report_digest } => {
                let mut subjects = vec![
                    derivation,
                    content,
                    request_subject(request.request.request_id()),
                    request_subject(request.request_id),
                ];
                subjects.extend(action_subject);
                self.record_taint(&subjects, report_digest, recipe)?;
            }
            KvReuseTaint::Exact if run.kv_report.is_none() => {
                let recompute = recompute_digest(&recipe, text.as_bytes(), action.as_ref());
                self.clear_matching(&recipe, derivation, &[content], action_subject, recompute)?;
            }
            KvReuseTaint::Exact => {}
        }
        self.taint
            .remember_minted(derivation.1, *segment.content_digest());
        Ok(DerivedRun {
            segment,
            taint,
            recipe_digest: recipe,
            resumable: ResumableRun {
                model_events: run.events.clone(),
                observations: Vec::new(),
                action,
                kv_taint: taint,
            },
        })
    }

    /// Clear the Derivation subject if it is live under `recipe`, and the content and action
    /// subjects among those given that are live under that same recipe.
    fn clear_matching(
        &mut self,
        recipe: &Digest,
        derivation: TaintSubject,
        content: &[TaintSubject],
        action: Option<TaintSubject>,
        recompute: Digest,
    ) -> Result<(), RuntimeError> {
        let live_under_recipe = |runtime: &Self, subject: &TaintSubject| {
            runtime
                .subject_taint(subject.0, subject.1)
                .is_some_and(|entry| entry.recipe_digest == *recipe)
        };
        if !live_under_recipe(self, &derivation) {
            return Ok(());
        }
        let mut subjects = vec![derivation];
        subjects.extend(
            content
                .iter()
                .copied()
                .chain(action)
                .filter(|subject| live_under_recipe(self, subject)),
        );
        self.clear_taint(&subjects, *recipe, recompute)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_derived_id_round_trips_and_malformed_ids_are_refused() {
        let digest = [0xab; 32];
        let id = format!("{DERIVED_ID_PREFIX}{}", hex(&digest));
        assert_eq!(parse_derived_id(&id), Some(digest));
        assert_eq!(parse_derived_id("kv-derived:zz"), None);
        assert_eq!(parse_derived_id(&hex(&digest)), None);
        assert_eq!(
            parse_derived_id(&format!(
                "{DERIVED_ID_PREFIX}{}",
                hex(&digest).to_uppercase()
            )),
            None
        );
    }
}
