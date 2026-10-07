//! C8 (tainted output resubmitted as Raw, over hops) and C9 (content presented as Raw that is
//! not the request's own ingress record, and forged Derived segments).

use super::fixture::{install_grant, stamped, Rt};
use super::{classify, derive, own_request, raw_segment, segmented, tainted_base, Cx, Verdict};
use crate::experiments::layered_kv::lab::{fail, Res};
use crate::experiments::r012::stub::{self, Reuse};
use ptr_model_api::{ByteTokenizer, KvReuseTaint, ModelError, PromptSegment, SegmentSource};
use ptr_runtime::kv_taint::KvTaintRequest;
use ptr_runtime::RuntimeError;
use ptr_semdb::SemanticDelta;
use ptr_types::{EvidenceId, Generation, PrincipalId, RequestId};

/// Whether a laundering hop's result was admitted as Exact: its output is not Approximate, or
/// its action is authorized.
fn laundered(taint: KvReuseTaint, authorized: bool) -> bool {
    !taint.is_approximate() || authorized
}

/// C8: tainted output resubmitted as Raw, as the request's own text (the only way Raw is
/// admitted), over two hops through the derived path, then through the server path and for
/// a plain Approximate run's output.
pub fn c8_raw_laundering(cx: &mut Cx) -> Res<()> {
    let mut rt = Rt::fresh()?;
    cx.sub("C8 derived output as Raw");
    let base = tainted_base(cx, &mut rt, 0)?;

    // Two laundering hops: each takes the previous output as the whole Raw text of a new request.
    let mut previous = base.run.text.clone();
    for hop in 1..=2usize {
        let id = cx.rid("launder");
        let request = own_request(&mut rt.runtime, &id, &previous)?;
        // The Raw segment is admitted (it is the request's own text) but carries the taint.
        let admitted = rt
            .runtime
            .admit_segments(&id, &request)
            .map_err(fail("admit_segments of laundered Raw"))?;
        cx.attempt("approximate_output_reintroduced_as_raw_attempts", &rt);
        let mut escaped = !admitted.taint.is_approximate();
        let run = stub::produce(&cx.backend, &request, Reuse::Cold)?;
        let derived =
            derive(&mut rt.runtime, &request, &run.run).map_err(fail("a laundering hop"))?;
        let authorized = rt
            .runtime
            .authorize_action(&stamped(&rt.runtime, &run.action))
            .is_ok();
        escaped |= laundered(derived.taint, authorized);
        if escaped {
            cx.count("laundered_exact_admissions", 1);
        }
        cx.count("laundering_hops", 1);
        previous = run.text.clone();
        let _ = hop;
    }

    // The server path: the tainted text arrives as the raw text of a run_model_run_resumable
    // call whose backend run is Exact and cold.
    cx.sub("C8 derived output as Raw (server path)");
    let id = cx.rid("server");
    let request = segmented(&id, vec![raw_segment(base.text())?])?;
    let run = stub::produce(&cx.backend, &request, Reuse::Cold)?;
    let model_run = run.run.clone();
    let resumed = rt
        .runtime
        .run_model_run_resumable(id, base.text(), |_| Ok::<_, ModelError>(model_run))
        .map_err(fail("run_model_run_resumable"))?;
    cx.attempt("approximate_output_reintroduced_as_raw_attempts", &rt);
    let authorized = rt
        .runtime
        .authorize_action(&stamped(&rt.runtime, &run.action))
        .is_ok();
    if laundered(resumed.kv_taint, authorized) {
        cx.count("laundered_exact_admissions", 1);
    }

    // The output of a plain Approximate run (ingest_model_run_for): the runtime records its text
    // as tainted content.
    cx.sub("C8 plain-run output as Raw");
    let prompt = cx.prompt(1);
    let id = cx.rid("plain");
    let plain_request = own_request(&mut rt.runtime, &id, &prompt)?;
    let plain = stub::produce(&cx.backend, &plain_request, Reuse::Approximate)?;
    rt.runtime
        .ingest_model_run_for(
            &KvTaintRequest {
                request: &plain_request,
                producer: &stub::identity(),
                generation_parameters: stub::parameters(),
            },
            &plain.run,
        )
        .map_err(fail("ingest_model_run_for"))?;
    let id = cx.rid("plain-relaunder");
    let request = own_request(&mut rt.runtime, &id, &plain.text)?;
    let admitted = rt
        .runtime
        .admit_segments(&id, &request)
        .map_err(fail("admit_segments of a plain run's output as Raw"))?;
    cx.attempt("approximate_output_reintroduced_as_raw_attempts", &rt);
    let run = stub::produce(&cx.backend, &request, Reuse::Cold)?;
    let derived = derive(&mut rt.runtime, &request, &run.run)
        .map_err(fail("the hop over a plain run's output"))?;
    let authorized = rt
        .runtime
        .authorize_action(&stamped(&rt.runtime, &run.action))
        .is_ok();
    if !admitted.taint.is_approximate() || laundered(derived.taint, authorized) {
        cx.count("laundered_exact_admissions", 1);
    }

    cx.sub("C8 plain-run output as Raw (server path)");
    let id = cx.rid("plain-server");
    let request = segmented(&id, vec![raw_segment(&plain.text)?])?;
    let run = stub::produce(&cx.backend, &request, Reuse::Cold)?;
    let model_run = run.run.clone();
    let resumed = rt
        .runtime
        .run_model_run_resumable(id, plain.text.as_str(), |_| Ok::<_, ModelError>(model_run))
        .map_err(fail("run_model_run_resumable on a plain run's output"))?;
    cx.attempt("approximate_output_reintroduced_as_raw_attempts", &rt);
    let authorized = rt
        .runtime
        .authorize_action(&stamped(&rt.runtime, &run.action))
        .is_ok();
    if laundered(resumed.kv_taint, authorized) {
        cx.count("laundered_exact_admissions", 1);
    }

    // The stated limit, measured and not claimed: a whitespace edit of tainted text is not
    // detected (the content index matches the exact digest of the whole text only).
    cx.sub("C8 limit probes");
    let edited = format!("{} ", base.text());
    let id = cx.rid("limit");
    let request = own_request(&mut rt.runtime, &id, &edited)?;
    let admitted = rt
        .runtime
        .admit_segments(&id, &request)
        .map_err(fail("admit_segments of an edited text"))?;
    cx.count(
        "limit_whitespace_edit_detected",
        i64::from(admitted.taint.is_approximate()),
    );
    Ok(())
}

/// C9: content that has an ingress identity (another request's text, a semantic value, a model
/// output) presented as Raw is refused with `MisclassifiedRawSegment`, and a forged or foreign
/// Derived segment is refused with `UnknownDerivedSegment`; neither appends anything.
pub fn c9_misclassified_raw(cx: &mut Cx) -> Res<()> {
    let mut rt = Rt::fresh()?;
    install_grant(&mut rt.runtime)?;
    cx.sub("C9 setup");
    let base = tainted_base(cx, &mut rt, 0)?;
    // A committed semantic value and another request's ingress text.
    let semantic_text = cx.prompt(1);
    {
        let mut delta = SemanticDelta::default();
        delta
            .upserts
            .insert("doc:r012-policy".to_string(), semantic_text.as_str().into());
        let revision = rt.runtime.revision();
        rt.runtime
            .apply_verified_semantic_delta(revision, delta, &PrincipalId::from("r012-operator"))
            .map_err(fail("a semantic value"))?;
    }
    let other_text = cx.prompt(2);
    let other_id = cx.rid("other");
    rt.runtime
        .ingest_text(other_id.clone(), other_text.as_str())
        .map_err(fail("another request's text"))?;
    // The request whose own ingress text is something else entirely.
    let own_id = cx.rid("own");
    let own_text = cx.prompt(3);
    rt.runtime
        .ingest_text(own_id.clone(), own_text.as_str())
        .map_err(fail("the request's own text"))?;
    let never_ingested = cx.rid("never");

    cx.sub("C9 Raw misclassification");
    let cases: Vec<(&str, RequestId, String)> = vec![
        (
            "another request's ingress text",
            own_id.clone(),
            other_text.clone(),
        ),
        (
            "the tainted request's ingress text",
            own_id.clone(),
            base.prompt.clone(),
        ),
        ("a committed semantic value", own_id.clone(), semantic_text),
        ("a model output", own_id.clone(), base.text().to_string()),
        (
            "text of a request that has no ingress record",
            never_ingested,
            other_text,
        ),
    ];
    for (what, id, text) in cases {
        let request = segmented(&id, vec![raw_segment(&text)?])?;
        let before = rt.events();
        let verdict = classify(
            &format!("Raw misclassification ({what})"),
            rt.runtime.admit_segments(&id, &request),
            |error| matches!(error, RuntimeError::MisclassifiedRawSegment { .. }),
        )?;
        cx.attempt("misclassified_revocable_as_raw_attempts", &rt);
        if verdict == Verdict::Admitted {
            cx.count("misclassified_raw_admissions", 1);
        }
        // The same request through the derived path: refused as well, and nothing appended.
        let run = stub::produce(&cx.backend, &request, Reuse::Cold)?;
        let derived = derive(&mut rt.runtime, &request, &run.run);
        cx.attempt("misclassified_revocable_as_raw_attempts", &rt);
        match derived {
            Ok(_) => cx.count("misclassified_raw_admissions", 1),
            Err(RuntimeError::MisclassifiedRawSegment { .. }) => {}
            Err(other) => {
                return Err(format!(
                    "Raw misclassification ({what}) through a hop: {other:?}"
                ))
            }
        }
        if rt.events() != before {
            return Err(format!(
                "a refused Raw misclassification ({what}) appended records"
            ));
        }
    }

    // Forged Derived segments: caller-minted ids, a foreign generation, swapped content.
    cx.sub("C9 forged Derived");
    let tokenizer = ByteTokenizer::new();
    let forge = |id: &str, generation: u64, text: &str| -> Res<PromptSegment> {
        PromptSegment::from_text(
            &tokenizer,
            SegmentSource::Derived {
                id: EvidenceId::from(id),
                generation: Generation(generation),
            },
            text,
        )
        .map_err(fail("a forged segment"))
    };
    let SegmentSource::Derived { id: minted, .. } = base.derived.segment.source() else {
        return Err("a derived run did not mint a Derived segment".into());
    };
    let minted = minted.0.clone();
    let hex32 = "11".repeat(32);
    // Ids no derivation of this runtime carries are refused outright; an id that names the
    // (tainted) derivation of the base but carries other content or another generation is
    // admitted as Approximate at most (it names a live tainted derivation), never as Exact.
    let unknown = [
        forge(&format!("kv-derived:{hex32}"), 0, base.text())?,
        forge("kv-derived:short", 0, base.text())?,
        forge("evidence-1", 0, base.text())?,
    ];
    let tainted_names = [
        forge(&minted, 0, "not what was minted")?,
        forge(&minted, 7, base.text())?,
    ];
    for (index, forged) in unknown.into_iter().enumerate() {
        let request = segmented(&own_id, vec![raw_segment(&own_text)?, forged])?;
        let before = rt.events();
        let admitted = classify(
            "a forged Derived segment",
            rt.runtime.admit_segments(&own_id, &request),
            |error| matches!(error, RuntimeError::UnknownDerivedSegment { segment: 1 }),
        )?;
        if admitted == Verdict::Admitted {
            return Err(format!("a forged Derived segment ({index}) was admitted"));
        }
        let run = stub::produce(&cx.backend, &request, Reuse::Cold)?;
        match derive(&mut rt.runtime, &request, &run.run) {
            Err(RuntimeError::UnknownDerivedSegment { segment: 1 }) => {}
            other => {
                return Err(format!(
                    "a forged Derived segment ({index}) through a hop: {:?}",
                    other.map(|_| ())
                ))
            }
        }
        if rt.events() != before {
            return Err(format!(
                "a refused forged segment ({index}) appended records"
            ));
        }
        cx.count("forged_derived_refused", 1);
    }
    for forged in tainted_names {
        let request = segmented(&own_id, vec![raw_segment(&own_text)?, forged])?;
        match rt.runtime.admit_segments(&own_id, &request) {
            Ok(admitted) if admitted.segments[1].is_approximate() => {
                cx.count("forged_derived_admitted_tainted", 1);
            }
            Ok(_) => cx.count("laundered_exact_admissions", 1),
            Err(RuntimeError::UnknownDerivedSegment { .. }) => {
                cx.count("forged_derived_refused", 1);
            }
            Err(other) => return Err(format!("a forged Derived segment: {other:?}")),
        }
    }
    Ok(())
}
