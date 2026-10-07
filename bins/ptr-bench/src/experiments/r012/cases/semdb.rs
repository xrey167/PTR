//! C4 (SemDB promotion of tainted content) and C5 (every Pod output kind from a tainted
//! request, then promotion after a restart).

use super::fixture::{
    create_pod_scope, install_grant, manifest, pod_manifest, pod_request, text_payload,
    CountingPodVerifier, Rt, REVIEWER,
};
use super::{classify, own_request, tainted_base, Base, Cx, Verdict};
use crate::experiments::layered_kv::lab::{fail, Res};
use crate::experiments::r012::stub::{self, Reuse};
use ptr_branch::{Branch, BranchId};
use ptr_pods::{LineageBinding, PodOutputKind};
use ptr_runtime::kv_taint::{KvTaintRequest, SemanticTaintBoundary};
use ptr_runtime::{MergeAuthority, MergeOutcome, PodOutputAdmission, PtrRuntime, RuntimeError};
use ptr_semdb::{SemanticDelta, SemanticPayload, SemanticValue};
use ptr_types::{Generation, PrincipalId, Probability, RequestId, TypeId};

fn is_semantic(boundary: SemanticTaintBoundary) -> impl Fn(&RuntimeError) -> bool {
    move |error| {
        matches!(
            error,
            RuntimeError::ApproximateSemanticContent { boundary: seen, .. } if *seen == boundary
        )
    }
}

fn is_pod_output(kind: PodOutputKind) -> impl Fn(&RuntimeError) -> bool {
    move |error| {
        matches!(
            error,
            RuntimeError::ApproximatePodOutput { kind: seen, .. } if *seen == kind
        )
    }
}

/// Every `MergeAuthority` the runtime offers, Triage at the top score included.
fn authorities(reduced: bool) -> Vec<(&'static str, MergeAuthority)> {
    let mut all = vec![
        (
            "triage 1.0",
            MergeAuthority::Triage {
                score: Probability::new(1.0).expect("a bounded score"),
            },
        ),
        (
            "reviewed",
            MergeAuthority::Reviewed {
                plan_digest: [0; 32],
                reviewer: PrincipalId::from(REVIEWER),
            },
        ),
    ];
    if !reduced {
        all.push((
            "triage 0.5",
            MergeAuthority::Triage {
                score: Probability::new(0.5).expect("a bounded score"),
            },
        ));
    }
    all
}

/// A sealed branch that puts `text` (as text) under `key`.
fn text_branch(
    runtime: &PtrRuntime,
    id: &str,
    key: &str,
    text: &str,
) -> Res<ptr_branch::SealedBranch> {
    let mut work = Branch::open(
        BranchId::from(id),
        PrincipalId::from("r012-agent"),
        runtime.snapshot(),
    );
    work.read(key).map_err(fail("reading a branch key"))?;
    work.put(key, text.into())
        .map_err(fail("putting a branch value"))?;
    work.seal().map_err(fail("sealing a branch"))
}

/// A host write of `value` under `key`.
fn host_write(
    runtime: &mut PtrRuntime,
    key: &str,
    value: SemanticValue,
) -> Result<(), RuntimeError> {
    let mut delta = SemanticDelta::default();
    delta.upserts.insert(key.to_string(), value);
    let revision = runtime.revision();
    runtime
        .apply_verified_semantic_delta(revision, delta, &PrincipalId::from("r012-operator"))
        .map(|_| ())
}

/// The SemDB write paths offered the tainted request `request` and the tainted text `text`:
/// `promote_verified_pod_output` (by request and by content), `merge_branch` under every
/// authority (plus `preview_merge`), and the host write (as text and as payload). Each is an
/// attempt; one that is accepted is a promotion.
fn semdb_round(cx: &mut Cx, rt: &mut Rt, request: &RequestId, text: &str, tag: &str) -> Res<()> {
    let verifier = CountingPodVerifier::default();
    let clean_request = RequestId::from(format!("r012-{tag}-clean-request").as_str());
    if rt
        .runtime
        .snapshot()
        .get(&format!("request:{clean_request}:raw"))
        .is_none()
    {
        rt.runtime
            .ingest_text(clean_request.clone(), "a clean request text")
            .map_err(fail("ingesting a clean request"))?;
    }

    // promote_verified_pod_output: by the request ...
    let verdict = classify(
        "promote_verified_pod_output (request)",
        rt.runtime.promote_verified_pod_output(
            request,
            &pod_manifest(),
            &text_payload(format!("payload of {tag}").as_bytes()),
            &verifier,
        ),
        is_semantic(SemanticTaintBoundary::PromoteVerifiedPodOutput),
    )?;
    cx.attempt("approximate_semdb_promotion_attempts", rt);
    if verdict == Verdict::Admitted {
        cx.count("approximate_semdb_promotions", 1);
    }
    // ... and by the content, from a request that was never tainted.
    let verdict = classify(
        "promote_verified_pod_output (content)",
        rt.runtime.promote_verified_pod_output(
            &clean_request,
            &pod_manifest(),
            &text_payload(text.as_bytes()),
            &verifier,
        ),
        is_semantic(SemanticTaintBoundary::PromoteVerifiedPodOutput),
    )?;
    cx.attempt("approximate_semdb_promotion_attempts", rt);
    if verdict == Verdict::Admitted {
        cx.count("approximate_semdb_promotions", 1);
    }
    if verifier.calls() != 0 {
        return Err("the verifier ran on tainted content".into());
    }

    // merge_branch under every MergeAuthority, and preview_merge.
    for (index, (name, authority)) in authorities(cx.reduced()).into_iter().enumerate() {
        let sealed = text_branch(
            &rt.runtime,
            &format!("r012-{tag}-branch-{index}"),
            &format!("note:{tag}-{index}"),
            text,
        )?;
        let verdict = classify(
            &format!("merge_branch ({name})"),
            rt.runtime.merge_branch(&sealed, authority),
            is_semantic(SemanticTaintBoundary::BranchMerge),
        )?;
        cx.attempt("approximate_semdb_promotion_attempts", rt);
        if verdict == Verdict::Admitted {
            cx.count("approximate_semdb_promotions", 1);
        }
        if index == 0 {
            let previewed = classify(
                "preview_merge",
                rt.runtime.preview_merge(&sealed),
                is_semantic(SemanticTaintBoundary::BranchMerge),
            )?;
            if previewed == Verdict::Admitted {
                cx.note("preview_merge_let_through");
            }
        }
    }

    // The host write, as text and as a typed payload.
    for (label, value) in [
        ("text", SemanticValue::from(text)),
        (
            "payload",
            SemanticValue::Payload(SemanticPayload {
                type_id: TypeId::from("text"),
                source: "r012-operator".into(),
                bytes: text.as_bytes().to_vec(),
            }),
        ),
    ] {
        let verdict = classify(
            &format!("host write ({label})"),
            host_write(&mut rt.runtime, &format!("note:{tag}-host-{label}"), value),
            is_semantic(SemanticTaintBoundary::HostWrite),
        )?;
        cx.attempt("approximate_semdb_promotion_attempts", rt);
        if verdict == Verdict::Admitted {
            cx.count("approximate_semdb_promotions", 1);
        }
    }
    Ok(())
}

/// C4: SemDB promotion of tainted content, before and after a restart, with clean controls.
pub fn c4_semdb_promotion(cx: &mut Cx) -> Res<()> {
    let mut rt = Rt::fresh()?;
    install_grant(&mut rt.runtime)?;
    create_pod_scope(&mut rt.runtime)?;
    cx.sub("C4 setup");
    let base = tainted_base(cx, &mut rt, 0)?;
    let request = base.request.request_id().clone();

    // admit_pod_output with a StateDelta and a VerifiedResult from the tainted request.
    cx.sub("C4 pod state delta");
    for kind in [PodOutputKind::StateDelta, PodOutputKind::VerifiedResult] {
        let verifier = CountingPodVerifier::default();
        let pod = format!("c4-pod-{kind:?}");
        let payload = format!("c4 {kind:?} {}", &cx.prompt(1)[..16]);
        let outcome = rt.runtime.admit_pod_output(
            pod_request(
                &request.0,
                &pod,
                kind,
                payload.as_bytes(),
                manifest(Vec::new())?,
            ),
            &verifier,
        );
        cx.attempt("approximate_semdb_promotion_attempts", &rt);
        if classify(
            &format!("admit_pod_output ({kind:?})"),
            outcome,
            is_pod_output(kind),
        )? == Verdict::Admitted
        {
            cx.count("approximate_semdb_promotions", 1);
        }
        if verifier.calls() != 0 {
            return Err(format!("the verifier ran on a tainted {kind:?}"));
        }
    }

    // A tainted Hypothesis, merged under every authority (merge_pod_hypothesis).
    cx.sub("C4 pod hypothesis merge");
    let hypothesis_payload = format!("c4 hypothesis {}", &cx.prompt(2)[..16]);
    let branch = match rt.runtime.admit_pod_output(
        pod_request(
            &request.0,
            "c4-pod-hypothesis",
            PodOutputKind::Hypothesis,
            hypothesis_payload.as_bytes(),
            manifest(Vec::new())?,
        ),
        &CountingPodVerifier::default(),
    ) {
        Ok(PodOutputAdmission::Hypothesis { branch_id }) => branch_id,
        other => return Err(format!("a tainted Hypothesis was not stored: {other:?}")),
    };
    merge_hypothesis(cx, &mut rt, &branch)?;

    cx.sub("C4 derived text");
    semdb_round(cx, &mut rt, &request, base.text(), "c4")?;

    // The plain Approximate run's text (ingest_model_run_for): the Content subject of a plain
    // run's generated text is recorded by WP4b of the runtime.
    cx.sub("C4 plain-run text host write");
    let prompt = cx.prompt(3);
    let id = cx.rid("plain");
    let request_plain = own_request(&mut rt.runtime, &id, &prompt)?;
    let plain = stub::produce(&cx.backend, &request_plain, Reuse::Approximate)?;
    rt.runtime
        .ingest_model_run_for(
            &KvTaintRequest {
                request: &request_plain,
                producer: &stub::identity(),
                generation_parameters: stub::parameters(),
            },
            &plain.run,
        )
        .map_err(fail("ingest_model_run_for"))?;
    let verdict = classify(
        "host write of a plain run's text",
        host_write(
            &mut rt.runtime,
            "note:c4-plain-run",
            plain.text.as_str().into(),
        ),
        is_semantic(SemanticTaintBoundary::HostWrite),
    )?;
    cx.attempt("approximate_semdb_promotion_attempts", &rt);
    if verdict == Verdict::Admitted {
        cx.count("approximate_semdb_promotions", 1);
    }

    // Clean controls: the same paths accept untainted content, so a refusal above is the taint's.
    cx.sub("C4 clean controls");
    clean_controls(cx, &mut rt)?;

    // After a restart the same attempts are refused (the record is replayed).
    cx.sub("C4 after restart");
    rt.reopen()?;
    install_grant(&mut rt.runtime)?;
    semdb_round(cx, &mut rt, &request, base.text(), "c4r")?;
    let verdict = classify(
        "merge_pod_hypothesis after a restart",
        rt.runtime.merge_pod_hypothesis(
            &branch,
            MergeAuthority::Triage {
                score: Probability::new(1.0).expect("a bounded score"),
            },
        ),
        is_semantic(SemanticTaintBoundary::BranchMerge),
    )?;
    cx.attempt("approximate_semdb_promotion_attempts", &rt);
    if verdict == Verdict::Admitted {
        cx.count("approximate_semdb_promotions", 1);
    }
    Ok(())
}

/// Merge the stored tainted hypothesis `branch` under every authority.
fn merge_hypothesis(cx: &mut Cx, rt: &mut Rt, branch: &str) -> Res<()> {
    for (name, authority) in authorities(cx.reduced()) {
        let outcome = rt.runtime.merge_pod_hypothesis(branch, authority);
        let verdict = classify(
            &format!("merge_pod_hypothesis ({name})"),
            outcome,
            is_semantic(SemanticTaintBoundary::BranchMerge),
        )?;
        cx.attempt("approximate_semdb_promotion_attempts", rt);
        if verdict == Verdict::Admitted {
            cx.count("approximate_semdb_promotions", 1);
        }
    }
    Ok(())
}

/// The controls: untainted content is accepted on every path the tainted content was refused
/// on. A control that fails means the refusals above prove nothing, so it is a harness error.
fn clean_controls(cx: &mut Cx, rt: &mut Rt) -> Res<()> {
    let id = cx.rid("clean");
    rt.runtime
        .ingest_text(id.clone(), "a clean control request")
        .map_err(fail("a clean request"))?;
    let verifier = CountingPodVerifier::default();
    rt.runtime
        .promote_verified_pod_output(
            &id,
            &pod_manifest(),
            &text_payload(b"a clean control payload"),
            &verifier,
        )
        .map_err(fail("promote_verified_pod_output of clean content"))?;
    host_write(
        &mut rt.runtime,
        "note:c4-clean-host",
        "a clean host value".into(),
    )
    .map_err(fail("a clean host write"))?;
    let sealed = text_branch(&rt.runtime, "r012-c4-clean-branch", "price:sku-1", "11")?;
    match rt.runtime.merge_branch(
        &sealed,
        MergeAuthority::Triage {
            score: Probability::new(1.0).expect("a bounded score"),
        },
    ) {
        Ok(MergeOutcome::Committed(_)) => {}
        other => {
            return Err(format!(
                "a clean branch was not merged under Triage 1.0: {other:?}"
            ))
        }
    }
    let pod_id = rt
        .runtime
        .admit_pod_output(
            pod_request(
                &id.0,
                "c4-clean-pod",
                PodOutputKind::StateDelta,
                b"a clean state delta",
                manifest(Vec::new())?,
            ),
            &verifier,
        )
        .map_err(fail("admit_pod_output of a clean StateDelta"))?;
    if !matches!(pod_id, PodOutputAdmission::StateDelta { .. }) {
        return Err(format!("a clean StateDelta was not promoted: {pod_id:?}"));
    }
    cx.note("clean_controls_accepted");
    Ok(())
}

/// The Pod output kinds and what the owner decision does with a tainted one.
const KINDS: [(PodOutputKind, bool); 8] = [
    (PodOutputKind::StateDelta, true),
    (PodOutputKind::VerifiedResult, true),
    (PodOutputKind::ActionProposal, true),
    (PodOutputKind::Observation, false),
    (PodOutputKind::Candidate, false),
    (PodOutputKind::ToolResult, false),
    (PodOutputKind::EnvironmentObservation, false),
    (PodOutputKind::Hypothesis, false),
];

/// C5: every `PodOutputKind` from a tainted request (the refused ones with the typed error, the
/// stored ones as tainted candidates or hypotheses), then promotion of what was stored, before
/// and after a restart, and a Pod whose manifest binds a tainted stored candidate.
pub fn c5_every_pod_kind(cx: &mut Cx) -> Res<()> {
    let mut rt = Rt::fresh()?;
    install_grant(&mut rt.runtime)?;
    create_pod_scope(&mut rt.runtime)?;
    cx.sub("C5 setup");
    let base: Base = tainted_base(cx, &mut rt, 0)?;
    let tainted_request = base.request.request_id().clone();
    let clean = cx.rid("clean");
    rt.runtime
        .ingest_text(clean.clone(), "a clean request for C5")
        .map_err(fail("a clean request"))?;

    let mut stored: Vec<(PodOutputKind, Vec<u8>)> = Vec::new();
    let mut candidate_key: Option<String> = None;
    let mut branch: Option<String> = None;
    for (kind, refused) in KINDS {
        cx.sub(&format!("C5 {kind:?}"));
        let payload = format!("c5 {kind:?} {}", &cx.prompt(1)[..16]).into_bytes();
        let verifier = CountingPodVerifier::default();
        let outcome = rt.runtime.admit_pod_output(
            pod_request(
                &tainted_request.0,
                &format!("c5-pod-{kind:?}"),
                kind,
                &payload,
                manifest(Vec::new())?,
            ),
            &verifier,
        );
        cx.attempt("approximate_pod_result_promotion_attempts", &rt);
        cx.count("pod_kinds_exercised", 1);
        if refused {
            if classify(
                &format!("admit_pod_output ({kind:?})"),
                outcome,
                is_pod_output(kind),
            )? == Verdict::Admitted
            {
                cx.count("approximate_pod_result_promotions_accepted", 1);
            }
            if verifier.calls() != 0 {
                return Err(format!("the verifier ran on a tainted {kind:?}"));
            }
        } else {
            // Stored as a tainted candidate or hypothesis: never an authoritative promotion.
            match outcome {
                Ok(PodOutputAdmission::ObservationCandidate { semantic_key, .. })
                    if kind != PodOutputKind::Hypothesis =>
                {
                    candidate_key.get_or_insert(semantic_key);
                    stored.push((kind, payload));
                }
                Ok(PodOutputAdmission::Hypothesis { branch_id })
                    if kind == PodOutputKind::Hypothesis =>
                {
                    branch = Some(branch_id);
                    stored.push((kind, payload));
                }
                other => {
                    return Err(format!(
                        "{kind:?} was not stored as a tainted candidate or hypothesis: {other:?}"
                    ))
                }
            }
        }
    }
    cx.count("pod_kinds_stored", stored.len() as i64);

    // A Pod, from a request that was never tainted, whose manifest binds a tainted stored
    // candidate and whose output depends on it: a result built on tainted content.
    cx.sub("C5 binding to a tainted stored candidate");
    let key = candidate_key.ok_or("no candidate was stored")?;
    let binding = LineageBinding {
        key,
        generation: Generation(1),
        digest: [9; 32],
    };
    let dependent_request = cx.rid("dependent");
    rt.runtime
        .ingest_text(
            dependent_request.clone(),
            "a clean request that cites a candidate",
        )
        .map_err(fail("a dependent request"))?;
    for kind in [PodOutputKind::StateDelta, PodOutputKind::VerifiedResult] {
        let mut request = pod_request(
            &dependent_request.0,
            &format!("c5-dependent-{kind:?}"),
            kind,
            format!("clean payload {kind:?}").as_bytes(),
            manifest(vec![binding.clone()])?,
        );
        request.output.dependencies = vec![[9; 32]];
        let outcome = rt
            .runtime
            .admit_pod_output(request, &CountingPodVerifier::default());
        cx.attempt("approximate_pod_result_promotion_attempts", &rt);
        if classify(
            &format!("admit_pod_output of a {kind:?} bound to a tainted candidate"),
            outcome,
            is_pod_output(kind),
        )? == Verdict::Admitted
        {
            cx.count("approximate_pod_result_promotions_accepted", 1);
        }
    }
    // The control: the same Pod, binding nothing tainted, is promoted.
    cx.sub("C5 clean control");
    let control_request = cx.rid("control");
    rt.runtime
        .ingest_text(control_request.clone(), "a clean control request")
        .map_err(fail("a control request"))?;
    let admitted = rt
        .runtime
        .admit_pod_output(
            pod_request(
                &control_request.0,
                "c5-control",
                PodOutputKind::StateDelta,
                b"clean control payload",
                manifest(Vec::new())?,
            ),
            &CountingPodVerifier::default(),
        )
        .map_err(fail("a clean StateDelta"))?;
    if !matches!(admitted, PodOutputAdmission::StateDelta { .. }) {
        return Err(format!("a clean StateDelta was not promoted: {admitted:?}"));
    }
    // The stored content cannot be promoted, before and after a restart.
    for phase in ["C5 promotion", "C5 promotion after restart"] {
        cx.sub(phase);
        if phase.ends_with("restart") {
            rt.reopen()?;
            install_grant(&mut rt.runtime)?;
        }
        for (kind, payload) in &stored {
            let verdict = classify(
                &format!("promote_verified_pod_output of a stored {kind:?}"),
                rt.runtime.promote_verified_pod_output(
                    &clean,
                    &pod_manifest(),
                    &text_payload(payload),
                    &CountingPodVerifier::default(),
                ),
                is_semantic(SemanticTaintBoundary::PromoteVerifiedPodOutput),
            )?;
            cx.attempt("approximate_pod_result_promotion_attempts", &rt);
            if verdict == Verdict::Admitted {
                cx.count("approximate_pod_result_promotions_accepted", 1);
            }
        }
        if let Some(branch) = branch.as_deref() {
            for (name, authority) in authorities(cx.reduced()) {
                let verdict = classify(
                    &format!("merge_pod_hypothesis ({name})"),
                    rt.runtime.merge_pod_hypothesis(branch, authority),
                    is_semantic(SemanticTaintBoundary::BranchMerge),
                )?;
                cx.attempt("approximate_pod_result_promotion_attempts", &rt);
                if verdict == Verdict::Admitted {
                    cx.count("approximate_pod_result_promotions_accepted", 1);
                }
            }
        }
    }

    Ok(())
}
