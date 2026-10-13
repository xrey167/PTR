use ptr_model_api::{
    KvReuseMode, KvReusePolicy, KvReuseReport, KvReuseTaint, ModelEvent, ModelRun, Tainted,
    TokenContractError,
};

fn approx(hit: usize) -> KvReuseReport {
    KvReuseReport::new(
        hit,
        KvReuseMode::Approximate,
        64,
        KvReusePolicy::AllowApproximate,
    )
    .unwrap()
}

fn exact(hit: usize) -> KvReuseReport {
    KvReuseReport::new(hit, KvReuseMode::Exact, 64, KvReusePolicy::ExactOnly).unwrap()
}

#[test]
fn report_taint_follows_mode() {
    assert_eq!(exact(16).taint(), KvReuseTaint::Exact);
    let taint = approx(16).taint();
    assert_eq!(
        taint,
        KvReuseTaint::Approximate {
            report_digest: approx(16).digest()
        }
    );
    assert_ne!(approx(16).digest(), approx(32).digest());
    assert_ne!(approx(16).digest(), exact(16).digest());
}

#[test]
fn model_run_without_report_is_exact_and_events_inherit_taint() {
    let clean = ModelRun {
        events: vec![ModelEvent::Finished],
        kv_report: None,
    };
    assert_eq!(clean.taint(), KvReuseTaint::Exact);
    let dirty = ModelRun {
        events: vec![ModelEvent::Token("a".into()), ModelEvent::Finished],
        kv_report: Some(approx(8)),
    };
    let events = dirty.into_tainted_events();
    assert_eq!(events.len(), 2);
    assert!(events.iter().all(|event| event.taint().is_approximate()));
}

#[test]
fn taint_survives_two_derivation_hops() {
    let dirty = Tainted::from_report(2u32, Some(&approx(8)));
    let clean = Tainted::from_report(3u32, Some(&exact(8)));
    let hop1 = dirty.combine(clean, |a, b| a + b);
    assert!(hop1.taint().is_approximate());
    let hop2 = hop1.map(|v| v * 10);
    assert!(hop2.taint().is_approximate());
    let hop3 = Tainted::exact(1u32).combine(hop2, |a, b| a + b);
    assert!(hop3.taint().is_approximate());
    assert_eq!(hop3.into_exact(), Err(TokenContractError::ApproximateTaint));
}

#[test]
fn join_is_commutative_associative_idempotent_and_exact_is_identity() {
    let a = approx(8).taint();
    let b = approx(16).taint();
    let c = approx(24).taint();
    assert_eq!(a.join(b), b.join(a));
    assert_eq!(a.join(b).join(c), a.join(b.join(c)));
    assert_eq!(a.join(a), a);
    assert_eq!(KvReuseTaint::Exact.join(a), a);
    assert_eq!(a.join(KvReuseTaint::Exact), a);
    assert_eq!(
        KvReuseTaint::Exact.join(KvReuseTaint::Exact),
        KvReuseTaint::Exact
    );
}

#[test]
fn tainted_value_cannot_be_downgraded_by_rewrapping() {
    // The only public ways to build a Tainted are `exact` (fresh clean value),
    // `from_report`, `map` and `combine`; none lets a tainted value drop its taint.
    let dirty = Tainted::from_report("x".to_string(), Some(&approx(8)));
    let rewrapped = dirty.clone().map(Tainted::exact);
    assert!(rewrapped.taint().is_approximate());
    let derived = dirty.map(|v| v.len());
    assert!(derived.taint().is_approximate());
    assert!(derived.into_exact().is_err());
}

#[test]
fn exact_value_is_released() {
    let clean = Tainted::from_report(5u8, Some(&exact(8)));
    assert_eq!(clean.into_exact(), Ok(5));
    assert_eq!(Tainted::exact(6u8).into_exact(), Ok(6));
}
