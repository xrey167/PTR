use ptr_model_api::{
    KvReuseMode, KvReusePolicy, KvReuseReport, KvReuseTaint, PageTaintError, TaintedCatalogEntry,
    TaintedPage,
};
use ptr_types::Generation;

fn approx_report(hit: usize) -> KvReuseReport {
    KvReuseReport::new(
        hit,
        KvReuseMode::Approximate,
        64,
        KvReusePolicy::AllowApproximate,
    )
    .unwrap()
}

fn page(id: &str, n: u8, taint: KvReuseTaint) -> TaintedPage {
    TaintedPage::new(id, [n; 32], [n.wrapping_add(100); 32], taint).unwrap()
}

fn samples() -> Vec<TaintedPage> {
    vec![
        page("a", 1, KvReuseTaint::Exact),
        page("b", 2, approx_report(8).taint()),
        page("c", 3, approx_report(16).taint()),
        page("a", 1, approx_report(8).taint()),
    ]
}

#[test]
fn join_of_tainted_and_exact_is_tainted() {
    let exact = page("a", 1, KvReuseTaint::Exact);
    let dirty = page("b", 2, approx_report(8).taint());
    for joined in [exact.join(&dirty).unwrap(), dirty.join(&exact).unwrap()] {
        assert!(joined.taint().is_approximate());
        assert!(!joined.is_exact());
        assert_eq!(joined.parts().len(), 2);
    }
}

#[test]
fn lattice_laws_hold_on_samples() {
    let s = samples();
    for a in &s {
        assert_eq!(a.join(a).unwrap(), *a, "idempotent");
        for b in &s {
            assert_eq!(a.join(b).unwrap(), b.join(a).unwrap(), "commutative");
            for c in &s {
                let left = a.join(b).unwrap().join(c).unwrap();
                let right = a.join(&b.join(c).unwrap()).unwrap();
                assert_eq!(left, right, "associative");
            }
        }
    }
}

#[test]
fn join_never_downgrades() {
    let dirty = page("b", 2, approx_report(8).taint());
    let exact = page("a", 1, KvReuseTaint::Exact);
    let joined = dirty.join(&exact).unwrap().join(&exact).unwrap();
    assert!(joined.taint().is_approximate());
    assert!(joined.require_exact().is_err());
}

#[test]
fn require_exact_accepts_exact_and_refuses_tainted_with_typed_error() {
    let exact = page("a", 1, KvReuseTaint::Exact);
    assert_eq!(exact.require_exact(), Ok(&exact));
    let report = approx_report(8);
    let dirty = TaintedPage::from_report("b", [2; 32], [3; 32], Some(&report)).unwrap();
    let joined = dirty.join(&exact).unwrap();
    assert_eq!(
        joined.require_exact(),
        Err(PageTaintError::Tainted {
            report_digest: report.digest(),
            page_ids: vec!["a".into(), "b".into()],
        })
    );
    let clean = TaintedPage::from_report("c", [4; 32], [5; 32], None).unwrap();
    assert!(clean.is_exact());
}

#[test]
fn catalog_entry_carries_taint_and_generation() {
    let dirty = page("b", 2, approx_report(8).taint());
    let entry = TaintedCatalogEntry::new("k", Generation(7), dirty).unwrap();
    assert_eq!(entry.key(), "k");
    assert_eq!(entry.generation(), Generation(7));
    assert!(entry.page().taint().is_approximate());
    assert!(matches!(
        entry.require_exact(),
        Err(PageTaintError::Tainted { .. })
    ));
    let clean =
        TaintedCatalogEntry::new("k", Generation(1), page("a", 1, KvReuseTaint::Exact)).unwrap();
    assert!(clean.require_exact().is_ok());
}

#[test]
fn lineage_of_each_part_survives_join() {
    let a = page("a", 1, KvReuseTaint::Exact);
    let b = page("b", 2, KvReuseTaint::Exact);
    let joined = a.join(&b).unwrap();
    let lineages: Vec<_> = joined.parts().iter().map(|p| p.lineage_digest).collect();
    assert_eq!(lineages, vec![[1; 32], [2; 32]]);
}
