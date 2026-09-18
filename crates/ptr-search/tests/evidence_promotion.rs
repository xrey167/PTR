use ptr_search::{EvidencePromotionError, EvidenceStage, SearchHit};
use ptr_types::{CapsuleId, Generation};

#[test]
fn retrieval_cannot_jump_directly_to_known() {
    let mut hit = SearchHit::new(CapsuleId::from("c1"), Generation(3), 0.9, "test");
    assert_eq!(
        hit.promote_verified(true),
        Err(EvidencePromotionError::WrongStage)
    );

    hit.mark_possible().unwrap();
    assert_eq!(
        hit.observe(Generation(4), true),
        Err(EvidencePromotionError::StaleGeneration)
    );
    assert_eq!(
        hit.observe(Generation(3), false),
        Err(EvidencePromotionError::SourceNotResolved)
    );
    hit.observe(Generation(3), true).unwrap();
    assert_eq!(hit.stage(), EvidenceStage::Observed);
    hit.promote_verified(true).unwrap();
    assert_eq!(hit.stage(), EvidenceStage::VerifiedKnown);
}
