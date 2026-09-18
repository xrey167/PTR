use ptr_search::{EvidenceStage, SearchIndex, TantivyLexicalIndex};
use ptr_types::{CapsuleId, Generation};

#[test]
fn tantivy_update_delete_and_generation_metadata_are_lifecycle_safe() {
    let index = TantivyLexicalIndex::new_in_ram().unwrap();

    index
        .upsert(CapsuleId::from("a"), Generation(1), "red apple orchard")
        .unwrap();
    index
        .upsert(CapsuleId::from("b"), Generation(2), "blue ocean current")
        .unwrap();

    let apple = index.search_result("apple", 10).unwrap();
    assert_eq!(apple.len(), 1);
    assert_eq!(apple[0].capsule, CapsuleId::from("a"));
    assert_eq!(apple[0].generation, Generation(1));
    assert_eq!(apple[0].backend, "tantivy");
    assert_eq!(apple[0].stage(), EvidenceStage::SearchCandidate);

    index
        .upsert(CapsuleId::from("a"), Generation(3), "green pear orchard")
        .unwrap();

    assert!(index.search("apple", 10).is_empty());
    let pear = index.search_result("pear", 10).unwrap();
    assert_eq!(pear.len(), 1);
    assert_eq!(pear[0].capsule, CapsuleId::from("a"));
    assert_eq!(pear[0].generation, Generation(3));

    index.delete(&CapsuleId::from("a")).unwrap();
    assert!(index.search("pear", 10).is_empty());
}

#[test]
fn tantivy_empty_query_or_zero_limit_is_empty() {
    let index = TantivyLexicalIndex::new_in_ram().unwrap();
    index
        .upsert(CapsuleId::from("a"), Generation(1), "hello world")
        .unwrap();
    assert!(index.search_result("", 10).unwrap().is_empty());
    assert!(index.search_result("hello", 0).unwrap().is_empty());
}
