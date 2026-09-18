use ptr_memory::MemoryClass;
#[test]
fn procedural_memory_is_distinct() {
    assert_ne!(MemoryClass::Procedural, MemoryClass::Semantic);
}
