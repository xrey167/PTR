use ptr_core::PtrCoreConfig;
#[test] fn default_core_has_semantic_slots(){ assert!(PtrCoreConfig::default().semantic_slots>0); }
