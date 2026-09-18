use ptr_search::reciprocal_rank_fusion;
#[test] fn empty_fusion_is_empty(){ assert!(reciprocal_rank_fusion(&[],60.0).is_empty()); }
