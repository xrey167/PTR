use ptr_state::MaterializedState;
#[test] fn default_state_starts_unapplied(){ assert_eq!(MaterializedState::default().last_applied,0); }
