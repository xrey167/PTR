use ptr_feedback::Population;
#[test] fn empty_population_has_no_best(){ let p:Population<String>=Population::default(); assert!(p.best_verified().is_none()); }
