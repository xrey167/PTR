use ptr_router::RoutingPolicy;
#[test]
fn default_router_has_parallel_budget() {
    assert!(RoutingPolicy::default().max_parallel > 0);
}
