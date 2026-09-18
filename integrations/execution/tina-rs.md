# tina-rs

Reference for bounded isolate/effect/replay semantics; evaluate dependency versus adopting semantics internally.

## PTR rule
Implement behind an internal trait/contract and benchmark in `evaluations/components/`. Do not let backend-specific types leak into semantic domain types.
