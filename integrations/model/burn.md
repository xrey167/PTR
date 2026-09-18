# burn

Rust-native tensor/training/inference framework candidate for PTR Core.

## PTR rule
Implement behind an internal trait/contract and benchmark in `evaluations/components/`. Do not let backend-specific types leak into semantic domain types.
