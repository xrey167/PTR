# raft-rs

Consensus module for cluster mode; pair with a durable log, state machine and transport.

## PTR rule
Implement behind an internal trait/contract and benchmark in `evaluations/components/`. Do not let backend-specific types leak into semantic domain types.
