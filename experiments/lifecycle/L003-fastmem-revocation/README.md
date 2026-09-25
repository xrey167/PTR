# L003-fastmem-revocation

## Hypothesis
Can a revoked input ever influence a fast-memory readout after revocation, replay, restore or crash?

## Primary metrics
bit-identity failures; resurrected reads; writes replayed per revocation

## Baseline
memory rebuilt from scratch without the revoked writes

## Falsification
Any bit difference from the never-saw-it fold, or any admitted read that depends on a revoked input, is a hard failure.

## Design
Mechanism and threat model: [35 — Agentic substrate](../../../docs/architecture/35-agentic-substrate.md).

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
