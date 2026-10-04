# E007-split-pod-kv

## Hypothesis
splitting a model's layers across two pods, with KV kept on the hosting pod and only activation frames crossing PodWire, is bit-identical to a single-pod run.

## Primary metrics
bitwise_mismatch_count, wire_kv_handle_count, chain_mismatch_stores

## Baseline
single-pod run

## Falsification
any output difference, any PodWire message carrying a KV handle or page bytes, or any stage storing pages under a mismatched upstream chain.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
