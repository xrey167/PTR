# R002-unseen-pod-generalization

## Hypothesis
Can the model use unseen Pod names from known semantic contracts?

## Primary metrics
task success; name invariance

## Design note: Symlink-LoRA (design only, no code)

Research (`NEURAL-SYMLINK.md`) trained a small LoRA per pod cluster that emits a stable link (`LINK n CLUSTER m`) from alias and question phrasings; the link is resolved to the pod's current head, so the pod's identity survives value and generation changes (see [43 — Pod linking](../../../docs/architecture/43-pod-linking.md)). PTR keeps the link out of the model: identity is value-free and its head is a ledger fact. What would be left for a model is the step from an unseen phrasing to a known identity, which this experiment may test as a registered candidate (`symlink-lora` in `evaluations/components/pod-routing`). Nothing here implements it.

Kill criterion for the reflex use: if the base model resolves no alias in 5 seeds (0 of the held-out alias set each time), the learned link adds nothing to measure and the reflex path stays off; the research repo's own reflex-dispatch check was red at 0 of 132 hits for the same reason.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
