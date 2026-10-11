# C003-attention-schema-routing

## Hypothesis
[CPU stage A on a toy router, GPU stage B on OLMoE L1 routing; stage B needs a GPU] Indicator AST-1 as a functional property: does a predictive model of the router's own routing (an attention schema: it predicts which experts or Pods the router will select and how loaded they will be) that is also used to control routing improve routing quality and stability over the same router without the schema? Control use is the point: a schema that only describes but does not control is a separate ablation.

## Primary metrics
task reward under the verifier; expert load coefficient of variation; routing churn between policy versions; routing entropy; schema prediction accuracy of the next routing decision; time to recover after an expert is revoked or overloaded; paired differences over seeds

## Baseline
same router without a schema; a router with a schema that is trained but not used for control (descriptive-only ablation); a heuristic load balancer (aux-loss-free bias)

## Falsification
Falsified if the schema-controlled router does not beat both the schema-free router and the heuristic balancer on reward at equal compute by a pinned margin with a one-sided 95% paired bound over at least 5 seeds, or if the control arm beats the describe-only arm by no more than the pinned margin (then the schema's control role is not shown). The schema has no authority over expert promotion or revocation.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
