# Reasoning Architecture

PTR treats reasoning as a mixture of typed computation modes rather than a single free-form text process.

- semantic reasoning — meaning, entities, relations, intent;
- deductive/symbolic — hard constraints and formal transformations;
- probabilistic — hypotheses and calibrated distributions;
- statistical — learned specialist models such as tree ensembles/time-series models;
- temporal/causal — ordered and causal state relations;
- search — branch/frontier expansion with explicit cost;
- optimization/simulation — numerical or environment-backed computation;
- external Pod — typed cognitive extension.

The operator router is itself trainable, but runtime hard invariants are not.
