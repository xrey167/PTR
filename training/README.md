# PTR Training Workspace

Training is separate from the production Rust runtime.

## Stages

1. semantic typing / raw↔typed verification;
2. PodWire + ActionIR;
3. epistemic calibration;
4. operator routing;
5. verifier + repair;
6. environment RL;
7. distillation / QAT / deployment optimization.

Python tooling is managed independently (uv is the preferred environment candidate). NeMo RL, Unsloth, Data Designer, DSPy and GEPA are candidates behind experiment configs rather than runtime dependencies.
