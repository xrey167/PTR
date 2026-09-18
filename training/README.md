# Training Workspace

Python remains a research/training plane, managed with `uv` when available. The production runtime does not depend on this package.

Suggested progression:

0. continued pretraining only if domain/protocol vocabulary requires it;
1. raw→typed semantic SFT;
2. PodWire / typed ActionIR SFT;
3. epistemic calibration and uncertainty tasks;
4. operator-routing supervision;
5. verifier/critique/repair training;
6. preference optimization / RL in executable environments;
7. distillation into smaller models and specialist heads;
8. quantization/QAT only after PTR-specific eval gates pass.

`configs/` contains stage definitions. `runs/` and `checkpoints/` are ignored by git.
