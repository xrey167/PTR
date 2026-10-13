# Third-party notices for research/consciousness

`scripts/check_notices.py` reads Rust lockfiles only. The Python environment of the C-series
experiments is `research/consciousness/uv.lock` (isolated from `training/uv.lock`): PyTorch
2.7.1 (CPU wheels from the PyTorch index) and its dependencies as pinned there. Nothing from
those packages is redistributed in this repository. PyTorch is BSD-3-Clause licensed; the
licences of the remaining packages are those of the locked versions.

No model weights and no dataset are used: C001 generates its task from a seed.
