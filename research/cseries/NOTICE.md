# Third-party notices for research/cseries

`scripts/check_notices.py` reads Rust lockfiles only. The Python environment of these experiments is the locked environment of
`research/consciousness` (`research/consciousness/uv.lock`, isolated from `training/uv.lock`): PyTorch 2.7.1 (CPU wheels from the PyTorch index)
and its dependencies as pinned there. This directory has no lock file of its own and adds no dependency; nothing from those packages is
redistributed in this repository. PyTorch is BSD-3-Clause licensed; the licences of the remaining packages are those of the locked versions.

No model weights and no dataset are used: C004, C007 and C008 generate their data from a seed.
