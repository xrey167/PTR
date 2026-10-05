"""MoE training backend (`trl-moe`).

Importing this package must stay free of heavy dependencies (torch, transformers,
trl, peft). Pure-Python submodules (`config`, `tasks`, `reward`, `routing_metrics`)
work on the base install; the others import torch lazily.
"""
