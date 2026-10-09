# Third-party notices for research/embeddings and research/baselines/strong_retrieval

`scripts/check_notices.py` reads Rust lockfiles only. The Python packages and model
weights used here are not in any Rust lockfile and are not redistributed in this
repository: only vectors computed with them are committed. This file is the hand-written
record. The exact package set is `research/embeddings/uv.lock` (isolated from
`training/uv.lock`); the model files are pinned by sha256 in `models.toml`.

## Models (downloaded from the Hugging Face Hub, never committed)

| Model | Revision | Licence | Use |
|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 | Apache-2.0 | document and query vectors for Q003 and M008 (384-d) |
| BAAI/bge-m3 | 5617a9f61b028005a4858fdac845db406aefb181 | MIT | dense vectors for strong_retrieval (1024-d) |
| BAAI/bge-reranker-v2-m3 | 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e | Apache-2.0 | cross-encoder reranker for strong_retrieval (loaded at run time only) |

The committed `vectors-*.f16` files are numerical outputs of these models over the
synthetic text pools in this directory. all-MiniLM-L6-v2 and bge-reranker-v2-m3 are
Apache-2.0 (attribution and the licence text apply to any redistribution of the
weights, which this repository does not do); bge-m3 is MIT (copyright BAAI).

## Python packages (installed from `uv.lock`, not committed)

| Package | Licence |
|---|---|
| torch (CPU wheel from download.pytorch.org/whl/cpu) | BSD-3-Clause |
| transformers | Apache-2.0 |
| tokenizers | Apache-2.0 |
| safetensors | Apache-2.0 |
| numpy | BSD-3-Clause |
| huggingface-hub | Apache-2.0 |

Transitive packages (filelock, fsspec, jinja2, networkx, sympy, PyYAML, regex, requests,
tqdm, and so on) carry permissive licences (MIT, BSD, Apache-2.0, PSF); read them from
the installed wheels' metadata when auditing. No copyleft package is intentionally
depended on, but the transitive licences were not individually audited here.
