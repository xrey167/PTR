"""Frozen text encoders for Dragonfly address training.

The encoder is never trained here: a pod's address is a vector `z` and a bias over its output.
`HashedBagOfWordsEncoder` is deterministic, needs no download and runs everywhere, so tests and
the committed fixture use it. `SentenceTransformerEncoder` is the production path and imports its
library lazily (extra `dragonfly`), so the base install and the test suite never need it.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from typing import Protocol, Sequence

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


class Encoder(Protocol):
    """A frozen text encoder with a pinned identity."""

    encoder_id: str
    encoder_revision: str
    dims: int

    def encode(self, text: str) -> list[float]:
        """The L2-normalised embedding of `text`."""


def normalized(text: str) -> str:
    """Case-folded, accent-stripped, whitespace-collapsed text (as the research router did)."""
    folded = unicodedata.normalize("NFKD", text.casefold())
    stripped = "".join(c for c in folded if not unicodedata.combining(c))
    return " ".join(stripped.split())


def l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        raise ValueError("cannot normalise the zero vector")
    return [v / norm for v in vector]


class HashedBagOfWordsEncoder:
    """Deterministic stand-in encoder: hashed word and character-trigram counts.

    Each normalised word and each trigram of the word padded with `^`/`$` is hashed to a bucket
    with a sign, so near spellings of one name share features without any model.
    """

    def __init__(self, dims: int = 64, *, revision: str = "1") -> None:
        if dims < 8:
            raise ValueError("dims must be at least 8")
        self.dims = dims
        self.encoder_id = f"hashed-bow-{dims}"
        self.encoder_revision = revision

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.sha256(f"{self.encoder_id}:{feature}".encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % self.dims
        sign = 1.0 if digest[4] & 1 else -1.0
        return index, sign

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        words = _WORD.findall(normalized(text))
        if not words:
            raise ValueError("text has no encodable words")
        for word in words:
            features = [f"w:{word}"]
            padded = f"^{word}$"
            features += [f"t:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
            for feature in features:
                index, sign = self._bucket(feature)
                vector[index] += sign
        return l2_normalize(vector)


class SentenceTransformerEncoder:
    """Production encoder (extra `dragonfly`). Never downloads implicitly in tests."""

    def __init__(self, model_name: str, revision: str, *, device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "SentenceTransformerEncoder needs the optional 'dragonfly' extra: "
                "pip install 'ptr-training[dragonfly]'"
            ) from error
        if not revision:
            raise ValueError("a pinned encoder revision is required")
        self._model = SentenceTransformer(model_name, revision=revision, device=device)
        self.encoder_id = model_name
        self.encoder_revision = revision
        self.dims = int(self._model.get_sentence_embedding_dimension())

    def encode(self, text: str) -> list[float]:  # pragma: no cover - optional dependency
        vector = self._model.encode(text, normalize_embeddings=True)
        return [float(v) for v in vector]
