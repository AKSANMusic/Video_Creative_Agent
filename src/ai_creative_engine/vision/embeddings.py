"""Local CPU embedding + binary quantization.

Pipeline
--------
1. Embed the concatenated caption text with a local ``sentence-transformers``
   model on CPU (no network after the model is downloaded).
2. Bring the vector to exactly ``target_dim`` dimensions:
   - if ``model_dim > target_dim``: Matryoshka-style truncation (take the first
     ``target_dim`` components) followed by L2 normalization of the slice;
   - if ``model_dim < target_dim``: zero-pad to ``target_dim``;
   - if equal: use as-is.
3. Sign-quantize: each component >= 0 -> 1 bit, < 0 -> 0 bit.
4. Pack bits MSB-first into ``target_dim / 8`` bytes.

Result: a deterministic 64-byte (for dim=512) binary vector with no ongoing
API cost and full local control of the math.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

from ..errors import EmbeddingError

# User explicitly requested we disable PyTorch/GPU usage 
# to prevent memory overhead or crashes on CPU systems.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

log = logging.getLogger(__name__)


class Embedder:
    """Lazily-initialized local sentence-embedding model with binary quantization.

    Use :meth:`embed_many` to embed a whole run's captions in a single inference
    call (sentence-transformers batches internally); :meth:`embed` is a thin
    wrapper kept for single-caption callers.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        target_dim: int = 384,
        device: str = "cpu",
        batch_size: int = 32,
        preload: bool = True,
    ) -> None:
        if target_dim <= 0 or target_dim % 8 != 0:
            raise ValueError("target_dim must be a positive multiple of 8")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model_name = model_name
        self.target_dim = target_dim
        self.device = device
        self.batch_size = int(batch_size)
        self._model = None
        if preload:
            self._load_model()

    def _load_model(self):
        if self._model is None:
            log.info(f"Loading embedding model {self.model_name} on {self.device} ...")
            try:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(self.model_name)
                log.info("Embedding model ready.")
            except Exception as exc:
                raise EmbeddingError(f"Failed to load fastembed model: {exc}") from exc

    def embed(self, text: str) -> bytes:
        """Embed a single ``text`` and return packed binary bytes.

        Convenience wrapper around :meth:`embed_many`; new code should prefer
        ``embed_many`` to batch a whole run in one inference call.
        """
        if not isinstance(text, str):
            raise EmbeddingError("embed() requires a string")
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[bytes]:
        """Embed a batch of ``texts``, chunked to bound peak memory.

        Parameters
        ----------
        texts:
            Captions to embed. Empty/whitespace strings are replaced with a
            single space (some models error on empty input). An empty list is
            a valid no-op and returns ``[]`` without loading the model.

        Returns
        -------
        list[bytes]
            One ``target_dim / 8`` packed binary blob per input, in order.

        Raises
        ------
        EmbeddingError
            If the input is not a list of strings or inference fails.
        """
        if not isinstance(texts, (list, tuple)):
            raise EmbeddingError("embed_many() requires a list of strings")
        if len(texts) == 0:
            return []
        for t in texts:
            if not isinstance(t, str):
                raise EmbeddingError("embed_many() requires a list of strings")

        cleaned = [(t or "").strip() or " " for t in texts]
        self._load_model()

        out: list[bytes] = []
        for start in range(0, len(cleaned), self.batch_size):
            chunk = cleaned[start : start + self.batch_size]
            out.extend(self._encode_chunk(chunk))
        return out

    def _encode_chunk(self, chunk: list[str]) -> list[bytes]:
        """Run one model.encode call over ``chunk`` and quantize the results."""
        try:
            # fastembed.embed returns a generator of numpy arrays, so we convert it to a list
            mat = list(self._model.embed(chunk))
        except Exception as exc:
            raise EmbeddingError(f"Embedding inference failed: {exc}") from exc

        out: list[bytes] = []
        for i in range(len(mat)):
            import numpy as np
            flat = np.array(mat[i], dtype="float64")
            adjusted = self._adjust_dim(flat)
            out.append(self._quantize(adjusted))
        return out

    # --- math ----------------------------------------------------------------

    def _adjust_dim(self, vec: Any) -> list[float]:
        """Truncate (Matryoshka-style) or zero-pad to ``target_dim``."""
        n = int(vec.shape[0])
        if n == self.target_dim:
            return [float(x) for x in vec]

        if n > self.target_dim:
            # Matryoshka truncation + re-normalization of the prefix.
            truncated = vec[: self.target_dim]
            return [float(x) for x in self._l2_normalize(truncated)]

        # n < target_dim -> zero-pad.
        out = [float(x) for x in vec] + [0.0] * (self.target_dim - n)
        return out

    @staticmethod
    def _l2_normalize(vec: Any) -> Any:
        norm = float(math.sqrt(float((vec * vec).sum())))
        if norm < 1e-12:
            return vec
        return vec / norm

    @staticmethod
    def _quantize(vec: list[float]) -> bytes:
        """Sign-quantize and MSB-first bit-pack to ``len(vec) // 8`` bytes."""
        if len(vec) % 8 != 0:
            raise EmbeddingError("quantize requires len(vec) to be a multiple of 8")

        bits = [1 if v >= 0.0 else 0 for v in vec]
        out = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if bits[i + j]:
                    byte |= 1 << (7 - j)  # MSB-first
            out.append(byte)
        return bytes(out)


def embedding_to_hex(b: bytes) -> str:
    """Return the lowercase hex string of an embedding byte string."""
    return b.hex()
