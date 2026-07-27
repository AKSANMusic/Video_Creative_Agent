"""Optional CLAP-based cross-modal mood/energy tagging.

CLAP (Contrastive Language-Audio Pretraining) maps audio and text into a
shared embedding space. We use it to pick, for each section, the best-matching
word from a small fixed vocabulary of mood/energy descriptors. The result is
attached to :class:`AudioMap.cross_modal`.

Stability mandate
------------------
CLAP is heavy (large model download, optional GPU) and its Python packaging is
fragile. This adapter is **strictly optional**:

- If ``enabled=False`` (default in config), :meth:`tag_sections` returns
  ``None`` and the audio map is produced without cross-modal tags.
- If ``enabled=True`` but the CLAP model/wheel is unavailable, the adapter
  logs a warning and returns ``None`` rather than crashing the pipeline.

This keeps the core librosa pipeline lightweight and the whole stage robust.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from ..errors import CreativeEngineError
from .audio_map import CrossModalTag, Section

log = logging.getLogger(__name__)

# Small, stable vocabulary. Kept tiny on purpose: the sequencer only needs a
# coarse handle on per-section character.
DEFAULT_MOOD_VOCAB = [
    "calm", "tense", "bright", "dark", "energetic", "mellow",
    "triumphant", "melancholic", "aggressive", "peaceful",
]
DEFAULT_ENERGY_VOCAB = ["low", "medium", "high", "rising", "falling"]


class CLAPAdapter:
    """Optional CLAP cross-modal tagger.

    Parameters
    ----------
    enabled:
        Master switch. When False, all calls return None immediately.
    model_name:
        HuggingFace LAION CLAP checkpoint to load.
    device:
        ``'cpu'`` by default to honor the no-GPU mandate.
    """

    def __init__(
        self,
        enabled: bool = False,
        model_name: str = "laion/larger_clap_general",
        device: str = "cpu",
        mood_vocab: Optional[list[str]] = None,
        energy_vocab: Optional[list[str]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.model_name = model_name
        self.device = device
        self.mood_vocab = list(mood_vocab or DEFAULT_MOOD_VOCAB)
        self.energy_vocab = list(energy_vocab or DEFAULT_ENERGY_VOCAB)
        self._model: Optional[Any] = None
        self._available: Optional[bool] = None

    # --- availability probe --------------------------------------------------

    def _ensure_model(self) -> bool:
        """Lazily load CLAP. Returns False if unavailable (never raises)."""
        if not self.enabled:
            return False
        if self._available is not None:
            return self._available
        try:
            # LAION CLAP's recommended import path.
            from laion_clap import CLAP_Module  # type: ignore

            model = CLAP_Module(enable_fusion=False)
            model.load_ckpt(self.model_name)  # may download on first use
            model.to(self.device)
            self._model = model
            self._available = True
            log.info("CLAP model loaded: %s on %s", self.model_name, self.device)
        except Exception as exc:
            log.warning(
                "CLAP unavailable (%s); cross-modal tags will be omitted. "
                "Install laion_clap or set ACE_CLAP_ENABLED=false.",
                exc,
            )
            self._available = False
        return self._available

    # --- public API ----------------------------------------------------------

    def tag_sections(
        self,
        audio_path: Path | str,
        sections: list[Section],
    ) -> Optional[list[CrossModalTag]]:
        """Return one :class:`CrossModalTag` per section, or ``None``.

        Each section's [start, end] window is sliced from the audio and
        compared (cosine similarity) against the text embeddings of the
        mood/energy vocabularies; the top word + its similarity are returned.
        """
        if not self.enabled or not sections:
            return None
        if not self._ensure_model():
            return None

        import numpy as np

        audio_path = Path(audio_path)
        try:
            import librosa

            y, sr = librosa.load(str(audio_path), sr=48000, mono=True)
        except Exception as exc:
            log.warning("CLAP could not reload audio for tagging (%s); skipping", exc)
            return None

        tags: list[CrossModalTag] = []
        for sec in sections:
            i0 = max(0, int(sec.start * sr))
            i1 = min(len(y), int(sec.end * sr))
            window = y[i0:i1]
            if len(window) < 2:
                tags.append(CrossModalTag(section_index=sec.index, mood="", energy_word="", similarity=0.0))
                continue
            mood, mood_sim = self._best_word(window, self.mood_vocab)
            energy, energy_sim = self._best_word(window, self.energy_vocab)
            tags.append(
                CrossModalTag(
                    section_index=sec.index,
                    mood=mood,
                    energy_word=energy,
                    similarity=float(max(mood_sim, energy_sim)),
                )
            )
        return tags

    # --- internals -----------------------------------------------------------

    def _best_word(self, window: Any, vocab: list[str]) -> tuple[str, float]:
        import numpy as np

        assert self._model is not None  # ensured by _ensure_model
        try:
            audio_emb = self._model.get_audio_embedding_from_data(
                x=np.asarray(window, dtype="float32").reshape(1, -1),
                use_tensor=False,
            )
            text_emb = self._model.get_text_embedding(vocab)
            audio_emb = np.asarray(audio_emb)
            text_emb = np.asarray(text_emb)
            # Cosine similarity (CLAP embeddings are already normalized-ish).
            a = audio_emb / (np.linalg.norm(audio_emb) + 1e-12)
            t = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-12)
            sims = (t @ a.ravel()).ravel()
            best = int(np.argmax(sims))
            return vocab[best], float(np.clip(sims[best], 0.0, 1.0))
        except Exception as exc:
            log.warning("CLAP embedding failed for a section (%s); defaulting", exc)
            return "", 0.0
