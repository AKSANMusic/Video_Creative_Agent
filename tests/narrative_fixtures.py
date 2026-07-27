"""Synthetic fixtures for Stage-3 tests: deterministic image metadata + audio maps.

Embeddings are constructed so that pairwise Hamming distances are known and
controllable, which lets us assert exact sequencer ordering.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from ai_creative_engine.audio.audio_map import AudioMap, Section
from ai_creative_engine.models import EMBEDDING_HEX_LEN, ImageMetadata


def _embedding_hex_from_int(seed: int) -> str:
    """64-byte embedding whose bits are derived from sha1(seed) -> deterministic."""
    digest = hashlib.sha1(str(seed).encode()).digest()  # 20 bytes
    buf = bytearray()
    while len(buf) < 64:
        buf.extend(digest)
    return bytes(buf[:64]).hex().ljust(EMBEDDING_HEX_LEN, "0")[:EMBEDDING_HEX_LEN]


def make_image(
    image_id_seed: int,
    tension: float = 0.5,
    mood: str = "neutral",
    file_path: Optional[str] = None,
) -> ImageMetadata:
    """Build a valid ImageMetadata with a deterministic 512-bit embedding."""
    image_id = hashlib.sha1(f"img-{image_id_seed}".encode()).hexdigest()
    return ImageMetadata(
        image_id=image_id,
        file_path=file_path or f"/tmp/img_{image_id_seed}.png",
        width=64,
        height=48,
        florence_caption=f"image {image_id_seed}",
        objects=[],
        bounding_boxes=[],
        llava_mood=mood,
        llava_tension=tension,
        llava_symbolism="",
        embedding_hex=_embedding_hex_from_int(image_id_seed),
    )


def make_images(n: int, tensions: Optional[list[float]] = None) -> list[ImageMetadata]:
    tensions = tensions or ([0.2, 0.8] * ((n // 2) + 1))[:n]
    return [make_image(i + 1, tension=tensions[i]) for i in range(n)]


def make_audio_map(
    duration: float = 8.0,
    bpm: float = 120.0,
    beats_per_section: int = 4,
    n_sections: int = 2,
    energies: Optional[list[float]] = None,
) -> AudioMap:
    """Build an audio map with a regular beat grid and equal-length sections.

    At 120 bpm the beat period is 0.5s. With beats_per_section=4 each section
    spans 2.0s. Downbeats land every 4th beat.
    """
    beat_period = 60.0 / bpm
    total_beats = int(round(duration / beat_period))
    beats = [round(i * beat_period, 6) for i in range(1, total_beats + 1)]
    downbeats = [b for i, b in enumerate(beats) if i % beats_per_section == 0]

    # Sections must tile the FULL [0, duration] range so the sequencer's
    # section-assignment phase covers every cut slot. We divide duration evenly
    # into n_sections pieces (independent of the beat grid) to guarantee that.
    energies = energies or ([0.2, 0.8] * ((n_sections // 2) + 1))[:n_sections]
    sections: list[Section] = []
    for i in range(n_sections):
        start = duration * i / n_sections
        end = duration * (i + 1) / n_sections
        sections.append(
            Section(
                index=i,
                start=start,
                end=end,
                label="",
                mean_energy=energies[i % len(energies)],
            )
        )

    import hashlib

    audio_id = hashlib.sha1(b"synthetic-audio").hexdigest()
    return AudioMap(
        audio_id=audio_id,
        file_path="/tmp/track.wav",
        duration=duration,
        sample_rate=22050,
        bpm=bpm,
        beats=beats,
        downbeats=downbeats,
        onsets=beats,
        rms_curve=[0.5] * 64,
        spectral_contrast_curve=[0.5] * 64,
        sections=sections,
        cross_modal=None,
    )
