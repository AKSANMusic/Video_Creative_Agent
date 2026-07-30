"""Application configuration via pydantic-settings.

Settings are resolved from (in order of precedence):
    1. Explicit constructor arguments.
    2. Process environment variables (e.g. ``REPLICATE_API_TOKEN``).
    3. A local ``.env`` file in the current working directory.

All ``ACE_*`` keys can be overridden via environment or ``.env`` without
touching the code.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import ConfigError


class Settings(BaseSettings):
    """Strongly-typed runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ACE_",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Secrets (no ACE_ prefix; the canonical env name is REPLICATE_API_TOKEN) ---
    replicate_api_token: str = Field(
        default="",
        description="Replicate API token. Required for real inference calls.",
        validation_alias="REPLICATE_API_TOKEN",
    )

    # --- Model identifiers on Replicate (owner/model:version_hash) ---
    # Version hashes MUST be pinned. Without them, Replicate returns 404 for
    # cold models even when the model name is valid. Get hashes from the model's
    # "Versions" tab on replicate.com.
    florence_model: str = Field(
        default="lucataco/florence-2-large:da53547e17d45b9cfb48174b2f18af8b83ca020fa76db62136bf9c6616762595",
        description="Florence-2 model identifier on Replicate (owner/model:version_hash).",
    )
    llava_model: str = Field(
        default="yorickvp/llava-v1.6-vicuna-13b:0603dec596080fa084e26f0ae6d605fc5788ed2b1a0358cd25010619487eae63",
        description="LLaVA-NeXT model identifier on Replicate (owner/model:version_hash).",
    )

    # --- Local embedding model ---
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Local sentence-transformers model used for binary embeddings.",
    )
    embedding_dim: int = Field(
        default=384,
        description="Target embedding dimension (Matryoshka truncation / zero-pad target).",
    )
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        description=(
            "Max captions passed to the embedder in a single encode call. "
            "Smaller = lower peak memory; tune down if you hit OOM on very large folders."
        ),
    )

    # --- Networking ---
    request_timeout_s: float = Field(
        default=60.0,
        description="Per-call API timeout in seconds.",
    )

    # --- Storage ---
    db_path: Path = Field(
        default=Path("creative_engine.db"),
        description="SQLite database path (relative to CWD or absolute).",
    )

    # --- Audio analysis (Stage 2) ---
    audio_sample_rate: int = Field(
        default=22050,
        description="librosa resample target sample rate (Hz). Lower = faster, less detail.",
    )
    audio_hop_length: int = Field(
        default=512,
        ge=64,
        description="librosa hop length (frames). Larger = faster, coarser time grid.",
    )
    audio_n_segments: int = Field(
        default=8,
        ge=2,
        description="Target number of structural sections for agglomerative segmentation.",
    )
    clap_enabled: bool = Field(
        default=False,
        description="Enable optional CLAP cross-modal tagging (heavy; off by default).",
    )
    clap_model: str = Field(
        default="laion/larger_clap_general",
        description="LAION CLAP checkpoint to use when clap_enabled is true.",
    )
    clap_device: str = Field(
        default="cpu",
        description="Device for CLAP inference ('cpu' honors the no-GPU mandate).",
    )

    # --- Narrative sequencer (Stage 3) ---
    sequencer_dp_threshold: int = Field(
        default=12,
        ge=2,
        description="Per-section image count above which exact DP falls back to greedy.",
    )
    sequencer_continuity_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Blend weight for color continuity vs embedding similarity in transition cost.",
    )
    sequencer_cut_on: str = Field(
        default="downbeats",
        description="Beat grid to lock cuts to: 'downbeats' or 'beats'.",
    )

    # --- Render (Stage 4) ---
    video_codec: str = Field(
        default="libx264",
        description=(
            "ffmpeg video encoder. Examples: libx264, libx265, h264_nvenc, "
            "hevc_nvenc, h264_vaapi, hevc_vaapi, h264_videotoolbox, "
            "hevc_videotoolbox. Validated at load time against the known registry."
        ),
    )

    @field_validator("embedding_dim")
    @classmethod
    def _dim_positive_multiple_of_8(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("embedding_dim must be positive")
        if v % 8 != 0:
            raise ValueError("embedding_dim must be a multiple of 8 (bit-packing constraint)")
        return v
 
    @field_validator("embedding_batch_size")
    @classmethod
    def _batch_size_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("embedding_batch_size must be >= 1")
        return v

    @field_validator("video_codec")
    @classmethod
    def _video_codec_known(cls, v: str) -> str:
        # Fail fast at config-load time rather than mid-render. Uses the render
        # codec registry as the single source of truth for known encoders.
        from .render.codec_profile import resolve_codec_profile

        resolve_codec_profile(v)  # raises ConfigError if unknown
        return v

    def require_api_token(self) -> str:
        """Return the Replicate token or raise :class:`ConfigError`.

        Call this from any code path that actually performs inference so that a
        missing token fails fast with an actionable message.
        """
        if not self.replicate_api_token:
            raise ConfigError(
                "REPLICATE_API_TOKEN is not set. "
                "Copy .env.example to .env and fill it in, or export the variable."
            )
        return self.replicate_api_token


def get_settings() -> Settings:
    """Build a :class:`Settings` instance from env / .env."""
    return Settings()
