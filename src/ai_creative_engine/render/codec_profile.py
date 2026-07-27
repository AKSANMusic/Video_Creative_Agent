"""FFmpeg codec profiles: map a user-facing codec name to its quality/preset flags.

Different ffmpeg encoders express "quality" with different flags:
    - libx264 / libx265        -> ``-crf``
    - h264_nvenc / hevc_nvenc  -> ``-cq``  (constant quality)
    - h264_vaapi / hevc_vaapi  -> ``-qp``
    - *_videotoolbox           -> ``-q:v`` (and no -preset flag)

This module is the single place that knows those quirks. Adding a new codec
is a one-line registry entry; :func:`quality_args` stays generic.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ConfigError


@dataclass(frozen=True)
class CodecProfile:
    """How a given ffmpeg video encoder expresses quality and preset.

    Attributes:
        video_codec: the string passed to ffmpeg's ``-c:v``.
        quality_flag: the flag name for constant-quality (``crf``, ``cq``,
            ``qp``, or ``q:v``). Rendered as ``-<quality_flag> <quality>``.
        preset_flag: the flag name for preset (default ``preset``). Empty
            string means the encoder ignores ``-preset`` (e.g. videotoolbox)
            and it is omitted entirely.
    """

    video_codec: str
    quality_flag: str
    preset_flag: str = "preset"


# Registry of supported codecs. Keys are the user-facing names (also the
# ``-c:v`` values). Add a row here to support a new encoder.
CODEC_PROFILES: dict[str, CodecProfile] = {
    "libx264": CodecProfile("libx264", "crf"),
    "libx265": CodecProfile("libx265", "crf"),
    "h264_nvenc": CodecProfile("h264_nvenc", "cq"),
    "hevc_nvenc": CodecProfile("hevc_nvenc", "cq"),
    "h264_vaapi": CodecProfile("h264_vaapi", "qp"),
    "hevc_vaapi": CodecProfile("hevc_vaapi", "qp"),
    "h264_videotoolbox": CodecProfile("h264_videotoolbox", "q:v", preset_flag=""),
    "hevc_videotoolbox": CodecProfile("hevc_videotoolbox", "q:v", preset_flag=""),
}


def resolve_codec_profile(name: str) -> CodecProfile:
    """Return the :class:`CodecProfile` for ``name`` or raise :class:`ConfigError`.

    Fails fast with an actionable message listing known codecs, so a typo in
    ``ACE_VIDEO_CODEC`` is caught at config-load time, not mid-render.
    """
    key = (name or "").strip()
    profile = CODEC_PROFILES.get(key)
    if profile is None:
        known = ", ".join(sorted(CODEC_PROFILES))
        raise ConfigError(
            f"Unknown video codec {key!r}. Known: {known}."
        )
    return profile


def quality_args(profile: CodecProfile, quality, preset: str) -> list[str]:
    """Build the ffmpeg argv fragment for quality + preset flags.

    Pure and side-effect-free; the unit tests target this directly.
    """
    args: list[str] = ["-c:v", profile.video_codec]
    if profile.preset_flag:
        args += ["-" + profile.preset_flag, str(preset)]
    args += ["-" + profile.quality_flag, str(quality)]
    return args


__all__ = [
    "CodecProfile",
    "CODEC_PROFILES",
    "resolve_codec_profile",
    "quality_args",
]
