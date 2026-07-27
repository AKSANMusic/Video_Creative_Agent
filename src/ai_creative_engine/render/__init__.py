"""Render subsystem (Stage 4): turn a timeline.json into an MP4 via FFmpeg.

Consumes Stage-3 timeline + the source audio and produces a beat-locked
music video. Strict KPI: A/V sync drift < 40ms over a 3-minute track.
"""
