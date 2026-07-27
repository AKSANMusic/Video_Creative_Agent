"""Audio subsystem: Librosa analysis + optional CLAP cross-modal mapping.

Stage 2 of the pipeline. Produces a compact `audio_map.json` describing the
beat grid, structural sections, energy curve, and (optionally) per-section
mood/energy tags for the Stage-3 narrative sequencer.
"""
