# AI Creative Engine

Automated narrative engine that turns static image collections into cinema-grade,
beat-synchronized music videos.

Stages implemented so far:
- **Stage 1 (Week 1):** vision extraction (Florence-2 + LLaVA-NeXT via Replicate),
  local CPU binary embeddings, SQLite caching.
- **Stage 2 (Week 2):** audio signal analysis (Librosa), optional CLAP
  cross-modal mood/energy tagging, `audio_map.json` export, SQLite caching.
- **Stage 3 (Week 3):** narrative sequencer (DP / greedy Hamiltonian path) that
  matches image tension to audio energy and beat-locks cuts to downbeats,
  emitting `timeline.json`.
- **Stage 4 (Week 4):** render pipeline that turns `timeline.json` into an MP4
  via FFmpeg — frame-accurate EDL, Ken Burns zoom/pan, xfade stitching, CFR
  H.264/AAC encode, and an A/V drift auditor enforcing < 40ms sync.

## Architecture

```
Stage 1 (vision)                          Stage 2 (audio)
--------------------------------------    ---------------------------------------
images_dir                                audio file (mp3/wav/flac/...)
   |                                          |
   v                                          v
pipeline.run()                            AudioPipeline.run()
   +- sha1 image_id -> cache hit? skip      +- sha1 audio_id -> cache hit? skip
   +- pillow: w/h + EXIF                    +- AudioAnalyzer (librosa):
   +- florence.extract() [Replicate]        |     tempo/bpm, beats, downbeats,
   +- llava.extract()   [Replicate]         |     onsets, RMS curve,
   +- embeddings.embed() (local CPU,        |     spectral contrast, sections
   |     512-d sign-quantized -> 64B)       +- CLAPAdapter (optional, off by default):
   +- cache.upsert() -> SQLite WAL          |     per-section mood/energy words
                                            +- cache.upsert() -> SQLite WAL
                                            +- export audio_map.json

Stage 3 (narrative sequencer)
---------------------------------------
image cache (Stage 1) + audio_map.json (Stage 2)
   |
   v
NarrativePipeline.run()
   +- load ImageMetadata records + AudioMap
   +- timeline cache hit (composite key)? skip
   +- NarrativeSequencer:
   |     Phase A: assign images to sections (tension <-> energy)
   |     Phase B: per-section ordering (exact DP for <=12 images, else greedy)
   |               minimizing transition cost (Hamming embedding distance +
   |               JSD color-continuity proxy); slots locked to downbeats.
   +- safety net: guarantee full [0, duration] coverage
   +- cache + export timeline.json
```

## Install

```bash
cd /home/keyhani/ai_creative_engine
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then fill REPLICATE_API_TOKEN for Stage 1
```

## Run

Stage 1 — vision extraction (needs `REPLICATE_API_TOKEN`):
```bash
ai-creative-engine extract --images ./sample_images --db ./out.db
```

Stage 2 — audio analysis (fully offline, no API key needed):
```bash
ai-creative-engine analyze-audio --audio ./track.mp3 --out audio_map.json --no-clap
```

Stage 3 — narrative sequencing (fully offline; reads Stages 1 & 2 caches):
```bash
ai-creative-engine sequence --audio-map audio_map.json --out timeline.json
```
Re-running any command on the same inputs produces nothing new (cache hits).

Stage 4 — render (needs `ffmpeg` + `ffprobe` on PATH):
```bash
ai-creative-engine render --timeline timeline.json --out output.mp4
```
Exits non-zero if A/V drift exceeds the 40ms KPI.

## Design Mandates

- **API-first / zero-DevOps:** heavy ML on Replicate; librosa + math + caching local.
- **Never re-pay:** every result (image metadata, audio map) cached in SQLite
  immediately, keyed by sha1 of raw bytes.
- **Defensive:** tenacity retries + typed exceptions; safe-default parsing;
  fail-soft batch; CLAP degrades to "off" if unavailable rather than crashing.
- **Compact:** image payloads < 0.5 KB; embeddings are 64-byte binary (512-d
  sign-quantized); audio curves downsampled to 64 buckets (~5 KB per map).

## Configuration

All settings are overridable via env vars or `.env` (prefix `ACE_`). Key ones:

| Variable | Default | Purpose |
|---|---|---|
| `REPLICATE_API_TOKEN` | (none) | Required for Stage 1 inference. |
| `ACE_EMBEDDING_BATCH_SIZE` | `32` | Max captions per encode call (OOM guard for large folders). |
| `ACE_DB_PATH` | `creative_engine.db` | SQLite cache path. |
| `ACE_AUDIO_SAMPLE_RATE` | `22050` | librosa resample target. |
| `ACE_AUDIO_HOP_LENGTH` | `512` | librosa hop length (frames). |
| `ACE_AUDIO_N_SEGMENTS` | `8` | Structural section count target. |
| `ACE_CLAP_ENABLED` | `false` | Enable optional CLAP cross-modal tagging. |
| `ACE_SEQUENCER_DP_THRESHOLD` | `12` | Per-section image count above which exact DP falls back to greedy. |
| `ACE_SEQUENCER_CONTINUITY_WEIGHT` | `0.5` | Blend weight for color continuity vs embedding similarity. |
| `ACE_SEQUENCER_CUT_ON` | `downbeats` | Beat grid to lock cuts to (`downbeats` or `beats`). |
| `ACE_VIDEO_CODEC` | `libx264` | ffmpeg video encoder (validated against known registry). |

### Per-section cut steering (`continuity_weight_override`)

An external director (LLM or human) can override the global continuity weight
**per section** by setting `continuity_weight_override` on a `Section` in
`audio_map.json` (range `[0,1]`, or `null` to use the global). Lower = harder,
more dynamic cuts; higher = smoother, color-stable cuts. Only the directed
section's DP/greedy ordering changes; others are unaffected (the override also
ripples into the next section's entry image via the cross-section seam).

### Video codec selection (`ACE_VIDEO_CODEC` / `--codec`)

The renderer routes the chosen encoder to the correct ffmpeg quality flags via
a `CodecProfile` registry: `libx264`/`libx265` → `-crf`, `h264_nvenc`/`hevc_nvenc`
→ `-cq`, `h264_vaapi`/`hevc_vaapi` → `-qp`, `*_videotoolbox` → `-q:v` (and omits
`-preset`). Set `ACE_VIDEO_CODEC` or pass `--codec` to `render`; unknown names
fail fast at config load. (Runtime availability probing is a follow-up — see
HANDOVER.md §5.)

## Tests

```bash
pytest
```

All tests run fully offline:
- Stage 1: Replicate + embeddings are mocked/faked.
- Stage 2: audio analysis runs on **synthetic click tracks** (deterministic,
  no network); CLAP is tested via a fake model + an unavailable-model path.
- Stage 3: sequencer runs on **synthetic image metadata + audio maps** with
  known embeddings, asserting contiguity, beat-locking, and determinism.
 
Stage 4 — render (needs `ffmpeg` + `ffprobe` on PATH):
```bash
ai-creative-engine render --timeline timeline.json --out output.mp4
```
Exits non-zero if A/V drift exceeds the 40ms KPI.
Stage 4 — render (needs `ffmpeg` + `ffprobe` on PATH):
```bash
ai-creative-engine render --timeline timeline.json --out output.mp4
```
Exits non-zero if A/V drift exceeds the 40ms KPI.
