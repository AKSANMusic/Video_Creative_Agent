# Architecture — AI Creative Engine

> Reference document for human lead engineers and the receiving autonomous AI agent.
> Pair with [`HANDOVER.md`](./HANDOVER.md) for the task queue and AI-integration seams.

The AI Creative Engine turns a folder of still images + one music track into a
cinema-grade, beat-synchronized music video across **four deterministic stages**.
Heavy ML runs on Replicate (API-first); everything else — audio analysis,
sequencing math, caching, rendering — runs locally on CPU.

---

## 1. Four-Stage Data Flow

```
                Stage 1 (Vision)                    Stage 2 (Audio)
        ┌─────────────────────────┐         ┌─────────────────────────┐
images ─▶│ ExtractionPipeline.run │  audio ─▶│ AudioPipeline.run       │
        │  florence.extract()     │         │  AudioAnalyzer (librosa)│
        │  llava.extract()        │         │  CLAPAdapter (optional) │
        │  embedder.embed_many()  │         │                         │
        └───────────┬─────────────┘         └───────────┬─────────────┘
                    │ image_metadata (SQLite)            │ audio_map.json
                    ▼                                    ▼
        ┌─────────────────────────────────────────────────────────┐
        │              Stage 3 (Narrative Sequencer)               │
        │   NarrativePipeline.run(image_cache, audio_map)          │
        │     1. assign images to sections (tension ↔ energy)      │
        │     2. per-section ordering: exact DP ≤12, else greedy   │
        │     3. beat-lock cuts to downbeats; cover [0, duration]  │
        └─────────────────────────┬───────────────────────────────┘
                                  │ timeline.json
                                  ▼
        ┌─────────────────────────────────────────────────────────┐
        │                 Stage 4 (Render)                         │
        │   RenderPipeline.run(timeline, out.mp4)                  │
        │     PlanBuilder → FilterGraphBuilder → FFmpegEncoder     │
        │     → DriftAuditor (KPI: A/V drift < 40 ms)              │
        └─────────────────────────┬───────────────────────────────┘
                                  ▼
                              output.mp4
```

Each stage is **independent and cacheable**. Re-running any stage on identical
inputs is a no-op (cache hit). The only cross-stage contract files are
`audio_map.json` (Stage 2 → 3) and `timeline.json` (Stage 3 → 4).

---

## 2. Module Layout

```
src/ai_creative_engine/
├── cli.py                 # Typer dispatch; the only entrypoint surface
├── config.py              # AppConfig (pydantic-settings); all ACE_* knobs
├── errors.py              # Typed exception hierarchy
├── models.py              # ImageMetadata (Stage-1 JSON contract)
├── cache.py               # MetadataCache (image SQLite, WAL)
├── pipeline.py            # ExtractionPipeline (Stage 1 orchestrator)
├── retries.py             # tenacity helpers
├── vision/
│   ├── florence.py        # Florence-2 client (Replicate)
│   ├── llava.py           # LLaVA-NeXT client (Replicate)
│   └── embeddings.py      # Embedder: local ST + sign-quantize (batched)
├── audio/
│   ├── analyzer.py        # librosa signal analysis
│   ├── audio_map.py       # AudioMap pydantic schema (Stage-2 contract)
│   ├── audio_cache.py     # AudioCache (SQLite, WAL)
│   ├── clap_adapter.py    # optional CLAP cross-modal tagging
│   ├── exporter.py        # load/export audio_map.json
│   └── pipeline.py        # AudioPipeline (Stage 2 orchestrator)
├── narrative/
│   ├── timeline.py        # Timeline pydantic schema (Stage-3 contract)
│   ├── scoring.py         # tension/energy + Hamming + JSD transition cost
│   ├── sequencer.py       # NarrativeSequencer (DP / greedy Hamiltonian)
│   ├── cache.py           # TimelineCache (SQLite, WAL)
│   ├── exporter.py        # load/export timeline.json
│   └── pipeline.py        # NarrativePipeline (Stage 3 orchestrator)
└── render/
    ├── plan.py            # PlanBuilder → frame-accurate EDL + Ken Burns
    ├── filtergraph.py     # ffmpeg filter_complex builder (zoompan + xfade)
    ├── encoder.py         # FFmpegEncoder (argv assembly + run, mockable)
    ├── drift.py           # DriftAuditor (<40ms KPI)
    └── pipeline.py        # RenderPipeline (Stage 4 orchestrator)
```

**Single-responsibility rule:** `cli.py` is dispatch-only. Business logic lives
in the package. Reporters/renderers never touch the network or the DB directly.

---

## 3. Core JSON Contracts

All contracts are `pydantic v2` models with `extra="forbid"` so schema drift
fails loudly. Round-trip helpers live in each `exporter.py`.

### 3.1 `ImageMetadata` (Stage 1 → SQLite `image_metadata` row)

Stored as `payload_json` + a separate `embedding_blob` (raw 64 bytes). Serialized
payload is capped at **512 bytes**.

```jsonc
{
  "image_id": "a3f1...40hexchars",
  "file_path": "/abs/path/to/img.png",
  "width": 1920,
  "height": 1080,
  "exif": { "Make": "Canon" },            // may be null
  "florence_caption": "a red square on white",
  "objects": ["square"],
  "bounding_boxes": [[0.1, 0.1, 0.9, 0.9]],
  "llava_mood": "tense",
  "llava_tension": 0.82,                  // [0.0, 1.0]
  "llava_symbolism": "a warning sign",
  "embedding_hex": "ab12...128hexchars"   // 64-byte sign-quantized vector
}
```

### 3.2 `audio_map.json` (Stage 2 → Stage 3)

```jsonc
{
  "audio_id": "1111...40hexchars",
  "file_path": "/abs/path/to/track.mp3",
  "duration": 187.42,
  "sample_rate": 22050,
  "bpm": 128.0,
  "beats": [0.464, 0.928, /* ... */],
  "downbeats": [0.464, 2.32, /* every 4th beat */],
  "onsets": [0.1, 0.52, /* ... */],
  "rms_curve": [/* 64 floats in [0,1] */],
  "spectral_contrast_curve": [/* 64 floats in [0,1] */],
  "sections": [
    { "index": 0, "start": 0.0, "end": 23.1, "label": "intro", "mean_energy": 0.21 }
  ],
  "cross_modal": [                        // null when CLAP disabled
    { "section_index": 0, "mood": "dark", "energy_word": "calm", "similarity": 0.71 }
  ]
}
```

Curves are downsampled to **64 buckets** (`CURVE_BUCKETS`) regardless of track
length, keeping the file ~5 KB.

### 3.3 `timeline.json` (Stage 3 → Stage 4)

```jsonc
{
  "audio_id": "1111...40hexchars",
  "audio_file": "/abs/path/to/track.mp3",
  "duration": 187.42,
  "bpm": 128.0,
  "downbeats_used": [0.464, 2.32, /* cut anchors */],
  "entries": [
    {
      "index": 0,
      "image_id": "a3f1...40hexchars",
      "file_path": "/abs/path/to/img.png",
      "section_index": 0,
      "start": 0.0,
      "end": 2.32,
      "transition": { "type": "cut", "duration_s": 0.0 }   // cut|dissolve|zoom
    }
  ]
}
```

Invariants enforced by the model:
- `entries` are contiguous (no gaps/overlaps beyond 1e-6).
- The last entry's `end` equals `duration` (±1e-3).
- `transition.duration_s ≤ (end - start)`.

---

## 4. Caching & State Strategy

Three independent SQLite caches, all WAL + `synchronous=NORMAL`:

| Cache | Table | PK | Keyed by | Module |
|---|---|---|---|---|
| image metadata | `image_metadata` | `image_id` | sha1(raw image bytes) | `cache.py` |
| audio map | `audio_map` | `audio_id` | sha1(raw audio bytes) | `audio/audio_cache.py` |
| timeline | `timeline` | `timeline_key` | composite of image-set + audio hashes | `narrative/cache.py` |

**Content-addressed → renames never invalidate; identical inputs never re-pay.**

### 4.1 The `user_version` Schema Nuance (⚠ important)

Every cache stamps `PRAGMA user_version = SCHEMA_VERSION` on connect — but
**today nothing reads or migrates it**. Current resilience comes from the fact
that the row body is a versioned pydantic model serialized to JSON, so:

- **Adding optional fields** → safe. Old rows deserialize into the new model;
  pydantic fills defaults.
- **Changing a field type or removing one** → old rows crash `model_validate`
  (wrapped as `CacheError`).

**Future work:** add a version-gated `migrate(old, new)` step in each cache's
`_connect()`. Until then, treat `SCHEMA_VERSION` as a marker, not an enforced
contract.

---

## 5. Testing Philosophy — Offline-First

- **148 tests pass** in ~7 s, fully offline (`pytest -q`).
- No test ever hits the network or a real GPU. Mocks/fakes:
  - Replicate clients → `FakeFlorence` / `FakeLLaVA` (`tests/conftest.py`).
  - `sentence-transformers` → `DeterministicEmbedder` + a batched fake ST
    monkeypatched via `monkeypatch.setitem(sys.modules, "sentence_transformers", …)`.
  - Audio analysis → synthetic click tracks; CLAP → fake model + an
    unavailable-model path.
  - ffmpeg/ffprobe → encoder runner + auditor's `_probe_duration` monkeypatched.
- Run a single module: `pytest tests/test_render.py`; by name: `pytest -k chunk`.
- **Convention:** every new code path gets a test in the matching
  `tests/test_<module>.py`, including the empty/edge case.

---

## 6. Error Handling — Fail-Soft, Typed

- A single typed hierarchy in `errors.py` (`CreativeEngineError` → `AuthError`,
  `CacheError`, `EmbeddingError`, `AudioAnalysisError`, …).
- `cli.py` translates each into a Rich panel + exit code; **no raw tracebacks**.
- **Fail-soft batches:** `ExtractionPipeline.run` wraps every per-image step in
  `try/except`; failures increment `stats.failed` and the run continues. The
  batched embed step is also guarded: if `embed_many` raises, all pending images
  are marked failed rather than crashing the run.
- **Graceful degradation:** silent audio → empty beat grid + flat curves;
  missing CLAP → tagged as "off"; missing ffmpeg → `CreativeEngineError` with an
  actionable message.

---

## 7. Performance Characteristics

- **Embedding** (the previously dominant per-item cost) is now **batched**:
  `Embedder.embed_many(texts)` chunks at `batch_size` (default 32, configurable
  via `ACE_EMBEDDING_BATCH_SIZE`) and issues one `model.encode` per chunk.
- **Sequencer DP** is `O(2ⁿ·n²)` per section; `ACE_SEQUENCER_DP_THRESHOLD=12`
  falls back to greedy above that. The hard scaling risk is *many images in one
  section*, not total image count. The per-section `continuity_weight_override`
  (LLM cut-steering) is **zero-cost**: it only swaps which weight feeds
  `transition_cost` during the existing pairwise-cost matrix build — no extra
  passes, no change to DP complexity.
- **Render** is CPU ffmpeg (libx264). A 3-min/30fps/720p timeline with ~90
  zoompan windows is a few minutes of encode. GPU is **not wired** (see Roadmap).

---

## 8. Conventions (must follow for any edit)

- Python ≥ 3.10; `from __future__ import annotations` at top of every module.
- Type hints on all public functions/dataclasses; Google-style docstrings.
- **Config via `AppConfig` only** — never read env vars inside a module.
- Boxed comment banners (`# ════...`) separate major sections in large modules.
- Preserve incremental-skip, retry/backoff, and fail-soft semantics in the scraper.
