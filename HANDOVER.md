# Handover — AI Creative Engine

> **Audience:** the autonomous AI agent picking up this project, and the human
> lead engineer reviewing its work. Read [`ARCHITECTURE.md`](./ARCHITECTURE.md)
> first for the system map; this file is the **task queue + integration guide**.

**Status (snapshot):** 4 stages implemented and wired end-to-end via the CLI.
**162 tests passing**, fully offline, ~7 s. Last shipped work: `video_codec`
config + `CodecProfile` registry (Stage 4, §4.1) — routes libx264/NVENC/
VideoToolbox/VAAPI to the correct quality flags (`-crf`/`-cq`/`-q:v`/`-qp`);
name-validated at config load; `--codec` CLI override.

---

## TL;DR — First Commands to Run

```bash
cd /home/keyhani/ai_creative_engine
source .venv/bin/activate
pytest -q                 # expect: 148 passed in ~7s
ai-creative-engine --help # shows: extract | info | analyze-audio | sequence | render
```

If `pytest` is not green, **stop and fix before any new work.** The suite is
fully offline; failures are never "environmental."

---

## 1. Configuration Surface

### 1.1 CLI Entry Points

All commands are subcommands of `ai-creative-engine` (`src/ai_creative_engine/cli.py`).
Global options come **before** the subcommand: `--config`, `--no-proxy` are not
used here; the only global is via env/`.env`.

| Command | Purpose | Key flags |
|---|---|---|
| `extract` | Stage 1: vision + embeddings → SQLite | `--images DIR` (req), `--db PATH`, `--max-images N`, `--verbose` |
| `info` | Cache stats for a DB | `--db PATH` (req) |
| `analyze-audio` | Stage 2: librosa + optional CLAP → `audio_map.json` | `--audio PATH` (req), `--out PATH`, `--db PATH`, `--[no-]clap`, `--verbose` |
| `sequence` | Stage 3: beat-locked timeline from cache + audio map | `--audio-map PATH` (req), `--out PATH`, `--db PATH`, `--verbose` |
| `render` | Stage 4: `timeline.json` → MP4 via ffmpeg | `--timeline PATH` (req), `--out PATH`, `--fps FLOAT`, `--width INT`, `--height INT`, `--verbose` |

> **Note on `--embed-batch-size`:** the batch size is currently **env-only**
> (`ACE_EMBEDDING_BATCH_SIZE`). If you want a per-run CLI override, see
> *Open Follow-ups* §5 — it is **not** the immediate next task.

### 1.2 Environment Variables → `AppConfig`

All `ACE_*` vars are read by `pydantic-settings` (`src/ai_creative_engine/config.py`),
precedence: process env → `.env` → defaults. Validators reject bad values fast.

| Variable | Default | Maps to | Notes |
|---|---|---|---|
| `REPLICATE_API_TOKEN` | (none) | `replicate_api_token` | **Required** for Stage 1; no `ACE_` prefix. |
| `ACE_FLORENCE_MODEL` | `microsoft/florence-2` | `florence_model` | |
| `ACE_LLAVA_MODEL` | `yorickvp/llava-v1.6-vicuna-13b` | `llava_model` | |
| `ACE_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | `embedding_model` | |
| `ACE_EMBEDDING_DIM` | `512` | `embedding_dim` | must be a positive multiple of 8. |
| `ACE_EMBEDDING_BATCH_SIZE` | `32` | `embedding_batch_size` | **≥1**; OOM guard for big folders. |
| `ACE_REQUEST_TIMEOUT_S` | `60` | `request_timeout_s` | per Replicate call. |
| `ACE_DB_PATH` | `creative_engine.db` | `db_path` | |
| `ACE_AUDIO_SAMPLE_RATE` | `22050` | `audio_sample_rate` | |
| `ACE_AUDIO_HOP_LENGTH` | `512` | `audio_hop_length` | ≥64. |
| `ACE_AUDIO_N_SEGMENTS` | `8` | `audio_n_segments` | ≥2. |
| `ACE_CLAP_ENABLED` | `false` | `clap_enabled` | heavy; off by default. |
| `ACE_CLAP_MODEL` | `laion/larger_clap_general` | `clap_model` | |
| `ACE_CLAP_DEVICE` | `cpu` | `clap_device` | |
| `ACE_SEQUENCER_DP_THRESHOLD` | `12` | `sequencer_dp_threshold` | DP→greedy cutoff per section. |
| `ACE_SEQUENCER_CONTINUITY_WEIGHT` | `0.5` | `sequencer_continuity_weight` | `[0,1]`. |
| `ACE_SEQUENCER_CUT_ON` | `downbeats` | `sequencer_cut_on` | `downbeats` or `beats`. |

---

## 2. AI-to-AI Integration Seams

These are the **deliberate hooks** an external LLM / agent can use today, and
the **planned** ones (see Roadmap).

### 2.1 Stage 2 — `cross_modal` tags (✅ available now)

`AudioMap.sections[i].cross_modal` is a CLAP-derived `{mood, energy_word,
similarity}` triple. An external agent can **mutate the JSON directly** before
Stage 3:

```python
from ai_creative_engine.audio.exporter import load_audio_map, export_audio_map
am = load_audio_map("audio_map.json")
# External LLM rewrites mood/energy based on a semantic prompt.
am.cross_modal = [ ...agent-decided tags... ]
export_audio_map(am, "audio_map.directed.json")
```

These tags flow into section assignment via `tension_energy_score`
(`narrative/scoring.py`). **Today they do not alter transition cost** — that is
the planned hook in §4.

### 2.2 Stage 1 — embeddings exportable as a feature matrix (✅ trivial)

The 64-byte sign-quantized vectors decode to a clean `(N, 512)` binary matrix
suitable for conditioning a diffusion/transformer video model. The pairwise
Hamming cost matrix in `NarrativeSequencer._hamiltonian_dp` is an "edit graph."
Neither is exported yet; both are ~30-line helpers away (see §5).

### 2.3 SQLite + `timeline.json` as shared memory (✅ shaped for it)

`timeline.json` is a pure, `extra="forbid"` pydantic model with round-trip
helpers. The timeline cache key is a composite content hash, so a mutated
timeline auto-creates a new key — **free versioning for an iterative agent edit
loop**. Multi-agent shared state can use the `timeline` table as a blackboard.

### 2.4 Stage 3 — `continuity_weight_override` (✅ available now)

The seam that lets an LLM steer **how aggressively cuts transition** per
section. A `Section` now carries an optional `continuity_weight_override`
(`None` by default → falls back to the global `ACE_SEQUENCER_CONTINUITY_WEIGHT`).
When set, the Stage-3 DP/greedy ordering uses it **instead of** the global
weight for that section's pairwise `transition_cost`, leaving all other
sections byte-identical. Held-Karp DP math is unchanged; only the edge-weight
function is section-aware.

An external director agent mutates `audio_map.json` between Stage 2 and 3:

```python
from ai_creative_engine.audio.exporter import load_audio_map, export_audio_map
am = load_audio_map("audio_map.json")
for s in am.sections:
    if s.label == "chorus":
        s.continuity_weight_override = 0.2   # hard, dynamic cuts
    elif s.label == "verse":
        s.continuity_weight_override = 0.9   # smooth, color-stable
export_audio_map(am, "audio_map.directed.json")
ai-creative-engine sequence --audio-map audio_map.directed.json --out tl.json
```

**Forward propagation:** changing section *N*'s ordering changes its exit
image, which becomes `prev_exit_bytes` for section *N+1*'s `_pick_entry`. So a
chorus override ripples into the following verse's *entry* image (desirable —
the director's decision influences the cross-section handoff). Within-section
ordering of non-directed sections is otherwise stable and deterministic.

---

## 3. Roadmap

Ordered by impact × readiness (top = next):

1. **Item #3a — `video_codec` on `FFmpegEncoder`** — **✅ DONE** (see §4.1).
   Routes NVENC / VideoToolbox / VAAPI to correct quality flags via a
   `CodecProfile` registry; no filtergraph changes.
2. **Item #3b — per-section `continuity_weight_override`** + LLM cut-steering
   hook — **✅ DONE** (see §2.4). Opens the prompt → cut-dynamics channel.
3. **`user_version`-gated migration** in each cache's `_connect()` — closes the
   schema-evolution gap (see ARCHITECTURE.md §4.1).
4. **ffmpeg path/codec probe** in the render CLI so a missing NVENC fails fast
   with an actionable message instead of a stderr tail.
5. **Optional follow-ups** (low priority): `--embed-batch-size` CLI flag;
   export helpers (`dump_embeddings.npy`, `export_edit_graph.json`);
   non-4/4 downbeat handling.
   
   > **Note:** items below were renumbered after #3b shipped. The next active
   > task is **Item #3a — `video_codec`** (see §4.1).
3. **`user_version`-gated migration** in each cache's `_connect()` — closes the
   schema-evolution gap (see ARCHITECTURE.md §4.1).
4. **ffmpeg path/codec probe** in the render CLI so a missing NVENC fails fast
   with an actionable message instead of a stderr tail.
5. **Optional follow-ups** (low priority): `--embed-batch-size` CLI flag;
   export of `(N,512)` feature matrix + edit-graph JSON for downstream training.

---

## 4. Immediate Next Task (Item #3) — Detailed Spec for the Receiving Agent

> **Your first job.** Implement both halves of Item #3. Do not branch into the
> lower-priority roadmap items until this is green.

### 4.1 Part A — Expose `video_codec` for Hardware Acceleration

**Goal:** let the renderer use NVENC / VideoToolbox / VAAPI instead of software
`libx264`, without rewriting the filtergraph.

**Where to edit:**

1. `src/ai_creative_engine/config.py` — add to `Settings`:
   ```python
   video_codec: str = Field(
       default="libx264",
       description="ffmpeg video encoder. Examples: libx264, h264_nvenc, h264_videotoolbox, h264_vaapi.",
   )
   ```
   Add a matching validator if desired (no-op passthrough is acceptable for v1).
2. `src/ai_creative_engine/render/encoder.py`:
   - `FFmpegEncoder.__init__` already takes `preset`, `crf`, `pix_fmt`. Add
     `video_codec: str = "libx264"` and store it.
   - In `_build_argv`, replace the hard-coded `"-c:v", "libx264"` with
     `"-c:v", self.video_codec`.
   - **Important:** `crf` is x264-specific. For NVENC/VT, fall back to a
     quality-equivalent (`-cq` for NVENC, `-q:v` for VT) when the codec is not
     x264/x265. A simple `if self.video_codec.startswith(("h264_nvenc", "hevc_nvenc")): …`
     branch is sufficient for v1.
3. `src/ai_creative_engine/cli.py` — thread `video_codec=settings.video_codec`
   into the `RenderPipeline` → `FFmpegEncoder` construction inside the `render`
   command. Also add `--codec` as an optional CLI flag overriding the setting.
4. `.env.example` + `README.md` config table — document `ACE_VIDEO_CODEC`.

**Tests to add** (`tests/test_render.py`):
- `test_encoder_argv_uses_configured_codec` — asserts `-c:v h264_nvenc` appears
  when `FFmpegEncoder(video_codec="h264_nvenc")`.
- `test_encoder_nvenc_uses_cq_not_crf` — verifies the NVENC branch drops `-crf`.
- `test_render_cli_codec_flag_override` — `--codec h264_nvenc` reaches the
  encoder (mock `RenderPipeline.run` as in the existing CLI tests).

### 4.2 Part B — `continuity_weight_override` (LLM Cut-Steering Hook)

**Goal:** let a per-section weight override the global
`ACE_SEQUENCER_CONTINUITY_WEIGHT`, so an external LLM can direct cut feel.

**Where to edit:**

1. `src/ai_creative_engine/audio/audio_map.py` — add an optional field to
   `Section`:
   ```python
   continuity_weight_override: Optional[float] = Field(
       default=None, ge=0.0, le=1.0,
       description="Per-section override of the global continuity weight; "
                   "set by an external director LLM. None = use global.",
   )
   ```
   `extra="forbid"` means old `audio_map.json` files still validate (the field
   is optional). Add a test in `tests/test_audio_map_schema.py`.
2. `src/ai_creative_engine/narrative/scoring.py` — change `transition_cost` to
   accept an optional `continuity_weight_override` and use it when provided
   (else fall back to the existing global argument).
3. `src/ai_creative_engine/narrative/sequencer.py` — in `_pair_cost` (or
   wherever `transition_cost` is invoked), pass
   `section.continuity_weight_override` through. Keep the global default
   behavior identical when the override is `None`.
4. **Document the LLM hook** in `HANDOVER.md` §2.4 (flip ⛔ → ✅) and add an
   example showing an agent mutating `audio_map.json` to soften cuts in a
   chorus (`continuity_weight_override: 0.2`) and harden them in a verse (`0.9`).

**Tests to add:**
- `tests/test_audio_map_schema.py` — section accepts and round-trips the field;
  old JSON without it still loads.
- `tests/test_narrative_scoring.py` — `transition_cost` honors the override.
- `tests/test_sequencer.py` — a timeline built from an audio map with overrides
  produces measurably different ordering than the global-weight baseline.

### 4.3 Definition of Done

- `pytest -q` → **all green** (expect ~160+ tests after the additions).
- No existing test modified to weaken its assertion.
- `.env.example`, `README.md`, `ARCHITECTURE.md` (§7 Performance), and
  `HANDOVER.md` (§2.4) updated.
- A short note added to the *Open Follow-ups* §5 below documenting anything
  deferred.

---

## 5. Open Follow-ups (not the immediate task)

- `--embed-batch-size` CLI flag on `extract` (currently env-only).
- Export helpers: `dump_embeddings.npy` (Stage-1 feature matrix) and
  `export_edit_graph.json` (Stage-3 pairwise cost + chosen permutation) — for
  downstream generative-video model training.
- `user_version`-gated migration in each cache `_connect()`.
- ffmpeg codec availability probe in the render CLI for friendlier failures.
  **Name validation already ships** (`resolve_codec_profile` + the
  `video_codec` pydantic validator reject unknown codec names at config load).
  The deferred piece is a *runtime* probe (`ffmpeg -hide_banner -encoders`)
  that rejects codecs the installed binary doesn't actually support (e.g.
  `h264_nvenc` on a machine without an NVIDIA GPU) before the encode starts.
  NVENC bitrate/2-pass modes are also deferred (v1 is constant-quality `cq`,
  mirroring `crf` semantics).
- Non-4/4 time-signature downbeat handling (currently assumes 4/4 via
  `beats[::4]` in `audio/analyzer.py`).

---

## 6. Agent Operating Rules (non-negotiable)

1. **Run `pytest -q` before declaring done.** All offline; no excuses.
2. **Never make a real network or GPU call in a test.** Mock Replicate,
   sentence-transformers, librosa's audio decode, ffmpeg, and ffprobe.
3. **Config via `AppConfig` only.** Do not read env vars inside package modules.
4. **Preserve fail-soft + incremental-skip semantics.** A single bad image or a
   failed embed chunk must never abort the whole run.
5. **Surgical edits.** Do not rename public APIs or reflow unrelated code.
6. **Update these two docs** whenever you change the config surface, a JSON
   contract, or add an AI-integration seam.
