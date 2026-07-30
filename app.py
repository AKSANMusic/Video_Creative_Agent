"""Gradio Graphical User Interface for the AI Creative Engine.

Directly imports and orchestrates the four pipeline stages:
1. ExtractionPipeline (Stage 1: Vision Extraction & Binary Embedding)
2. AudioPipeline (Stage 2: Rhythm & Structural Audio Analysis)
3. NarrativePipeline (Stage 3: Beat-Locked Dynamic Sequencing)
4. RenderPipeline (Stage 4: Ken Burns & FFmpeg Rendering)

UI Features:
- Custom Deep Black & Fine Gold theme.
- Thread-based heartbeat generator streaming to keep WebSocket/SSE alive indefinitely.
- Real-time live log streaming console (GradioLogHandler + generator yields).
- Asynchronous execution queue (demo.queue()).
- Typography controls disabled by default.
"""

from __future__ import annotations

import json
import logging
import os

# User explicitly requested we disable PyTorch/GPU usage 
# to prevent memory overhead or crashes on CPU systems.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PATH"] = os.getcwd() + os.pathsep + os.environ.get("PATH", "")

import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Generator, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr

# Ensure src is on Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_creative_engine.audio.analyzer import AudioAnalyzer
from ai_creative_engine.audio.audio_cache import AudioCache
from ai_creative_engine.audio.audio_map import AudioMap
from ai_creative_engine.audio.clap_adapter import CLAPAdapter
from ai_creative_engine.audio.exporter import export_audio_map, load_audio_map
from ai_creative_engine.audio.pipeline import AudioPipeline
from ai_creative_engine.cache import MetadataCache
from ai_creative_engine.config import Settings, get_settings
from ai_creative_engine.errors import CreativeEngineError
from ai_creative_engine.narrative.cache import TimelineCache
from ai_creative_engine.narrative.exporter import export_timeline, load_timeline
from ai_creative_engine.narrative.pipeline import NarrativePipeline
from ai_creative_engine.narrative.sequencer import NarrativeSequencer
from ai_creative_engine.pipeline import ExtractionPipeline
from ai_creative_engine.render.encoder import FFmpegEncoder
from ai_creative_engine.render.pipeline import RenderPipeline
from ai_creative_engine.render.plan import PlanBuilder
from ai_creative_engine.render.social_formats import SocialFormatter, PRESETS, Strategy

from ai_creative_engine.vision.embeddings import Embedder
from ai_creative_engine.vision.florence import FlorenceClient
from ai_creative_engine.vision.llava import LLaVAClient
from ai_creative_engine.vision.rate_limiter import REPLICATE_LIMITER

# Initialize GLOBAL_EMBEDDER on the main thread when the module loads.
# This prevents silent C-level PyTorch crashes when worker threads try to load the model.
print("[INFO] Pre-loading embedding model on main thread...")
_settings = get_settings()
GLOBAL_EMBEDDER = Embedder(
    model_name=_settings.embedding_model,
    target_dim=_settings.embedding_dim,
    preload=True,
)
print("[INFO] Embedding model ready.")


# --- Live Logging Handler --------------------------------------------------

class GradioLogHandler(logging.Handler):
    """In-memory logging handler that captures log lines for live streaming."""

    def __init__(self) -> None:
        super().__init__()
        self.logs: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.logs.append(msg)
        except Exception:
            pass

    def get_logs(self) -> str:
        if not self.logs:
            return "Awaiting pipeline execution..."
        return "\n".join(self.logs[-200:])

    def add_line(self, line: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {line}")

    def clear(self) -> None:
        self.logs.clear()


gradio_logger = GradioLogHandler()
log_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
gradio_logger.setFormatter(log_formatter)

root_log = logging.getLogger()
root_log.addHandler(gradio_logger)
root_log.setLevel(logging.INFO)

# Do NOT add gradio_logger to the child logger - it already propagates to root.
# Adding it to both causes every log line to appear twice.
engine_log = logging.getLogger("ai_creative_engine")
engine_log.setLevel(logging.INFO)
engine_log.propagate = True


# --- Custom Dark Black & Gold Theme ---------------------------------------

class GoldBlackTheme(gr.themes.Base):
    def __init__(self):
        super().__init__()
        self.primary_hue = gr.themes.colors.amber
        self.secondary_hue = gr.themes.colors.gray
        self.neutral_hue = gr.themes.colors.neutral
        
        self.background_fill_primary = "#0a0a0c"
        self.background_fill_secondary = "#121216"
        self.block_background_fill = "#16161c"
        self.block_border_color = "#2a2a35"
        self.border_color_primary = "#3a3a48"
        
        self.color_accent = "#d4af37"
        self.color_accent_soft = "#e5c158"
        
        self.body_text_color = "#e0e0e8"
        self.block_label_text_color = "#d4af37"
        self.button_primary_background_fill = "#d4af37"
        self.button_primary_background_fill_hover = "#e5c158"
        self.button_primary_text_color = "#0a0a0c"


# Global workspace directory
WORK_DIR = Path("workspace_runs")
WORK_DIR.mkdir(exist_ok=True)


# --- Stage Handlers (Threaded Generators for SSE Heartbeats) --------------

def run_stage_1(
    images: list[Any] | None,
    images_dir_str: str,
    db_path_str: str,
    api_token: str,
    replicate_delay: float = 11.0,
    progress: gr.Progress = gr.Progress(),
) -> Generator[str, None, None]:
    """Execute Stage 1 Vision Extraction & Binary Embedding with live streaming logs."""
    gradio_logger.clear()
    gradio_logger.add_line("▶ Starting Stage 1: Vision Extraction & Binary Embedding")
    yield gradio_logger.get_logs()

    if api_token:
        os.environ["REPLICATE_API_TOKEN"] = api_token
        gradio_logger.add_line("✓ Configured REPLICATE_API_TOKEN from UI input")
        yield gradio_logger.get_logs()
        
    settings = get_settings()
    db_path = Path(db_path_str) if db_path_str else settings.db_path
    settings = settings.model_copy(update={"db_path": db_path})

    # Sync the UI slider value to the global rate limiter so both Florence
    # and LLaVA honour the same inter-request interval.
    REPLICATE_LIMITER.set_interval(replicate_delay)
    gradio_logger.add_line(f"ℹ Replicate rate limiter set to {replicate_delay:.1f}s between calls")
    yield gradio_logger.get_logs()
    
    target_dir = WORK_DIR / "input_images"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    if images_dir_str and Path(images_dir_str).is_dir():
        src_dir = Path(images_dir_str)
        gradio_logger.add_line(f"Copying images from directory: {src_dir}")
        yield gradio_logger.get_logs()
        for f in src_dir.iterdir():
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                try:
                    shutil.copy(f, target_dir / f.name)
                except shutil.SameFileError:
                    pass
    elif images:
        gradio_logger.add_line(f"Receiving batch of {len(images)} uploaded images...")
        yield gradio_logger.get_logs()
        for item in images:
            file_path = Path(item.name if hasattr(item, "name") else item)
            try:
                shutil.copy(file_path, target_dir / file_path.name)
            except shutil.SameFileError:
                pass

    files = [p for p in target_dir.iterdir() if p.is_file()]
    if not files:
        gradio_logger.add_line("❌ Error: No valid image files provided or found.")
        yield gradio_logger.get_logs()
        return

    gradio_logger.add_line(f"Initializing models for {len(files)} target images...")
    yield gradio_logger.get_logs()

    progress(0.1, desc="Initializing Vision Models & Cache...")
    
    try:
        token = settings.require_api_token()
        florence = FlorenceClient(model=settings.florence_model, api_token=token)
        llava = LLaVAClient(model=settings.llava_model, api_token=token)
        
        # Use the globally pre-loaded embedder
        embedder = GLOBAL_EMBEDDER
        cache = MetadataCache(db_path=settings.db_path)
        pipeline = ExtractionPipeline(
            florence=florence,
            llava=llava,
            embedder=embedder,
            cache=cache,
            request_delay_s=replicate_delay,
        )
        
        result_holder: dict[str, Any] = {}
        error_holder: list[Exception] = []

        def _worker():
            try:
                def _on_progress(idx: int, total: int, filename: str):
                    p_val = float(idx) / float(max(1, total))
                    progress(p_val, desc=f"Processing [{idx}/{total}]: {filename}")
                    gradio_logger.add_line(f"Processed image [{idx}/{total}]: {filename}")

                result_holder["stats"] = pipeline.run(target_dir, progress_callback=_on_progress)
            except Exception as exc:
                error_holder.append(exc)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        # Heartbeat loop: yield every 2s to keep SSE WebSocket connection alive continuously
        while t.is_alive():
            yield gradio_logger.get_logs()
            time.sleep(2.0)

        t.join()
        yield gradio_logger.get_logs()

        if error_holder:
            gradio_logger.add_line(f"❌ Stage 1 Error: {error_holder[0]}")
            yield gradio_logger.get_logs()
            return

        stats = result_holder.get("stats")
        if stats:
            progress(1.0, desc="Stage 1 Complete!")
            gradio_logger.add_line(
                f"✅ Stage 1 Complete: Total={stats.total}, New={stats.new}, "
                f"Cached={stats.cached}, Failed={stats.failed}"
            )
            yield gradio_logger.get_logs()
    except Exception as exc:
        gradio_logger.add_line(f"❌ Stage 1 Initialization Error: {exc}")
        yield gradio_logger.get_logs()


def run_stage_2(
    audio_file: Any | None,
    db_path_str: str,
    enable_clap: bool,
    progress: gr.Progress = gr.Progress(),
) -> Generator[tuple[str, str], None, None]:
    """Execute Stage 2 Audio Analysis with live streaming logs."""
    gradio_logger.clear()
    gradio_logger.add_line("▶ Starting Stage 2: Audio Analysis")
    yield gradio_logger.get_logs(), "{}"

    if not audio_file:
        gradio_logger.add_line("❌ Error: Please upload an audio file first.")
        yield gradio_logger.get_logs(), "{}"
        return
        
    audio_path = Path(audio_file.name if hasattr(audio_file, "name") else audio_file)
    gradio_logger.add_line(f"Loading audio file: {audio_path.name}")
    yield gradio_logger.get_logs(), "{}"

    settings = get_settings()
    db_path = Path(db_path_str) if db_path_str else settings.db_path
    settings = settings.model_copy(update={"db_path": db_path, "clap_enabled": enable_clap})
    
    progress(0.2, desc="Analyzing Audio Rhythm & Structure...")
    try:
        analyzer = AudioAnalyzer(
            sample_rate=settings.audio_sample_rate,
            hop_length=settings.audio_hop_length,
            n_segments=settings.audio_n_segments,
        )
        cache = AudioCache(db_path=settings.db_path)
        clap = CLAPAdapter(enabled=settings.clap_enabled, model_name=settings.clap_model)
        pipeline = AudioPipeline(analyzer=analyzer, cache=cache, clap=clap)
        
        result_holder: dict[str, Any] = {}
        error_holder: list[Exception] = []

        def _worker():
            try:
                audio_map = pipeline.run(audio_path)
                result_holder["audio_map"] = audio_map
            except Exception as exc:
                error_holder.append(exc)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        while t.is_alive():
            yield gradio_logger.get_logs(), "{}"
            time.sleep(2.0)

        t.join()

        if error_holder:
            gradio_logger.add_line(f"❌ Stage 2 Error: {error_holder[0]}")
            yield gradio_logger.get_logs(), "{}"
            return

        audio_map = result_holder.get("audio_map")
        if audio_map:
            out_map_path = WORK_DIR / "audio_map.json"
            export_audio_map(audio_map, out_map_path)
            
            progress(1.0, desc="Stage 2 Complete!")
            gradio_logger.add_line(
                f"✅ Stage 2 Complete: BPM={audio_map.bpm:.1f}, "
                f"Duration={audio_map.duration:.2f}s, Beats={len(audio_map.beats)}, "
                f"Sections={len(audio_map.sections)}"
            )
            json_content = out_map_path.read_text(encoding="utf-8")
            yield gradio_logger.get_logs(), json_content
    except Exception as exc:
        gradio_logger.add_line(f"❌ Stage 2 Error: {exc}")
        yield gradio_logger.get_logs(), "{}"


def run_stage_3(
    db_path_str: str,
    continuity_weight: float,
    cut_grid: str,
    progress: gr.Progress = gr.Progress(),
) -> Generator[tuple[str, str], None, None]:
    """Execute Stage 3 Narrative Sequencing with live streaming logs."""
    gradio_logger.clear()
    gradio_logger.add_line("▶ Starting Stage 3: Beat-Locked Narrative Sequencing")
    yield gradio_logger.get_logs(), "{}"

    audio_map_path = WORK_DIR / "audio_map.json"
    if not audio_map_path.exists():
        gradio_logger.add_line("❌ Error: audio_map.json missing. Please run Stage 2 first.")
        yield gradio_logger.get_logs(), "{}"
        return
        
    settings = get_settings()
    db_path = Path(db_path_str) if db_path_str else settings.db_path
    settings = settings.model_copy(update={
        "db_path": db_path,
        "sequencer_continuity_weight": continuity_weight,
        "sequencer_cut_on": cut_grid,
    })
    
    progress(0.3, desc="Sequencing Beat-Locked Timeline...")
    try:
        audio_map = load_audio_map(audio_map_path)
        image_cache = MetadataCache(db_path=settings.db_path)
        timeline_cache = TimelineCache(db_path=settings.db_path)
        sequencer = NarrativeSequencer(
            dp_threshold=settings.sequencer_dp_threshold,
            continuity_weight=settings.sequencer_continuity_weight,
            cut_on=settings.sequencer_cut_on,
        )
        pipeline = NarrativePipeline(image_cache=image_cache, timeline_cache=timeline_cache, sequencer=sequencer)
        
        result_holder: dict[str, Any] = {}
        error_holder: list[Exception] = []

        def _worker():
            try:
                timeline = pipeline.run(audio_map)
                result_holder["timeline"] = timeline
            except Exception as exc:
                error_holder.append(exc)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        while t.is_alive():
            yield gradio_logger.get_logs(), "{}"
            time.sleep(2.0)

        t.join()

        if error_holder:
            gradio_logger.add_line(f"❌ Stage 3 Error: {error_holder[0]}")
            yield gradio_logger.get_logs(), "{}"
            return

        timeline = result_holder.get("timeline")
        if timeline:
            out_timeline_path = WORK_DIR / "timeline.json"
            export_timeline(timeline, out_timeline_path)
            
            cuts = sum(1 for e in timeline.entries if e.transition.type == "cut")
            dissolves = sum(1 for e in timeline.entries if e.transition.type == "dissolve")
            
            progress(1.0, desc="Stage 3 Complete!")
            gradio_logger.add_line(
                f"✅ Stage 3 Complete: Entries={len(timeline.entries)}, "
                f"Duration={timeline.duration:.2f}s, Cuts={cuts}, Dissolves={dissolves}"
            )
            json_content = out_timeline_path.read_text(encoding="utf-8")
            yield gradio_logger.get_logs(), json_content
    except Exception as exc:
        gradio_logger.add_line(f"❌ Stage 3 Error: {exc}")
        yield gradio_logger.get_logs(), "{}"


def run_stage_4(
    fps: float,
    width: int,
    height: int,
    video_codec: str,
    social_format: str = "custom",
    progress: gr.Progress = gr.Progress(),
) -> Generator[tuple[str, str | None], None, None]:
    """Execute Stage 4 Video Rendering via FFmpeg with live streaming logs."""
    gradio_logger.clear()
    gradio_logger.add_line("▶ Starting Stage 4: FFmpeg Render")
    yield gradio_logger.get_logs(), None

    timeline_path = WORK_DIR / "timeline.json"
    if not timeline_path.exists():
        gradio_logger.add_line("❌ Error: timeline.json missing. Please run Stage 3 first.")
        yield gradio_logger.get_logs(), None
        return
        
    progress(0.2, desc="Assembling Ken Burns Filtergraph & Encoding MP4...")
    try:
        timeline = load_timeline(timeline_path)
        
        # Determine base dimensions
        final_width = int(width)
        final_height = int(height)
        if "custom" not in social_format.lower():
            preset_key = social_format.split(" ")[0].lower().strip("—").strip("-").strip()
            matched_key = None
            for key in PRESETS:
                if key.startswith(preset_key) or preset_key.startswith(key):
                    matched_key = key
                    break
            if matched_key:
                preset = PRESETS[matched_key]
                final_width = preset.width
                final_height = preset.height
                gradio_logger.add_line(f"ℹ Using {preset.label} native resolution: {final_width}x{final_height}")
        else:
            gradio_logger.add_line(f"ℹ Using custom native resolution: {final_width}x{final_height}")

        plan_builder = PlanBuilder(fps=float(fps), width=final_width, height=final_height)
        encoder = FFmpegEncoder(fps=float(fps), video_codec=video_codec)
        pipeline = RenderPipeline(plan_builder=plan_builder, encoder=encoder)
        
        out_mp4_path = WORK_DIR / "output.mp4"
        result_holder: dict[str, Any] = {}
        error_holder: list[Exception] = []

        def _worker():
            try:
                res = pipeline.run(timeline, out_mp4_path)
                result_holder["result"] = res
            except Exception as exc:
                error_holder.append(exc)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        while t.is_alive():
            yield gradio_logger.get_logs(), None
            time.sleep(2.0)

        t.join()

        if error_holder:
            gradio_logger.add_line(f"❌ Stage 4 Error: {error_holder[0]}")
            yield gradio_logger.get_logs(), None
            return

        result = result_holder.get("result")
        if result:
            progress(1.0, desc="Stage 4 Complete!")
            drift_msg = "PASS" if result.drift.passed else "WARN"
            gradio_logger.add_line(
                f"🎬 Stage 4 Render Complete: {out_mp4_path} ({width}x{height} @ {fps}fps, "
                f"Drift={result.drift.max_drift_ms:.1f}ms [{drift_msg}])"
            )
            yield gradio_logger.get_logs(), str(out_mp4_path)
    except Exception as exc:
        gradio_logger.add_line(f"❌ Stage 4 Error: {exc}")
        yield gradio_logger.get_logs(), None

def run_social_export(
    format_name: str,
    strategy_name: str,
    progress: gr.Progress = gr.Progress(),
) -> Generator[tuple[str, str | None], None, None]:
    """Export the Stage 4 output.mp4 to a social-media aspect ratio."""
    gradio_logger.clear()
    gradio_logger.add_line("▶ Starting Social Media Export")
    yield gradio_logger.get_logs(), None

    source = WORK_DIR / "output.mp4"
    if not source.exists():
        gradio_logger.add_line("❌ Error: output.mp4 missing. Run Stage 4 first.")
        yield gradio_logger.get_logs(), None
        return

    # Parse strategy from human-readable label.
    strat = Strategy.CROP if "Crop" in strategy_name else Strategy.LETTERBOX
    # Parse preset name from the dropdown label (first word is the key).
    preset_key = format_name.split(" ")[0].lower().strip("—").strip("-").strip()
    # Match against known keys.
    matched_key = None
    for key in PRESETS:
        if key.startswith(preset_key) or preset_key.startswith(key):
            matched_key = key
            break
    if not matched_key:
        matched_key = list(PRESETS.keys())[0]

    preset = PRESETS[matched_key]
    gradio_logger.add_line(
        f"ℹ Format: {preset.label}  |  Strategy: {strat.value}  |  "
        f"Target: {preset.width}×{preset.height}"
    )
    yield gradio_logger.get_logs(), None

    progress(0.3, desc=f"Exporting {preset.name} ({strat.value})...")
    try:
        fmt = SocialFormatter()
        out_dir = WORK_DIR / "social_exports"

        error_holder: list[Exception] = []
        result_holder: dict = {}

        def _worker():
            try:
                res = fmt.export(source, matched_key, strat, output_dir=out_dir)
                result_holder["result"] = res
            except Exception as exc:
                error_holder.append(exc)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        while t.is_alive():
            yield gradio_logger.get_logs(), None
            time.sleep(1.0)
        t.join()

        if error_holder:
            gradio_logger.add_line(f"❌ Export Error: {error_holder[0]}")
            yield gradio_logger.get_logs(), None
            return

        result = result_holder.get("result")
        if result and result.succeeded:
            progress(1.0, desc="Export Complete!")
            gradio_logger.add_line(
                f"✅ Export Complete: {result.output_path.name}  "
                f"({preset.width}×{preset.height}, {result.duration_s:.1f}s)"
            )
            yield gradio_logger.get_logs(), str(result.output_path)
        else:
            err = result.stderr_tail[:300] if result else "Unknown error"
            gradio_logger.add_line(f"❌ Export Failed: {err}")
            yield gradio_logger.get_logs(), None

    except Exception as exc:
        gradio_logger.add_line(f"❌ Export Error: {exc}")
        yield gradio_logger.get_logs(), None


def run_full_pipeline(
    images: list[Any] | None,
    images_dir_str: str,
    audio_file: Any | None,
    db_path_str: str,
    api_token: str,
    enable_clap: bool,
    continuity_weight: float,
    cut_grid: str,
    fps: float,
    width: int,
    height: int,
    video_codec: str,
    social_format: str = "custom",
    replicate_delay: float = 1.5,
    progress: gr.Progress = gr.Progress(),
) -> Generator[tuple[str, str, str, str | None], None, None]:
    """Execute all four stages sequentially with live continuous log streaming."""
    gradio_logger.clear()
    gradio_logger.add_line("🚀 STARTING FULL PIPELINE (STAGES 1 - 4)")
    yield gradio_logger.get_logs(), "{}", "{}", None

    # Stage 1
    progress(0.1, desc="Stage 1: Vision Extraction...")
    for logs in run_stage_1(images, images_dir_str, db_path_str, api_token, replicate_delay=replicate_delay, progress=progress):
        yield logs, "{}", "{}", None
        
    # Stage 2
    progress(0.35, desc="Stage 2: Audio Analysis...")
    am_json = "{}"
    for res in run_stage_2(audio_file, db_path_str, enable_clap, progress=progress):
        logs, am_json = res
        yield logs, am_json, "{}", None
        
    # Stage 3
    progress(0.65, desc="Stage 3: Narrative Sequencing...")
    tl_json = "{}"
    for res in run_stage_3(db_path_str, continuity_weight, cut_grid, progress=progress):
        logs, tl_json = res
        yield logs, am_json, tl_json, None
        
    # Stage 4
    progress(0.85, desc="Stage 4: FFmpeg Video Render...")
    video_path = None
    for res in run_stage_4(fps, width, height, video_codec, social_format, progress=progress):
        logs, video_path = res
        yield logs, am_json, tl_json, video_path


# --- Gradio Interface Construction ----------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="AI Creative Engine Workspace") as demo:
        gr.Markdown(
            "<h1 class='gold-title'>🎬 AI CREATIVE ENGINE WORKSPACE</h1>"
            "<p style='text-align: center; color: #a0a0b0;'>"
            "Beat-Synced Music Video Generator • Direct Python Pipeline Architecture</p>"
        )

        with gr.Row():
            # Left Panel: Input Setup & Settings
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Pipeline Inputs")
                input_images = gr.File(
                    label="Source Images (Upload Batch)",
                    file_count="multiple",
                    file_types=["image"],
                )
                input_dir = gr.Textbox(
                    label="Source Images Directory Path (Optional)",
                    placeholder="e.g. C:/path/to/images",
                )
                input_audio = gr.File(
                    label="Audio Track (MP3 / WAV / FLAC)",
                    file_count="single",
                    file_types=["audio"],
                )
                
                with gr.Accordion("⚙️ Advanced Pipeline Parameters", open=False):
                    replicate_token = gr.Textbox(
                        label="Replicate API Token (REPLICATE_API_TOKEN)",
                        type="password",
                        placeholder="r8_...",
                    )
                    db_path_input = gr.Textbox(
                        label="SQLite Database Path",
                        value="creative_engine.db",
                    )
                    enable_clap = gr.Checkbox(
                        label="Enable CLAP Cross-Modal Tagging",
                        value=False,
                    )
                    continuity_weight = gr.Slider(
                        label="Color Continuity Weight vs Semantic Similarity",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                    )
                    cut_grid = gr.Radio(
                        label="Beat Cut Alignment Grid",
                        choices=["downbeats", "beats"],
                        value="downbeats",
                    )
                    
                    replicate_delay = gr.Slider(
                        label="Replicate API Inter-Request Delay (seconds)",
                        info="Replicate free tier: 6 req/min, burst=1 → minimum 11s recommended",
                        minimum=0.0,
                        maximum=30.0,
                        value=11.0,
                        step=0.5,
                    )
                    
                    # Typography control: Text overlays disabled by default
                    enable_text_overlays = gr.Checkbox(
                        label="Automated Typography Overlays (Disabled by Default)",
                        value=False,
                        interactive=False,
                    )
                    
                    fps_input = gr.Number(label="Video Framerate (FPS)", value=30.0)
                    width_input = gr.Number(label="Render Width (px)", value=1280)
                    height_input = gr.Number(label="Render Height (px)", value=720)
                    codec_input = gr.Dropdown(
                        label="FFmpeg Video Codec",
                        choices=["libx264", "libx265", "h264_nvenc", "hevc_nvenc"],
                        value="libx264",
                    )

            # Right Panel: Control Triggers & Outputs
            with gr.Column(scale=2):
                gr.Markdown("### ⚡ Execution & Pipeline Controls")
                
                with gr.Row():
                    btn_stage1 = gr.Button("1️⃣ Stage 1: Vision")
                    btn_stage2 = gr.Button("2️⃣ Stage 2: Audio")
                    btn_stage3 = gr.Button("3️⃣ Stage 3: Timeline")
                    btn_stage4 = gr.Button("4️⃣ Stage 4: Render")
                    
                btn_full = gr.Button("🚀 RUN FULL PIPELINE (STAGES 1 - 4)", elem_classes=["gold-btn"])

                # Prominent Live Log Console
                status_output = gr.Code(
                    label="📜 Real-Time System & Pipeline Log Console",
                    language="markdown",
                    lines=12,
                    interactive=False,
                )
                
                with gr.Tabs():
                    with gr.Tab("🎬 Output Video Player"):
                        video_output = gr.Video(label="Final Beat-Synced MP4 Render", interactive=False)
                    with gr.Tab("🎵 Stage 2: audio_map.json"):
                        json_audio = gr.Code(label="Audio Analysis Output", language="json")
                    with gr.Tab("⏱️ Stage 3: timeline.json"):
                        json_timeline = gr.Code(label="Beat Timeline Output", language="json")

                # --- Social Media Export Panel ---
                gr.Markdown("### 📱 Social Media Export")
                with gr.Row():
                    social_format = gr.Dropdown(
                        label="Target Format",
                        choices=[
                            "vertical — 9:16 (Reels / TikTok / Shorts)",
                            "portrait — 4:5 (Instagram Feed)",
                            "square — 1:1 (General Feed)",
                            "landscape — 16:9 (YouTube / Desktop)",
                            "custom — Use Advanced Settings width/height",
                        ],
                        value="vertical — 9:16 (Reels / TikTok / Shorts)",
                    )
                    social_strategy = gr.Radio(
                        label="Adaptation Strategy",
                        choices=["Center Crop (Fill)", "Letterbox (Fit + Pad)"],
                        value="Center Crop (Fill)",
                    )
                btn_social = gr.Button("📱 Export for Social Media", elem_classes=["gold-btn"])
                social_video_output = gr.Video(label="Social Export Preview", interactive=False)

        # --- Event Wiring (Live Generator Log Streaming Enabled) -------------
        btn_stage1.click(
            fn=run_stage_1,
            inputs=[input_images, input_dir, db_path_input, replicate_token, replicate_delay],
            outputs=[status_output],
        )
        btn_stage2.click(
            fn=run_stage_2,
            inputs=[input_audio, db_path_input, enable_clap],
            outputs=[status_output, json_audio],
        )
        btn_stage3.click(
            fn=run_stage_3,
            inputs=[db_path_input, continuity_weight, cut_grid],
            outputs=[status_output, json_timeline],
        )
        btn_stage4.click(
            fn=run_stage_4,
            inputs=[fps_input, width_input, height_input, codec_input, social_format],
            outputs=[status_output, video_output],
        )
        btn_full.click(
            fn=run_full_pipeline,
            inputs=[
                input_images, input_dir, input_audio, db_path_input, replicate_token,
                enable_clap, continuity_weight, cut_grid, fps_input, width_input,
                height_input, codec_input, social_format, replicate_delay,
            ],
            outputs=[status_output, json_audio, json_timeline, video_output],
        )
        btn_social.click(
            fn=run_social_export,
            inputs=[social_format, social_strategy],
            outputs=[status_output, social_video_output],
        )

    return demo


if __name__ == "__main__":
    theme = GoldBlackTheme()
    custom_css = """
    .gradio-container { background-color: #0a0a0c !important; }
    .gold-title { color: #d4af37 !important; font-family: 'Cinzel', 'Georgia', serif; font-weight: 700; text-align: center; }
    .gold-btn { background: linear-gradient(135deg, #d4af37 0%, #aa820a 100%) !important; color: #0a0a0c !important; font-weight: bold !important; border: none !important; }
    .gold-btn:hover { background: linear-gradient(135deg, #e5c158 0%, #c49817 100%) !important; }
    """
    app = build_app()
    app.queue(default_concurrency_limit=10)
    try:
        app.launch(server_name="127.0.0.1", server_port=7861, theme=theme, css=custom_css, share=False)
    except OSError:
        app.launch(server_name="127.0.0.1", server_port=7862, theme=theme, css=custom_css, share=False)
