"""Command-line interface for the AI Creative Engine.

Stage 1 (vision) and Stage 2 (audio) subcommands.

Usage
-----
    ai-creative-engine extract --images ./sample_images --db ./out.db
    ai-creative-engine analyze-audio --audio ./track.mp3 --out audio_map.json
    python -m ai_creative_engine <command> ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from .audio.audio_cache import AudioCache
from .audio.audio_map import AudioMap
from .audio.analyzer import AudioAnalyzer
from .audio.clap_adapter import CLAPAdapter
from .audio.exporter import export_audio_map
from .audio.pipeline import AudioPipeline
from .narrative.cache import TimelineCache
from .narrative.exporter import export_timeline
from .narrative.exporter import load_timeline
from .narrative.pipeline import NarrativePipeline
from .narrative.sequencer import NarrativeSequencer
from .render.pipeline import RenderPipeline
from .render.plan import PlanBuilder
from .render.encoder import FFmpegEncoder
from .cache import MetadataCache
from .config import Settings, get_settings
from .errors import CreativeEngineError
from .pipeline import ExtractionPipeline
from .vision.embeddings import Embedder
from .vision.florence import FlorenceClient
from .vision.llava import LLaVAClient

app = typer.Typer(
    name="ai-creative-engine",
    help="AI Creative Engine: vision + audio analysis for beat-synced video.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


def _build_pipeline(settings: Settings, max_images: Optional[int]) -> ExtractionPipeline:
    """Wire up vision clients + cache from settings."""
    token = settings.require_api_token()  # fail fast on missing token
    florence = FlorenceClient(
        model=settings.florence_model,
        api_token=token,
        timeout_s=settings.request_timeout_s,
    )
    llava = LLaVAClient(
        model=settings.llava_model,
        api_token=token,
        timeout_s=settings.request_timeout_s,
    )
    embedder = Embedder(
        model_name=settings.embedding_model,
        target_dim=settings.embedding_dim,
        batch_size=settings.embedding_batch_size,
    )
    cache = MetadataCache(db_path=settings.db_path)
    return ExtractionPipeline(
        florence=florence,
        llava=llava,
        embedder=embedder,
        cache=cache,
        max_images=max_images,
    )


def _build_audio_pipeline(settings: Settings) -> AudioPipeline:
    """Wire up the analyzer + optional CLAP + cache from settings."""
    analyzer = AudioAnalyzer(
        sample_rate=settings.audio_sample_rate,
        hop_length=settings.audio_hop_length,
        n_segments=settings.audio_n_segments,
    )
    cache = AudioCache(db_path=settings.db_path)
    clap = CLAPAdapter(
        enabled=settings.clap_enabled,
        model_name=settings.clap_model,
        device=settings.clap_device,
    )
    return AudioPipeline(analyzer=analyzer, cache=cache, clap=clap)


# ---------------------------------------------------------------------------
# Stage 1: vision extraction
# ---------------------------------------------------------------------------


@app.command()
def extract(
    images: Path = typer.Option(
        ..., "--images", "-i", help="Directory containing source images.",
    ),
    db: Optional[Path] = typer.Option(
        None, "--db", "-d", help="SQLite database path (overrides ACE_DB_PATH).",
    ),
    max_images: Optional[int] = typer.Option(
        None, "--max-images", "-n", help="Process at most N images (for testing).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Extract vision metadata + binary embeddings for all images in a directory."""
    _configure_logging(verbose)

    settings = get_settings()
    if db is not None:
        settings = settings.model_copy(update={"db_path": db})

    try:
        pipeline = _build_pipeline(settings, max_images)
    except CreativeEngineError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=2)

    try:
        stats = pipeline.run(images)
    except CreativeEngineError as exc:
        console.print(f"[red]Pipeline error:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(
        f"\n[bold green]Done.[/bold green] "
        f"new={stats.new} cached={stats.cached} failed={stats.failed} total={stats.total}"
    )
    if stats.failed_files:
        console.print("[yellow]Failed files:[/yellow]")
        for name in stats.failed_files:
            console.print(f"  - {name}")

    if stats.new == 0 and stats.failed > 0:
        raise typer.Exit(code=1)


@app.command()
def info(
    db: Path = typer.Option(..., "--db", "-d", help="SQLite database path."),
) -> None:
    """Show basic stats about a cache database."""
    cache = MetadataCache(db_path=db)
    try:
        n = cache.count()
    except CreativeEngineError as exc:
        console.print(f"[red]Cache error:[/red] {exc}")
        raise typer.Exit(code=1)
    console.print(f"Cached images in {db}: {n}")


# ---------------------------------------------------------------------------
# Stage 2: audio analysis
# ---------------------------------------------------------------------------


@app.command(name="analyze-audio")
def analyze_audio(
    audio: Path = typer.Option(
        ..., "--audio", "-a", help="Path to the audio file (mp3/wav/flac/ogg/...).",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Write audio_map.json to this path (default: alongside the DB).",
    ),
    db: Optional[Path] = typer.Option(
        None, "--db", "-d", help="SQLite database path (overrides ACE_DB_PATH).",
    ),
    clap: bool = typer.Option(
        None, "--clap/--no-clap", help="Force-enable or force-disable CLAP tagging (overrides ACE_CLAP_ENABLED).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Analyze an audio track and emit an audio_map.json."""
    _configure_logging(verbose)

    settings = get_settings()
    updates: dict = {}
    if db is not None:
        updates["db_path"] = db
    if clap is not None:
        updates["clap_enabled"] = clap
    if updates:
        settings = settings.model_copy(update=updates)

    try:
        pipeline = _build_audio_pipeline(settings)
    except CreativeEngineError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=2)

    try:
        audio_map: AudioMap = pipeline.run(audio)
    except FileNotFoundError as exc:
        console.print(f"[red]File not found:[/red] {exc}")
        raise typer.Exit(code=2)
    except CreativeEngineError as exc:
        console.print(f"[red]Audio analysis error:[/red] {exc}")
        raise typer.Exit(code=1)

    if out is None:
        out = Path("audio_map.json")
    try:
        export_audio_map(audio_map, out)
    except OSError as exc:
        console.print(f"[red]Failed to write {out}:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(
        f"\n[bold green]Audio map written:[/bold green] {out}\n"
        f"  bpm={audio_map.bpm:.1f}  duration={audio_map.duration:.2f}s  "
        f"beats={len(audio_map.beats)}  sections={len(audio_map.sections)}  "
        f"cross_modal={'on' if audio_map.cross_modal else 'off'}"
    )

# ---------------------------------------------------------------------------
# Stage 3: narrative sequencer
# ---------------------------------------------------------------------------


def _build_narrative_pipeline(settings: Settings) -> NarrativePipeline:
    """Wire up image cache + timeline cache + sequencer from settings."""
    image_cache = MetadataCache(db_path=settings.db_path)
    timeline_cache = TimelineCache(db_path=settings.db_path)
    sequencer = NarrativeSequencer(
        dp_threshold=settings.sequencer_dp_threshold,
        continuity_weight=settings.sequencer_continuity_weight,
        cut_on=settings.sequencer_cut_on,
    )
    return NarrativePipeline(
        image_cache=image_cache,
        timeline_cache=timeline_cache,
        sequencer=sequencer,
    )


@app.command()
def sequence(
    audio_map_path: Path = typer.Option(
        ..., "--audio-map", "-a", help="Path to the Stage-2 audio_map.json.",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Write timeline.json to this path (default: ./timeline.json).",
    ),
    db: Optional[Path] = typer.Option(
        None, "--db", "-d", help="SQLite database path (overrides ACE_DB_PATH).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Build a beat-locked timeline.json from cached images + an audio map."""
    _configure_logging(verbose)

    settings = get_settings()
    if db is not None:
        settings = settings.model_copy(update={"db_path": db})

    if not audio_map_path.is_file():
        console.print(f"[red]Audio map not found:[/red] {audio_map_path}")
        raise typer.Exit(code=2)

    try:
        from .audio.exporter import load_audio_map

        audio_map = load_audio_map(audio_map_path)
    except Exception as exc:
        console.print(f"[red]Failed to load audio map:[/red] {exc}")
        raise typer.Exit(code=2)

    try:
        pipeline = _build_narrative_pipeline(settings)
    except CreativeEngineError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=2)

    try:
        timeline = pipeline.run(audio_map)
    except CreativeEngineError as exc:
        console.print(f"[red]Sequencer error:[/red] {exc}")
        raise typer.Exit(code=1)

    if out is None:
        out = Path("timeline.json")
    try:
        export_timeline(timeline, out)
    except OSError as exc:
        console.print(f"[red]Failed to write {out}:[/red] {exc}")
        raise typer.Exit(code=1)

    cuts = sum(1 for e in timeline.entries if e.transition.type == "cut")
    dissolves = sum(1 for e in timeline.entries if e.transition.type == "dissolve")
    console.print(
        f"\n[bold green]Timeline written:[/bold green] {out}\n"
        f"  entries={len(timeline.entries)}  duration={timeline.duration:.2f}s  "
        f"bpm={timeline.bpm:.1f}\n"
        f"  cuts={cuts}  dissolves={dissolves}  downbeats_used={len(timeline.downbeats_used)}"
    )


# ---------------------------------------------------------------------------
# Stage 4: render
# ---------------------------------------------------------------------------

@app.command()
def render(
    timeline_path: Path = typer.Option(
        ..., "--timeline", "-t", help="Path to the Stage-3 timeline.json.",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Output MP4 path (default: ./output.mp4).",
    ),
    fps: float = typer.Option(
        30.0, "--fps", help="Target frames per second (constant frame rate).",
    ),
    width: int = typer.Option(1280, "--width", help="Output video width (px)."),
    height: int = typer.Option(720, "--height", help="Output video height (px)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    codec: Optional[str] = typer.Option(
        None,
        "--codec",
        help=(
            "ffmpeg video encoder overriding ACE_VIDEO_CODEC. "
            "Examples: libx264, h264_nvenc, h264_videotoolbox, h264_vaapi."
        ),
    ),
) -> None:
    """Render a beat-locked timeline.json into an MP4 via FFmpeg."""
    _configure_logging(verbose)

    settings = get_settings()
    video_codec = codec if codec is not None else settings.video_codec

    if not timeline_path.is_file():
        console.print(f"[red]Timeline not found:[/red] {timeline_path}")
        raise typer.Exit(code=2)

    try:
        timeline = load_timeline(timeline_path)
    except Exception as exc:
        console.print(f"[red]Failed to load timeline:[/red] {exc}")
        raise typer.Exit(code=2)

    try:
        plan_builder = PlanBuilder(fps=fps, width=width, height=height)
        encoder = FFmpegEncoder(fps=fps, video_codec=video_codec)
        pipeline = RenderPipeline(plan_builder=plan_builder, encoder=encoder)
    except (CreativeEngineError, ValueError) as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=2)

    if out is None:
        out = Path("output.mp4")

    try:
        result = pipeline.run(timeline, out)
    except CreativeEngineError as exc:
        console.print(f"[red]Render error:[/red] {exc}")
        raise typer.Exit(code=1)

    drift_status = (
        "[green]OK[/green]" if result.drift.passed else "[red]FAILED[/red]"
    )
    console.print(
        f"\n[bold green]Render written:[/bold green] {out}\n"
        f"  frames={result.plan.total_frames}  fps={result.plan.fps:.1f}  "
        f"{result.plan.width}x{result.plan.height}\n"
        f"  windows={len(result.plan.windows)}  "
        f"duration={result.encode.duration_s:.2f}s\n"
        f"  drift={result.drift.max_drift_ms:.1f}ms "
        f"(mode={result.drift.mode}, KPI 40ms) -> {drift_status}"
    )
    if not result.drift.passed:
        raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
