"""Offline tests for Stage-4 render subsystem.

No real ffmpeg/ffprobe is invoked: the encoder runner and the auditor's
duration probe are monkeypatched. Timeline + image fixtures are synthesized on
disk so file-existence validation passes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_creative_engine.cli import app
from ai_creative_engine.errors import CreativeEngineError
from ai_creative_engine.narrative.timeline import Timeline, TimelineEntry, Transition
from ai_creative_engine.render.drift import DriftAuditor
from ai_creative_engine.render.encoder import EncodeResult, FFmpegEncoder
from ai_creative_engine.render.filtergraph import FilterGraphBuilder
from ai_creative_engine.render.plan import PlanBuilder
from ai_creative_engine.render.pipeline import RenderPipeline

IMG_ID = "0" * 40
AUDIO_ID = "1" * 40


def _entry(index, start, end, path, transition="cut"):
    return TimelineEntry(
        index=index,
        image_id=IMG_ID,
        file_path=path,
        section_index=0,
        start=start,
        end=end,
        transition=Transition(type=transition, duration_s=0.0 if transition == "cut" else 0.25),
    )


def _timeline(entries, audio_file, duration):
    return Timeline(
        audio_id=AUDIO_ID,
        audio_file=audio_file,
        duration=duration,
        bpm=120.0,
        downbeats_used=[e.start for e in entries],
        entries=entries,
    )


def _make_timeline_on_disk(tmp_path, *, n_entries=3, duration=3.0):
    seg = duration / n_entries
    entries = []
    for i in range(n_entries):
        img = tmp_path / f"img_{i}.png"
        img.write_bytes(b"\x89PNG" + bytes([i]))
        entries.append(_entry(i, i * seg, (i + 1) * seg, str(img)))
    audio = tmp_path / "track.wav"
    audio.write_bytes(b"RIFF")
    return _timeline(entries, str(audio), duration)


class TestPlanBuilder:
    def test_build_covers_full_track_contiguously(self, tmp_path):
        tl = _make_timeline_on_disk(tmp_path, n_entries=3, duration=3.0)
        plan = PlanBuilder(fps=30.0, width=1280, height=720).build(tl)
        assert plan.fps == 30.0
        assert plan.total_frames == 90
        assert plan.windows[0].start_frame == 0
        assert plan.windows[-1].end_frame == plan.total_frames
        for prev, cur in zip(plan.windows, plan.windows[1:]):
            assert cur.start_frame == prev.end_frame
        plan.validate()

    def test_rejects_non_positive_fps(self):
        with pytest.raises(ValueError):
            PlanBuilder(fps=0)
        with pytest.raises(ValueError):
            PlanBuilder(fps=-1)

    def test_rejects_bad_geometry(self):
        with pytest.raises(ValueError):
            PlanBuilder(width=0)
        with pytest.raises(ValueError):
            PlanBuilder(height=-10)

    def test_rejects_inverted_zoom(self):
        with pytest.raises(ValueError):
            PlanBuilder(zoom_start=1.2, zoom_end=1.0)

    def test_validate_rejects_empty_windows(self, tmp_path):
        tl = _make_timeline_on_disk(tmp_path, n_entries=1, duration=1.0)
        plan = PlanBuilder().build(tl)
        plan.windows.clear()
        with pytest.raises(CreativeEngineError):
            plan.validate()


class TestFilterGraph:
    def test_graph_inputs_and_xfade(self, tmp_path):
        tl = _make_timeline_on_disk(tmp_path, n_entries=3, duration=3.0)
        plan = PlanBuilder(fps=30.0, width=640, height=360).build(tl)
        graph = FilterGraphBuilder().build(plan, audio_path=tl.audio_file)
        assert len(graph.input_order) == 3
        assert graph.audio_index == 3
        assert graph.video_output_label == "[vout]"
        assert "xfade" in graph.filter_complex
        for i in range(3):
            assert f"[{i}:v]" in graph.filter_complex

    def test_single_entry_skips_xfade(self, tmp_path):
        tl = _make_timeline_on_disk(tmp_path, n_entries=1, duration=1.0)
        plan = PlanBuilder(fps=30.0, width=640, height=360).build(tl)
        graph = FilterGraphBuilder().build(plan, audio_path=tl.audio_file)
        assert len(graph.input_order) == 1
        assert graph.audio_index == 1
        assert "xfade" not in graph.filter_complex


class TestEncoderArgv:
    def test_argv_assembles_expected_flags(self, tmp_path):
        tl = _make_timeline_on_disk(tmp_path, n_entries=2, duration=2.0)
        plan = PlanBuilder(fps=24.0, width=1280, height=720).build(tl)
        graph = FilterGraphBuilder().build(plan, audio_path=tl.audio_file)

        enc = FFmpegEncoder(fps=24.0, crf=20, preset="fast")
        argv = enc._build_argv(graph, Path(tl.audio_file), tmp_path / "out.mp4")

        assert argv[0] == "ffmpeg" or argv[0].endswith("ffmpeg.exe")
        assert "-map" in argv
        assert "-r" in argv and "24.0" in argv
        assert "-c:v" in argv and "libx264" in argv
        assert "-crf" in argv and "20" in argv
        assert "-preset" in argv and "fast" in argv
        assert "-shortest" in argv
        assert argv[-1] == str(tmp_path / "out.mp4")
        assert argv.count("-i") == 3


    def test_encoder_argv_uses_configured_codec(self, tmp_path):
        """A non-default codec reaches the assembled ffmpeg argv via -c:v."""
        from ai_creative_engine.render.codec_profile import resolve_codec_profile

        tl = _make_timeline_on_disk(tmp_path, n_entries=2, duration=2.0)
        plan = PlanBuilder(fps=30.0, width=640, height=360).build(tl)
        graph = FilterGraphBuilder().build(plan, audio_path=tl.audio_file)

        enc = FFmpegEncoder(video_codec="h264_nvenc", crf=20)
        # Construction resolves the profile; the encoder stores it.
        assert enc.codec_profile == resolve_codec_profile("h264_nvenc")
        argv = enc._build_argv(graph, Path(tl.audio_file), tmp_path / "out.mp4")
        assert "-c:v" in argv and "h264_nvenc" in argv

    def test_encoder_nvenc_uses_cq_not_crf(self, tmp_path):
        """NVENC must express quality with -cq and must NOT emit -crf."""
        tl = _make_timeline_on_disk(tmp_path, n_entries=2, duration=2.0)
        plan = PlanBuilder(fps=30.0, width=640, height=360).build(tl)
        graph = FilterGraphBuilder().build(plan, audio_path=tl.audio_file)

        enc = FFmpegEncoder(video_codec="h264_nvenc", crf=22)
        argv = enc._build_argv(graph, Path(tl.audio_file), tmp_path / "out.mp4")

        assert "-cq" in argv and "22" in argv
        assert "-crf" not in argv
        # videotoolbox omits -preset entirely and uses -q:v.
        enc_vt = FFmpegEncoder(video_codec="h264_videotoolbox", crf=50)
        argv_vt = enc_vt._build_argv(graph, Path(tl.audio_file), tmp_path / "out2.mp4")
        assert "-q:v" in argv_vt and "50" in argv_vt
        assert "-preset" not in argv_vt

    def test_encoder_rejects_unknown_codec(self):
        """An unknown codec name fails fast at encoder construction."""
        from ai_creative_engine.errors import CreativeEngineError

        with pytest.raises(CreativeEngineError):
            FFmpegEncoder(video_codec="not-a-real-codec")


class TestDriftAuditor:
    def test_duration_mode_pass_when_within_kpi(self, tmp_path, monkeypatch):
        tl = _make_timeline_on_disk(tmp_path, n_entries=2, duration=2.0)
        plan = PlanBuilder(fps=30.0, width=1280, height=720).build(tl)

        auditor = DriftAuditor(kpi_ms=40.0)
        monkeypatch.setattr(auditor, "_probe_duration", lambda _p: plan.total_frames / plan.fps)
        report = auditor.audit(plan, tmp_path / "out.mp4")

        assert report.mode == "duration"
        assert report.passed is True
        assert report.max_drift_ms <= 40.0
        assert report.per_cut
        for _idx, _exp, _obs, drift in report.per_cut:
            assert drift <= (1000.0 / plan.fps) + 1e-6

    def test_unavailable_mode_when_probe_returns_zero(self, tmp_path, monkeypatch):
        tl = _make_timeline_on_disk(tmp_path, n_entries=2, duration=2.0)
        plan = PlanBuilder(fps=30.0, width=1280, height=720).build(tl)

        auditor = DriftAuditor()
        monkeypatch.setattr(auditor, "_probe_duration", lambda _p: 0.0)
        report = auditor.audit(plan, tmp_path / "missing.mp4")

        assert report.mode == "unavailable"
        assert report.passed is False


class _FakeEncoder:
    def __init__(self, fps=30.0):
        self.fps = fps

    def encode(self, graph, audio_path, output_path, extra_inputs=None):
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"FAKE_MP4")
        return EncodeResult(
            output_path=out, argv=["ffmpeg", "..."], returncode=0,
            duration_s=0.0, succeeded=True,
        )


class TestRenderPipeline:
    def test_run_happy_path(self, tmp_path, monkeypatch):
        tl = _make_timeline_on_disk(tmp_path, n_entries=3, duration=3.0)
        plan_builder = PlanBuilder(fps=30.0, width=640, height=360)
        auditor = DriftAuditor()
        monkeypatch.setattr(auditor, "_probe_duration", lambda _p: 3.0)

        pipeline = RenderPipeline(plan_builder=plan_builder, encoder=_FakeEncoder(), auditor=auditor)
        out = tmp_path / "out.mp4"
        result = pipeline.run(tl, out)

        assert result.encode.succeeded is True
        assert result.drift.mode == "duration"
        assert result.drift.passed is True
        assert out.read_bytes() == b"FAKE_MP4"
        assert len(result.plan.windows) == 3

    def test_run_raises_when_audio_missing(self, tmp_path):
        tl = _make_timeline_on_disk(tmp_path, n_entries=2, duration=2.0)
        tl.audio_file = str(tmp_path / "nope.wav")
        pipeline = RenderPipeline(plan_builder=PlanBuilder(), encoder=_FakeEncoder())
        with pytest.raises(CreativeEngineError):
            pipeline.run(tl, tmp_path / "out.mp4")

    def test_run_raises_when_image_missing(self, tmp_path):
        tl = _make_timeline_on_disk(tmp_path, n_entries=2, duration=2.0)
        Path(tl.entries[0].file_path).unlink()
        pipeline = RenderPipeline(
            plan_builder=PlanBuilder(), encoder=_FakeEncoder(), auditor=DriftAuditor(),
        )
        with pytest.raises(CreativeEngineError):
            pipeline.run(tl, tmp_path / "out.mp4")


def _patch_render_run(monkeypatch, *, drift_ms, passed):
    from ai_creative_engine.render import pipeline as render_pipeline_mod
    from ai_creative_engine.render.drift import DriftReport

    def _fake_run(self, timeline, output_path):
        plan = self.plan_builder.build(timeline)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"FAKE_MP4")
        encode = EncodeResult(
            output_path=out, argv=["ffmpeg"], returncode=0,
            duration_s=timeline.duration, succeeded=True,
        )
        drift = DriftReport(
            mode="duration", max_drift_ms=drift_ms, passed=passed,
            expected_duration_s=timeline.duration, observed_duration_s=timeline.duration,
        )
        return render_pipeline_mod.RenderResult(plan=plan, graph=None, encode=encode, drift=drift)

    monkeypatch.setattr(render_pipeline_mod.RenderPipeline, "run", _fake_run)


class TestRenderCLI:

    def test_render_cli_codec_flag_reaches_encoder(self, tmp_path, monkeypatch):
        """--codec h264_nvenc must be threaded into the FFmpegEncoder construction."""
        from ai_creative_engine.narrative.exporter import export_timeline
        from ai_creative_engine.render import encoder as encoder_mod

        tl = _make_timeline_on_disk(tmp_path, n_entries=2, duration=2.0)
        timeline_path = tmp_path / "timeline.json"
        export_timeline(tl, timeline_path)

        _patch_render_run(monkeypatch, drift_ms=5.0, passed=True)

        captured: dict = {}
        real_init = encoder_mod.FFmpegEncoder.__init__

        def _spy(self, *args, **kwargs):
            captured["video_codec"] = kwargs.get("video_codec", "libx264")
            return real_init(self, *args, **kwargs)

        monkeypatch.setattr(encoder_mod.FFmpegEncoder, "__init__", _spy)

        runner = CliRunner()
        out_mp4 = tmp_path / "out.mp4"
        result = runner.invoke(
            app,
            ["render", "--timeline", str(timeline_path),
             "--out", str(out_mp4), "--codec", "h264_nvenc"],
        )
        assert result.exit_code == 0, result.output
        assert captured["video_codec"] == "h264_nvenc"

    def test_render_requires_existing_timeline(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(app, ["render", "--timeline", str(tmp_path / "missing.json")])
        assert result.exit_code == 2
        assert "Timeline not found" in result.output

    def test_render_happy_path(self, tmp_path, monkeypatch):
        from ai_creative_engine.narrative.exporter import export_timeline

        tl = _make_timeline_on_disk(tmp_path, n_entries=2, duration=2.0)
        timeline_path = tmp_path / "timeline.json"
        export_timeline(tl, timeline_path)

        _patch_render_run(monkeypatch, drift_ms=5.0, passed=True)

        runner = CliRunner()
        out_mp4 = tmp_path / "out.mp4"
        result = runner.invoke(
            app, ["render", "--timeline", str(timeline_path), "--out", str(out_mp4)],
        )
        assert result.exit_code == 0, result.output
        assert "Render written" in result.output
        assert out_mp4.exists()

    def test_render_nonzero_when_drift_fails(self, tmp_path, monkeypatch):
        from ai_creative_engine.narrative.exporter import export_timeline

        tl = _make_timeline_on_disk(tmp_path, n_entries=2, duration=2.0)
        timeline_path = tmp_path / "timeline.json"
        export_timeline(tl, timeline_path)

        _patch_render_run(monkeypatch, drift_ms=80.0, passed=False)

        runner = CliRunner()
        out_mp4 = tmp_path / "out.mp4"
        result = runner.invoke(
            app, ["render", "--timeline", str(timeline_path), "--out", str(out_mp4)],
        )
        assert result.exit_code == 1
        assert "FAILED" in result.output
