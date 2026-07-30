"""Phase 3 Visual Audit Runner.

This script automates the 4-step execution of the AI Creative Engine 
for a visual audit of the new Phase 2 Cinematic Director features 
(saliency, Ken Burns, J/L-cuts).

Usage:
    python scripts/run_visual_audit.py --images path/to/images --audio path/to/audio.mp3 --out visual_audit.mp4
"""

import argparse
import subprocess
import sys
from pathlib import Path
import os

def run_cmd(cmd: list[str], step_name: str):
    print(f"\n{'='*80}\n🚀 RUNNING STEP: {step_name}\n{'='*80}")
    print(f"> {' '.join(cmd)}\n")
    
    # Ensure PYTHONPATH is set to src
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED AT STEP: {step_name}")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run the Phase 3 Visual Audit")
    parser.add_argument("--images", required=True, help="Path to directory containing test images (mixed media)")
    parser.add_argument("--audio", required=True, help="Path to test audio track (ideally ~30s)")
    parser.add_argument("--out", default="visual_audit_render.mp4", help="Output MP4 path")
    parser.add_argument("--db", default="audit_cache.db", help="Temporary database for this audit")
    
    args = parser.parse_args()
    
    # 1. Vision Extraction
    run_cmd([
        "python", "-m", "ai_creative_engine.cli", "extract",
        "--images", args.images,
        "--db", args.db
    ], "Stage 1: Vision Extraction (Features, Saliency, Roles)")
    
    # 2. Audio Analysis
    run_cmd([
        "python", "-m", "ai_creative_engine.cli", "analyze-audio",
        "--audio", args.audio,
        "--db", args.db,
        "--out", "audit_audio_map.json"
    ], "Stage 2: Audio Analysis (Energy Curves, Beats)")
    
    # 3. Sequencing & Directing
    run_cmd([
        "python", "-m", "ai_creative_engine.cli", "sequence",
        "--audio-map", "audit_audio_map.json",
        "--db", args.db,
        "--out", "audit_timeline.json"
    ], "Stage 3: Narrative Sequencing & Cinematic Directing")
    
    # 4. Rendering
    run_cmd([
        "python", "-m", "ai_creative_engine.cli", "render",
        "--timeline", "audit_timeline.json",
        "--out", args.out,
        "--fps", "30",
        "--width", "1080",
        "--height", "1920"
    ], "Stage 4: Rendering (FFmpeg Filtergraph & Ken Burns)")
    
    print(f"\n✅ Visual Audit Render Complete: {args.out}")
    print("Please review the video to audit the saliency-driven focal points and transition pacing.")

if __name__ == "__main__":
    main()
