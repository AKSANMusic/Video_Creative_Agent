# AI Creative Engine

Automated narrative engine that turns static image collections and a music track into cinema-grade, beat-synchronized music videos.

Designed for content creators and editors, it natively supports exporting to social media formats (like Instagram Reels and TikTok) while utilizing advanced semantic AI to match the *mood* of your images to the *energy* of your music.

## Documentation

Comprehensive documentation has been split out into the `docs/` folder:

- 📖 **[User Guide](docs/USER_GUIDE.md)**: A step-by-step tutorial on how to use the Gradio interface to generate your videos.
- 🏗️ **[Architecture](docs/ARCHITECTURE.md)**: A deep dive into the 4-stage pipeline, SQLite caching models, and local ML sub-processes.
- 📱 **[Social Exports & Aspect Ratios](docs/SOCIAL_EXPORTS.md)**: Details on the dynamic resolution engine and adapting footage for mobile platforms.

---

## Quick Start

### 1. Installation

Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone https://github.com/AKSANMusic/Video_Creative_Agent.git
cd Video_Creative_Agent

# Set up the virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

*Note: You must have `ffmpeg` and `ffprobe` binaries in your system PATH or located in the project root.*

### 2. Configuration

Copy the example environment file and set up your Replicate API token (required for the Stage 1 vision extraction models).

```bash
cp .env.example .env
```
Open `.env` and add your token: `REPLICATE_API_TOKEN=r8_your_token_here`

### 3. Run the App

The primary way to use the engine is via the Gradio Web UI:

```bash
python app.py
```
Open `http://127.0.0.1:7861` in your browser. Upload your images and audio track, select your target aspect ratio (e.g., Vertical 9:16), and click **🚀 Run Full Pipeline**.

---

## 4-Stage Pipeline Overview

The engine operates on a robust, heavily-cached, 4-stage deterministic flow:

1. **Vision Extraction:** Automatically captions images (Florence-2) and assigns tension/mood scores (LLaVA-NeXT) via Replicate API. Computes sub-process text embeddings.
2. **Audio Analysis:** Uses Librosa to map BPM, beats, onsets, and structural sections. Optionally uses CLAP for semantic cross-modal tagging.
3. **Narrative Sequencer:** Assigns images to audio sections using energy/tension matching. Calculates the perfect order of images via dynamic programming to preserve visual color continuity. Locks cuts to downbeats.
4. **Render:** Uses FFmpeg to construct the video. Applies Ken Burns zooming, crossfades, and letterboxing/cropping to your selected Target Aspect Ratio.
