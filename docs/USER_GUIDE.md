# AI Creative Engine — User Guide

Welcome to the **AI Creative Engine**, an automated tool designed to turn static image collections and a music track into cinema-grade, beat-synchronized music videos.

The primary way to use the engine is via the robust Gradio Web Interface.

## Starting the Application

Ensure your environment is set up and activated. Then, simply start the Gradio app:

```bash
python app.py
```

Navigate to `http://127.0.0.1:7861` (or whichever port the console prints) in your web browser.

---

## 1. Input Setup & Settings

On the left panel, you'll see the **Pipeline Inputs**.

1. **Source Images:** Upload a batch of images via the file picker, OR type in the absolute path to an image directory in the **Source Images Directory Path** text box (e.g. `C:/path/to/my_images`).
2. **Audio Track:** Upload the music track you want the video synchronized to (MP3, WAV, FLAC).

### Advanced Pipeline Parameters
Under the **⚙️ Advanced Pipeline Parameters** accordion, you can fine-tune the engine's behavior:
- **Replicate API Token:** Required to run the cloud-based vision models (Florence-2 for captioning, LLaVA for mood tagging). Set your `REPLICATE_API_TOKEN` here.
- **SQLite Database Path:** The local file where metadata is cached (`creative_engine.db` by default).
- **Enable CLAP:** Enables cross-modal audio tagging. It maps the semantic mood of sections in the audio to matching images.
- **Color Continuity Weight:** Slider [0.0 - 1.0]. A higher value (e.g., 0.8) prioritizes smooth visual transitions by matching similar colors between consecutive images. A lower value prioritizes semantic and mood similarity regardless of visual color shock.
- **Cut Grid:** Choose to lock cuts strictly to **Downbeats** (every 4th beat, slower pacing) or **Beats** (every beat, extremely fast pacing).
- **Target Format:** Choose the final aspect ratio of your video. The base canvas will adapt natively (e.g., 1080x1920 for Vertical). If you select "custom", the engine will use the manual Width and Height options.

---

## 2. Running the Pipeline

You have two choices for executing the video creation:

### Option A: The "One-Click" Full Pipeline
Click the large **🚀 Run Full Pipeline** button to automatically execute all four stages in sequence. The live log viewer on the right will update continuously, showing progress and detailed diagnostics. 

### Option B: Step-by-Step Execution
If you prefer fine-grained control or want to inspect intermediate data, run each stage individually using the smaller buttons:

1. **Stage 1 (Vision Extraction):** Sends your images to Replicate for captioning and mood analysis, and computes local semantic embeddings. Caches results in SQLite.
2. **Stage 2 (Audio Analysis):** Analyzes the audio track using Librosa (and optionally CLAP). Identifies beats, energy curves, and structure.
3. **Stage 3 (Narrative Sequencing):** Uses a complex algorithm to match the tension of your images to the energy of the audio, assigning them to structural sections. Computes the optimal ordering for transitions and locks them to the beat grid.
4. **Stage 4 (FFmpeg Render):** Assembles the final video using FFmpeg. Generates Ken Burns zooming effects, crossfades, and applies dynamic aspect ratio targeting.

*Note on Caching:* Every stage is deterministic and strongly cached. If you re-run a stage with identical inputs, it will finish instantly.

---

## 3. Social Media Export

Once your `output.mp4` is rendered, you might want it in multiple formats without re-running the heavy ML pipeline. 

Scroll down to the **📱 Social Media Export** section. Here you can take your finished video and format it for different platforms.

1. **Target Format:** Pick your desired ratio (Vertical, Square, Landscape, Portrait).
2. **Adaptation Strategy:**
   - **Center Crop (Fill):** Zooms into the video to fill the new frame completely, cropping the edges. No black bars.
   - **Letterbox (Fit + Pad):** Shrinks the video to fit inside the new frame entirely, padding the empty space with a cinematic black background.

Click **📱 Export for Social Media** to instantly format the video.

See [SOCIAL_EXPORTS.md](./SOCIAL_EXPORTS.md) for more technical details on resolution handling.
