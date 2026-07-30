# Social Media Exports & Dynamic Resolution

The AI Creative Engine handles aspect ratios via a **Dynamic Canvas Resolution** model. Instead of hardcoding a 16:9 canvas and cropping afterwards (which destroys quality and breaks framing), the pipeline constructs the primary video timeline natively at the dimensions you select.

## 1. Dynamic Canvas Native Resolution (Stage 4)

In the Gradio app, the dropdown labeled **Target Format** determines the base canvas resolution used by the `PlanBuilder` and `FFmpegEncoder` during Stage 4.

| Format Preset | Aspect Ratio | Dimensions | Use Cases |
|---|---|---|---|
| **Vertical** | 9:16 | 1080x1920 | Instagram Reels, TikTok, YouTube Shorts, SnapChat |
| **Portrait** | 4:5 | 1080x1350 | Instagram Feed (Standard Portrait) |
| **Square** | 1:1 | 1080x1080 | Facebook Feed, General Posts |
| **Landscape** | 16:9 | 1920x1080 | YouTube, Desktop Web |

### The "Custom" Override
If you select `custom — Use Advanced Settings width/height` from the Target Format dropdown, the pipeline will ignore the presets and read the manual `Width` and `Height` parameters located in the Advanced Settings accordion. This is useful for ultra-wide monitors, website headers, or non-standard display panels.

## 2. Adaptation Strategies

When images of varying aspect ratios are placed into a unified timeline, they must be adapted to fit the canvas. The engine handles this during the Ken Burns filtergraph construction (`filtergraph.py`) using two primary strategies, which you can choose when exporting:

### Strategy A: Center Crop (Fill)
- **What it does:** Scales the image up until the *smallest* dimension matches the target frame, then crops the overflow from the edges.
- **Pros:** Completely fills the screen. No black bars. Highly immersive.
- **Cons:** If you put a wide image into a vertical video, the extreme left and right sides will be heavily cropped out.

### Strategy B: Letterbox (Fit + Pad)
- **What it does:** Scales the image down until the *largest* dimension matches the target frame, preserving the entire image. The empty space is padded with a deep, cinematic black (`#0a0a0c`).
- **Pros:** 1080% safe. No visual information is lost or cropped. Perfect framing.
- **Cons:** Introduces black bars on the top/bottom (letterboxing) or sides (pillarboxing) for mis-matched images.

*(By default, the primary render pipeline currently utilizes Letterbox (Fit+Pad) internally to guarantee no visual data is lost, ensuring the final video captures the entire source image, regardless of format).*

## 3. Post-Render Export Formatting

If you have already rendered your video in 16:9 (Landscape) but suddenly need a 9:16 (Vertical) version for TikTok, you do **not** need to re-run the heavy ML pipeline. 

The **📱 Export for Social Media** section in the UI uses the `SocialFormatter` to take the existing `output.mp4` and rapidly adapt it to any of the other presets using FFmpeg scaling math. Note that this *will* use cropping or padding on the baked video rather than natively rendering the source frames, so for the absolute highest quality, it is still recommended to run the full pipeline natively in your desired target format.
