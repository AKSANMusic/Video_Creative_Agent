"""Local, efficient visual composition and color feature extractor (Phase 2).

Extracts composition, saliency, color metrics, and classifies image roles
without external API calls or GPU requirements.
"""

from __future__ import annotations

import colorsys
import math
from pathlib import Path
from typing import Any
from PIL import Image, ImageFilter, ImageStat

# A-roll keywords for classifier
A_ROLL_KEYWORDS = {
    "person", "man", "woman", "girl", "boy", "people", "face", "human",
    "cat", "dog", "bird", "animal", "car", "vehicle", "plane", "keychain",
    "action", "playing", "holding", "walking", "running", "jumping"
}


class VisualFeatureExtractor:
    """Extracts metadata from images locally using PIL, with cv2 fallbacks if available."""

    def extract(self, image_path: Path | str, metadata_so_far: dict[str, Any] | None = None) -> dict[str, Any]:
        """Extract all Phase 2 features for a given image path."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        img = Image.open(path)
        
        # 1. Color Metrics (PIL-native)
        dominant_colors = self._extract_dominant_colors(img)
        mean_luminance, mean_saturation, color_temp = self._extract_color_stats(img)
        
        # 2. Saliency & Composition (PIL-native with fallback)
        sal_x, sal_y = self._estimate_saliency_center(img)
        edge_complexity = self._estimate_edge_complexity(img)
        quadrant = self._get_quadrant(sal_x, sal_y)
        
        # 3. Image Role Classification (A-roll vs B-roll)
        role = self._classify_role(metadata_so_far or {}, sal_x, sal_y, dominant_colors)

        return {
            "image_role": role,
            "dominant_colors": dominant_colors,
            "mean_luminance": round(mean_luminance, 4),
            "mean_saturation": round(mean_saturation, 4),
            "color_temperature": round(color_temp, 4),
            "saliency_center_x": round(sal_x, 4),
            "saliency_center_y": round(sal_y, 4),
            "edge_complexity": round(edge_complexity, 4),
            "subject_quadrant": quadrant,
        }

    def _extract_dominant_colors(self, img: Image.Image) -> list[str]:
        """Extract top 3 dominant colors as hex strings using PIL fast quantization."""
        try:
            # Downsample for speed
            small_img = img.resize((100, 100))
            # Quantize down to 3 colors
            quantized = small_img.quantize(colors=3, method=Image.Quantize.FASTOCTREE)
            palette = quantized.getpalette()
            
            # Extract unique RGB triplets
            hex_colors = []
            if palette:
                for i in range(3):
                    r = palette[i * 3]
                    g = palette[i * 3 + 1]
                    b = palette[i * 3 + 2]
                    hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")
            return hex_colors
        except Exception:
            return ["#7f7f7f", "#7f7f7f", "#7f7f7f"]

    def _extract_color_stats(self, img: Image.Image) -> tuple[float, float, float]:
        """Extract mean luminance, mean saturation, and estimated color temperature."""
        try:
            # Convert to RGB if not already
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            stat = ImageStat.Stat(img)
            # Normalised mean RGB
            mean_r = stat.mean[0] / 255.0
            mean_g = stat.mean[1] / 255.0
            mean_b = stat.mean[2] / 255.0
            
            # 1. Luminance (standard BT.709 weights)
            luminance = 0.2126 * mean_r + 0.7152 * mean_g + 0.0722 * mean_b
            
            # 2. Saturation
            h, s, v = colorsys.rgb_to_hsv(mean_r, mean_g, mean_b)
            
            # 3. Color Temperature (simplified warm/cool balance)
            # R is warm, B is cool. Ratio of (R-B)/(R+B) normalized to [0,1]
            denom = (mean_r + mean_b)
            color_temp = 0.5
            if denom > 0.01:
                color_temp = 0.5 + 0.5 * ((mean_r - mean_b) / denom)
                
            return luminance, s, color_temp
        except Exception:
            return 0.5, 0.5, 0.5

    def _estimate_saliency_center(self, img: Image.Image) -> tuple[float, float]:
        """Estimate visual focus center. Falls back to PIL-based high-pass filter if cv2 is absent."""
        try:
            # Try importing OpenCV for a more advanced spectral saliency
            import cv2
            import numpy as np
            
            # Convert PIL Image to OpenCV NumPy array
            open_cv_image = np.array(img.convert("RGB"))
            open_cv_image = open_cv_image[:, :, ::-1].copy() # RGB to BGR
            
            # Downsample for performance
            h, w = open_cv_image.shape[:2]
            small_img = cv2.resize(open_cv_image, (128, 128))
            
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliency_map = saliency.computeSaliency(small_img)
            
            if success:
                # Find centroid of saliency map
                m = cv2.moments((saliency_map * 255).astype(np.uint8))
                if m["m00"] > 0:
                    cx = m["m10"] / m["m00"] / 128.0
                    cy = m["m01"] / m["m00"] / 128.0
                    return cx, cy
        except ImportError:
            pass
        except Exception:
            pass

        # PIL Fallback: Contrast variance detection
        try:
            # Resize for speed
            small_img = img.resize((64, 64)).convert("L")
            # Apply FIND_EDGES high-pass filter
            edges = small_img.filter(ImageFilter.FIND_EDGES)
            pixels = list(edges.getdata())
            
            # Compute center of mass of edge pixels
            sum_x = 0.0
            sum_y = 0.0
            total_weight = 0.0
            
            for y in range(64):
                for x in range(64):
                    weight = pixels[y * 64 + x]
                    if weight > 30:  # Threshold noise
                        sum_x += x * weight
                        sum_y += y * weight
                        total_weight += weight
            
            if total_weight > 0:
                return (sum_x / total_weight) / 64.0, (sum_y / total_weight) / 64.0
        except Exception:
            pass
            
        return 0.5, 0.5

    def _estimate_edge_complexity(self, img: Image.Image) -> float:
        """Estimate image edge complexity (detail density) using Canny-like filter."""
        try:
            small_img = img.resize((100, 100)).convert("L")
            edges = small_img.filter(ImageFilter.FIND_EDGES)
            stat = ImageStat.Stat(edges)
            # Average intensity of edge pixels normalized
            return stat.mean[0] / 255.0
        except Exception:
            return 0.5

    def _get_quadrant(self, x: float, y: float) -> str:
        """Categorise coordinates into rule-of-thirds focal quadrants."""
        if 0.33 <= x <= 0.66 and 0.33 <= y <= 0.66:
            return "center"
        
        horiz = "left" if x < 0.33 else ("right" if x > 0.66 else "")
        vert = "top" if y < 0.33 else ("bottom" if y > 0.66 else "")
        
        if horiz and vert:
            return f"{vert}_{horiz}"
        return horiz or vert or "center"

    def _classify_role(self, metadata: dict[str, Any], sal_x: float, sal_y: float, dominant_colors: list[str]) -> str:
        """Heuristically classify an image as A-roll or B-roll."""
        reasons = 0
        
        # 1. Subject keyword presence
        caption = metadata.get("florence_caption", "").lower()
        symbolism = metadata.get("llava_symbolism", "").lower()
        combined_text = f"{caption} {symbolism}"
        
        matched_keywords = [w for w in A_ROLL_KEYWORDS if w in combined_text]
        if matched_keywords:
            reasons += 1
            if "person" in matched_keywords or "people" in matched_keywords:
                reasons += 1  # Person presence is strong A-roll indicator
                
        # 2. Object detection bounding boxes
        boxes = metadata.get("bounding_boxes", [])
        if boxes:
            reasons += 1
            # Check for large focal object
            for box in boxes:
                if len(box) == 4:
                    area = abs(box[2] - box[0]) * abs(box[3] - box[1])
                    if area > 0.08:
                        reasons += 1
                        break

        # 3. Artistic tension
        tension = metadata.get("llava_tension", 0.5)
        if tension > 0.65:
            reasons += 1
            
        # Classify as A-roll if there are at least two strong cues
        if reasons >= 2:
            return "a_roll"
        return "b_roll"
