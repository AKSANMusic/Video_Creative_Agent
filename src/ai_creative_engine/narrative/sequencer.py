"""The narrative sequencer (Stage 3 core).

Algorithm (two-phase, deterministic)
-------------------------------------
Phase A - Section assignment:
    For each image compute its best-fitting audio section via
    ``tension_energy_score(image.llava_tension, section.mean_energy)``.
    Images are bucketed per section. Ties broken by section order for stability.

Phase B - Per-section image ordering via DP / greedy:
    Within a section we have a set of cut slots (downbeats inside that section).
    We pick an ordering of images that minimizes the total transition cost
    (embedding similarity + color continuity) along the sequence.

    - Small sets (n <= ``dp_threshold``, default 12): exact DP over subsets
      (Held-Karp style Hamiltonian path) -> optimal ordering.
    - Larger sets: greedy nearest-neighbor from a fixed start (section entry),
      deterministic and O(n^2).

    The path's *entry* image is the one whose embedding is most similar to the
    previous section's *exit* image (for the first section, the globally
    best-tension match), providing cross-section continuity.

Output
------
A list of (image, start, end, section_index) tuples whose ``start`` times are
exactly the chosen downbeats and whose union covers the full track duration.
The caller wraps these into :class:`Timeline` / :class:`TimelineEntry`.

Determinism
-----------
No randomness anywhere; identical inputs => identical timeline. This is
essential for caching and reproducible renders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from ..audio.audio_map import AudioMap, Section
from ..errors import CreativeEngineError
from ..models import ImageMetadata
from . import scoring
from .timeline import Timeline, TimelineEntry, Transition

log = logging.getLogger(__name__)

# Above this image-set size per section we fall back from exact DP to greedy.
DEFAULT_DP_THRESHOLD = 12


@dataclass(frozen=True)
class _Placement:
    image: ImageMetadata
    start: float
    end: float
    section_index: int


class NarrativeSequencer:
    """Builds a beat-locked :class:`Timeline` from images + an audio map."""

    def __init__(
        self,
        dp_threshold: int = DEFAULT_DP_THRESHOLD,
        continuity_weight: float = 0.5,
        cut_on: str = "downbeats",
        min_shot_duration: float = 1.5,
    ) -> None:
        if dp_threshold < 2:
            raise ValueError("dp_threshold must be >= 2")
        if cut_on not in ("downbeats", "beats"):
            raise ValueError("cut_on must be 'downbeats' or 'beats'")
        if min_shot_duration <= 0:
            raise ValueError("min_shot_duration must be positive")
        self.dp_threshold = int(dp_threshold)
        self.continuity_weight = float(continuity_weight)
        self.cut_on = cut_on
        self.min_shot_duration = float(min_shot_duration)

    # --- public API ----------------------------------------------------------

    def sequence(
        self,
        images: list[ImageMetadata],
        audio_map: AudioMap,
    ) -> Timeline:
        """Return a fully-populated :class:`Timeline`."""
        if not images:
            raise CreativeEngineError("Cannot build a timeline from zero images.")
        if audio_map.duration <= 0:
            raise CreativeEngineError("Audio map has non-positive duration.")

        anchors = self._select_anchors(audio_map)
        if len(anchors) < 2:
            # Degenerate track: emit a single full-duration entry with the
            # best-tension image so the renderer still has something valid.
            return self._single_entry_timeline(images, audio_map, anchors)

        # Phase A: assign images to sections.
        buckets = self._assign_to_sections(images, audio_map.sections)

        # Phase B: order images within each section and slot them.
        placements: list[_Placement] = []
        prev_exit_bytes: Optional[bytes] = None
        section_index_of_prev_exit: Optional[int] = None

        for section in audio_map.sections:
            bucket = buckets.get(section.index, [])
            slots = self._section_slots(section, anchors)
            if not slots:
                continue

            ordered = self._order_section(
                bucket=bucket,
                slots=slots,
                prev_exit_bytes=prev_exit_bytes,
                fallback_pool=images,
                section=section,
            )

            # If section has more slots than images, repeat from the ordered set
            # cyclically (keeps continuity). If fewer slots, trim.
            for i, (s_start, s_end) in enumerate(slots):
                img = ordered[i % len(ordered)]
                placements.append(_Placement(img, s_start, s_end, section.index))

            prev_exit_bytes = ordered[(len(slots) - 1) % len(ordered)].embedding_bytes
            section_index_of_prev_exit = section.index

        if not placements:
            return self._single_entry_timeline(images, audio_map, anchors)

        # Safety net: guarantee the timeline tiles [0, duration] exactly.
        placements = self._ensure_full_coverage(placements, audio_map, anchors)

        entries = self._placements_to_entries(placements, images, audio_map)
        return Timeline(
            audio_id=audio_map.audio_id,
            audio_file=audio_map.file_path,
            duration=audio_map.duration,
            bpm=audio_map.bpm,
            downbeats_used=[p[0] for p in self._section_slots_all(audio_map.sections, anchors)],
            entries=entries,
        )

    def _ensure_full_coverage(
        self,
        placements: list[_Placement],
        audio_map: AudioMap,
        anchors: list[float],
    ) -> list[_Placement]:
        """Guarantee placements tile [0, duration] with no gaps/overlaps.

        Defensive: if sections did not perfectly cover the track (or the last
        anchor was not exactly `duration`), this extends/trims placements so
        the resulting Timeline passes its contiguity invariant. The image
        choice for each slot is preserved; only time bounds are normalized.
        """
        if not placements:
            return placements
        placements = sorted(placements, key=lambda p: (p.start, p.section_index))
        duration = audio_map.duration
        # Build a clean contiguous grid from the placement start times, forcing
        # the first to 0 and the last end to `duration`.
        starts = [0.0] + [p.start for p in placements[1:]]
        ends = starts[1:] + [duration]
        out: list[_Placement] = []
        for p, start, end in zip(placements, starts, ends):
            # Keep at least a tiny positive window; if two placements shared a
            # start, collapse by extending to the following boundary.
            if end <= start:
                end = start + 1e-3
            out.append(_Placement(image=p.image, start=start, end=end, section_index=p.section_index))
        return out

    # --- anchor / slot selection --------------------------------------------

    def _select_anchors(self, audio_map: AudioMap) -> list[float]:
        """Return the cut-anchor times, always including 0 and duration."""
        grid = audio_map.downbeats if self.cut_on == "downbeats" else audio_map.beats
        grid = sorted(set(float(t) for t in grid))
        
        # Enforce minimum shot duration (thin out anchors)
        if grid:
            thinned = [grid[0]]
            for t in grid[1:]:
                if t - thinned[-1] >= self.min_shot_duration:
                    thinned.append(t)
            grid = thinned
            
        if not grid or grid[0] > 1e-6:
            grid = [0.0] + grid
        if grid[-1] < audio_map.duration - 1e-6:
            # Force the last anchor if we need to close the track.
            # But if the previous anchor was too close to duration, replace it?
            if len(grid) > 1 and audio_map.duration - grid[-1] < self.min_shot_duration / 2:
                grid[-1] = audio_map.duration
            else:
                grid.append(audio_map.duration)
                
        # Drop any anchors beyond duration (defensive).
        grid = [t for t in grid if t <= audio_map.duration + 1e-6]
        return grid

    def _section_slots(
        self, section: Section, anchors: list[float]
    ) -> list[tuple[float, float]]:
        """Cut slots (start,end) that fall strictly inside this section."""
        out: list[tuple[float, float]] = []
        for i in range(len(anchors) - 1):
            s_start, s_end = anchors[i], anchors[i + 1]
            # A slot belongs to a section if its midpoint is inside [start,end).
            mid = (s_start + s_end) / 2.0
            if section.start <= mid < section.end:
                out.append((s_start, s_end))
        return out

    def _section_slots_all(
        self, sections: list[Section], anchors: list[float]
    ) -> list[tuple[float, float]]:
        slots: list[tuple[float, float]] = []
        for sec in sections:
            slots.extend(self._section_slots(sec, anchors))
        return slots

    # --- Phase A: assignment -------------------------------------------------

    def _assign_to_sections(
        self, images: list[ImageMetadata], sections: list[Section]
    ) -> dict[int, list[ImageMetadata]]:
        if not sections:
            return {}
        buckets: dict[int, list[ImageMetadata]] = {s.index: [] for s in sections}
        
        # Group by mood for semantic clustering
        from collections import defaultdict
        mood_groups = defaultdict(list)
        for img in images:
            mood_groups[img.llava_mood].append(img)
            
        # Assign each mood group to the section that best matches its average tension
        for mood, group in mood_groups.items():
            avg_tension = sum(img.llava_tension for img in group) / len(group)
            
            best_idx = sections[0].index
            best_score = -1.0
            for s in sections:
                sc = scoring.tension_energy_score(avg_tension, s.mean_energy)
                if sc > best_score:
                    best_score = sc
                    best_idx = s.index
            
            buckets[best_idx].extend(group)
            
        # Guarantee every section that will have slots has at least one image.
        # (Empty buckets are handled by _order_section via fallback_pool.)
        return buckets

    # --- Phase B: ordering ---------------------------------------------------

    def _order_section(
        self,
        bucket: list[ImageMetadata],
        slots: list[tuple[float, float]],
        prev_exit_bytes: Optional[bytes],
        fallback_pool: list[ImageMetadata],
        section: Section,
    ) -> list[ImageMetadata]:
        """Return an ordered list of images to fill ``slots`` for a section."""
        # Ensure the working set is non-empty.
        pool = list(bucket) if bucket else list(fallback_pool)
        # Cap working set to the number of slots (we only need that many).
        k = min(len(pool), max(1, len(slots)))

        # Pick the entry image: most similar to previous section's exit, else
        # the best tension-energy match in this section.
        start = self._pick_entry(pool[:k], prev_exit_bytes, section)

       # Restrict the path set to k images including the chosen start.
        working = self._build_working_set(pool, start, k)

        if len(working) <= self.dp_threshold:
            ordered = self._hamiltonian_dp(working, start, section)
        else:
            ordered = self._greedy_nearest_neighbor(working, start, section)
        return ordered

    def _pick_entry(
        self,
        candidates: list[ImageMetadata],
        prev_exit_bytes: Optional[bytes],
        section: Section,
    ) -> ImageMetadata:
        if not candidates:
            raise CreativeEngineError("empty candidate set for section entry")
        if prev_exit_bytes is not None:
            best = max(
                candidates,
                key=lambda im: scoring.embedding_similarity(prev_exit_bytes, im.embedding_bytes),
            )
            return best
        return max(
            candidates,
            key=lambda im: scoring.tension_energy_score(im.llava_tension, section.mean_energy),
        )

    def _build_working_set(
        self, pool: list[ImageMetadata], start: ImageMetadata, k: int
    ) -> list[ImageMetadata]:
        """Pick k images including ``start``; deterministic by image_id ordering."""
        seen = {start.image_id}
        working = [start]
        for img in sorted(pool, key=lambda im: im.image_id):
            if len(working) >= k:
                break
            if img.image_id in seen:
                continue
            working.append(img)
            seen.add(img.image_id)
        return working

   # --- exact DP (Held-Karp) ------------------------------------------------

    def _hamiltonian_dp(
       self, items: list[ImageMetadata], start: ImageMetadata, section: Optional[Section] = None
    ) -> list[ImageMetadata]:
        """Exact minimum-cost path visiting every item once, beginning at ``start``.

        Complexity O(n^2 * 2^n) -> only used for n <= dp_threshold.
        """
        n = len(items)
        if n == 1:
            return list(items)
        start_idx = items.index(start)
        # Precompute pairwise costs.
        cost = [
            [self._pair_cost(items[i], items[j], section) if i != j else 0.0 for j in range(n)]
            for i in range(n)
        ]

        # dp[mask][j] = (cost, prev_j) of the cheapest path that visits the set
        # `mask` and ends at node j. We seed with the start node.
        INF = float("inf")
        dp = [[(INF, -1)] * n for _ in range(1 << n)]
        dp[1 << start_idx][start_idx] = (0.0, -1)

        for mask in range(1 << n):
            for j in range(n):
                if not (mask & (1 << j)):
                    continue
                cur_cost, _ = dp[mask][j]
                if cur_cost == INF:
                    continue
                for k in range(n):
                    if mask & (1 << k):
                        continue
                    nxt_mask = mask | (1 << k)
                    cand = cur_cost + cost[j][k]
                    if cand < dp[nxt_mask][k][0]:
                        dp[nxt_mask][k] = (cand, j)

        # Pick the cheapest full mask ending anywhere.
        full = (1 << n) - 1
        best_end = min(range(n), key=lambda j: dp[full][j][0])
        # Reconstruct path.
        path: list[int] = []
        mask = full
        j = best_end
        while j != -1:
            path.append(j)
            prev = dp[mask][j][1]
            mask_without_j = mask & ~(1 << j)
            j = prev
            mask = mask_without_j
        path.reverse()
        return [items[i] for i in path]

    # --- greedy fallback -----------------------------------------------------

    def _greedy_nearest_neighbor(
        self, items: list[ImageMetadata], start: ImageMetadata, section: Optional[Section] = None
    ) -> list[ImageMetadata]:
        """Deterministic nearest-neighbor path from ``start``."""
        remaining = {im.image_id: im for im in items}
        ordered: list[ImageMetadata] = [start]
        del remaining[start.image_id]
        cur = start
        while remaining:
            nxt = min(
                remaining.values(),
                key=lambda im: (
                    self._pair_cost(cur, im, section),
                    im.image_id,  # deterministic tie-break
                ),
            )
            ordered.append(nxt)
            del remaining[nxt.image_id]
            cur = nxt
        return ordered

    # --- cost helpers --------------------------------------------------------

    def _pair_cost(
        self,
        a: ImageMetadata,
        b: ImageMetadata,
        section: Optional[Section] = None,
    ) -> float:
        override = section.continuity_weight_override if section is not None else None
        return scoring.transition_cost(
            a.embedding_bytes,
            b.embedding_bytes,
            continuity_weight=self.continuity_weight,
            continuity_weight_override=override,
            prev_hist=getattr(a, "color_histogram", None),
            cur_hist=getattr(b, "color_histogram", None),
        )

    # --- placements -> entries ----------------------------------------------

    def _placements_to_entries(
        self,
        placements: list[_Placement],
        all_images: list[ImageMetadata],
        audio_map: AudioMap,
    ) -> list[TimelineEntry]:
        entries: list[TimelineEntry] = []
        for idx, p in enumerate(placements):
            # Transition: hard cut by default; dissolve if the slot is long
            # enough and the previous image was very similar (smooth).
            slot_len = p.end - p.start
            trans = Transition(type="cut", duration_s=0.0)
            if idx > 0 and slot_len >= 1.0:
                prev_img = placements[idx - 1].image
                sim = scoring.embedding_similarity(prev_img.embedding_bytes, p.image.embedding_bytes)
                if sim >= 0.85:
                    trans = Transition(type="dissolve", duration_s=min(0.4, slot_len * 0.3))
            entries.append(
                TimelineEntry(
                    index=idx,
                    image_id=p.image.image_id,
                    file_path=p.image.file_path,
                    section_index=p.section_index,
                    start=p.start,
                    end=p.end,
                    transition=trans,
                )
            )
        return entries

    # --- degenerate fallback -------------------------------------------------

    def _single_entry_timeline(
        self,
        images: list[ImageMetadata],
        audio_map: AudioMap,
        anchors: list[float],
    ) -> Timeline:
        """One entry spanning the whole track (used when no beat grid exists)."""
        best = max(
            images,
            key=lambda im: (
                scoring.tension_energy_score(
                    im.llava_tension,
                    audio_map.sections[0].mean_energy if audio_map.sections else 0.5,
                ),
                im.image_id,
            ),
        )
        entry = TimelineEntry(
            index=0,
            image_id=best.image_id,
            file_path=best.file_path,
            section_index=0,
            start=0.0,
            end=audio_map.duration,
            transition=Transition(type="cut", duration_s=0.0),
        )
        return Timeline(
            audio_id=audio_map.audio_id,
            audio_file=audio_map.file_path,
            duration=audio_map.duration,
            bpm=audio_map.bpm,
            downbeats_used=[0.0],
            entries=[entry],
        )
