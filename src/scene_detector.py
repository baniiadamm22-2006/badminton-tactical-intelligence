"""Detect abrupt shot changes with frame difference and color histogram.

A cut means a new temporal segment, not necessarily a new rally.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


@dataclass(frozen=True)
class SceneConfig:
    pixel_threshold: float = 0.016
    histogram_threshold: float = 0.06
    min_segment_frames: int = 12
    sample_width: int = 160
    sample_height: int = 160


def frame_features(frame: np.ndarray, bounds: tuple[int, int, int, int]):
    left, top, right, bottom = bounds
    roi = frame[top:bottom, left:right]
    if roi.size == 0:
        raise ValueError("ROI scene detector kosong")
    image = cv2.resize(roi, (160, 160), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    return gray, hist


def compare_features(previous, current) -> tuple[float, float]:
    prev_gray, prev_hist = previous
    gray, hist = current
    pixel_diff = float(cv2.absdiff(prev_gray, gray).mean() / 255.0)
    histogram_diff = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
    return pixel_diff, histogram_diff


def detect_cuts(video_path: str | Path, bounds: tuple[int, int, int, int],
                max_frames: int | None = None, config: SceneConfig = SceneConfig()):
    """Return frames where a new scene *starts* (frame 0 is implicit)."""
    if config.min_segment_frames < 1:
        raise ValueError("min_segment_frames harus >= 1")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video tidak bisa dibuka: {video_path}")
    cuts = []
    previous = None
    index = 0
    last_cut = 0
    try:
        while max_frames is None or index < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            current = frame_features(frame, bounds)
            if previous is not None:
                pixel_diff, hist_diff = compare_features(previous, current)
                if (pixel_diff >= config.pixel_threshold and
                        hist_diff >= config.histogram_threshold and
                        index - last_cut >= config.min_segment_frames):
                    cuts.append({
                        "frame_index": index,
                        "pixel_difference": round(pixel_diff, 5),
                        "histogram_difference": round(hist_diff, 5),
                    })
                    last_cut = index
            previous = current
            index += 1
    finally:
        cap.release()
    return cuts


def segment_for_frame(frame_index: int, cut_frames: list[int]) -> int:
    return 1 + sum(frame_index >= cut for cut in cut_frames)
