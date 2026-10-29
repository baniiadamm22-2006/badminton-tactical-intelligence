"""Video inspection, sampling, and metadata reporting."""

from __future__ import annotations

import json
from pathlib import Path

import cv2


class VideoProcessor:
    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video tidak ditemukan: {self.video_path}")

    def inspect(self) -> dict:
        capture = cv2.VideoCapture(str(self.video_path))
        try:
            if not capture.isOpened():
                raise RuntimeError("Video tidak dapat dibuka.")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps <= 0 or width <= 0 or height <= 0:
                raise RuntimeError("Metadata FPS/resolusi tidak valid.")
            ok, _ = capture.read()
            if not ok:
                raise RuntimeError("Frame pertama tidak dapat dibaca.")
            return {
                "filename": self.video_path.name,
                "width": width,
                "height": height,
                "fps": round(fps, 3),
                "frame_count_reported": count,
                "duration_seconds_estimated": round(count / fps, 2) if count > 0 else None,
            }
        finally:
            capture.release()

    def save_metadata(self, path: str | Path) -> dict:
        metadata = self.inspect()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    def extract_frames(self, directory: str | Path, interval_seconds: float = 1.0,
                       max_frames: int | None = None) -> int:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds harus > 0")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames harus > 0")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise RuntimeError("Video tidak dapat dibuka.")
        saved = 0
        frame_index = 0
        next_time = 0.0
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                raise RuntimeError("FPS tidak valid.")
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                time_seconds = frame_index / fps
                if time_seconds + 1e-8 >= next_time:
                    target = directory / f"frame_{frame_index:08d}.jpg"
                    if not cv2.imwrite(str(target), frame):
                        raise RuntimeError(f"Gagal menyimpan {target}")
                    saved += 1
                    next_time += interval_seconds
                    if max_frames is not None and saved >= max_frames:
                        break
                frame_index += 1
        finally:
            capture.release()
        return saved
