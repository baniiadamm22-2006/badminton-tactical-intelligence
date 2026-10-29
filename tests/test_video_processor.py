from pathlib import Path

import cv2
import numpy as np
import pytest

from src.video_processor import VideoProcessor


def make_sample_video(path: Path, frames: int = 12, fps: float = 6.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v codec tidak tersedia")
    for index in range(frames):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.circle(image, (20 + index * 4, 60), 10, (255, 255, 255), -1)
        writer.write(image)
    writer.release()


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoProcessor(tmp_path / "missing.mp4")


def test_inspect_and_extract(tmp_path):
    video = tmp_path / "sample.mp4"
    make_sample_video(video)
    processor = VideoProcessor(video)
    metadata = processor.save_metadata(tmp_path / "metadata.json")
    assert metadata["width"] == 160
    assert metadata["height"] == 120
    assert metadata["fps"] == 6.0
    count = processor.extract_frames(tmp_path / "frames", interval_seconds=0.5)
    assert count == 4
    assert len(list((tmp_path / "frames").glob("*.jpg"))) == 4


def test_invalid_interval(tmp_path):
    video = tmp_path / "sample.mp4"
    make_sample_video(video)
    with pytest.raises(ValueError):
        VideoProcessor(video).extract_frames(tmp_path / "frames", 0)
