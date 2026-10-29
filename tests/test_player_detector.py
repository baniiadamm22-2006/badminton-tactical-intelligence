from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

pytest.importorskip("ultralytics")
from src.player_detector import PlayerDetector


class MockTensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class MockModel:
    def predict(self, **kwargs):
        assert kwargs["classes"] == [0]
        return [SimpleNamespace(boxes=SimpleNamespace(
            xyxy=MockTensor([[10, 20, 40, 100]]),
            conf=MockTensor([0.9]),
        ))]


def fake_detector():
    detector = PlayerDetector.__new__(PlayerDetector)
    detector.model = MockModel()
    detector.confidence = 0.35
    detector.device = "cpu"
    detector.image_size = 640
    return detector


def test_detection_geometry():
    detections = fake_detector().detect(np.zeros((120, 160, 3), dtype=np.uint8))
    assert len(detections) == 1
    assert detections[0]["foot_x"] == 25.0
    assert detections[0]["foot_y"] == 100.0
    assert detections[0]["confidence"] == 0.9


def test_csv_and_video_outputs(tmp_path):
    input_path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(input_path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (160, 120))
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v codec tidak tersedia")
    for _ in range(5):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()
    output_path = tmp_path / "results"
    report = fake_detector().process_video(input_path, output_path, max_frames=3)
    assert report["processed_frames"] == 3
    assert report["total_person_detections"] == 3
    assert (output_path / "detections.csv").exists()
    assert (output_path / "detections.mp4").exists()
    assert len((output_path / "detections.csv").read_text().splitlines()) == 4


def test_invalid_stride(tmp_path):
    with pytest.raises(ValueError):
        fake_detector().process_video(tmp_path / "no_video.mp4", tmp_path / "out", stride=0)
