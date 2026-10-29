"""Regression: cropped detections must not mutate inference tensors."""
from types import SimpleNamespace
import pytest


def test_extract_tracks_clones_tensors_and_offsets():
    torch = pytest.importorskip("torch")
    from src.player_tracker import extract_tracks

    class AcceptAll:
        polygon = [(0, 0), (476, 0), (476, 848), (0, 848)]

        def accepts(self, detection):
            return True

    with torch.inference_mode():
        xyxy = torch.tensor([[10., 20., 30., 60.]])
        ids = torch.tensor([1.])
        confidence = torch.tensor([.75])

    original = xyxy.clone()
    result = SimpleNamespace(boxes=SimpleNamespace(
        xyxy=xyxy, id=ids, conf=confidence,
    ))
    detections = extract_tracks(result, AcceptAll(), offset_x=100, offset_y=200)
    assert len(detections) == 1
    assert detections[0]["x1"] == 110.0
    assert detections[0]["x2"] == 130.0
    assert detections[0]["y1"] == 220.0
    assert detections[0]["y2"] == 260.0
    assert torch.equal(xyxy, original)
