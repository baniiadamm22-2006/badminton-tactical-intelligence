import json
import pytest
from src.court_filter import CourtFilter, validate_polygon


def test_court_filter(tmp_path):
    path = tmp_path / "court.json"
    path.write_text(json.dumps({"width": 640, "height": 480,
        "points": [[100, 100], [500, 100], [500, 400], [100, 400]]}))
    roi = CourtFilter(path, margin_px=10)
    roi.ensure_resolution(640, 480)
    assert roi.accepts({"foot_x": 250, "foot_y": 250})
    assert not roi.accepts({"foot_x": 250, "foot_y": 80})
    assert roi.accepts({"foot_x": 250, "foot_y": 95})
    assert not roi.accepts({"foot_x": 250, "foot_y": 89})
    with pytest.raises(ValueError):
        roi.ensure_resolution(1280, 720)


def test_invalid_polygon():
    with pytest.raises(ValueError):
        validate_polygon([[1, 1], [2, 2], [3, 3], [4, 4]])
