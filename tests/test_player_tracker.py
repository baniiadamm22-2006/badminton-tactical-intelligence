import numpy as np
from src.player_tracker import court_side, extract_tracks


POLYGON = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)


def test_court_side():
    assert court_side(50, 20, POLYGON) == 'far'
    assert court_side(50, 80, POLYGON) == 'near'


def test_extract_tracks_without_ids():
    class Result:
        boxes = None
    class Filter:
        polygon = POLYGON
    assert extract_tracks(Result(), Filter()) == []
