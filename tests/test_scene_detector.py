import numpy as np
from src.scene_detector import SceneConfig, compare_features, frame_features, segment_for_frame


def test_detect_visual_change():
    a = np.zeros((100, 100, 3), dtype=np.uint8)
    b = np.full((100, 100, 3), (0, 200, 100), dtype=np.uint8)
    pixel, hist = compare_features(frame_features(a, (0, 0, 100, 100)),
                                   frame_features(b, (0, 0, 100, 100)))
    assert pixel > 0.016
    assert hist > 0.06


def test_segment_boundaries():
    assert segment_for_frame(248, [249]) == 1
    assert segment_for_frame(249, [249]) == 2
    assert segment_for_frame(300, [249]) == 2


def test_same_frame_no_cut():
    a = np.full((40, 60, 3), 100, dtype=np.uint8)
    pixel, hist = compare_features(frame_features(a, (0, 0, 60, 40)),
                                   frame_features(a, (0, 0, 60, 40)))
    assert pixel == 0
    assert hist == 0
