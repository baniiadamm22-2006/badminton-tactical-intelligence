import csv
import json
from pathlib import Path

import pytest

from src.rally_annotation import Rally, export_annotations, load_annotations, save_annotations, validate_rallies


def test_export_with_scene_cut(tmp_path):
    path = tmp_path / "ann.json"
    rallies = [Rally(1, 10, 20, "far", "verified"), Rally(2, 240, 255)]
    save_annotations(path, "video.mp4", 30, 300, rallies)
    scenes = tmp_path / "cuts.json"
    scenes.write_text(json.dumps({"cuts": [{"frame_index": 249}]}))
    report = export_annotations(path, tmp_path / "output", scenes)
    assert report["annotated_rallies"] == 2
    assert report["rallies_crossing_scene_cut"] == 1
    assert load_annotations(path, 300) == rallies
    with (tmp_path / "output" / "rallies.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[1]["crosses_scene_cut"] == "True"


def test_overlap_rejected():
    with pytest.raises(ValueError):
        validate_rallies([Rally(1, 0, 10), Rally(2, 10, 20)], 30)


def test_outside_frames_rejected():
    with pytest.raises(ValueError):
        validate_rallies([Rally(1, 0, 30)], 30)


def test_duplicate_id_rejected():
    with pytest.raises(ValueError):
        validate_rallies([Rally(1, 0, 5), Rally(1, 7, 10)], 30)


def test_missing_returns_empty(tmp_path):
    assert load_annotations(tmp_path / "missing.json", 30) == []
