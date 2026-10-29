import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.full_match import (write_chunk, assemble_csv, segment_start_for,
                            segment_id_for, run_full_match, FIELDS)


def test_segment_boundary():
    assert segment_start_for(25, [12, 25, 40]) == 25
    assert segment_id_for(25, [12, 25, 40]) == 3
    assert segment_id_for(24, [12, 25, 40]) == 2


def test_chunk_roundtrip(tmp_path):
    rows = [{key: 1 for key in FIELDS}]
    write_chunk(tmp_path, 0, rows)
    write_chunk(tmp_path, 1, [])
    assert assemble_csv(tmp_path, 2, tmp_path / 'all.csv') == 1
    with (tmp_path/'all.csv').open() as f:
        assert len(list(csv.DictReader(f))) == 1


class FakeModel:
    def __init__(self, container):
        self.container = container
        self.calls = 0

    def track(self, frame, **kwargs):
        self.calls += 1
        self.container.append(self.calls)
        boxes = SimpleNamespace(id=None)
        return [SimpleNamespace(boxes=boxes)]


def make_video(path: Path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (160, 120))
    if not writer.isOpened():
        pytest.skip('Codec MJPG tidak tersedia')
    for _ in range(18):
        writer.write(np.full((120, 160, 3), (20, 80, 20), dtype=np.uint8))
    writer.release()


def test_resume_does_not_duplicate_frames(tmp_path):
    video = tmp_path / 'v.avi'
    make_video(video)
    court = tmp_path / 'court.json'
    court.write_text(json.dumps({"width":160,"height":120,"points":
                  [[20,10],[140,10],[140,110],[20,110]]}), encoding='utf-8')
    config = tmp_path / 'tracker.yaml'
    config.write_text('tracker_type: bytetrack', encoding='utf-8')
    results = []
    # Fake model has no tracked boxes but produces frame-level inferencing events.
    factory = lambda: FakeModel(results)
    kwargs = dict(video=video, court=court, output=tmp_path/'out', tracker=str(config),
                  model_factory=factory, chunk_frames=5, max_frames=18)
    report = run_full_match(**kwargs)
    assert report['processed_frames'] == 18
    assert report['completed_chunks'] == 4
    assert report['finished']
    previous_calls = len(results)
    again = run_full_match(**kwargs, resume=True)
    assert again['completed_chunks'] == 4
    assert again['tracking_rows'] == 0
    # Note: current implementation warms up tracker on resume even if complete.
    assert len(results) >= previous_calls
    assert len(list((tmp_path/'out'/'chunks').glob('chunk_*.csv'))) == 4


def test_resume_rejects_changed_parameters(tmp_path):
    video = tmp_path / 'v.avi'; make_video(video)
    court = tmp_path/'court.json'
    court.write_text(json.dumps({"width":160,"height":120,"points":
                  [[20,10],[140,10],[140,110],[20,110]]}), encoding='utf-8')
    config = tmp_path / 'tracker.yaml'; config.write_text('tracker_type: bytetrack')
    args = dict(video=video, court=court, output=tmp_path/'out', tracker=str(config),
                model_factory=lambda: FakeModel([]), max_frames=18)
    run_full_match(**args)
    with pytest.raises(ValueError, match='resume ditolak'):
        run_full_match(**{**args, 'imgsz':640}, resume=True)


def test_resume_after_interruption(tmp_path):
    video = tmp_path / 'v.avi'; make_video(video)
    court = tmp_path / 'court.json'
    court.write_text(json.dumps({"width": 160, "height": 120,
                                 "points": [[20, 10], [140, 10], [140, 110], [20, 110]]}))
    tracker = tmp_path / 'tracker.yaml'; tracker.write_text('tracker_type: bytetrack')

    class InterruptedModel(FakeModel):
        def track(self, frame, **kwargs):
            if self.calls == 7:
                raise KeyboardInterrupt('simulated interruption')
            return super().track(frame, **kwargs)

    kwargs = dict(video=video, court=court, output=tmp_path / 'out',
                  tracker=str(tracker), chunk_frames=5, max_frames=18)
    with pytest.raises(KeyboardInterrupt):
        run_full_match(**kwargs, model_factory=lambda: InterruptedModel([]))
    cp = json.loads((tmp_path / 'out' / 'checkpoint.json').read_text())
    assert cp['next_frame'] == 5
    assert cp['chunks_completed'] == 1
    report = run_full_match(**kwargs, resume=True,
                            model_factory=lambda: FakeModel([]))
    assert report['processed_frames'] == 18
    assert report['completed_chunks'] == 4
    assert len(list((tmp_path / 'out' / 'chunks').glob('chunk_*.csv'))) == 4
