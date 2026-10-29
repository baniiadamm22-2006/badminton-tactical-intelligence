import csv
import json
from pathlib import Path
import numpy as np
import pytest
from src.court_mapper import build_homography, map_tracking_csv, project_point

CAL = {'width': 476, 'height': 848, 'points': [[96, 281], [376, 282], [467, 539], [9, 535]]}

def test_four_corners_map_to_metric_rectangle():
    H = build_homography(CAL)
    target = [(0, 0), (6.1, 0), (6.1, 13.4), (0, 13.4)]
    for src, expected in zip(CAL['points'], target):
        assert np.allclose(project_point(H, *src), expected, atol=1e-3)

def test_invalid_polygon_rejected():
    with pytest.raises(ValueError):
        build_homography({'points': [[1, 1], [1, 1], [2, 2], [3, 3]]})

def test_csv_preserves_segment_ids_and_excludes_outside(tmp_path):
    cal = tmp_path / 'court.json'
    cal.write_text(json.dumps(CAL), encoding='utf-8')
    path = tmp_path / 'tracks.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame_index','segment_id','track_id','foot_x','foot_y'])
        writer.writeheader()
        writer.writerow({'frame_index':0,'segment_id':1,'track_id':1,'foot_x':96,'foot_y':281})
        writer.writerow({'frame_index':249,'segment_id':2,'track_id':1,'foot_x':376,'foot_y':282})
        writer.writerow({'frame_index':250,'segment_id':2,'track_id':1,'foot_x':-200,'foot_y':-200})
    report = map_tracking_csv(path, cal, tmp_path / 'out')
    assert report['inside_court'] == 2
    assert report['outside_court'] == 1
    assert report['unique_segment_track_pairs'] == 2
    assert (tmp_path/'out'/'court_map.png').is_file()
    with (tmp_path/'out'/'court_positions.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert rows[2]['mapping_status']=='outside'
