import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest
from src.movement_intelligence import analyze, zone_for_position


def test_zone_orientation():
    assert zone_for_position(0.5, 'far') == 'rear'
    assert zone_for_position(6.2, 'far') == 'front'
    assert zone_for_position(12.9, 'near') == 'rear'
    assert zone_for_position(7.1, 'near') == 'front'
    assert zone_for_position(4, 'near') == 'out_of_side'


def test_analyze_keeps_segments_isolated(tmp_path):
    data = pd.DataFrame([
        {'frame_index': 0, 'segment_id': 1, 'track_id': 1, 'court_side': 'far',
         'clean_x_m': 2, 'clean_y_m': 1, 'eligible_for_movement': True},
        {'frame_index': 1, 'segment_id': 1, 'track_id': 1, 'court_side': 'far',
         'clean_x_m': 2.1, 'clean_y_m': 1.1, 'eligible_for_movement': False},
        {'frame_index': 2, 'segment_id': 2, 'track_id': 1, 'court_side': 'near',
         'clean_x_m': 2, 'clean_y_m': 12, 'eligible_for_movement': True},
    ])
    source = tmp_path / 'source.csv'
    data.to_csv(source, index=False)
    report = analyze(source, tmp_path / 'out', min_samples=2)
    assert report['valid_location_rows'] == 2
    assert len(report['tracks']) == 2
    assert [x['valid_rows'] for x in report['tracks']] == [1, 1]
    assert all(not x['sufficient_samples'] for x in report['tracks'])
    zones = pd.read_csv(tmp_path / 'out' / 'zone_statistics.csv')
    assert len(zones) == 6


def test_missing_columns_fail(tmp_path):
    source = tmp_path / 'invalid.csv'
    pd.DataFrame({'abc': [1]}).to_csv(source, index=False)
    with pytest.raises(ValueError, match='Kolom'):
        analyze(source, tmp_path / 'out')
