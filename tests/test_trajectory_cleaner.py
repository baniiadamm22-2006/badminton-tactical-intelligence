import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.trajectory_cleaner import clean_positions

def test_scene_boundaries_not_connected(tmp_path):
    rows=[]
    for seg in (1,2):
        for i in range(12):
            rows.append(dict(frame_index=i+(seg-1)*12,segment_id=seg,track_id=1,timestamp_seconds=(i+(seg-1)*12)/30,court_x_m=1.0 if seg==1 else 5.0,court_y_m=1.0, court_side='far',confidence=.9))
    src=tmp_path/'in.csv';pd.DataFrame(rows).to_csv(src,index=False)
    result=clean_positions(src,tmp_path/'out')
    assert result['speed_jump_review_points']==0
    assert result['eligible_points']==24

def test_jump_flagged_not_interpolated(tmp_path):
    rows=[]
    for i in range(15):
        rows.append(dict(frame_index=i,segment_id=1,track_id=1,timestamp_seconds=i/30,court_x_m=5.0 if i==7 else 1.0,court_y_m=1.0,court_side='far',confidence=.9))
    src=tmp_path/'in.csv';pd.DataFrame(rows).to_csv(src,index=False)
    result=clean_positions(src,tmp_path/'out')
    assert result['speed_jump_review_points']>=2
    df=pd.read_csv(tmp_path/'out'/'cleaned_positions.csv')
    assert pd.isna(df.loc[7,'clean_x_m'])
