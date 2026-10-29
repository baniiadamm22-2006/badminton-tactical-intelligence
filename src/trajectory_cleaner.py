"""Audit and smooth badminton court trajectories without joining scene cuts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REQUIRED = {'frame_index', 'timestamp_seconds', 'segment_id', 'track_id', 'court_x_m', 'court_y_m', 'court_side', 'confidence'}

def clean_positions(source: str | Path, out_dir: str | Path, *, max_speed: float = 12.0, min_track_frames: int = 10, median_window: int = 5) -> dict:
    if max_speed <= 0 or min_track_frames < 1 or median_window < 3 or median_window % 2 != 1:
        raise ValueError('Parameters invalid: speed > 0, track frames >= 1, odd median window >= 3')
    df = pd.read_csv(source)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f'CSV missing required columns: {sorted(missing)}')
    df = df.copy().sort_values(['segment_id','track_id','frame_index']).reset_index(drop=True)
    if df.duplicated(['segment_id','track_id','frame_index']).any():
        raise ValueError('Duplicate segment_id/track_id/frame_index records')
    df['quality_flag'] = 'ok'
    df['step_speed_mps'] = np.nan
    df['clean_x_m'] = np.nan
    df['clean_y_m'] = np.nan
    df['eligible_for_movement'] = False
    details = []
    for (segment, track), idx in df.groupby(['segment_id','track_id'], sort=True).groups.items():
        part = df.loc[idx].sort_values('frame_index')
        ids = part.index.to_numpy()
        xy = part[['court_x_m','court_y_m']].to_numpy(dtype=float)
        times = part['timestamp_seconds'].to_numpy(dtype=float)
        frames = part['frame_index'].to_numpy(dtype=int)
        if not np.isfinite(xy).all() or not np.isfinite(times).all():
            raise ValueError(f'Non-finite coordinates/times in segment {segment}, track {track}')
        if len(times)>1 and np.any(np.diff(times)<=0):
            raise ValueError(f'Non-increasing timestamps in segment {segment}, track {track}')
        if len(part) < min_track_frames:
            df.loc[ids,'quality_flag'] = 'short_track_review'
            details.append({'segment_id':int(segment),'track_id':int(track),'rows':len(part),'status':'short_track_review','large_steps':0})
            continue
        flags = np.zeros(len(part), dtype=bool)
        speeds = np.full(len(part), np.nan)
        for i in range(1,len(part)):
            if frames[i]-frames[i-1] != 1:
                continue
            speeds[i] = float(np.linalg.norm(xy[i]-xy[i-1])/(times[i]-times[i-1]))
            if speeds[i] > max_speed:
                flags[i] = True
                flags[i-1] = True  # ambiguous which endpoint is erroneous
        df.loc[ids,'step_speed_mps'] = speeds
        df.loc[ids[flags],'quality_flag'] = 'speed_jump_review'
        # Only smooth contiguous, unflagged stretches, never across cuts, gaps or flagged samples.
        groups = []
        run = []
        for i in range(len(part)):
            if flags[i] or (run and frames[i] != frames[run[-1]]+1):
                if run: groups.append(run)
                run=[]
            if not flags[i]: run.append(i)
        if run: groups.append(run)
        for run in groups:
            original = pd.DataFrame(xy[run], columns=['x','y'])
            smoothed = original.rolling(window=median_window, center=True, min_periods=1).median().to_numpy()
            df.loc[ids[run],'clean_x_m'] = smoothed[:,0]
            df.loc[ids[run],'clean_y_m'] = smoothed[:,1]
            df.loc[ids[run],'eligible_for_movement'] = True
        details.append({'segment_id':int(segment),'track_id':int(track),'rows':len(part),'status':'review_required' if flags.any() else 'candidate','large_steps':int(np.sum(speeds>max_speed)), 'flagged_points':int(flags.sum())})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    df.to_csv(out_dir/'cleaned_positions.csv',index=False,float_format='%.5f')
    report = {'input_rows':int(len(df)), 'eligible_points':int(df.eligible_for_movement.sum()), 'speed_jump_review_points':int((df.quality_flag=='speed_jump_review').sum()), 'short_track_review_points':int((df.quality_flag=='short_track_review').sum()), 'max_speed_mps_screening':max_speed, 'median_window':median_window, 'important':'Screening threshold is heuristic, not a physiological limit. Cleaned positions do not establish athlete identity, ground-truth accuracy or reliable distance totals.', 'tracks':details}
    (out_dir/'cleaning_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig, axes = plt.subplots(1,2,figsize=(10,8),sharex=True,sharey=True)
    for ax,colx,coly,name in [(axes[0],'court_x_m','court_y_m','Raw court positions'),(axes[1],'clean_x_m','clean_y_m','Valid segments after screening')]:
        for (seg,tr), g in df.groupby(['segment_id','track_id']):
            if len(g)<min_track_frames: continue
            # split path at missing points and missing frame indexes
            g=g.sort_values('frame_index')
            valid = g[colx].notna() & g[coly].notna()
            breaks=(g.frame_index.diff().fillna(1)!=1)|(~valid)
            for _,h in g[valid].groupby(breaks.cumsum()[valid]):
                if len(h)>1: ax.plot(h[colx],h[coly],linewidth=0.9,alpha=.8,label=f'S{seg} T{tr}')
        ax.set_xlim(0,6.1);ax.set_ylim(13.4,0);ax.set_aspect('equal');ax.set_title(name);ax.axhline(6.7,color='gray',lw=1);ax.grid(alpha=.15)
    axes[0].set_ylabel('Court y (m)');fig.tight_layout()
    fig.savefig(out_dir/'trajectory_comparison.png',dpi=170);plt.close(fig)
    return report

def main():
    parser=argparse.ArgumentParser(description='Court trajectory quality screening')
    parser.add_argument('--input',default='outputs/court_mapping/court_positions.csv')
    parser.add_argument('--output',default='outputs/trajectory_cleaning')
    parser.add_argument('--max-speed',type=float,default=12.0)
    parser.add_argument('--min-track-frames',type=int,default=10)
    parser.add_argument('--median-window',type=int,default=5)
    a=parser.parse_args()
    print(json.dumps(clean_positions(a.input,a.output,max_speed=a.max_speed,min_track_frames=a.min_track_frames,median_window=a.median_window),indent=2))

if __name__=='__main__':main()
