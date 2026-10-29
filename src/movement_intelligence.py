"""BTI Stage 05: audited location heatmaps and court zones (no speed estimates)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
REQUIRED = {'frame_index', 'segment_id', 'track_id', 'court_side',
            'clean_x_m', 'clean_y_m', 'eligible_for_movement'}


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])


def zone_for_position(y: float, side: str) -> str:
    if side not in ('far', 'near') or not np.isfinite(y):
        return 'unknown'
    if side == 'far':
        depth = y
    else:
        depth = COURT_LENGTH_M - y
    if not (0 <= depth <= COURT_LENGTH_M / 2):
        return 'out_of_side'
    third = (COURT_LENGTH_M / 2) / 3
    if depth < third:
        return 'rear'
    if depth < 2 * third:
        return 'middle'
    return 'front'


def load_positions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f'Kolom CSV tidak lengkap: {sorted(missing)}')
    df['eligible'] = truthy(df['eligible_for_movement'])
    for col in ('clean_x_m', 'clean_y_m'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['valid_location'] = (df['eligible'] &
                            df['clean_x_m'].between(0, COURT_WIDTH_M) &
                            df['clean_y_m'].between(0, COURT_LENGTH_M))
    df['zone'] = 'excluded'
    mask = df['valid_location']
    df.loc[mask, 'zone'] = [zone_for_position(y, side) for y, side in zip(
        df.loc[mask, 'clean_y_m'], df.loc[mask, 'court_side'])]
    # Only locations on the indicated half-court are used.
    df['valid_location'] &= df['zone'].isin(['rear', 'middle', 'front'])
    df.loc[~df['valid_location'], 'zone'] = 'excluded'
    return df


def court_axes(ax):
    ax.set_xlim(0, COURT_WIDTH_M)
    ax.set_ylim(COURT_LENGTH_M, 0)
    ax.set_aspect('equal')
    ax.set_xlabel('Court x (m)')
    ax.set_ylabel('Court y (m)')
    ax.plot([0, COURT_WIDTH_M, COURT_WIDTH_M, 0, 0],
            [0, 0, COURT_LENGTH_M, COURT_LENGTH_M, 0], color='black', linewidth=1)
    ax.axhline(COURT_LENGTH_M / 2, color='black', linewidth=1)
    ax.grid(alpha=.18)


def plot_track(group: pd.DataFrame, output_dir: Path, segment: int, track: int):
    valid = group[group['valid_location']].sort_values('frame_index')
    if valid.empty:
        return
    suffix = f'segment_{segment}_track_{track}'
    # 2D occupancy histogram: number of valid samples, NOT probability of tactical weakness.
    grid, _, _ = np.histogram2d(valid['clean_x_m'], valid['clean_y_m'],
                               bins=[np.linspace(0, COURT_WIDTH_M, 13),
                                     np.linspace(0, COURT_LENGTH_M, 27)])
    fig, ax = plt.subplots(figsize=(5, 8))
    img = ax.imshow(grid.T, origin='lower', extent=[0, COURT_WIDTH_M, COURT_LENGTH_M, 0],
                    aspect='equal', interpolation='nearest')
    court_axes(ax)
    ax.set_title(f'Heatmap | segment {segment}, track {track} (n={len(valid)})')
    fig.colorbar(img, ax=ax, label='Valid frame samples per bin')
    fig.tight_layout()
    fig.savefig(output_dir / f'{suffix}_heatmap.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 8))
    # Avoid connecting across missing/flagged frames, and do not bridge segment boundaries.
    chunks = (valid['frame_index'].diff().ne(1)).cumsum()
    for _, part in valid.groupby(chunks):
        ax.plot(part['clean_x_m'], part['clean_y_m'], linewidth=1.3)
    ax.scatter(valid['clean_x_m'].iloc[0], valid['clean_y_m'].iloc[0], s=35, marker='o')
    court_axes(ax)
    ax.set_title(f'Trajectory | segment {segment}, track {track}')
    fig.tight_layout()
    fig.savefig(output_dir / f'{suffix}_trajectory.png', dpi=150)
    plt.close(fig)


def analyze(input_path: str | Path, output_dir: str | Path, min_samples: int = 20) -> dict:
    if min_samples < 1:
        raise ValueError('min_samples harus >= 1')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = load_positions(input_path)
    df.to_csv(output / 'positions_with_zones.csv', index=False)
    rows = []
    report = {'input_rows': int(len(df)), 'valid_location_rows': int(df['valid_location'].sum()),
              'excluded_rows': int((~df['valid_location']).sum()),
              'minimum_valid_samples_for_visualization': min_samples,
              'court_width_m': COURT_WIDTH_M, 'court_length_m': COURT_LENGTH_M,
              'limitation': ('Frequencies represent valid sampled frames, not independent rallies, '
                             'movement speed, athlete identity, or tactical weakness.'),
              'tracks': []}
    for (seg, track), g in df.groupby(['segment_id', 'track_id'], sort=True):
        valid = g[g['valid_location']]
        cnt = valid['zone'].value_counts()
        n = len(valid)
        stats = {'segment_id': int(seg), 'track_id': int(track),
                 'court_side_labels': sorted(g['court_side'].dropna().unique().tolist()),
                 'total_rows': int(len(g)), 'valid_rows': int(n),
                 'excluded_rows': int(len(g)-n),
                 'sufficient_samples': bool(n >= min_samples),
                 'zones': {z: {'samples': int(cnt.get(z, 0)),
                               'percent_of_valid': round(100 * int(cnt.get(z, 0)) / n, 2) if n else 0.0}
                           for z in ('front', 'middle', 'rear')}}
        report['tracks'].append(stats)
        for zone, values in stats['zones'].items():
            rows.append({'segment_id': int(seg), 'track_id': int(track),
                         'zone': zone, **values, 'valid_rows': int(n),
                         'sufficient_samples': stats['sufficient_samples']})
        if n >= min_samples:
            plot_track(g, output, int(seg), int(track))
    pd.DataFrame(rows).to_csv(output / 'zone_statistics.csv', index=False)
    (output / 'movement_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description='BTI Movement Intelligence Stage 05')
    parser.add_argument('--input', type=Path, default=Path('outputs/trajectory_cleaning/cleaned_positions.csv'))
    parser.add_argument('--output', type=Path, default=Path('outputs/movement'))
    parser.add_argument('--min-samples', type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.output, args.min_samples), indent=2))


if __name__ == '__main__':
    main()
