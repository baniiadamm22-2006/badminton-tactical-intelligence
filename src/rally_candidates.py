"""Heuristic proposals for offline badminton rally annotation.

IMPORTANT: scene segments are NOT automatically rallies. Every proposal needs review.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def scene_intervals(frame_count: int, cuts: list[int]) -> list[tuple[int, int]]:
    if frame_count <= 0:
        raise ValueError('frame_count harus positif')
    bounds = [0] + sorted({int(x) for x in cuts if 0 < int(x) < frame_count}) + [frame_count]
    return [(a, b - 1) for a, b in zip(bounds, bounds[1:])]


def load_tracking(path: Path) -> dict[int, dict[str, set[int]]]:
    frames: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        required = {'frame_index', 'segment_id', 'track_id', 'court_side'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'CSV harus mempunyai kolom {sorted(required)}')
        for row in reader:
            side = row['court_side'].lower().strip()
            if side in {'far', 'near'}:
                frames[int(row['frame_index'])][side].add(int(row['track_id']))
    return frames


def smooth_boolean(values: list[bool], gap: int) -> list[bool]:
    """Fill short internal missing intervals, without extending at edges."""
    result = values[:]
    previous = None
    for index, active in enumerate(values):
        if not active:
            continue
        if previous is not None and 0 < index - previous - 1 <= gap:
            for j in range(previous + 1, index):
                result[j] = True
        previous = index
    return result


def contiguous_runs(values: list[bool]) -> list[tuple[int, int]]:
    starts = []
    begin = None
    for i, active in enumerate(values + [False]):
        if active and begin is None:
            begin = i
        elif not active and begin is not None:
            starts.append((begin, i - 1))
            begin = None
    return starts


def propose(frame_count: int, cuts: list[int], tracking: dict, fps: float,
            min_seconds: float = 1.5, fill_gap_frames: int = 12,
            min_coverage: float = .55) -> list[dict]:
    if fps <= 0 or min_seconds <= 0 or not (0 <= min_coverage <= 1):
        raise ValueError('Parameter tidak valid')
    proposals = []
    for scene_id, (left, right) in enumerate(scene_intervals(frame_count, cuts), start=1):
        both = [bool(tracking.get(i, {}).get('near')) and bool(tracking.get(i, {}).get('far'))
                for i in range(left, right + 1)]
        filled = smooth_boolean(both, fill_gap_frames)
        for start, end in contiguous_runs(filled):
            length = end - start + 1
            if length < min_seconds * fps:
                continue
            coverage = sum(both[start:end+1]) / length
            if coverage < min_coverage:
                continue
            start_frame, end_frame = left + start, left + end
            proposals.append({
                'proposal_id': len(proposals) + 1,
                'segment_id': scene_id,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_seconds': round(start_frame / fps, 3),
                'end_seconds': round(end_frame / fps, 3),
                'two_player_coverage': round(coverage, 4),
                'score': round(coverage, 4),
                'status': 'needs_review',
                'winner': 'unknown',
                'reason': 'Kontinuitas deteksi kedua sisi lapangan; tidak membuktikan shuttle aktif',
            })
    return proposals


def intersection_over_union(a: dict, b: dict) -> float:
    common = max(0, min(a['end_frame'], b['end_frame']) - max(a['start_frame'], b['start_frame']) + 1)
    union = (a['end_frame'] - a['start_frame'] + 1) + (b['end_frame'] - b['start_frame'] + 1) - common
    return common / union if union else 0.


def evaluate(proposals: list[dict], annotations: list[dict], threshold: float = .5) -> dict:
    """Greedy one-to-one interval IoU matching, descriptive only (same video)."""
    references = [r for r in annotations if r.get('annotation_status') == 'verified']
    pairs = sorted(((intersection_over_union(p, ref), pi, ri)
                    for pi, p in enumerate(proposals) for ri, ref in enumerate(references)), reverse=True)
    used_p, used_r, matches = set(), set(), []
    for overlap, pi, ri in pairs:
        if overlap < threshold or pi in used_p or ri in used_r:
            continue
        used_p.add(pi); used_r.add(ri)
        matches.append({'proposal_id': proposals[pi]['proposal_id'],
                        'rally_id': references[ri]['rally_id'], 'iou': round(overlap, 4)})
    precision = len(matches) / len(proposals) if proposals else 0.
    recall = len(matches) / len(references) if references else 0.
    return {'verified_reference_rallies': len(references), 'proposal_count': len(proposals),
            'matched_at_iou_0_5': len(matches), 'precision_at_iou_0_5': round(precision, 4),
            'recall_at_iou_0_5': round(recall, 4), 'matches': matches,
            'note': 'Evaluasi in-sample pada satu video; bukan validasi generalisasi model.'}


def write_outputs(output: Path, proposals: list[dict], evaluation: dict, video: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / 'rally_candidates.json').write_text(json.dumps({
        'source_video': video, 'method': 'two-sided track continuity heuristic',
        'warning': 'Kandidat wajib ditinjau. Scene cut dan dua pemain terlihat bukan bukti rally berlangsung.',
        'candidates': proposals}, indent=2, ensure_ascii=False), encoding='utf-8')
    (output / 'evaluation.json').write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding='utf-8')
    columns = ['proposal_id', 'segment_id', 'start_frame', 'end_frame', 'start_seconds',
               'end_seconds', 'two_player_coverage', 'score', 'status', 'winner', 'reason']
    with (output / 'rally_candidates.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader(); writer.writerows(proposals)


def main() -> None:
    p = argparse.ArgumentParser(description='BTI rally candidate baseline (offline)')
    p.add_argument('--tracks', type=Path, required=True)
    p.add_argument('--scenes', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('outputs/rally_candidates'))
    p.add_argument('--min-seconds', type=float, default=1.5)
    p.add_argument('--gap-frames', type=int, default=12)
    args = p.parse_args()
    refs = json.loads(args.annotations.read_text(encoding='utf-8'))
    scenes = json.loads(args.scenes.read_text(encoding='utf-8'))
    tracks = load_tracking(args.tracks)
    frame_count = int(refs['frame_count'])
    fps = float(refs['fps'])
    candidates = propose(frame_count, [c['frame_index'] for c in scenes['cuts']], tracks,
                         fps, args.min_seconds, args.gap_frames)
    result = evaluate(candidates, refs['rallies'])
    write_outputs(args.output, candidates, result, refs.get('video', ''))
    print(json.dumps({'candidate_count': len(candidates), 'evaluation': result,
                      'output': str(args.output)}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
