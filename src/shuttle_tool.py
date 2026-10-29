"""Shuttlecock ground-truth annotation, prediction normalization, evaluation.

This is NOT an automatic detector. It supports manual annotations and reading
TrackNet-style predictions produced by a separately installed model.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

FIELDS = ('frame_index', 'timestamp_seconds', 'visible', 'x', 'y', 'source')


def save_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item['frame_index'])):
            writer.writerow({key: row.get(key, '') for key in FIELDS})


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8-sig') as file:
        return list(csv.DictReader(file))


def normalize_prediction_csv(source: Path, output: Path, fps: float, width: int, height: int) -> int:
    """Convert TrackNet-like CSV containing Frame, Visibility, X, Y into BTI format.

    Assumes coordinates are in original video pixels. Never rescales silently.
    """
    if fps <= 0:
        raise ValueError('FPS must be positive')
    with source.open(newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        columns = {field.lower().strip(): field for field in (reader.fieldnames or [])}
        frame_key = columns.get('frame') or columns.get('frame_index')
        visibility_key = columns.get('visibility') or columns.get('visible')
        x_key = columns.get('x')
        y_key = columns.get('y')
        if not (frame_key and visibility_key and x_key and y_key):
            raise ValueError('Prediction CSV requires Frame, Visibility, X, Y columns')
        rows = []
        previous = -1
        for raw in reader:
            index = int(raw[frame_key])
            if index <= previous:
                raise ValueError('Frame numbers must be unique and increasing')
            previous = index
            visible = int(float(raw[visibility_key])) != 0
            x, y = float(raw[x_key]), float(raw[y_key])
            if visible and not (0 <= x < width and 0 <= y < height):
                raise ValueError(f'Coordinates out of bounds at frame {index}: {(x, y)}')
            rows.append({'frame_index': index, 'timestamp_seconds': round(index / fps, 6),
                         'visible': int(visible), 'x': x if visible else '',
                         'y': y if visible else '', 'source': 'model'})
    save_rows(output, rows)
    return len(rows)


def evaluate(reference: Path, prediction: Path, tolerance_px: float = 8.0) -> dict:
    """Evaluate exactly on manually labeled frames (including invisible frames)."""
    if tolerance_px <= 0:
        raise ValueError('tolerance_px must be positive')
    ref_rows = {int(row['frame_index']): row for row in load_rows(reference)}
    pred_rows = {int(row['frame_index']): row for row in load_rows(prediction)}
    if not ref_rows:
        raise ValueError('Reference annotations are empty')
    tp = fp = fn = tn = 0
    distance_sum = 0.0
    localization_pairs = 0
    for frame, gt in ref_rows.items():
        pred = pred_rows.get(frame)
        gt_visible = int(gt['visible']) == 1
        pred_visible = pred is not None and int(pred['visible']) == 1
        if gt_visible and pred_visible:
            dx = float(gt['x']) - float(pred['x'])
            dy = float(gt['y']) - float(pred['y'])
            distance = (dx * dx + dy * dy) ** 0.5
            distance_sum += distance
            localization_pairs += 1
            if distance <= tolerance_px:
                tp += 1
            else:
                fp += 1
                fn += 1
        elif gt_visible:
            fn += 1
        elif pred_visible:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'annotated_frames': len(ref_rows), 'true_positive': tp, 'false_positive': fp,
            'false_negative': fn, 'true_negative': tn,
            'precision': round(precision, 4), 'recall': round(recall, 4),
            'f1': round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
            'mean_localization_error_px_when_both_visible': round(distance_sum / localization_pairs, 3) if localization_pairs else None,
            'tolerance_px': tolerance_px,
            'note': 'Metrics measured only on manually annotated frames; include negative/invisible labels.'}


def annotate(video: Path, output: Path, start_frame: int, interval: int, max_samples: int) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Unable to read video: {video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or interval <= 0 or max_samples <= 0 or not (0 <= start_frame < total):
        cap.release()
        raise ValueError('Invalid video metadata or annotation parameters')
    annotated = {int(r['frame_index']): r for r in load_rows(output)}
    state = {'click': None}
    window = 'BTI Shuttle Ground Truth | Click=visible | N=invisible | S=save | Q=quit'

    def onclick(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            state['click'] = (x, y)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, onclick)
    cursor = start_frame
    shown = 0
    try:
        while cursor < total and shown < max_samples:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cursor)
            ok, frame = cap.read()
            if not ok:
                break
            state['click'] = None
            original = frame.copy()
            cv2.putText(frame, f'Frame {cursor}/{total-1}  {cursor/fps:.2f}s | click shuttle; N not visible; Q quit',
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 255, 255), 1)
            cv2.imshow(window, frame)
            while True:
                key = cv2.waitKey(30) & 0xFF
                if state['click'] is not None:
                    x, y = state['click']
                    annotated[cursor] = {'frame_index': cursor, 'timestamp_seconds': round(cursor/fps, 6),
                                         'visible': 1, 'x': x, 'y': y, 'source': 'manual'}
                    cv2.circle(original, (x, y), 7, (0, 255, 255), 2)
                    cv2.imshow(window, original)
                    break
                if key == ord('n'):
                    annotated[cursor] = {'frame_index': cursor, 'timestamp_seconds': round(cursor/fps, 6),
                                         'visible': 0, 'x': '', 'y': '', 'source': 'manual'}
                    break
                if key in (ord('q'), 27):
                    save_rows(output, list(annotated.values()))
                    print(f'Saved {len(annotated)} annotations: {output}')
                    return
                if key == ord('s'):
                    save_rows(output, list(annotated.values()))
                    print(f'Saved: {output}')
            shown += 1
            cursor += interval
    finally:
        save_rows(output, list(annotated.values()))
        cap.release()
        cv2.destroyAllWindows()
    print(f'Saved {len(annotated)} annotations: {output}')


def main():
    parser = argparse.ArgumentParser(description='BTI Stage 07 shuttlecock annotation/evaluation')
    sub = parser.add_subparsers(dest='command', required=True)
    annotate_parser = sub.add_parser('annotate')
    annotate_parser.add_argument('--video', type=Path, default=Path('data/raw/match_001.mp4'))
    annotate_parser.add_argument('--output', type=Path, default=Path('data/annotations/shuttle_ground_truth.csv'))
    annotate_parser.add_argument('--start-frame', type=int, default=0)
    annotate_parser.add_argument('--interval', type=int, default=5)
    annotate_parser.add_argument('--max-samples', type=int, default=80)
    import_parser = sub.add_parser('import-tracknet')
    import_parser.add_argument('--input', type=Path, required=True)
    import_parser.add_argument('--output', type=Path, default=Path('outputs/shuttle/predictions.csv'))
    import_parser.add_argument('--fps', type=float, required=True)
    import_parser.add_argument('--width', type=int, required=True)
    import_parser.add_argument('--height', type=int, required=True)
    eval_parser = sub.add_parser('evaluate')
    eval_parser.add_argument('--reference', type=Path, default=Path('data/annotations/shuttle_ground_truth.csv'))
    eval_parser.add_argument('--predictions', type=Path, required=True)
    eval_parser.add_argument('--output', type=Path, default=Path('outputs/shuttle/evaluation.json'))
    eval_parser.add_argument('--tolerance-px', type=float, default=8.0)
    args = parser.parse_args()
    if args.command == 'annotate':
        annotate(args.video, args.output, args.start_frame, args.interval, args.max_samples)
    elif args.command == 'import-tracknet':
        count = normalize_prediction_csv(args.input, args.output, args.fps, args.width, args.height)
        print(f'Imported {count} predictions: {args.output}')
    else:
        report = evaluate(args.reference, args.predictions, args.tolerance_px)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
