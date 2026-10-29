"""Stage 04: project validated pixel foot-points onto a metric court plane."""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

# BWF doubles outer rectangle: width 6.10 m, length 13.40 m.
COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40


def build_homography(config: dict) -> np.ndarray:
    points = np.asarray(config['points'], dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError('Empat titik kalibrasi wajib valid.')
    if not cv2.isContourConvex(points.reshape(-1, 1, 2)) or cv2.contourArea(points) < 100:
        raise ValueError('Urutan harus TL, TR, BR, BL dan membentuk poligon valid.')
    destination = np.array([
        [0, 0], [COURT_WIDTH_M, 0],
        [COURT_WIDTH_M, COURT_LENGTH_M], [0, COURT_LENGTH_M],
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points, destination)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError('Homography tidak valid.')
    return matrix


def project_point(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    projected = cv2.perspectiveTransform(
        np.array([[[x, y]]], dtype=np.float32), matrix
    )[0, 0]
    return float(projected[0]), float(projected[1])


def map_tracking_csv(tracks_path: str | Path, calibration_path: str | Path,
                     output_dir: str | Path, allow_outside: bool = False) -> dict:
    source = Path(tracks_path)
    config = json.loads(Path(calibration_path).read_text(encoding='utf-8'))
    matrix = build_homography(config)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    with source.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        required = {'frame_index', 'track_id', 'foot_x', 'foot_y'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f'Kolom CSV wajib: {sorted(required)}')
        input_rows = list(reader)
        input_columns = list(reader.fieldnames or [])
    columns = input_columns + [c for c in ('segment_id', 'court_x_m', 'court_y_m', 'inside_court', 'mapping_status') if c not in input_columns]
    counts = defaultdict(int)
    paths: dict[tuple[int, int], list[tuple[int, float, float]]] = defaultdict(list)
    output_rows = []
    for row in input_rows:
        segment = int(row.get('segment_id') or 1)
        track = int(row['track_id'])
        x, y = project_point(matrix, float(row['foot_x']), float(row['foot_y']))
        inside = -1e-3 <= x <= COURT_WIDTH_M + 1e-3 and -1e-3 <= y <= COURT_LENGTH_M + 1e-3
        # No clamping: points outside court should remain visible as invalid or uncertain.
        row['segment_id'] = segment
        row['court_x_m'] = round(x, 4)
        row['court_y_m'] = round(y, 4)
        row['inside_court'] = int(inside)
        row['mapping_status'] = 'inside' if inside else 'outside'
        counts[row['mapping_status']] += 1
        if inside or allow_outside:
            paths[(segment, track)].append((int(row['frame_index']), x, y))
        output_rows.append(row)
    with (folder / 'court_positions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output_rows)
    render_court(paths, folder / 'court_map.png')
    report = {
        'input_rows': len(output_rows), 'inside_court': counts['inside'],
        'outside_court': counts['outside'],
        'unique_segment_track_pairs': len({(int(r['segment_id']), int(r['track_id'])) for r in output_rows}),
        'court_width_m': COURT_WIDTH_M, 'court_length_m': COURT_LENGTH_M,
        'coordinate_system': 'origin=far-left corner; x across court; y toward near baseline',
        'calibration_image_size': [config.get('width'), config.get('height')],
        'warning': 'Track IDs are segment-local. Foot points approximate ground contact. Physical accuracy requires independent landmark validation. Do not connect across cuts.'
    }
    (folder / 'mapping_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def render_court(paths: dict, target: Path) -> None:
    scale = 65
    margin = 48
    width = round(COURT_WIDTH_M * scale) + 2 * margin
    height = round(COURT_LENGTH_M * scale) + 2 * margin
    canvas = np.full((height, width, 3), (49, 102, 63), dtype=np.uint8)

    def pt(x: float, y: float) -> tuple[int, int]:
        return round(margin + x * scale), round(margin + y * scale)

    white = (235, 235, 235)
    cv2.rectangle(canvas, pt(0, 0), pt(COURT_WIDTH_M, COURT_LENGTH_M), white, 2)
    # Singles sidelines; doubles outer lines are used for homography.
    cv2.line(canvas, pt(.46, 0), pt(.46, COURT_LENGTH_M), white, 1)
    cv2.line(canvas, pt(5.64, 0), pt(5.64, COURT_LENGTH_M), white, 1)
    cv2.line(canvas, pt(0, 6.70), pt(COURT_WIDTH_M, 6.70), (0, 220, 240), 2)
    cv2.line(canvas, pt(0, 1.98), pt(COURT_WIDTH_M, 1.98), white, 1)
    cv2.line(canvas, pt(0, 11.42), pt(COURT_WIDTH_M, 11.42), white, 1)
    cv2.line(canvas, pt(0, 4.72), pt(COURT_WIDTH_M, 4.72), white, 1)
    cv2.line(canvas, pt(0, 8.68), pt(COURT_WIDTH_M, 8.68), white, 1)
    colors = [(20, 220, 255), (255, 130, 90), (210, 110, 235), (130, 230, 130)]
    for index, (key, coords) in enumerate(sorted(paths.items())):
        coords.sort(key=lambda item: item[0])
        color = colors[index % len(colors)]
        for (prev_frame, x1, y1), (frame, x2, y2) in zip(coords, coords[1:]):
            # Never link over skipped frames / tracker gaps.
            if frame == prev_frame + 1:
                cv2.line(canvas, pt(x1, y1), pt(x2, y2), color, 1)
        if coords:
            cv2.circle(canvas, pt(coords[-1][1], coords[-1][2]), 4, color, -1)
            cv2.putText(canvas, f'S{key[0]} T{key[1]}', (6, 16 + 17 * index),
                        cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1)
    if not cv2.imwrite(str(target), canvas):
        raise RuntimeError('Tidak dapat menyimpan visualisasi court_map.png')
