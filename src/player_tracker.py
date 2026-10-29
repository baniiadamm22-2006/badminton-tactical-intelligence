"""ByteTrack baseline with court filtering and explicit track-ID diagnostics."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from src.scene_detector import SceneConfig, detect_cuts, segment_for_frame
from pathlib import Path

import cv2


COLUMNS = [
    'frame_index', 'timestamp_seconds', 'segment_id', 'track_id', 'court_side',
    'confidence', 'x1', 'y1', 'x2', 'y2', 'foot_x', 'foot_y',
]


def court_side(foot_x: float, foot_y: float, polygon) -> str:
    """Approximate near/far side by interpolating the polygon net-side midline."""
    tl, tr, br, bl = polygon
    left_middle_y = (float(tl[1]) + float(bl[1])) / 2
    right_middle_y = (float(tr[1]) + float(br[1])) / 2
    left_x = (float(tl[0]) + float(bl[0])) / 2
    right_x = (float(tr[0]) + float(br[0])) / 2
    ratio = 0.5 if abs(right_x-left_x) < 1 else max(0.0, min(1.0, (foot_x-left_x)/(right_x-left_x)))
    mid_y = left_middle_y + ratio*(right_middle_y-left_middle_y)
    return 'near' if foot_y >= mid_y else 'far'


def extract_tracks(result, court_filter, offset_x=0, offset_y=0):
    """Return only tracked person boxes whose feet pass court ROI validation."""
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return []
    # Avoid modifying PyTorch inference tensors returned by Ultralytics.
    # Work on an independent CPU NumPy copy instead.
    coords = boxes.xyxy.detach().cpu().numpy().copy()
    coords[:, [0, 2]] += offset_x
    coords[:, [1, 3]] += offset_y
    ids = boxes.id.int().cpu().tolist()
    scores = boxes.conf.cpu().tolist()
    detections = []
    for (x1, y1, x2, y2), track_id, score in zip(coords, ids, scores):
        detection = {
            'track_id': int(track_id), 'confidence': round(float(score), 5),
            'x1': round(float(x1), 2), 'y1': round(float(y1), 2),
            'x2': round(float(x2), 2), 'y2': round(float(y2), 2),
            'foot_x': round(float(x1+x2)/2, 2), 'foot_y': round(float(y2), 2),
        }
        if court_filter.accepts(detection):
            detection['court_side'] = court_side(detection['foot_x'], detection['foot_y'], court_filter.polygon)
            detections.append(detection)
    return detections


class PlayerTracker:
    def __init__(self, model_name='yolo26n.pt', confidence=0.15, imgsz=640, device='cpu', tracker='bytetrack.yaml'):
        if not 0 < confidence <= 1:
            raise ValueError('confidence harus pada (0,1]')
        from ultralytics import YOLO
        self.model_name = model_name
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device
        self.tracker = tracker

    def process_video(self, video_path, output_dir, court_filter, max_frames=300, save_video=True, crop_mode='court', crop_pad_x=35, crop_pad_y=100, scene_aware=True, pixel_threshold=0.016, histogram_threshold=0.06, min_segment_frames=12):
        if crop_mode not in ('court', 'none') or crop_pad_x < 0 or crop_pad_y < 0:
            raise ValueError('Crop mode atau padding tidak valid')
        if max_frames is not None and max_frames <= 0:
            raise ValueError('max_frames harus positif')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f'Tidak bisa membuka video: {video_path}')
        writer = None
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps <= 0 or width <= 0 or height <= 0:
                raise RuntimeError('Metadata video tidak valid')
            court_filter.ensure_resolution(width, height)
            if crop_mode == 'court':
                polygon = court_filter.polygon
                left = max(0, int(polygon[:, 0].min()) - crop_pad_x)
                top = max(0, int(polygon[:, 1].min()) - crop_pad_y)
                right = min(width, int(polygon[:, 0].max()) + crop_pad_x + 1)
                bottom = min(height, int(polygon[:, 1].max()) + crop_pad_y + 1)
            else:
                left, top, right, bottom = 0, 0, width, height
            if right <= left or bottom <= top:
                raise ValueError('Area crop kosong')
            if save_video:
                writer = cv2.VideoWriter(str(output_dir/'tracking.avi'), cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError('Gagal membuat output tracking.avi: codec MJPG tidak tersedia')
            scene_config = SceneConfig(pixel_threshold=pixel_threshold,
                                       histogram_threshold=histogram_threshold,
                                       min_segment_frames=min_segment_frames)
            scene_bounds = (left, top, right, bottom)
            scene_cuts = detect_cuts(video_path, scene_bounds, max_frames, scene_config) if scene_aware else []
            cut_frames = [item['frame_index'] for item in scene_cuts]
            (output_dir/'scene_cuts.json').write_text(json.dumps({
                'cuts': scene_cuts, 'scene_aware': scene_aware,
                'thresholds': {'pixel': pixel_threshold, 'histogram': histogram_threshold,
                               'min_segment_frames': min_segment_frames},
            }, indent=2), encoding='utf-8')
            counts_by_segment = defaultdict(Counter)
            counts = Counter()
            per_side = defaultdict(set)
            frames_empty = 0
            frames_multi = 0
            processed = 0
            with (output_dir/'tracks.csv').open('w', newline='', encoding='utf-8') as handle:
                csv_writer = csv.DictWriter(handle, fieldnames=COLUMNS)
                csv_writer.writeheader()
                while max_frames is None or processed < max_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    # Never reuse tracker state across an edited/cut video boundary.
                    if processed in cut_frames:
                        from ultralytics import YOLO
                        self.model = YOLO(self.model_name)
                    segment_id = segment_for_frame(processed, cut_frames)
                    # Frames within each segment remain consecutive for persist=True.
                    cropped_frame = frame[top:bottom, left:right]
                    result = self.model.track(
                        cropped_frame, persist=True, tracker=self.tracker, classes=[0],
                        conf=self.confidence, imgsz=self.imgsz, device=self.device,
                        verbose=False,
                    )[0]
                    # Convert crop-local boxes into original-video coordinates.
                    # Never mutate result.boxes.xyxy (an inference-mode tensor).
                    detected = extract_tracks(result, court_filter, left, top)
                    if not detected:
                        frames_empty += 1
                    if len(detected) > 2:
                        frames_multi += 1
                    for detection in detected:
                        track_id = detection['track_id']
                        counts[track_id] += 1
                        counts_by_segment[segment_id][track_id] += 1
                        per_side[detection['court_side']].add(track_id)
                        csv_writer.writerow({
                            'frame_index': processed,
                            'segment_id': segment_id,
                            'timestamp_seconds': round(processed/fps, 4),
                            **detection,
                        })
                    if writer is not None:
                        court_filter.draw(frame)
                        for item in detected:
                            x1, y1, x2, y2 = (int(item[k]) for k in ('x1','y1','x2','y2'))
                            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                            cv2.putText(frame, f"S{segment_id} T{item['track_id']} {item['court_side']}",
                                        (x1, max(20,y1-7)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,255,0), 2)
                        writer.write(frame)
                    processed += 1
            report = {
                'frames_processed': processed,
                'scene_aware': scene_aware,
                'scene_cuts': scene_cuts,
                'segments': [
                    {'segment_id': index + 1, 'start_frame': start,
                     'end_frame_inclusive': end - 1,
                     'track_ids': sorted(counts_by_segment[index + 1]),
                     'unique_track_ids': len(counts_by_segment[index + 1])}
                    for index, (start, end) in enumerate(zip([0] + cut_frames,
                        cut_frames + [processed])) if start < end
                ],
                'unique_segment_track_pairs': sum(len(c) for c in counts_by_segment.values()),
                'crop_bounds_xyxy': [left, top, right, bottom],
                'tracker_config': self.tracker,
                'model_confidence': self.confidence,
                'inference_imgsz': self.imgsz,
                'frames_without_players': frames_empty,
                'frames_more_than_two_candidates': frames_multi,
                'unique_track_ids': len(counts),
                'frames_per_track_id': dict(sorted(counts.items())),
                'track_ids_by_court_side': {k: sorted(v) for k,v in per_side.items()},
                'warning': 'Track ID hanya berlaku dalam segment_id yang sama. Cut tidak sama dengan batas rally. Player ID permanen belum ditetapkan.',
            }
            (output_dir/'tracking_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        if save_video:
            output_video = output_dir / 'tracking.avi'
            verify = cv2.VideoCapture(str(output_video))
            try:
                if not verify.isOpened():
                    raise RuntimeError(f'Video keluaran tidak bisa dibuka: {output_video}')
                recorded_frames = int(verify.get(cv2.CAP_PROP_FRAME_COUNT))
                ok, _ = verify.read()
                if not ok or recorded_frames == 0:
                    raise RuntimeError(f'Video keluaran kosong/rusak: {output_video}')
            finally:
                verify.release()
        return report
