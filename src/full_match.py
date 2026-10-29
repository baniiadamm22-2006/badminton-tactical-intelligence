"""Offline whole-match tracking with scene-aware tracker warm-up and durable chunks.

A resume replays the active scene to restore ByteTrack state; completed scene
segments never replay. Chunk boundary never resets the tracker.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from collections import Counter

import cv2

from src.court_filter import CourtFilter
from src.player_tracker import extract_tracks
from src.scene_detector import SceneConfig, detect_cuts

FIELDS = ["frame_index", "timestamp_seconds", "segment_id", "track_id",
          "court_side", "confidence", "x1", "y1", "x2", "y2", "foot_x", "foot_y"]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def chunk_file(directory: Path, index: int) -> Path:
    return directory / f"chunk_{index:06d}.csv"


def write_chunk(directory: Path, index: int, rows: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = chunk_file(directory, index)
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def assemble_csv(directory: Path, chunks: int, output_path: Path) -> int:
    """Assemble saved chunks without loading whole-match rows into RAM."""
    count = 0
    temp = output_path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as dest:
        writer = csv.DictWriter(dest, fieldnames=FIELDS)
        writer.writeheader()
        for idx in range(chunks):
            with chunk_file(directory, idx).open(newline="", encoding="utf-8") as src:
                for row in csv.DictReader(src):
                    writer.writerow(row)
                    count += 1
    os.replace(temp, output_path)
    return count


def segment_start_for(frame: int, cuts: list[int]) -> int:
    return max((cut for cut in [0, *cuts] if cut <= frame), default=0)


def segment_id_for(frame: int, cuts: list[int]) -> int:
    return 1 + sum(c <= frame for c in cuts)


def run_full_match(video: str | Path, court: str | Path, output: str | Path,
                   *, model: str = "yolo26n.pt", tracker: str = "config/bytetrack_badminton.yaml",
                   device: str = "cpu", conf: float = .05, imgsz: int = 960,
                   chunk_frames: int = 300, margin: float = 12,
                   crop_pad_x: int = 35, crop_pad_y: int = 100,
                   resume: bool = False, max_frames: int | None = None,
                   model_factory=None) -> dict:
    if chunk_frames < 1 or not (0 < conf <= 1) or imgsz < 32:
        raise ValueError("chunk_frames/conf/imgsz tidak valid")
    video, court, output = Path(video), Path(court), Path(output)
    if not video.is_file():
        raise FileNotFoundError(video)
    if not court.is_file():
        raise FileNotFoundError(court)
    if not Path(tracker).is_file():
        raise FileNotFoundError(f"Konfigurasi ByteTrack tidak ditemukan: {tracker}")
    output.mkdir(parents=True, exist_ok=True)
    chunks_dir = output / "chunks"
    checkpoint_path = output / "checkpoint.json"
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("Gagal membuka video")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if min(width, height, fps) <= 0:
            raise RuntimeError("Metadata video tidak valid")
        filter_ = CourtFilter(court, margin_px=margin)
        filter_.ensure_resolution(width, height)
        polygon = filter_.polygon
        left = max(0, int(polygon[:, 0].min()) - crop_pad_x)
        top = max(0, int(polygon[:, 1].min()) - crop_pad_y)
        right = min(width, int(polygon[:, 0].max()) + crop_pad_x + 1)
        bottom = min(height, int(polygon[:, 1].max()) + crop_pad_y + 1)
        if right <= left or bottom <= top:
            raise ValueError("ROI crop kosong")
        limit = min(max_frames, total_reported) if max_frames is not None else total_reported
        if limit < 1:
            raise ValueError("max_frames harus positif")
        signature = hashlib.sha256(json.dumps({
            "video": str(video.resolve()), "size": video.stat().st_size,
            "mtime": video.stat().st_mtime_ns, "court": court.read_text(encoding="utf-8"),
            "tracker_content": Path(tracker).read_text(encoding="utf-8"),
            "model": model, "device": device, "conf": conf, "imgsz": imgsz,
            "chunk_frames": chunk_frames, "margin": margin,
            "crop_pad_x": crop_pad_x, "crop_pad_y": crop_pad_y, "limit": limit,
        }, sort_keys=True).encode()).hexdigest()

        # Prepass is independent of inference and allows deterministic scene IDs on resume.
        cfg = SceneConfig()
        cuts_metadata = detect_cuts(video, (left, top, right, bottom), limit, cfg)
        cuts = [int(item["frame_index"]) for item in cuts_metadata]
        atomic_json(output / "scene_cuts.json", {"cuts": cuts_metadata, "thresholds": cfg.__dict__})

        if resume:
            if not checkpoint_path.exists():
                raise FileNotFoundError("Checkpoint belum ada. Jalankan tanpa --resume terlebih dahulu.")
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("signature") != signature:
                raise ValueError("Parameter/video/kalibrasi berubah; resume ditolak. Gunakan output baru.")
            next_frame = int(checkpoint["next_frame"])
            completed = int(checkpoint["chunks_completed"])
            for idx in range(completed):
                if not chunk_file(chunks_dir, idx).exists():
                    raise RuntimeError(f"Chunk hilang: {idx}")
        else:
            if checkpoint_path.exists() or any(chunks_dir.glob("chunk_*.csv")):
                raise FileExistsError("Output sudah berisi pekerjaan lama. Gunakan --resume atau --output baru.")
            next_frame = 0
            completed = 0
            atomic_json(checkpoint_path, {"signature": signature, "next_frame": 0,
                                          "chunks_completed": 0, "finished": False})

        if model_factory is None:
            from ultralytics import YOLO
            model_factory = lambda: YOLO(model)
        active_segment = segment_id_for(next_frame, cuts)
        restart_from = segment_start_for(next_frame, cuts)
        cap.set(cv2.CAP_PROP_POS_FRAMES, restart_from)
        engine = model_factory()
        rows: list[dict] = []
        pending_frames = 0
        frame_no = restart_from
        last_seen = restart_from - 1
        while frame_no < limit:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_no in cuts and frame_no != restart_from:
                engine = model_factory()  # hard cut: new ByteTrack state
            segment_id = segment_id_for(frame_no, cuts)
            result = engine.track(frame[top:bottom, left:right], persist=True,
                                  tracker=tracker, classes=[0], conf=conf,
                                  imgsz=imgsz, device=device, verbose=False)[0]
            detections = extract_tracks(result, filter_, left, top)
            if frame_no >= next_frame:
                for det in detections:
                    rows.append({"frame_index": frame_no,
                                 "timestamp_seconds": round(frame_no/fps, 6),
                                 "segment_id": segment_id, **det})
            last_seen = frame_no
            frame_no += 1
            if frame_no > next_frame:
                pending_frames += 1
            if pending_frames >= chunk_frames:
                write_chunk(chunks_dir, completed, rows)
                completed += 1
                next_frame = frame_no
                rows = []
                pending_frames = 0
                atomic_json(checkpoint_path, {"signature": signature,
                            "next_frame": next_frame, "chunks_completed": completed,
                            "finished": False})
                print(f"[BTI] frame={frame_no}/{limit} chunk={completed} segmen={segment_id}", flush=True)
        if pending_frames:
            write_chunk(chunks_dir, completed, rows)
            completed += 1
            next_frame = frame_no
            atomic_json(checkpoint_path, {"signature": signature,
                        "next_frame": next_frame, "chunks_completed": completed,
                        "finished": False})
        # If a video ends before metadata count, checkpoint at the observed EOF.
        elif frame_no > next_frame:
            next_frame = frame_no
        row_count = assemble_csv(chunks_dir, completed, output / "tracks.csv")
        report = {
            "processed_frames": next_frame,
            "reported_video_frames": total_reported,
            "fps": fps, "cut_frames": cuts,
            "segments": len(cuts) + 1,
            "completed_chunks": completed,
            "tracking_rows": row_count,
            "crop_bounds_xyxy": [left, top, right, bottom],
            "finished": next_frame >= limit or last_seen + 1 < limit,
            "warning": "ID track hanya valid di segmen terkait. Scene cut bukan batas rally. Kalibrasi harus ditinjau jika sudut kamera berubah.",
        }
        atomic_json(output / "tracking_report.json", report)
        atomic_json(checkpoint_path, {"signature": signature, "next_frame": next_frame,
                                     "chunks_completed": completed, "finished": report["finished"]})
        return report
    finally:
        cap.release()
