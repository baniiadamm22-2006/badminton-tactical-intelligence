"""Frame-by-frame person detection; no identity tracking or badminton-player attribution."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


CSV_COLUMNS = [
    "frame_index", "timestamp_seconds", "detection_index", "confidence",
    "x1", "y1", "x2", "y2", "foot_x", "foot_y", "inside_court",
]


class PlayerDetector:
    def __init__(self, model_name: str = "yolo26n.pt", confidence: float = 0.35,
                 device: str = "cpu", image_size: int = 640) -> None:
        if not 0 < confidence <= 1:
            raise ValueError("confidence harus dalam rentang (0,1].")
        if image_size <= 0:
            raise ValueError("image_size harus > 0.")
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.device = device
        self.image_size = image_size

    def detect(self, frame) -> list[dict]:
        predictions = self.model.predict(
            source=frame,
            classes=[0],  # COCO: 0 = person
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        detections = []
        for result in predictions:
            if result.boxes is None:
                continue
            coords = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            for box, score in zip(coords, confidences):
                x1, y1, x2, y2 = (float(value) for value in box)
                detections.append({
                    "confidence": round(float(score), 5),
                    "x1": round(x1, 2), "y1": round(y1, 2),
                    "x2": round(x2, 2), "y2": round(y2, 2),
                    "foot_x": round((x1 + x2) / 2, 2),
                    "foot_y": round(y2, 2),
                })
        return detections

    def process_video(self, video_path: str | Path, output_dir: str | Path,
                      max_frames: int | None = 300, stride: int = 1,
                      save_video: bool = True, court_filter=None) -> dict:
        if stride < 1:
            raise ValueError("stride harus >= 1")
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames harus >= 1")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Gagal membuka video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError("FPS atau resolusi video tidak valid")

        if court_filter is not None:
            court_filter.ensure_resolution(width, height)

        writer = None
        if save_video:
            output_video = output_dir / "detections.mp4"
            writer = cv2.VideoWriter(
                str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride,
                (width, height),
            )
            if not writer.isOpened():
                capture.release()
                raise RuntimeError("Gagal membuat output video (codec mp4v). Gunakan --no-video jika codec tidak tersedia.")

        frame_index = 0
        processed = 0
        total_detections = 0
        total_accepted = 0
        csv_path = output_dir / "detections.csv"
        try:
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                csv_writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
                csv_writer.writeheader()
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    current_index = frame_index
                    frame_index += 1
                    if current_index % stride != 0:
                        continue
                    detections = self.detect(frame)
                    accepted = 0
                    if writer is not None and court_filter is not None:
                        court_filter.draw(frame)
                    for detection_index, detection in enumerate(detections):
                        keep = court_filter is None or court_filter.accepts(detection)
                        accepted += int(keep)
                        csv_writer.writerow({
                            "frame_index": current_index,
                            "timestamp_seconds": round(current_index / fps, 4),
                            "detection_index": detection_index,
                            **detection,
                            "inside_court": int(keep),
                        })
                        if writer is not None:
                            start = (int(detection["x1"]), int(detection["y1"]))
                            end = (int(detection["x2"]), int(detection["y2"]))
                            color = (0, 255, 0) if keep else (0, 0, 255)
                            cv2.rectangle(frame, start, end, color, 2)
                            label = "PLAYER CANDIDATE" if keep else "OUTSIDE ROI"
                            cv2.putText(frame, f"{label} {detection['confidence']:.2f}",
                                        (start[0], max(18, start[1] - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if writer is not None:
                        writer.write(frame)
                    total_detections += len(detections)
                    total_accepted += accepted
                    processed += 1
                    if processed % 50 == 0:
                        print(f"Diproses {processed} frame | {total_detections} deteksi")
                    if max_frames is not None and processed >= max_frames:
                        break
        finally:
            capture.release()
            if writer is not None:
                writer.release()
        return {
            "processed_frames": processed,
            "total_person_detections": total_detections,
            "total_player_candidates": total_accepted,
            "total_rejected": total_detections - total_accepted,
            "csv_path": str(csv_path),
            "annotated_video": str(output_dir / "detections.mp4") if save_video else None,
            "note": "Kandidat berdasarkan ROI, bukan identitas pemain terverifikasi; belum tracking.",
        }
