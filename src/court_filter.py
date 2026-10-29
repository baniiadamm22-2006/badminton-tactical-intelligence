"""Court polygon calibration and robust ROI decisions in image pixel coordinates."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np


def validate_polygon(points: list[list[float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.shape != (4, 2) or not np.isfinite(arr).all():
        raise ValueError("Polygon harus berisi empat titik [x,y] yang valid.")
    if not cv2.isContourConvex(arr.reshape(-1, 1, 2)):
        raise ValueError("Urutan titik tidak konveks; gunakan TL, TR, BR, BL.")
    if cv2.contourArea(arr) < 100:
        raise ValueError("Luas polygon terlalu kecil.")
    return arr


class CourtFilter:
    def __init__(self, config_path: str | Path, margin_px: float = 12) -> None:
        if margin_px < 0:
            raise ValueError("margin_px harus >= 0.")
        self.config_path = Path(config_path)
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.width = int(config["width"])
        self.height = int(config["height"])
        self.polygon = validate_polygon(config["points"])
        self.margin_px = float(margin_px)

    def ensure_resolution(self, width: int, height: int) -> None:
        if (width, height) != (self.width, self.height):
            raise ValueError(
                f"Resolusi video {width}x{height} tidak cocok kalibrasi "
                f"{self.width}x{self.height}; kalibrasikan ulang dari video ini."
            )

    def accepts(self, detection: dict) -> bool:
        foot = (float(detection["foot_x"]), float(detection["foot_y"]))
        distance = cv2.pointPolygonTest(self.polygon, foot, True)
        return distance >= -self.margin_px

    def draw(self, frame) -> None:
        cv2.polylines(frame, [self.polygon.astype(np.int32)], True, (0, 255, 255), 2)


def calibrate(video_path: str | Path, output_path: str | Path,
              second: float = 10.0) -> None:
    """Click TL, TR, BR, BL of the outer court boundary on an original video frame."""
    if second < 0:
        raise ValueError("second harus >= 0.")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Gagal membuka video {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Gagal membaca frame kalibrasi. Pilih --second yang valid.")
    finally:
        capture.release()
    height, width = frame.shape[:2]
    points: list[list[int]] = []
    title = "Klik TL, TR, BR, BL | U undo | S save | ESC cancel"
    def onclick(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, onclick)
    try:
        while True:
            preview = frame.copy()
            for idx, (x, y) in enumerate(points):
                cv2.circle(preview, (x, y), 6, (0, 255, 255), -1)
                cv2.putText(preview, f"{idx+1}", (x+8, y-8),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
            if len(points) >= 2:
                cv2.polylines(preview, [np.array(points, dtype=np.int32)],
                              len(points) == 4, (0, 255, 255), 2)
            cv2.imshow(title, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord('q')):
                raise KeyboardInterrupt("Kalibrasi dibatalkan.")
            if key == ord('u') and points:
                points.pop()
            if key == ord('s') and len(points) == 4:
                validate_polygon(points)
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"width": width, "height": height,
                    "points": points, "frame_second": second}, indent=2),encoding="utf-8")
                print(f"Kalibrasi tersimpan: {path}")
                return
    finally:
        cv2.destroyAllWindows()
