# Stage 03.1 — PyTorch Inference Tensor Fix

Bug: `RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed`.

Cause: `result.boxes.xyxy` is owned by Ultralytics and is an inference tensor; it must not be mutated in place.

Resolution: `extract_tracks(result, court_filter, offset_x, offset_y)` takes an independent NumPy copy and offsets that copy before applying Court ROI and exporting original-video coordinates. `main.py` remains unchanged.

Copy `src/player_tracker.py` from this package over your project file (or copy the full project code while preserving data and video). Restart Python, then run:

```powershell
python -m pytest -q
python main.py track --device cpu --max-frames 300 --confidence 0.05 --imgsz 960 --tracker config/bytetrack_badminton.yaml --crop-mode court --output outputs/tracking_crop
```

Verify `outputs/tracking_crop/tracking.avi`, `tracks.csv`, and `tracking_report.json`. This package was unit-tested without running actual YOLO inference on the user's video.
