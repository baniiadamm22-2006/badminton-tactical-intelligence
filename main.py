"""Command-line interface for badminton computer vision stages 01-02.5."""
import argparse
import json
from pathlib import Path
from src.video_processor import VideoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Badminton Tactical Intelligence")
    parser.add_argument("--video", type=Path, default=Path("data/raw/match_001.mp4"))
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--output", type=Path, default=Path("outputs/video_metadata.json"))
    extract = sub.add_parser("extract")
    extract.add_argument("--output", type=Path, default=Path("data/frames/match_001"))
    extract.add_argument("--interval", type=float, default=1.0)
    extract.add_argument("--max-frames", type=int, default=120)
    cal = sub.add_parser("calibrate", help="Click 4 court corners on an original video frame")
    cal.add_argument("--second", type=float, default=10)
    cal.add_argument("--output", type=Path, default=Path("data/calibration/court.json"))
    for name in ("detect", "filter"):
        det = sub.add_parser(name)
        det.add_argument("--output", type=Path,
                         default=Path("outputs/filtered") if name == "filter" else Path("outputs/detection"))
        det.add_argument("--model", default="yolo26n.pt")
        det.add_argument("--confidence", type=float, default=.35)
        det.add_argument("--imgsz", type=int, default=640)
        det.add_argument("--device", default="cpu")
        det.add_argument("--max-frames", type=int, default=300)
        det.add_argument("--stride", type=int, default=1)
        det.add_argument("--no-video", action="store_true")
        if name == "filter":
            det.add_argument("--court", type=Path, default=Path("data/calibration/court.json"))
            det.add_argument("--margin", type=float, default=12)
    trk = sub.add_parser("track", help="Tracking dua kandidat di ROI lapangan menggunakan ByteTrack")
    trk.add_argument("--court", type=Path, default=Path("data/calibration/court.json"))
    trk.add_argument("--output", type=Path, default=Path("outputs/tracking"))
    trk.add_argument("--model", default="yolo26n.pt")
    trk.add_argument("--confidence", type=float, default=0.15)
    trk.add_argument("--imgsz", type=int, default=640)
    trk.add_argument("--device", default="cpu")
    trk.add_argument("--max-frames", type=int, default=300)
    trk.add_argument("--margin", type=float, default=12)
    trk.add_argument("--no-video", action="store_true")
    trk.add_argument("--tracker", default="config/bytetrack_badminton.yaml")
    trk.add_argument("--crop-mode", choices=["court", "none"], default="court")
    trk.add_argument("--crop-pad-x", type=int, default=35)
    trk.add_argument("--crop-pad-y", type=int, default=100)
    trk.add_argument("--scene-aware", action=argparse.BooleanOptionalAction, default=True)
    trk.add_argument("--scene-pixel-thresh", type=float, default=0.016)
    trk.add_argument("--scene-hist-thresh", type=float, default=0.06)
    trk.add_argument("--scene-min-frames", type=int, default=12)
    mapping = sub.add_parser("map-court", help="Transform CSV tracking to metric court coordinates")
    mapping.add_argument("--tracks", type=Path, default=Path("outputs/tracking_scene/tracks.csv"))
    mapping.add_argument("--court", type=Path, default=Path("data/calibration/court.json"))
    mapping.add_argument("--output", type=Path, default=Path("outputs/court_mapping"))
    mapping.add_argument("--allow-outside", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "calibrate":
        from src.court_filter import calibrate
        calibrate(args.video, args.output, args.second)
    elif args.command == "inspect":
        print(json.dumps(VideoProcessor(args.video).save_metadata(args.output), indent=2))
    elif args.command == "extract":
        count = VideoProcessor(args.video).extract_frames(args.output, args.interval, args.max_frames)
        print(f"Berhasil menyimpan {count} frame ke {args.output}")
    elif args.command == "map-court":
        from src.court_mapper import map_tracking_csv
        report = map_tracking_csv(args.tracks, args.court, args.output, args.allow_outside)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.command == "track":
        from src.court_filter import CourtFilter
        from src.player_tracker import PlayerTracker
        court = CourtFilter(args.court, margin_px=args.margin)
        tracker = PlayerTracker(args.model, args.confidence, args.imgsz, args.device, args.tracker)
        result = tracker.process_video(args.video, args.output, court, args.max_frames, not args.no_video, args.crop_mode, args.crop_pad_x, args.crop_pad_y, args.scene_aware, args.scene_pixel_thresh, args.scene_hist_thresh, args.scene_min_frames)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        from src.player_detector import PlayerDetector
        court_filter = None
        if args.command == "filter":
            from src.court_filter import CourtFilter
            court_filter = CourtFilter(args.court, margin_px=args.margin)
        detector = PlayerDetector(args.model, args.confidence, args.device, args.imgsz)
        result = detector.process_video(args.video, args.output, args.max_frames,
                                        args.stride, not args.no_video, court_filter)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
