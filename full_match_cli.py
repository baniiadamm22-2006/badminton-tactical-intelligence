"""Tahap 05.1 command-line entrypoint; does not replace the previous main.py."""
import argparse
import json
from pathlib import Path
from src.full_match import run_full_match


def main():
    parser = argparse.ArgumentParser(description="BTI offline full-match tracker")
    parser.add_argument("--video", type=Path, default=Path("data/raw/match_001.mp4"))
    parser.add_argument("--court", type=Path, default=Path("data/calibration/court.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/full_match"))
    parser.add_argument("--tracker", default="config/bytetrack_badminton.yaml")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=.05)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--chunk-frames", type=int, default=300)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run_full_match(args.video, args.court, args.output,
                            tracker=args.tracker, model=args.model, device=args.device,
                            conf=args.confidence, imgsz=args.imgsz,
                            chunk_frames=args.chunk_frames, max_frames=args.max_frames,
                            resume=args.resume)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
