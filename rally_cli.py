"""BTI Stage 06 rally annotation GUI and CSV export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.rally_annotation import Rally, export_annotations, load_annotations, save_annotations


def annotate(video_path: Path, annotations_path: Path) -> None:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membaca video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or count <= 0:
        cap.release()
        raise RuntimeError("Metadata FPS/frame_count tidak valid")
    rallies = load_annotations(annotations_path, count)
    frame_no = 0
    playing = False
    start = None
    message = "SPACE play/pause | A/D frame | I start | O end | 1/2 winner | S save | U undo | Q quit"
    window = "BTI Rally Annotation"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if not ok:
                break
            frame = frame.copy()
            lines = [f"Frame {frame_no}/{count - 1}   Time {frame_no/fps:.2f}s   Rallies {len(rallies)}", message,
                     f"Current start: {start if start is not None else '-'}; winner: 1=near, 2=far, 0=unknown", "Press V to set last rally verified after manual review"]
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, line, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow(window, frame)
            key = cv2.waitKey(max(1, round(1000 / fps)) if playing else 0) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord(' '):
                playing = not playing
            elif key == ord('a'):
                playing = False
                frame_no = max(0, frame_no - 1)
            elif key == ord('d'):
                playing = False
                frame_no = min(count - 1, frame_no + 1)
            elif key == ord('i'):
                playing = False
                start = frame_no
            elif key == ord('o') and start is not None:
                playing = False
                candidate = Rally(rally_id=max([r.rally_id for r in rallies], default=0) + 1, start_frame=start, end_frame=frame_no)
                try:
                    validate_copy = rallies + [candidate]
                    from src.rally_annotation import validate_rallies
                    validate_rallies(validate_copy, count)
                except ValueError as exc:
                    print("Tidak disimpan:", exc)
                else:
                    rallies.append(candidate)
                    start = None
                    save_annotations(annotations_path, str(video_path), fps, count, rallies)
                    print("Rally tersimpan:", candidate.rally_id)
            elif key in (ord('1'), ord('2'), ord('0')) and rallies:
                rallies[-1].winner = {ord('1'): 'near', ord('2'): 'far', ord('0'): 'unknown'}[key]
                save_annotations(annotations_path, str(video_path), fps, count, rallies)
            elif key == ord('v') and rallies:
                rallies[-1].annotation_status = 'verified'
                save_annotations(annotations_path, str(video_path), fps, count, rallies)
            elif key == ord('u') and rallies:
                rallies.pop()
                save_annotations(annotations_path, str(video_path), fps, count, rallies)
            elif key == ord('s'):
                save_annotations(annotations_path, str(video_path), fps, count, rallies)
            if playing:
                frame_no = min(count - 1, frame_no + 1)
                if frame_no == count - 1:
                    playing = False
    finally:
        save_annotations(annotations_path, str(video_path), fps, count, rallies)
        cap.release()
        cv2.destroyAllWindows()
    print(f"Anotasi tersimpan: {annotations_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="BTI manual rally annotation")
    sub = p.add_subparsers(dest="command", required=True)
    gui = sub.add_parser("annotate")
    gui.add_argument("--video", type=Path, default=Path("data/raw/match_001.mp4"))
    gui.add_argument("--annotations", type=Path, default=Path("data/annotations/rally_annotations.json"))
    export = sub.add_parser("export")
    export.add_argument("--annotations", type=Path, default=Path("data/annotations/rally_annotations.json"))
    export.add_argument("--scenes", type=Path, default=Path("outputs/full_match/scene_cuts.json"))
    export.add_argument("--output", type=Path, default=Path("outputs/rallies"))
    args = p.parse_args()
    if args.command == "annotate":
        annotate(args.video, args.annotations)
    else:
        print(json.dumps(export_annotations(args.annotations, args.output, args.scenes), indent=2))


if __name__ == "__main__":
    main()
