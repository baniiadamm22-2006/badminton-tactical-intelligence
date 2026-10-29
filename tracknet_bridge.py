"""TrackNetV3 offline adapter for Badminton Tactical Intelligence.

Runs upstream inference in a separate Python environment and converts CSV output
using the Stage 07 existing shuttle_tool module; no model weights are bundled.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def infer(repo: Path, python_exe: Path, video: Path, output: Path,
          tracknet: Path, inpaintnet: Path | None, batch_size: int,
          large_video: bool) -> Path:
    repo, python_exe, video, output, tracknet = (
        repo.resolve(), python_exe.resolve(), video.resolve(),
        output.resolve(), tracknet.resolve()
    )
    if not (repo / 'predict.py').is_file():
        raise FileNotFoundError(f'TrackNetV3 predict.py not found: {repo}')
    for path in (python_exe, video, tracknet):
        if not path.is_file():
            raise FileNotFoundError(f'Required file missing: {path}')
    if inpaintnet is not None:
        inpaintnet = inpaintnet.resolve()
        if not inpaintnet.is_file():
            raise FileNotFoundError(f'InpaintNet checkpoint missing: {inpaintnet}')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    output.mkdir(parents=True, exist_ok=True)
    command = [str(python_exe), 'predict.py', '--video_file', str(video),
               '--tracknet_file', str(tracknet), '--save_dir', str(output),
               '--batch_size', str(batch_size)]
    if inpaintnet is not None:
        command.extend(['--inpaintnet_file', str(inpaintnet)])
    if large_video:
        command.append('--large_video')
    subprocess.run(command, cwd=repo, check=True)
    result = output / f'{video.stem}_ball.csv'
    if not result.is_file() or result.stat().st_size == 0:
        raise RuntimeError(f'Inference finished but prediction CSV is missing/empty: {result}')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='TrackNetV3 -> BTI offline adapter')
    parser.add_argument('--repo', type=Path, default=Path('external/TrackNetV3'))
    parser.add_argument('--python', dest='python_exe', required=True, type=Path,
                        help='Python executable from isolated TrackNet environment')
    parser.add_argument('--video', type=Path, default=Path('data/raw/match_001.mp4'))
    parser.add_argument('--tracknet', type=Path, required=True)
    parser.add_argument('--inpaintnet', type=Path)
    parser.add_argument('--ground-truth', type=Path,
                        default=Path('data/annotations/shuttle_ground_truth.csv'))
    parser.add_argument('--output', type=Path, default=Path('outputs/shuttle_tracknet'))
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--large-video', action='store_true')
    parser.add_argument('--tolerance-px', type=float, default=8.0)
    args = parser.parse_args()
    if args.tolerance_px <= 0:
        parser.error('--tolerance-px must be positive')

    import cv2
    from src.shuttle_tool import normalize_prediction_csv, evaluate

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video}')
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError('Video metadata is invalid')
    raw_csv = infer(args.repo, args.python_exe, args.video, args.output,
                    args.tracknet, args.inpaintnet, args.batch_size, args.large_video)
    normalized = args.output / 'predictions.csv'
    count = normalize_prediction_csv(raw_csv, normalized, fps, width, height)
    summary = {'video': str(args.video), 'fps': fps, 'resolution': [width, height],
               'prediction_rows': count, 'raw_predictions': str(raw_csv),
               'normalized_predictions': str(normalized),
               'evaluation_status': 'not_run'}
    if args.ground_truth.is_file():
        metrics = evaluate(args.ground_truth, normalized, args.tolerance_px)
        evaluation_file = args.output / 'evaluation.json'
        evaluation_file.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        summary['evaluation_status'] = 'completed'
        summary['evaluation_file'] = str(evaluation_file)
    summary_file = args.output / 'run_summary.json'
    summary_file.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
