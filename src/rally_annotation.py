"""Manual rally annotation, validation and export. No automatic rally claims."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Rally:
    rally_id: int
    start_frame: int
    end_frame: int
    winner: str = "unknown"
    annotation_status: str = "manual_unverified"
    notes: str = ""


def validate_rallies(rallies: list[Rally], frame_count: int) -> None:
    previous_end = -1
    for rally in sorted(rallies, key=lambda x: x.start_frame):
        if not (0 <= rally.start_frame <= rally.end_frame < frame_count):
            raise ValueError(f"Rentang frame tidak valid: rally {rally.rally_id}")
        if rally.start_frame <= previous_end:
            raise ValueError(f"Rally bertumpang tindih: {rally.rally_id}")
        if rally.winner not in {"near", "far", "unknown"}:
            raise ValueError("winner harus near, far, atau unknown")
        previous_end = rally.end_frame
    if len({r.rally_id for r in rallies}) != len(rallies):
        raise ValueError("rally_id harus unik")


def save_annotations(path: Path, video_path: str, fps: float, frame_count: int, rallies: list[Rally]) -> None:
    validate_rallies(rallies, frame_count)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "video": video_path,
        "fps": fps,
        "frame_count": frame_count,
        "rallies": [asdict(r) for r in rallies],
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def load_annotations(path: Path, frame_count: int) -> list[Rally]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("frame_count") != frame_count:
        raise ValueError("Frame count berbeda dari anotasi sebelumnya; gunakan file output lain")
    rallies = [Rally(**row) for row in raw.get("rallies", [])]
    validate_rallies(rallies, frame_count)
    return rallies


def export_annotations(annotation_file: Path, output_dir: Path, scene_file: Path | None = None) -> dict:
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    fps = float(payload["fps"])
    frame_count = int(payload["frame_count"])
    rallies = [Rally(**row) for row in payload["rallies"]]
    validate_rallies(rallies, frame_count)
    cuts = []
    if scene_file is not None and scene_file.exists():
        cuts = sorted({int(c["frame_index"]) for c in json.loads(scene_file.read_text(encoding="utf-8")).get("cuts", [])})
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in sorted(rallies, key=lambda x: x.start_frame):
        intersecting = [c for c in cuts if r.start_frame < c <= r.end_frame]
        rows.append({
            "rally_id": r.rally_id,
            "start_frame": r.start_frame,
            "end_frame": r.end_frame,
            "start_seconds": round(r.start_frame / fps, 4),
            "end_seconds": round((r.end_frame + 1) / fps, 4),
            "duration_seconds": round((r.end_frame - r.start_frame + 1) / fps, 4),
            "winner_side": r.winner,
            "annotation_status": r.annotation_status,
            "crosses_scene_cut": bool(intersecting),
            "scene_cut_frames": ";".join(map(str, intersecting)),
            "notes": r.notes,
        })
    with (output_dir / "rallies.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rally_id", "start_frame", "end_frame", "start_seconds", "end_seconds", "duration_seconds", "winner_side", "annotation_status", "crosses_scene_cut", "scene_cut_frames", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "video": payload["video"],
        "frames": frame_count,
        "fps": fps,
        "annotated_rallies": len(rows),
        "manual_unverified": sum(r.annotation_status == "manual_unverified" for r in rallies),
        "rallies_crossing_scene_cut": sum(bool(r["crosses_scene_cut"]) for r in rows),
        "rallies_without_winner": sum(r.winner == "unknown" for r in rallies),
        "note": "Anotasi manual, bukan prediksi model AI. Scene cut bukan batas rally. Label near/far bukan identitas atlet permanen.",
    }
    (output_dir / "rally_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
