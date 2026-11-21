"""Convert the TVSum benchmark (Song et al., CVPR 2015) to SceneSift annotations and a manifest.

TVSum stores, for each of 20 annotators per video, an importance score from 1 to 5 for every frame;
the score is constant within each 2-second shot. Each annotator's shot scores are kept for rank
correlation. For overlap scores, each annotator's highest-scored shots, up to a share of the video's
length, become that annotator's reference intervals.

    python experiments/convert_tvsum.py path/to/ydata-tvsum50-v1_1 --out experiments/tvsum
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SHOT_SECONDS = 2.0
CATEGORIES = {
    "VT": "changing vehicle tire", "VU": "getting vehicle unstuck", "GA": "grooming an animal",
    "MS": "making sandwich", "PK": "parkour", "PR": "parade", "FM": "flash mob gathering",
    "BK": "bee keeping", "BT": "attempting bike tricks", "DS": "dog show",
}


def shot_scores(frame_scores: np.ndarray, fps: float) -> np.ndarray:
    """One score per 2-second shot, read at the middle frame of each shot."""
    per_shot = SHOT_SECONDS * fps
    count = max(1, int(np.ceil(len(frame_scores) / per_shot - 1e-9)))
    middles = np.minimum(((np.arange(count) + 0.5) * per_shot).astype(int), len(frame_scores) - 1)
    return np.asarray(frame_scores, dtype=float)[middles]


def top_shot_intervals(scores: np.ndarray, duration: float, share: float) -> list[list[float]]:
    """The highest-scored shots filling `share` of the video, as merged [start, end) intervals.

    Annotators give many shots the same score. Ties go first to the shot whose neighbours (two on
    each side) score higher, then to the earlier shot.
    """
    keep = max(1, int(share * duration / SHOT_SECONDS))
    window = np.ones(5)
    neighbourhood = np.round(np.convolve(scores, window, mode="same")
                             / np.convolve(np.ones(len(scores)), window, mode="same"), 6)
    order = sorted(range(len(scores)), key=lambda k: (-scores[k], -neighbourhood[k], k))
    chosen = sorted(order[:keep])
    intervals: list[list[float]] = []
    for k in chosen:
        start, end = k * SHOT_SECONDS, min((k + 1) * SHOT_SECONDS, duration)
        if intervals and abs(intervals[-1][1] - start) < 1e-9:
            intervals[-1][1] = round(end, 3)
        else:
            intervals.append([round(start, 3), round(end, 3)])
    return intervals


def video_fps(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    if not fps or fps != fps or fps <= 0:
        raise ValueError(f"Could not read the frame rate of {path}")
    return float(fps)


def convert(tvsum_dir: Path, out_dir: Path, split: str = "test", share: float = 0.15) -> int:
    annotation_file = tvsum_dir / "data" / "ydata-tvsum50-anno.tsv"
    if not annotation_file.is_file():
        raise FileNotFoundError(f"{annotation_file} not found")
    per_video: dict[str, list[np.ndarray]] = defaultdict(list)
    categories: dict[str, str] = {}
    with open(annotation_file, newline="", encoding="utf-8") as handle:
        for video_id, category, scores in csv.reader(handle, delimiter="\t"):
            per_video[video_id].append(np.array(scores.split(","), dtype=float))
            categories[video_id] = CATEGORIES.get(category, category)
    annotation_dir = out_dir / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for video_id, annotators in sorted(per_video.items()):
        video = tvsum_dir / "video" / f"{video_id}.mp4"
        if not video.is_file():
            print(f"skipping {video_id}: {video} not found", file=sys.stderr)
            continue
        fps = video_fps(video)
        duration = len(annotators[0]) / fps
        shots = {f"user_{index + 1:02d}": shot_scores(scores, fps) for index, scores in enumerate(annotators)}
        annotation = {
            "video_id": video_id, "category": categories[video_id], "prompt": None,
            "annotators": {name: top_shot_intervals(scores, duration, share) for name, scores in shots.items()},
            "importance": {"step_seconds": SHOT_SECONDS,
                           "annotators": {name: scores.tolist() for name, scores in shots.items()}},
        }
        (annotation_dir / f"{video_id}.json").write_text(json.dumps(annotation), encoding="utf-8")
        rows.append([video_id, categories[video_id], str(video.resolve()),
                     str((annotation_dir / f"{video_id}.json").resolve()), split,
                     "TVSum: CC-BY videos, research use", ""])
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_id", "category", "path", "annotation", "split", "license", "prompt"])
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tvsum_dir", type=Path, help="Directory containing data/ and video/.")
    parser.add_argument("--out", type=Path, default=Path("experiments/tvsum"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--share", type=float, default=0.15,
                        help="Share of each video's length kept as an annotator's reference summary.")
    args = parser.parse_args()
    try:
        count = convert(args.tvsum_dir, args.out, args.split, args.share)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {count} annotations and {args.out / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
