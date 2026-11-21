"""Convert the SumMe benchmark (Gygli et al., ECCV 2014) to SceneSift annotations and a manifest.

SumMe stores, per video, a frames x users matrix whose non-zero entries mark the frames each person
kept. Each person's runs of kept frames become that annotator's intervals, in seconds.

    python experiments/convert_summe.py path/to/SumMe --out experiments/summe
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io


def user_intervals(selected: np.ndarray, fps: float) -> list[list[float]]:
    """Contiguous runs of selected frames as [start, end) intervals in seconds."""
    flags = np.concatenate([[0], (np.asarray(selected) > 0).astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(flags))
    return [[round(start / fps, 3), round(end / fps, 3)] for start, end in zip(edges[::2], edges[1::2])]


def convert(summe_dir: Path, out_dir: Path, split: str = "test") -> int:
    ground_truth = sorted((summe_dir / "GT").glob("*.mat"))
    if not ground_truth:
        raise FileNotFoundError(f"No GT/*.mat files under {summe_dir}")
    annotation_dir = out_dir / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mat_path in ground_truth:
        name = mat_path.stem
        video = summe_dir / "videos" / f"{name}.mp4"
        if not video.is_file():
            print(f"skipping {name}: {video} not found", file=sys.stderr)
            continue
        data = scipy.io.loadmat(mat_path)
        frames = int(np.ravel(data["nFrames"])[0])
        fps = frames / float(np.ravel(data["video_duration"])[0])  # the stored FPS field is rounded
        scores = data["user_score"]
        source_id = name.replace(" ", "_")
        annotation = {
            "video_id": source_id, "category": "summe", "prompt": None,
            "annotators": {f"user_{index + 1:02d}": user_intervals(scores[:, index], fps)
                           for index in range(scores.shape[1])},
        }
        (annotation_dir / f"{source_id}.json").write_text(json.dumps(annotation), encoding="utf-8")
        rows.append([source_id, "summe", str(video.resolve()),
                     str((annotation_dir / f"{source_id}.json").resolve()), split,
                     "SumMe: non-commercial research use", ""])
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_id", "category", "path", "annotation", "split", "license", "prompt"])
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("summe_dir", type=Path, help="Directory containing GT/ and videos/.")
    parser.add_argument("--out", type=Path, default=Path("experiments/summe"))
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    try:
        count = convert(args.summe_dir, args.out, args.split)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {count} annotations and {args.out / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
