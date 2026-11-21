# Running the SceneSift comparison

This folder holds the inputs and outputs of the experiment described in
[docs/EXPERIMENT.md](../docs/EXPERIMENT.md). No videos, annotations, or raw result files are committed; the one run so far is written up in
[SUMME_RESULTS.md](SUMME_RESULTS.md). The study uses SumMe and TVSum (see the SumMe section below; the
TVSum converter is not written yet). The manifest format in sections 1 and 2 also works for any other
annotated video.

## 1. Collect videos and annotations

Put authorized videos under `experiments/videos/` (git-ignored) and one annotation file per video
under `experiments/annotations/`.

An annotation file lists, for each annotator, the intervals they marked as highlights, in seconds on
the original video's timeline:

```json
{
  "video_id": "parkour_01",
  "category": "parkour",
  "prompt": null,
  "annotators": {
    "annotator_1": [[12.0, 20.5], [44.0, 51.0]],
    "annotator_2": [[13.5, 21.0]]
  }
}
```

`prompt` is the request shown to annotators for prompted tests; leave it `null` for generic highlights.
See [annotations/example.json](annotations/example.json).

## 2. Write the manifest

Copy [manifest.example.csv](manifest.example.csv) to `manifest.csv` and list every video:

| Column | Meaning |
| --- | --- |
| `source_id` | Unique name for the video |
| `category` | Any label, for example a TVSum category; used only for reporting |
| `path`, `annotation` | Paths relative to the manifest |
| `split` | `train`, `validation`, or `test`. Only one split is evaluated per run |
| `license` | Where the video came from and under what terms |
| `prompt` | Optional request; overrides the one in the annotation file |

Keep the split at the source-video level. Fit weights or settings on SumMe only, never on TVSum.

## 3. Run

```bash
python -m scenesift evaluate experiments/manifest.csv --out experiments/results \
    --budgets 30 60 120 --ablations
```

- Each video is analysed once (cached under `~/.cache/scenesift`), then every method is scored on it.
- A budget is a number of seconds (`30`) or a share of each video's length (`15%`).
- Budgets at least as long as a video are skipped for that video.
- Every run also scores a `uniform` reference: evenly spaced 5-second clips filling the same budget.
  It ignores the content, so it shows what a score looks like with no skill.
- `--ablations` adds the multimodal method with each modality removed, and without the diversity
  constraint.
- `--split validation` evaluates the validation videos instead.

## 4. Read the output

- `results.csv`: one row per video, method, variant, and budget.
- `summary.md`: mean ± sample standard deviation per category and pooled, with the number of videos.

| Metric | Definition |
| --- | --- |
| `precision`, `recall`, `f1` | Seconds of overlap between the summary and one annotator's intervals, averaged over annotators |
| `f1_best_annotator` | F1 against the closest annotator |
| `event_coverage` | Share of an annotator's intervals with at least half their duration in the summary, averaged over annotators |
| `redundancy` | Mean pairwise cosine similarity of the selected clips' visual embeddings (lower is less repetitive) |
| `seconds_per_source_minute` | Analysis time for the stages that method needs, per minute of input. Measured once on the shared analysis pass, so it estimates the cost of running that method alone |
| `annotator_agreement_f1` | Mean pairwise F1 between annotators, as a ceiling for interpreting the scores |

Rank correlation (EXPERIMENT.md section 8) is not implemented yet.

## SumMe benchmark

SumMe (Gygli et al., ECCV 2014; archive at <https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip>) has 25 user videos, each
summarized by 15 to 18 people. Its videos are provided for non-commercial research only, so they are
not redistributed here.

```bash
# After downloading and unpacking SumMe.zip (about 2.4 GB):
python experiments/convert_summe.py path/to/SumMe --out experiments/summe
python -m scenesift evaluate experiments/summe/manifest.csv --out experiments/results/summe \
    --budgets 15% --ablations
```

The converter turns each person's selected frames into intervals. `f1` is then the mean F-measure
against each person at a 15% budget, which is the SumMe protocol; `annotator_agreement_f1` averages
to 0.31 over the dataset, matching the human agreement reported in the paper. SumMe has no categories
or requests, so it tests generic highlights only.
