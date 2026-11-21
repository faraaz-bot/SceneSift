# Running the SceneSift comparison

This folder holds the inputs and outputs of the experiment described in
[docs/EXPERIMENT.md](../docs/EXPERIMENT.md). No videos, annotations, or raw result files are committed; the runs are written up in
[SUMME_RESULTS.md](SUMME_RESULTS.md) and [TVSUM_RESULTS.md](TVSUM_RESULTS.md). The study uses SumMe
and TVSum, each with its own converter (see the last two sections). The manifest format in sections 1
and 2 also works for any other annotated video.

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
An optional `importance` entry holds per-annotator scores on a regular grid, as TVSum provides:
`{"step_seconds": 2.0, "annotators": {"annotator_1": [4, 4, 1, ...]}}`. When present it is used for
rank correlation.
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
- The methods are `activity`, `semantic`, `multimodal`, `classical`, and `blend`; `--methods` picks a subset.
- `--ablations` adds the multimodal method with each modality removed, and without the diversity
  constraint.
- `--split validation` evaluates the validation videos instead.
- `--config` takes a YAML file of settings, for example the fitted blend weights.

To fit the blend's weights on development videos and write them to a config file:

```bash
python -m scenesift fit experiments/manifest.csv --split train --out configs/fitted.yaml
```

It prints the weights and the leave-one-video-out scores of the blend. To compare methods pairwise
from a results file:

```bash
python experiments/compare_methods.py experiments/results/results.csv --metric f1
```

## 4. Read the output

- `results.csv`: one row per video, method, variant, and budget.
- `summary.md`: mean ± sample standard deviation per category and pooled, with the number of videos.

| Metric | Definition |
| --- | --- |
| `precision`, `recall`, `f1` | Seconds of overlap between the summary and one annotator's intervals, averaged over annotators |
| `f1_best_annotator` | F1 against the closest annotator |
| `event_coverage` | Share of an annotator's intervals with at least half their duration in the summary, averaged over annotators |
| `kendall_tau`, `spearman_rho` | Rank correlation between the method's score over time and each annotator's scores, averaged over annotators. Uses the annotation's `importance` scores when present, otherwise the share of each second the annotator selected |
| `annotator_agreement_tau` | Mean pairwise Kendall's τ between annotators |
| `redundancy` | Mean pairwise cosine similarity of the selected clips' visual embeddings (lower is less repetitive) |
| `seconds_per_source_minute` | Analysis time for the stages that method needs, per minute of input. Measured once on the shared analysis pass, so it estimates the cost of running that method alone |
| `annotator_agreement_f1` | Mean pairwise F1 between annotators, as a ceiling for interpreting the scores |

The uniform reference has no scores, so it has no rank correlation.

## Showing how F1 works

`f1_demo.py` draws a labelled picture and a labelled video for one video, to explain the score:

```bash
python experiments/f1_demo.py experiments/tvsum/manifest.csv iVt07TCkFM0 --method classical \
    --annotator user_02 --title "Pure Fix TV: How to Wheelie" --credit "Source and licence line"
```

- The picture lines up one annotator's summary, the method's summary, and their overlap on a timeline,
  works through precision, recall, and F1 with the actual numbers, and then shows every annotator with
  the F1 against each and the mean.
- The video plays the source with a banner saying who picked the current moment (both, only the
  method, only the annotator, or neither) and a running precision, recall, and F1.
- Without `--annotator` it shows the annotator whose F1 is the median. `--no-video` skips the video.

Outputs go to `output/f1_demo/`, which is git-ignored. They contain frames of the source video, so
its licence applies: credit the source, and do not publish outputs made from videos that are for
research use only.

## SumMe benchmark

SumMe (Gygli et al., ECCV 2014; archive at <https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip>) has 25 user videos, each
summarized by 15 to 18 people. Its videos are provided for non-commercial research only, so they are
not redistributed here.

```bash
# After downloading and unpacking SumMe.zip (about 2.4 GB):
python experiments/convert_summe.py path/to/SumMe --out experiments/summe
python -m scenesift fit experiments/summe/manifest.csv --split test --out configs/fitted_summe.yaml
python -m scenesift evaluate experiments/summe/manifest.csv --out experiments/results/summe \
    --budgets 15% --ablations --config configs/fitted_summe.yaml
```

SumMe is the development set: the blend's weights are fitted on it, so only the leave-one-video-out
figures printed by `fit` say how the blend does on unseen SumMe videos.

The converter turns each person's selected frames into intervals. `f1` is then the mean F-measure
against each person at a 15% budget, which is the SumMe protocol; `annotator_agreement_f1` averages
to 0.31 over the dataset, matching the human agreement reported in the paper. SumMe has no categories
or requests, so it tests generic highlights only.

## TVSum benchmark

TVSum (Song et al., CVPR 2015; archive `tvsum50_ver_1_1.tgz` at
<https://people.csail.mit.edu/yalesong/tvsum/>, 640 MB) has 50 YouTube videos in 10 categories, each
scored by 20 people. It is provided for research use and is not redistributed here.

```bash
# After unpacking the archive, then ydata-tvsum50-data.zip and ydata-tvsum50-video.zip inside it:
python experiments/convert_tvsum.py path/to/ydata-tvsum50-v1_1 --out experiments/tvsum
python -m scenesift evaluate experiments/tvsum/manifest.csv --out experiments/results/tvsum \
    --budgets 15% --ablations --config configs/fitted_summe.yaml
```

The converter keeps each person's score for every 2-second shot, which is what rank correlation is
computed against. For overlap scores it also builds a reference summary per person: their
highest-scored shots up to 15% of the video, with ties going to the shot whose neighbours score
higher and then to the earlier shot. People give many shots the same score, so these reference
summaries depend on the tie rule, and published work has shown that overlap scores on TVSum barely
separate random selections from real methods. Rank correlation is the main measure on TVSum.

TVSum is the test set. Run it only with weights and settings that were fixed beforehand.
