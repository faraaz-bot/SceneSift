# Datasets and published results

SceneSift is evaluated on two public video-summarization benchmarks. Both come with highlight labels from many people per video, so no videos are collected and no annotators are recruited.

| | SumMe | TVSum |
| --- | --- | --- |
| Paper | Gygli et al., *Creating Summaries from User Videos*, ECCV 2014 | Song et al., *TVSum: Summarizing Web Videos Using Titles*, CVPR 2015 |
| Videos | 25 personal videos, unedited or minimally edited | 50 YouTube videos, 5 in each of 10 categories |
| Total length | 67 minutes | Not yet measured here; individual videos run from about 1 to 11 minutes |
| Annotators | 15 to 18 per video | 20 per video |
| Label | Each person selected the frames they would keep in a summary of at most 15% of the video | Each person scored every 2-second shot from 1 (not important) to 5 (very important) |
| Text requests | None | None (each video has a title and a category) |
| Terms | Videos for non-commercial research only; annotations free for research | Creative Commons CC-BY 3.0 |
| Role here | Development set; fusion weights are fitted on it | Held-out test set; per-category results |

Neither dataset is redistributed in this repository.

## SumMe

The archive (<https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip>, about 2.4 GB) contains:

- `videos/`: the 25 videos in WebM format. The content is everyday footage such as holidays, sports, and events: a plane landing, cooking, a base jump, kids playing in leaves, a paintball match.
- `GT/`: one MATLAB file per video with the frame rate, the number of frames, each person's selection for every frame (`user_score`), and the share of people who selected each frame (`gt_score`).
- `matlab/` and `python/`: the authors' evaluation code.

`experiments/convert_summe.py` turns each person's selected frames into intervals in seconds. Agreement between annotators, measured as the mean pairwise F-measure, is 0.31.

## TVSum

The dataset is listed at <https://github.com/yalesong/tvsum>, which points to <http://people.csail.mit.edu/yalesong/tvsum> for the download (644 MB). It has not been downloaded here yet, so the layout below is taken from the dataset's documentation and still needs checking against the files.

- 50 videos found on YouTube by searching for ten categories: changing a vehicle tire (VT), getting a vehicle unstuck (VU), grooming an animal (GA), making a sandwich (MS), parkour (PK), parade (PR), flash mob (FM), beekeeping (BK), bike tricks (BT), and dog show (DS).
- An annotation file with one row per video and annotator, holding that person's 1-to-5 score for every 2-second shot.
- A metadata file with each video's category, title, and length.

Because the labels are importance scores and not selected intervals, TVSum supports two kinds of measurement: overlap with a reference summary built from the scores, and rank correlation between a method's scores and each annotator's scores.

## Published results

These are the reference points for this project. Protocols differ between papers, so each table says how its numbers were computed.

### SumMe: F-measure averaged over annotators, 15% summary length

This is the protocol used in this repository, so the rows are comparable.

| Method | Kind | F-measure | Source |
| --- | --- | --- | --- |
| Uniform sampling | No content | 0.143 | Gygli et al. 2014 |
| Colour-histogram clustering | Classical | 0.163 | Gygli et al. 2014 |
| Visual attention | Classical | 0.167 | Gygli et al. 2014 |
| Superframes with interestingness (the SumMe paper's method) | Classical, hand-built features | 0.234 | Gygli et al. 2014 |
| Random scores with KTS segmentation | No content | 0.19 | Otani et al. 2019 |
| Agreement between annotators | Human | 0.311 | Gygli et al. 2014 |
| SceneSift uniform reference | No content | 0.172 | [SUMME_RESULTS.md](../experiments/SUMME_RESULTS.md) |
| SceneSift A: activity | Classical signals | 0.195 | same |
| SceneSift B: semantic | Pretrained features | 0.165 | same |
| SceneSift C: multimodal, untuned weights | Mixed | 0.206 | same |

The classical method from 2014 is ahead of the current untuned multimodal method. Reaching 0.234 on SumMe is the first target for the blend.

### F-measure as reported by deep-learning papers

Most later papers score SumMe against the best-matching annotator and segment videos with kernel temporal segmentation (KTS). Otani et al. (2019) showed that random importance scores do about as well as published methods under this protocol, so these numbers are not comparable with the table above and say little about a method's quality.

| Method | SumMe (best annotator) | TVSum (average) |
| --- | --- | --- |
| Random scores, KTS segmentation | 0.41 | 0.57 |
| dppLSTM (supervised) | 0.43 | 0.60 |
| DR-DSN (unsupervised, trained) | 0.41 | 0.58 |
| Human | 0.54 | 0.54 |

Source: Otani et al. 2019, Table 1.

### Rank correlation with annotators' importance scores

Kendall's τ and Spearman's ρ measure whether a method orders the moments of a video the way people do. Random scores give zero by construction, which makes this the more informative measure, and it is the main measure for the TVSum test in this project.

| Method | Training | SumMe τ | SumMe ρ | TVSum τ | TVSum ρ | Source |
| --- | --- | --- | --- | --- | --- | --- |
| Random | None | 0.000 | 0.000 | 0.000 | 0.000 | Otani et al. 2019 |
| DR-DSN | Trained, no labels | | | 0.020 | 0.026 | Otani et al. 2019 |
| dppLSTM | Supervised | | | 0.042 | 0.055 | Otani et al. 2019 |
| Zero-shot contrastive scoring | None (pretrained GoogLeNet features) | 0.096 | 0.117 | 0.135 | 0.178 | Pang et al. 2024 |
| CLIP-It | Supervised | | | 0.108 | 0.147 | as listed by Son et al. 2024 |
| A2Summ | Supervised | 0.108 | 0.129 | 0.137 | 0.165 | as listed by Son et al. 2024 |
| VASNet | Supervised | 0.160 | 0.170 | 0.160 | 0.170 | as listed by Son et al. 2024 |
| CSTA | Supervised | 0.246 | 0.274 | 0.194 | 0.255 | Son et al. 2024 |
| Human | | 0.205 | 0.213 | 0.177 | 0.204 | Otani et al. 2019; Son et al. 2024 |

The zero-shot row gives each dataset's best training-free variant in that paper. The supervised methods are trained and tested on splits of the same dataset with five-fold cross-validation. SceneSift trains no network, so the closest published comparison is the zero-shot row: τ of about 0.10 on SumMe and 0.13 on TVSum.

### Sources

- M. Gygli, H. Grabner, H. Riemenschneider, L. Van Gool. *Creating Summaries from User Videos.* ECCV 2014. The baseline rows (uniform, clustering, attention) were taken from an excerpt of Table 2 and should be checked against the paper before being quoted; the 0.23 and 0.31 figures are confirmed by Otani et al.
- Y. Song, J. Vallmitjana, A. Stent, A. Jaimes. *TVSum: Summarizing Web Videos Using Titles.* CVPR 2015.
- M. Otani, Y. Nakashima, E. Rahtu, J. Heikkilä. *Rethinking the Evaluation of Video Summaries.* CVPR 2019. <https://arxiv.org/abs/1903.11328>
- Z. Pang, Y. Nakashima, M. Otani, H. Nagahara. *Unleashing the Power of Contrastive Learning for Zero-Shot Video Summarization.* Journal of Imaging 10(9), 2024. <https://doi.org/10.3390/jimaging10090229>
- J. Son, J. Park, K. Kim. *CSTA: CNN-based Spatiotemporal Attention for Video Summarization.* 2024. <https://arxiv.org/abs/2405.11905>

## Data handling

1. Keep SumMe and TVSum apart: fit weights and make method choices on SumMe only, and do not consult TVSum annotations until methods and weights are frozen.
2. Use a summary budget of 15% of each video's length on both datasets.
3. Do not commit video files, raw annotation archives, or rendered reels to Git.
