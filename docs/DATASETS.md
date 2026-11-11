# Dataset selection for SceneSift

## Practical recommendation

**Start with TVSum** for reliable, annotated evaluation on clips between 2 and 10 minutes. Use Mr. HiSum for a much larger collection (but shorter clips), VideoXum for aligned video-text annotations, and evaluate the suitability/accessibility of LfVS-T for longer-form videos.

| Dataset | Approximate size | Typical duration | Labels | Caveats |
| --- | --- | --- | --- | --- |
| [TVSum](https://github.com/yalesong/tvsum) | 50 videos | 2–10 min | 20 human importance annotations/video | Not a large-scale training dataset; some older URLs may fail |
| [Mr. HiSum](https://github.com/MRHiSum/MR.HiSum) | 31,892 | ~2–5 min | Replay-derived importance | Replay interest is not always semantic importance; videos may require separate retrieval |
| [VideoXum](https://huggingface.co/datasets/jylins/videoxum) | 14,001 | ~2 min average; up to ~12.5 min | Visual and text summaries | Underlying ActivityNet videos are not necessarily packaged as MP4 files; most clips are short |
| [LfVS-T](https://openaccess.thecvf.com/content/CVPR2024/html/Argaw_Scaling_Up_Video_Summarization_Pretraining_with_Large_Language_Models_CVPR_2024_paper.html) | 1,200 | ~12.2 min average | Human summaries | Confirm download access, raw video availability, and annotation type before committing to it |
| [LfVS-P](https://openaccess.thecvf.com/content/CVPR2024/html/Argaw_Scaling_Up_Video_Summarization_Pretraining_with_Large_Language_Models_CVPR_2024_paper.html) | 250,000 | ~13.3 min average | Automatically produced summaries | Massive pretraining collection; not equivalent to human-annotated evaluation |

Sources: original dataset repositories and project papers linked above. Dataset size is not the number of files currently downloadable.

## Initial protocol

1. Audit available video URLs/files, licensing, and video durations before selecting samples.
2. Start with TVSum for reproducible comparison against importance annotations.
3. Build category-diverse supplementary tests with authorized basketball, gaming, interview, and travel footage.
4. For testing `~10 minute` videos, filter actual inspected durations, e.g. 8–12 min, rather than assuming a dataset's published range implies every video is that length.
5. Keep any train/validation/test partition at the *source video* level to avoid temporal leakage.
6. Match summary-duration budgets and document differences between frame-importance annotations and human-selected intervals.
7. Do not commit raw video files, private annotations or temporary output MP4s to Git.

## What is not yet verified

No dataset has been downloaded or audited in this PR. In particular the raw-video accessibility of LfVS-P/T, working links for TVSum, and availability of a sufficiently large set of legally usable ten-minute clips have not yet been validated.

## Suggested sample manifest (future)

Store a CSV with `source_id, category, duration_seconds, license, annotation_source, split, media_available`. Populate it only with verified downloaded/accessible inputs.
