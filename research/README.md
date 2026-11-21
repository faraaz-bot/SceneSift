# SceneSift research report

`main.tex` is the project report in IEEE conference format. Its tables are copied from
`experiments/SUMME_RESULTS.md` and `experiments/TVSUM_RESULTS.md`, and its figures are drawn from the
evaluation outputs; keep both in step when a run is repeated.

## Figures

`make_figures.py` draws every figure into `figures/` (PDF for the report, PNG for previews):

```bash
python research/make_figures.py --summe experiments/results/summe --tvsum experiments/results/tvsum \
    --tvsum-manifest experiments/tvsum/manifest.csv --config configs/fitted_summe.yaml
```

| File | Content |
| --- | --- |
| `pipeline` | Block diagram of the pipeline |
| `f1` | How temporal F1 is computed against one annotator |
| `results` | F1 and Kendall's τ per method on both benchmarks, with bootstrap intervals |
| `ablation` | Change in the multimodal method's τ when one feature is removed |
| `categories` | TVSum τ by category and method |
| `example` | One TVSum video: frames picked by two methods, annotator ratings, and every method's summary |

`f1` and `example` show a TVSum video, so they need the dataset; without `--tvsum-manifest` only the
charts are drawn. The two videos shown carry a Creative Commons BY licence and are credited in the
figure captions.

## Compile

From `research/`, run:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
Alternatively upload `main.tex`, `references.bib`, and `figures/` to Overleaf, or run
`tectonic main.tex`. The report was last compiled with Tectonic 0.15 (9 pages).

## Working convention

- Keep research claims grounded in implemented experiments.
- Document the split, annotation protocol, model versions, and system hardware before reporting numbers.
- Add actual tables and plots after experiments, never placeholder performance values.
- Update the bibliography when selecting additional datasets and methods.
