# SceneSift research manuscript

The LaTeX draft is an **IEEE-style working paper**, not a completed report. It deliberately states that results and discussion are pending.

## Compile

From `research/`, run:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
Alternatively upload `main.tex` and `references.bib` to Overleaf and choose IEEEtran.

## Working convention

- Keep research claims grounded in implemented experiments.
- Document the split, annotation protocol, model versions, and system hardware before reporting numbers.
- Add actual tables and plots after experiments, never placeholder performance values.
- Update the bibliography when selecting additional datasets and methods.
- This LaTeX source is tracked alongside UI development but paper writing can proceed independently.
