# Building the paper

`main.tex` targets arXiv's standard article class. Build with any TeX
distribution (TeX Live / MiKTeX):

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

No figures required; all results are tables. For the arXiv submission,
run `latexmk -pdf main.tex`, then upload `main.tex`, `references.bib`,
and the generated `main.bbl` (arXiv does not run bibtex, so include the
`.bbl` in the tarball).

Author/anonymity note: `main.tex` and `CITATION.cff` carry the release
author metadata. For a double-blind venue, temporarily replace the
`\author{...}` block in `main.tex` with `Anonymous` on a separate build
branch.
