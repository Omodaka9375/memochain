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

Author/anonymity note: the byline is currently `Anonymous` for a
double-blind submission. For a non-anonymous release (arXiv, blog),
replace the `\author{...}` block in `main.tex` and the author fields in
`CITATION.cff` at the repository root.
