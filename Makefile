PYTHON ?= python3

.PHONY: all analysis figures report slides check clean

all: analysis figures report slides

analysis:
	$(PYTHON) scripts/run_analysis.py

figures:
	$(PYTHON) scripts/make_figures.py

report: analysis figures
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex

slides: analysis figures
	cd slides && latexmk -xelatex -interaction=nonstopmode -halt-on-error slides.tex
	cd slides && latexmk -xelatex -interaction=nonstopmode -halt-on-error -jobname=slides_notes -usepretex='\def\shownotes{}' slides.tex

check:
	$(PYTHON) checks/run_checks.py

clean:
	cd report && latexmk -c report.tex
	cd slides && latexmk -c slides.tex && latexmk -c -jobname=slides_notes slides.tex
