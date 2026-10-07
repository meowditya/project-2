PYTHON ?= python3

.PHONY: all analysis figures report check clean

all: analysis figures report

analysis:
	$(PYTHON) scripts/run_analysis.py

figures:
	$(PYTHON) scripts/make_figures.py

report: analysis figures
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex

check:
	$(PYTHON) checks/run_checks.py

clean:
	cd report && latexmk -c report.tex
