filename = paper
macro_sources=$(wildcard macro_data/*.json)
macro_targets=$(patsubst macro_data/%.json,macro_data/%.tex,$(macro_sources))
TEXFILES := $(wildcard *.tex)

all: pdf
macros: $(macro_targets)

$(macro_targets): macro_data/%.tex: macro_data/%.json
	json2latex $^ $* $@
	scripts/macro/ensure_nl_eof.sh $@

macros_from_json.tex: $(macro_targets)
	cat $(macro_targets) > $@

pdf: $(TEXFILES) macros_from_json.tex
	latexmk -pdf ${filename}

gitID.txt:
	git describe --abbrev=8 --dirty --always --tags > gitID.txt

clean:
	rm -f *.pdf *.aux *.bbl *.blg *.log *.toc *.out *Notes.bib *.fdb* *.fls gitID.txt $(macro_targets) macros_from_json.tex

check: pdf
	checkcites -c ${filename}
