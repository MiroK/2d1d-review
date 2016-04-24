CORE=schur.tex
JUNK:=$(wildcard *.bbl *.aux *.blg *.log *.gnuplot *.table *.dvi *.md5 *.auxlock *.out)

all: $(CORE)
	latex --shell-escape schur.tex
#	bibtex schur.aux
#	latex --shell-escape schur.tex 
#	latex --shell-escape schur.tex
	dvipdf schur.dvi

view: schur.pdf
	gnome-open schur.pdf

clean: $(JUNK)
	rm $(JUNK)
