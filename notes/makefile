CORE=report.tex
JUNK:=$(wildcard *.bbl *.aux *.blg *.log *.gnuplot *.table *.dvi *.md5 *.auxlock *.out)

all: $(CORE)
	latex --shell-escape report.tex
	bibtex report.aux
	latex --shell-escape report.tex 
	latex --shell-escape report.tex
	dvipdf report.dvi

view: report.pdf
	gnome-open report.pdf

clean: $(JUNK)
	rm $(JUNK)
