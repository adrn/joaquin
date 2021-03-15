# WARNING: BIBTEX TURNED OFF FOR NOW

LATEX       = pdflatex -interaction nonstopmode -halt-on-error -file-line-error
CHECK_RERUN = grep "Rerun to get" $*.log
CRUFT_SUFFS = pdf aux bbl blg log dvi ps eps out brf fls fdb_latexmk synctex.gz bcf run.xml
NAME        = joaquin-dr17

all: ${NAME}.pdf

${NAME}.pdf: ${NAME}.tex # refs.bib
	${LATEX} ${NAME}.tex
# 	bibtex ${NAME}
#	${LATEX} ${NAME}.tex
#	${LATEX} ${NAME}.tex
	( ${CHECK_RERUN} && ${LATEX} $* ) || echo "Done."
	( ${CHECK_RERUN} && ${LATEX} $* ) || echo "Done."

clean:
	${RM} $(foreach suff, ${CRUFT_SUFFS}, ${NAME}.${suff})
	${RM} *Notes.bib
