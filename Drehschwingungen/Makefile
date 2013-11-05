all:
		make protocol

protocol:
		make tex
		make tex

tex: Protokoll.tex Latex/Preambel.tex
		pdflatex Protokoll.tex

clean:
		rm -f Latex/*.aux
		rm -f *.aux
		rm -f Latex/*.synctex
		rm -f *.synctex
		rm -f Latex/*.bbl
		rm -f *.bbl
		rm -f Latex/*.bcf
		rm -f *.bcf
		rm -f Latex/*.blg
		rm -f *.blg
		rm -f Latex/*.log
		rm -f *.log
		rm -f Latex/*.nav
		rm -f *.nav
		rm -f Latex/*.out		
		rm -f *.out		
		rm -f Latex/*.run.xml
		rm -f *.run.xml
		rm -f Latex/*.snm
		rm -f *.snm
		rm -f Latex/*.toc	
		rm -f *.toc	
		rm -f Latex/*.tmp	
		rm -f *.tmp	