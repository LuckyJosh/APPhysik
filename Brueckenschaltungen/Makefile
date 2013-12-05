Protokoll.pdf: Protokoll.tex  Latex/*.tex ../globales/latex/Literatur.bib Plots/*.pdf
	pdflatex Protokoll.tex
	biber Protokoll
	pdflatex Protokoll.tex
	
#Plots/*.pdf: *.py Messdaten/*.txt	
#	python *.py
	
clean:  
	rm -f Latex/*.aux
	rm -f *.aux
	rm -f Latex/*.synctex
	rm -f *.synctex	
	rm -f Latex/*.synctex.gz
	rm -f *.synctex.gz
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