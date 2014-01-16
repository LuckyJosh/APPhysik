new:
	mkdir $(filter-out $@,$(MAKECMDGOALS))
	cp -r _vorlage/.  $(filter-out $@,$(MAKECMDGOALS))
