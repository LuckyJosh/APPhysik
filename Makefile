new:
	mkdir $(filter-out $@,$(MAKECMDGOALS))
	cp -r Versuchsvorlage/.  $(filter-out $@,$(MAKECMDGOALS))
