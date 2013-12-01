new:
	mkdir $(filter-out $@,$(MAKECMDGOALS))
	cp -r vorlage/.  $(filter-out $@,$(MAKECMDGOALS))
