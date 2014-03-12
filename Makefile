new:
	$(MAKE) -C _vorlage/ clean
	#mkdir $(filter-out $@,$(MAKECMDGOALS))[]
	cp -r _vorlage/.  $(filter-out $@,$(MAKECMDGOALS)[' '])

done:
	mv $(filter-out $@,$(MAKECMDGOALS))[' '] $(filter-out $@,$(MAKECMDGOALS))[X]

	