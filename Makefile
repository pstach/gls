all:
	(cd misc && make all)
	(cd ularith && make all)
	(cd cofact && make all)
	(cd tests && make all)
	(cd tests/ocl && make all)
	(cd makefb && make all)
	(cd las && make all)
	
clean:
	(cd misc && make clean)
	(cd ularith && make clean)
	(cd cofact && make clean)
	(cd tests && make clean)
	(cd tests/ocl && make clean)
	(cd makefb && make clean)
	(cd las && make clean)
