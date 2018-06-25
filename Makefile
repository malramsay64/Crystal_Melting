#
# Makefile
# Malcolm Ramsay, 2018-03-20 17:19
#

all:
	@echo ""
	@echo "make melting      Compute melting rates of the simulations in the"
	@echo "                  directory data/simulations/melting"
	@echo "make dynamics     Compute the dynamics quantities of the simulations"
	@echo "                  in the directory data/simulations/dynamics"
	@echo "make relaxations  Compute the relaxations quantities of all values"
	@echo "                  in the file data/analysis/dynamics.h5"
	@echo ""

melting:
	python src/analysis/melting_rates.py -i data/simulations/melting -o data/analysis -s 100

dynamics:
	ls data/simulations/dynamics/trajectory-* | xargs -n1 sdanalysis comp_dynamics -o data/analysis

relaxations:
	python src/relaxations.py


# vim:ft=make
#
