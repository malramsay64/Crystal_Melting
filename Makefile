#
# Makefile
# Malcolm Ramsay, 2018-03-20 17:19
#

all:
	@echo "Makefile needs your attention"

analysis:
	python src/analysis/melting_rates.py -i data/simulations/melting -o data/analysis -s 100



# vim:ft=make
#
