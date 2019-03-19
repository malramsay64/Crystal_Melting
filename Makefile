#
# Makefile
# Malcolm Ramsay, 2018-03-20 17:19
#
# The excellent documentation of the functions in this makefile are adapted
# from a blogpost by François Zaninotto
# https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
#
.DEFAULT_GOAL := help

melting: ## Compute melting rates of the simulations in the directory data/simulations/melting
	python src/melting_rates.py -i data/simulations/interface/output -o data/analysis -s 100

dynamics: ## Compute the dynamics quantites of the simulations in the directory data/simulations/dynamics
	sdanalysis comp-dynamics -o data/analysis/ data/simulations/dynamics/output/trajectory-*

relaxations: ## Compute the relaxation quantities of all values in the file data/analysis/dynamics.h5
	sdanalysis comp-relaxations data/analysis/dynamics.h5

interface-dynamics: ## Compute the dynamics of a simulation with a liquid--crystal interface in data/simulations/2017-09-04-interface/
	sdanalysis comp-dynamics -o data/analysis/interface data/simulations/interface/output/dump-*

test: ## Test the functionality of the helper modules in src
	python -m pytest src

pack-dataset: ## Pack the relevant files from dataset into a tarball
	cd data/simulations/dataset/output && tar cvJf dataset.tar.xz dump-*.gsd && mv dataset.tar.xz ../../../

.PHONY: figures
figures: ## Generate all the figures in the figures directory
	@mkdir -p figures
	jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=-1 --execute notebooks/06_Defect_Creation.ipynb

report_targets := $(wildcard reports/*.md)

reports: $(report_targets:.md=.pdf) ## Generate reports
	echo $<

%.pdf: %.md
	cd $(dir $<); pandoc $(notdir $<) --filter pandoc-fignos -o $(notdir $@)

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# vim:ft=make
#
