#
# Makefile
# Malcolm Ramsay, 2018-03-20 17:19
#
# The excellent documentation of the functions in this makefile are adapted
# from a blogpost by François Zaninotto
# https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
#
.DEFAULT_GOAL := help

#
# Melting Rules
#

melting_sim = data/simulations/interface/output
melting_analysis_dir = data/analysis/melting

melting_trajectories = $(wildcard $(melting_sim)/dump-Trimer*.gsd)
melting_analysis = $(addprefix $(melting_analysis_dir)/, $(notdir $(melting_trajectories:.gsd=.h5)))

melting: data/analysis/melting.h5 ## Compute melting rates of the simulations in the directory data/simulations/melting

data/analysis/melting.h5: $(analysis_files)
	python3 src/melting_rates.py collate $@ $^

$(melting_analysis_dir)/dump-%.h5: $(melting_sim)/dump-%.gsd
	python src/melting_rates.py melting $< $@ -s 1000

#
# Dynamics Rules
#

dynamics_sim = data/simulations/dynamics/output
dynamics_analysis_dir = data/analysis/dynamics

dynamics_trajectories = $(wildcard $(dynamics_sim)/trajectory-Trimer-*.gsd)
dynamics_analysis = $(addprefix $(dynamics_analysis_dir)/, $(notdir $(dynamics_trajectories:.gsd=.h5)))

dynamics = data/analysis/dynamics.h5
dynamics_clean = data/analysis/dynamics_clean.h5

dynamics: ${dynamics_clean} ## Compute dynamics quantities for all parameters of the trimer molecule
	echo $(dynamics_analysis)

bootstrap: ${dynamics_clean}
	python src/dynamics_calc.py bootstrap $<

${dynamics_clean}: ${dynamics}
	# python src/dynamics_calc.py clean --min-samples 50 $<

${dynamics}: $(dynamics_analysis)
	echo $(dynamics_analysis)
	python3 src/dynamics_calc.py collate $@ $^

$(dynamics_analysis_dir)/trajectory-Trimer-P1.00-%.h5: $(dynamics_sim)/trajectory-Trimer-P1.00-%.gsd
	sdanalysis --keyframe-interval 20_000 --wave-number 2.80 comp-dynamics $< $@

$(dynamcis_analysis_dir)/trajectory-Trimer-P13.50-%.h5: $(dynamics_sim)/trajectory-Trimer-P13.50-%.gsd
	sdanalysis --keyframe-interval 20_000 --wave-number 2.90 comp-dynamics $< $@

#
# Other Rules
#

relaxations: ## Compute the relaxation quantities of all values in the file data/analysis/dynamics.h5
	sdanalysis comp-relaxations data/analysis/dynamics_clean.h5

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
