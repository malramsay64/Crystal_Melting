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
# Machine Learning Rules
#

ml_model = models/knn-trimer.pkl
ml_data_dir = data/simulations/dataset/output

.PHONY: model
model: $(ml_model) ## Train the machine learning model

$(ml_model):
	python3 src/models.py train-models $(ml_data_dir)

#
# Rates Rules
#

rates_sim = data/simulations/rates/output
rates_analysis_dir = data/analysis/rates

rates_trajectories = $(wildcard $(rates_sim)/dump-Trimer*.gsd)
rates_analysis = $(addprefix $(rates_analysis_dir)/, $(notdir $(rates_trajectories:.gsd=.h5)))

rates: data/analysis/rates_clean.h5 ## Compute the rate of melting
	python3 src/melting_rates.py rates $<

data/analysis/rates_clean.h5: data/analysis/rates.h5
	python3 src/melting_rates.py clean $<

data/analysis/rates.h5: $(rates_analysis)
	python3 src/melting_rates.py collate $@ $^

$(rates_analysis_dir)/dump-%.h5: $(rates_sim)/dump-%.gsd | $(ml_model)
	python src/melting_rates.py melting --skip-frames 1 $< $@

#
# Melting Rules
#

melting_sim = data/simulations/interface/output
melting_analysis_dir = data/analysis/interface

melting_trajectories = $(wildcard $(melting_sim)/dump-Trimer*.gsd)
melting_analysis = $(addprefix $(melting_analysis_dir)/, $(notdir $(melting_trajectories:.gsd=.h5)))

melting: data/analysis/melting_clean.h5 ## Compute melting rates of the simulations in the directory data/simulations/melting

data/analysis/melting_clean.h5: data/analysis/melting.h5
	python3 src/melting_rates.py clean $<

data/analysis/melting.h5: $(melting_analysis)
	python3 src/melting_rates.py collate $@ $^

$(melting_analysis_dir)/dump-%.h5: $(melting_sim)/dump-%.gsd | $(ml_model)
	python src/melting_rates.py melting --skip-frames 100 $< $@

#
# Dynamics Rules
#

dynamics_sim = data/simulations/dynamics/output
dynamics_analysis_dir = data/analysis/dynamics

dynamics_trajectories = $(wildcard $(dynamics_sim)/trajectory-Trimer-*.gsd)
dynamics_analysis = $(addprefix $(dynamics_analysis_dir)/, $(notdir $(dynamics_trajectories:.gsd=.h5)))

dynamics = data/analysis/dynamics.h5
dynamics_clean = data/analysis/dynamics_clean.h5

dynamics: bootstrap ## Compute dynamics quantities for all parameters of the trimer molecule

bootstrap: ${dynamics_clean}
	python src/dynamics_calc.py bootstrap $<

${dynamics_clean}: ${dynamics}
	python src/dynamics_calc.py clean --min-samples 50 $<

${dynamics}: $(dynamics_analysis)
	echo $(dynamics_analysis)
	python3 src/dynamics_calc.py collate $@ $^

$(dynamics_analysis_dir)/trajectory-Trimer-P1.00-%.h5: $(dynamics_sim)/trajectory-Trimer-P1.00-%.gsd
	sdanalysis --keyframe-interval 1_000_000 --linear-steps 100 --wave-number 2.80 comp-dynamics $< $@

$(dynamics_analysis_dir)/trajectory-Trimer-P13.50-%.h5: $(dynamics_sim)/trajectory-Trimer-P13.50-%.gsd
	sdanalysis --keyframe-interval 1_000_000 --linear-steps 100 --wave-number 2.90 comp-dynamics $< $@

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

#
# Notebook rules
#

all_notebooks = $(wildcard notebooks/*.md)

.PHONY: notebooks
notebooks: $(all_notebooks:.md=.ipynb)

.PHONY: sync
sync:
	jupytext --set-formats ipynb,md notebooks/*.md
	jupytext --set-formats ipynb,md notebooks/*.ipynb
	jupytext --sync --pipe black notebooks/*.ipynb

%.ipynb: %.md
	cd $(dir $<) && jupytext --to notebook --execute $(notdir $<)

.PHONY: figures
figures: notebooks ## Generate all the figures in the figures directory

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
