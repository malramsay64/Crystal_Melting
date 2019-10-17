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

rates-py: data/analysis/rates_clean.h5 ## Compute the rate of melting using slow python version
	python3 src/melting_rates.py rates $<

rates: data/analysis/rates_rs_clean.h5 ## Compute the rate of melting using rust for analysis
	python3 src/melting_rates.py rates $<

data/analysis/rates_clean.h5: data/analysis/rates.h5
	python3 src/melting_rates.py clean $<

data/analysis/rates_rs_clean.h5: data/analysis/rates_rs.h5
	python3 src/melting_rates.py clean $<

data/analysis/rates.h5: $(rates_analysis)
	python3 src/melting_rates.py collate $@ $^

data/analysis/rates_rs.h5: $(rates_analysis:.h5=.csv)
	python3 src/melting_rates.py collate $@ $^

$(rates_analysis_dir)/dump-%.h5: $(rates_sim)/dump-%.gsd | $(ml_model)
	python src/melting_rates.py melting --skip-frames 1 $< $@

$(rates_analysis_dir)/dump-%.csv: $(rates_sim)/dump-%.gsd | $(ml_model)
	trajedy --voronoi --skip-frames 1 $< $@ --training $(wildcard data/simulations/dataset/output/*.gsd)

#
# Melting Rules -> This is about understanding the interface for a range of crystals
#

melting_sim = data/simulations/interface/output
melting_analysis_dir = data/analysis/interface

melting_trajectories = $(wildcard $(melting_sim)/dump-Trimer*.gsd)
melting_analysis = $(addprefix $(melting_analysis_dir)/, $(notdir $(melting_trajectories:.gsd=.h5)))

melting-py: data/analysis/melting_clean.h5 ## Compute melting of the interface for a range of crystals using the slow python version

melting: data/analysis/melting_rs_clean.h5 ## Compute melting of the interface for a range of crystals

data/analysis/melting_rs_clean.h5: data/analysis/melting_rs.h5
	python3 src/melting_rates.py clean $<

data/analysis/melting_rs.h5: $(melting_analysis:.h5=.csv)
	python3 src/melting_rates.py collate $@ $^

data/analysis/melting_clean.h5: data/analysis/melting.h5
	python3 src/melting_rates.py clean $<

data/analysis/melting.h5: $(melting_analysis)
	python3 src/melting_rates.py collate $@ $^

$(melting_analysis_dir)/dump-%.h5: $(melting_sim)/dump-%.gsd | $(ml_model)
	python src/melting_rates.py melting --skip-frames 100 $< $@

$(melting_analysis_dir)/dump-%.csv: $(melting_sim)/dump-%.gsd | $(ml_model)
	trajedy --voronoi --skip-frames 100 $< $@ --training $(wildcard data/simulations/dataset/output/*.gsd)

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
# Fluctuation analysis results
#

thermo_sim = data/simulations/thermodynamics/output
dynamics_sim = data/simulations/dynamics/output
fluctuation_analysis_dir = data/analysis/fluctuation

fluctuation_trajectories = $(wildcard $(thermo_sim)/dump-Trimer*.gsd) $(wildcard $(dynamics_sim)/dump-Trimer*.gsd)
fluctuation_analysis = $(addprefix $(fluctuation_analysis_dir)/, $(notdir $(fluctuation_trajectories:.gsd=.h5)))
fluctuation_analysis_csv = $(addprefix $(fluctuation_analysis_dir)/, $(notdir $(fluctuation_trajectories:.gsd=.csv)))

fluctuation-py: data/analysis/fluctuation.h5 ## Compute values for the fluctuation of the particles using the slow python version

fluctuation: data/analysis/fluctuation_rs.h5 ## Compute values for the fluctuation of the particles

data/analysis/fluctuation_rs.h5: $(fluctuation_analysis_csv)
	python3 src/fluctuations.py collate $@ $^

data/analysis/fluctuation.h5: $(fluctuation_analysis)
	python3 src/fluctuations.py collate $@ $^

$(fluctuation_analysis_dir)/dump-%.h5: $(thermo_sim)/dump-%.gsd | $(fluctuation_analysis_dir)
	python3 src/fluctuations.py analyse $< $@

$(fluctuation_analysis_dir)/dump-%.h5: $(dynamics_sim)/dump-%.gsd | $(fluctuation_analysis_dir)
	python3 src/fluctuations.py analyse $< $@

$(fluctuation_analysis_dir)/dump-%.csv: $(thermo_sim)/dump-%.gsd | $(fluctuation_analysis_dir)
	trajedy $< $@ -n 100

$(fluctuation_analysis_dir)/dump-%.csv: $(dynamics_sim)/dump-%.gsd | $(fluctuation_analysis_dir)
	trajedy $< $@ -n 100

$(fluctuation_analysis_dir):
	mkdir -p $@

#
# Thermodynamics Analysis
#

thermo: data/analysis/thermodynamics.h5 ## Collate the thermodynamics into a single file

data/analysis/thermodynamics.h5: $(wildcard $(thermo_sim)/thermo*.log) $(wildcard $(dynamics_sim)/thermo*.log)
	python3 src/fluctuations.py thermodynamics $@ $^

#
# Other Rules
#

relaxations: dynamics ## Compute the relaxation quantities of all values in the file data/analysis/dynamics.h5
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
notebooks: $(all_notebooks:.md=.ipynb) ## Run all notebooks

.PHONY: sync
sync: ## Synchronise and format the juptyer and markdown representations of the notebooks
	jupytext --set-formats ipynb,md notebooks/*.md
	jupytext --set-formats ipynb,md notebooks/*.ipynb
	jupytext --sync --pipe black notebooks/*.ipynb

%.ipynb: %.md
	cd $(dir $<) && jupytext --to notebook --execute $(notdir $<)

.PHONY: figures
figures: notebooks ## Generate all the figures in the figures directory

report_targets := $(wildcard reports/*.md)
all_figures := $(wildcard figures/*.svg)

convert_figures: $(all_figures:.svg=.pdf)

reports: $(report_targets:.md=.pdf) ## Generate pdf reports
	echo $<

%.pdf: %.md $(all_figures:.svg=.pdf)
	cd $(dir $<); pandoc $(notdir $<) --filter pandoc-crossref -o $(notdir $@)

%.pdf: %.svg
	cairosvg $< -o $@

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# vim:ft=make
#
