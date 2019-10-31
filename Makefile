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
# All
#

.PHONY: analysis
analysis: model rates dynamics melting fluctuation fluctuation-disc thermo ## Perform all the analysis steps

#
# Machine Learning Rules
#

ml_data_dir = data/simulations/dataset/output

.PHONY: model
model: models/knn-trimer.pkl ## Train the machine learning model

models/knn-trimer.pkl:
	python3 src/models.py train-models $(ml_data_dir)

#
# Rates Rules
#

# Rates variables
#
# These describe the location of the experiments which determine the melting rate
# and where the resulting analysis will end up.

rates_sim = data/simulations/rates/output
rates_analysis_dir = data/analysis/rates

rates_trajectories = $(wildcard $(rates_sim)/dump-Trimer*.gsd)
rates_analysis = $(addprefix $(rates_analysis_dir)/, $(notdir $(rates_trajectories:.gsd=.h5)))

# Calculate the rate from the raw values
#
# This takes the raw values indicating the crystalline fraction and converts
# these into a melting rate.

.PHONY: rates rates-py
rates-py: data/analysis/rates_clean.h5 ## Compute the rate of melting using slow python version
	python3 src/melting_rates.py rates $<

rates: data/analysis/rates_rs_clean.h5 ## Compute the rate of melting using rust for analysis
	python3 src/melting_rates.py rates $<

# Clean up the analysis files
#
# This removes data which is deemed to be unusable. For the melting rates
# the primary reason for this is the particle is too small to behave like
# the bulk liquid. Another reason is the temperatures being above the
# spinodal point, where a melting rate doesn't make sense.

data/analysis/rates_clean.h5: data/analysis/rates.h5
	python3 src/melting_rates.py clean $<

data/analysis/rates_rs_clean.h5: data/analysis/rates_rs.h5
	python3 src/melting_rates.py clean $<

# Aggregate individual trajectories into a single file
#
# By having everything in a single file it is much simpler to perform further processing

data/analysis/rates.h5: $(rates_analysis)
	python3 src/melting_rates.py collate $@ $^

data/analysis/rates_rs.h5: $(rates_analysis:.h5=.csv)
	python3 src/melting_rates.py collate $@ $^

# Calculate the melting rates for each trajectory
#
# There are implementations for both using the python implementation,
# in addition to a rust implementation which is at least ~20x faster

$(rates_analysis_dir)/dump-%.h5: $(rates_sim)/dump-%.gsd | $(ml_model)
	python src/melting_rates.py melting --skip-frames 1 $< $@

$(rates_analysis_dir)/dump-%.csv: $(rates_sim)/dump-%.gsd | $(ml_model)
	trajedy --voronoi --skip-frames 1 $< $@ --training $(wildcard data/simulations/dataset/output/*.gsd)

#
# Melting Rules
#
# This is about understanding the interface for a range of crystals.
# While it is possible to calculate melting rates using these experiments
# there are no replications which makes it a poor estimate.
#
# The main purpose is to understand the behaviour of each of the
# different crystal structures.
#

# Melting Variables
#
# These specify the directory where the melting experiments take place
# and where the resulting analysis should end up.

melting_sim = data/simulations/interface/output
melting_analysis_dir = data/analysis/interface

melting_trajectories = $(wildcard $(melting_sim)/dump-Trimer*.gsd)
melting_analysis = $(addprefix $(melting_analysis_dir)/, $(notdir $(melting_trajectories:.gsd=.h5)))

# Commands
#
# These are the use facing commands for running the analysis of the melting.
# There are two sets of commands, one for a python version which has issues with performance,
# and another for a rust version which is at least ~20x faster.
# The python version is retained since it was used in developing the analysis and is more rigorous.

.PHONY: melting melting-py
melting-py: data/analysis/melting_clean.h5 ## Compute melting of the interface for a range of crystals using the slow python version
melting: data/analysis/melting_rs_clean.h5 ## Compute melting of the interface for a range of crystals

# Clean up the melting analysis
#
# This removes the values where the crystal is too small,
# so is likely to exhibit effects of the small size
# rather than a bulk crystal.
#
# It will also remove values which are above
# the spinodal point since their melting behaviour
# takes on very different properties.

data/analysis/melting_rs_clean.h5: data/analysis/melting_rs.h5
	python3 src/melting_rates.py clean $<

data/analysis/melting_clean.h5: data/analysis/melting.h5
	python3 src/melting_rates.py clean $<

# Collate the melting data
#
# This takes the data from each of the files analysing a single crystal
# and combines it to a single file which is much easier to work with.

data/analysis/melting_rs.h5: $(melting_analysis:.h5=.csv)
	python3 src/melting_rates.py collate $@ $^

data/analysis/melting.h5: $(melting_analysis)
	python3 src/melting_rates.py collate $@ $^

# Perform the melting analysis
#
# This is the step which calculates the size of the crystal at each point
# in time for a trajectory. It does each trajectory individually
# to better cope with errors arising in a simulation. When this does occur
# only a single simulation needs to be re-run.

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

# The minimum number of samples for each point in the resulting figure
# This is a trade-off between having points out to a long enough time period
# and having well averaged values.
dynamics_min_samples = 50

# Dynamics Command
#
# This defines the user facing commands to run, which is dynamics.
# There is the additional bootstrap rule, which performs
# the bootstrapping analysis, however this is more of an implementation
# detail of how the errors are estimated.

.PHONY: dynamics bootstrap
dynamics: bootstrap ## Compute dynamics quantities for all parameters of the trimer molecule
bootstrap: data/analysis/dynamics_clean_agg.h5

# Aggregation of values
#
# Values are aggregated using bootstrapping to estimate the mean
# and the confidence interval. Which is encoded as three values,
# mean, lower and upper, allowing for asymmetric errors.

data/analysis/dynamics_clean_agg.h5: data/analysis/dynamics_clean.h5
	python src/dynamics_calc.py bootstrap $<

# Cleaning Dataset
#
# Remove data points where there are fewer than a specified number of samples.
# This is to reduce the noise and ensure that the values are representative
# of the ensemble as a whole, rather than a few points within it.

data/analysis/dynamics_clean.h5: data/analysis/dynamics.h5
	python src/dynamics_calc.py clean --min-samples $(dynamics_min_samples) $<

# Collating Quantities
#
# Take the large number of input files and collate them to a single file
# which is much easier to deal with.

data/analysis/dynamics.h5: $(dynamics_analysis)
	python3 src/dynamics_calc.py collate $@ $^

# Dynamics Analysis
#
# Calculate the dynamic values for all the quantities.
# There are separate commands for each pressure, as I am using a different value
# of the wavenumber to track the structural relaxation for each quantity.

$(dynamics_analysis_dir)/trajectory-Trimer-P1.00-%.h5: $(dynamics_sim)/trajectory-Trimer-P1.00-%.gsd
	sdanalysis --keyframe-interval 1_000_000 --linear-steps 100 --wave-number 2.80 comp-dynamics $< $@

$(dynamics_analysis_dir)/trajectory-Trimer-P13.50-%.h5: $(dynamics_sim)/trajectory-Trimer-P13.50-%.gsd
	sdanalysis --keyframe-interval 1_000_000 --linear-steps 100 --wave-number 2.90 comp-dynamics $< $@

#
# Fluctuation analysis results
#

# Fluctuation Variables
#
# The variables for the analysis of the fluctuations. This requires two different
# experiments, the thermodynamics experiment which provides the thermodynamics
# experiment which provides the fluctuations of the Crystal while the dynamics
# experiment provides the fluctuations of the liquid.

thermo_sim = data/simulations/thermodynamics/output
dynamics_sim = data/simulations/dynamics/output
fluctuation_analysis_dir = data/analysis/fluctuation

fluctuation_trajectories = $(wildcard $(thermo_sim)/dump-Trimer*.gsd) $(wildcard $(dynamics_sim)/dump-Trimer*.gsd)
fluctuation_analysis = $(addprefix $(fluctuation_analysis_dir)/, $(notdir $(fluctuation_trajectories:.gsd=.h5)))
fluctuation_analysis_csv = $(addprefix $(fluctuation_analysis_dir)/, $(notdir $(fluctuation_trajectories:.gsd=.csv)))

disc_crystal_sim = data/simulations/disc_crystal/output
disc_liquid_sim = data/simulations/disc_liquid/output
fluctuation_disc_analysis_dir = data/analysis/fluctuation-disc

fluctuation_disc_trajectories = $(wildcard $(disc_crystal_sim)/dump-Disc-*.gsd) $(wildcard $(disc_liquid_sim)/dump-Disc-*.gsd)
fluctuation_disc_analysis_csv = $(addprefix $(fluctuation_disc_analysis_dir)/, $(notdir $(fluctuation_disc_trajectories:.gsd=.csv)))

# Commands
#
# There are two sets of commands, the python and the rust version. This analysis is very
# slow with the python analysis, and so the rust version is recommended for use.


.PHONY: fluctuation fluctuation-py
fluctuation-py: data/analysis/fluctuation.h5 ## Compute values for the fluctuation of the particles using the slow python version
fluctuation: data/analysis/fluctuation_rs.h5 ## Compute values for the fluctuation of the particles

fluctuation-disc: data/analysis/fluctuation_disc.h5

# Collation
#
# Take the large number of files generated and collate them into a single easy to use
# file.

data/analysis/fluctuation_rs.h5: $(fluctuation_analysis_csv)
	python3 src/fluctuations.py collate $@ $^

data/analysis/fluctuation.h5: $(fluctuation_analysis)
	python3 src/fluctuations.py collate $@ $^

data/analysis/fluctuation_disc.h5: $(fluctuation_disc_analysis_csv)
	python3 src/fluctuations.py collate-disc $@ $^

# Analysis
#
# There are rules for both the liquid and the crystal analysis,
# for each of the rust and python implementations.

$(fluctuation_analysis_dir)/dump-%.h5: $(thermo_sim)/dump-%.gsd | $(fluctuation_analysis_dir)
	python3 src/fluctuations.py analyse $< $@

$(fluctuation_analysis_dir)/dump-%.h5: $(dynamics_sim)/dump-%.gsd | $(fluctuation_analysis_dir)
	python3 src/fluctuations.py analyse $< $@

$(fluctuation_analysis_dir)/dump-%.csv: $(thermo_sim)/dump-%.gsd | $(fluctuation_analysis_dir)
	trajedy $< $@ -n 100

$(fluctuation_analysis_dir)/dump-%.csv: $(dynamics_sim)/dump-%.gsd | $(fluctuation_analysis_dir)
	trajedy $< $@ -n 100

$(fluctuation_disc_analysis_dir)/dump-%.csv: $(disc_crystal_sim)/dump-%.gsd | $(fluctuation_disc_analysis_dir)
	trajedy $< $@ -n 100

$(fluctuation_disc_analysis_dir)/dump-%.csv: $(disc_liquid_sim)/dump-%.gsd | $(fluctuation_disc_analysis_dir)
	trajedy $< $@ -n 100

$(fluctuation_analysis_dir):
	mkdir -p $@

$(fluctuation_disc_analysis_dir):
	mkdir -p $@

#
# Thermodynamics Analysis
#

.PHONY: thermo
thermo: data/analysis/thermodynamics.h5 ## Collate the thermodynamics into a single file

data/analysis/thermodynamics.h5: $(wildcard $(thermo_sim)/thermo*.log) $(wildcard $(dynamics_sim)/thermo*.log)
	python3 src/fluctuations.py thermodynamics $@ $^

#
# Other Rules
#

interface-dynamics: ## Compute the dynamics of a simulation with a liquid--crystal interface in data/simulations/2017-09-04-interface/
	sdanalysis comp-dynamics -o data/analysis/interface data/simulations/interface/output/dump-*

test: ## Test the functionality of the helper modules in src
	python -m pytest src

pack-dataset: ## Pack the relevant files from dataset into a tarball
	cd data/simulations/dataset/output && tar cvJf dataset.tar.xz dump-*.gsd && mv dataset.tar.xz ../../../

#
# Notebook rules
#

# Variables
#
# This finds all the notebooks in the notebooks directory

all_notebooks = $(wildcard notebooks/*.md)

.PHONY: notebooks
notebooks: analysis $(all_notebooks:.md=.ipynb) ## Run all notebooks

.PHONY: sync
sync: ## Synchronise and format the juptyer and markdown representations of the notebooks
	jupytext --set-formats ipynb,md notebooks/*.md
	jupytext --set-formats ipynb,md notebooks/*.ipynb
	jupytext --sync --pipe black notebooks/*.ipynb

%.ipynb: %.md
	cd $(dir $<) && jupytext --to notebook --execute $(notdir $<)

#
# Reports
#

# Variables
#
# These find all the different items which are going to become reports. I am converting
# all the notebooks to a pdf report, as well as markdown files in the reports directory.
#
# Additionally where there are figures which need converting from svg to pdf, this takes
# finds the targets.

report_targets = $(patsubst %.md, %.pdf, $(wildcard reports/*.md)) $(patsubst %.md, %.pdf, $(wildcard notebooks/*.md))
all_figures = $(wildcard figures/*.svg)

# Commands
#
# There are a couple of commands here. Generating all the figures and additionally
# creating the pdf reports.

.PHONY: figures reports
figures: notebooks ## Generate all the figures in the figures directory
reports: $(report_targets) analysis notebooks ## Generate pdf reports

# Conversions
#
# These are the rules to convert the different filetypes to a pdf

%.pdf: %.md
	cd $(dir $<); pandoc $(notdir $<) --pdf-engine=tectonic --filter ../src/pandoc-svg.py --filter pandoc-crossref -o $(notdir $@)

#
# Help
#
# The magic which provides some help output when running make by itself

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# vim:ft=make
#
