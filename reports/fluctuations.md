# Fluctuations

It seems that the fluctuations which are observed
within the crystal are really limited
making the Crystal stable at large supercoolings.
As a method of investigating this
I am looking at the range of fluctuations of local environments.

To measure the fluctuations,
I am reducing the local environment
to a 1D measure of order $O_\theta$,

$$ O_\theta = \frac{1}{N} \langle \sum_{i=1}^N \cos^2(\theta_i - \theta_0) \rangle $$

where $N$ is the number of nearest neighbours.
For this 2D case,
the optimal number of neighbours is six,
so this quantity is calculated over
the six closest neighbours of each particle.
The $\cos^2$ means that orientations of both 0 and 180
have the maximum score of 1.

This value $O_\theta$ can be calculated
for all local environments
in a simulation of the crystal configuration
and for the liquid configuration
at a range of simulation conditions.

## Thermodynamics Analysis

While the idea of the analysis of the fluctuations
is that the thermodynamics are not driving the stability,
the results are not in isolation,
and the potential energy difference between
the liquid and the crystal phases is important.

<div id="fig:thermodynamics" class="subfigures">

![P=13.50
](../figures/thermodynamics_potential_energy_P13.50.pdf){#fig:thermodynamics_potential_energy_P13.50}

![P=1.00
](../figures/thermodynamics_potential_energy_P1.00.pdf){#fig:thermodynamics_potential_energy_P2.00}

Thermodynamics as a function of temperature for both sets of pressures.

</div>

@Fig:thermodynamics shows the energy of the pg crystal
sits between that of the p2/p2gg crystals and the liquid.
Additionally it is interesting to note that both the p2 and p2gg crystals
have nearly identical potential energies.

## Fluctuation Histograms

![The distribution of orientational order for configurations over a range of
temperatures and pressures separated into liquid and crystal.
](../figures/fluctuation_distribution.pdf){#fig:fluctuation_distribution}

@Fig:fluctuation_distribution demonstrates the difference in configuration space
between the liquid and the crystal.
The liquid has a very wide distribution,
with most of the values lying between 0.30 and 0.85,
while nearly all values of the crystal
are between 0.95 and 1.0.
