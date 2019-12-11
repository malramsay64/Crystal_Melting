---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.0
  kernelspec:
    display_name: crystal
    language: python
    name: crystal
---

# Fluctuations

The fluctuations are an important part
of understanding the different materials I am working with.
It provides an indication of how easy it is
for a particle in one state to transition to another.

There are two different fluctuations I am measuring

1. The fluctuations of the trimer,
   which are measured using an orientational order parameter.
2. The fluctuations of 2D Lennard-Jones discs
   which are measured using a bond orientational order parameter.

```python
import pandas as pd
import altair as alt
import numpy as np
import sys

sys.path.append("../src")
import figures
import fluctuations

%matplotlib inline
import matplotlib.pyplot as plt
```

## Trimer Fluctuations

These are the fluctuations of the trimer molecule
which have been calculated using the command `make fluctuations`
generating the file `../data/analysis/fluctuation_rs.h5`.
This command calculates the orientational order parameter

$$ O_\theta = \frac{1}{N} \langle \sum_{i=1}^N \cos^2(\theta_i - \theta_0) \rangle $$

for each molecule many configurations.
The file we are analysing contains the distribution
of $O_\theta$ over all configurations.
Each different simulation condition has it's own
distribution of orientational order.

```python
# Read in and select the data that we want

file = "../data/analysis/fluctuation_rs.h5"
with pd.HDFStore(file) as src:
    df = src.get("ordering")
# The range of conditions in which we have reasonable values
df = df.query("pressure == 13.50 and 1.30 < temperature and temperature < 1.8")
# Only want to compare the liquid and the p2 crystal.
df = df[df["crystal"].isin(["liquid", "p2"])]
```

```python
c = (
    alt.Chart(df)
    .mark_line()
    .encode(
        x=alt.X("bins", title="Orientation Order"),
        y=alt.Y("mean(count)", title="Distribution"),
        color=alt.Color("crystal", title="Crystal"),
    )
)
with alt.data_transformers.enable("default"):
    c.save("../figures/fluctuation_distribution.svg")
```

![The distribution of the orientational order parameter for the trimer liquid
and the p2 crystal. The values for all the simulations
have been aggregated onto the same line
as a result of being nearly identical in this overview.
](../figures/fluctuation_distribution.svg)

## Disc Fluctuations

The fluctuations for the disc are measured using
the bond orientational order $\psi_k$
which is typically written as

$$ \psi_k = \frac{1}{k} \sum_j^n \exp{i k \theta} $$

however this gives a complex number,
which we can't use directly,
and many publications neglect to mention
the transformation to convert to a real number.
So I am going to include it in the formulation.

$$ \psi_k = |\frac{1}{k} \sum_j^n \exp{i k \theta}| $$

The command `make fluctuation-disc` will calculate
the fluctuations of the bond orientational order parameter
for all the simulations,
which are stored in the file `../data/analysis/fluctuation_disc.h5`.

```python
file = "../data/analysis/fluctuation_disc.h5"
df_disc = pd.read_hdf(file, "ordering")
df_disc.loc[df_disc.crystal.isna(), "crystal"] = "liquid"
df_disc = df_disc.query("temperature > 0.1")
```

```python
c = (
    alt.Chart(df_disc)
    .mark_line()
    .encode(x="bins", y="count", color="crystal", row="temperature")
)
with alt.data_transformers.enable("default"):
    c.save("figures/fluctuation_disc_all.svg", webdriver="firefox")
```

![The fluctuations of all the disc configurations shows that
nearly all the conditions only exhibit a single phase.
That is, the transformation from the liquid to the crystal
took place faster than I could characterise the metastable state.
](figures/fluctuation_disc_all.svg)

With only two configurations demonstrating separate behaviour
from the metastable phase,
for clarity I am going to plot just one of them,
being the temperature at 0.53.

```python
c = (
    alt.Chart(df_disc.query("temperature == 0.53"))
    .mark_line()
    .encode(x="bins", y="count", color="crystal")
)
with alt.data_transformers.enable("default"):
    c.save("figures/fluctuation_disc.svg", webdriver="firefox")
```

![Fluctuations of the bond orientational order parameter
of the disc at a temperature of 0.53
](figures/fluctuation_disc.svg)

## Conclusion

There is a distinct difference between
the fluctuation behaviour of the trimer
compared to the disc.
The trimer has a significantly narrower fluctuation,
with a very clear delineation
between the liquid and crystal phases.
This is not present in the disc,
with a significant overlap between the two distributions.
