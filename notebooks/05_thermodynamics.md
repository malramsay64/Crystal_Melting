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

# Thermodynamics of Liquids and Crystals

The thermodynamic quantities of the liquid and crystal phases
are important for understanding the rates of attachment to the interface.
This notebook provides a look at
the thermodynamic properties of the liquid and crystal phases.

The thermodynamics of the liquid phase
are taken from the production runs of the dynamics simulations,
while the thermodynamic properties of the crystal phase
are generated specifically in the `thermodynamics` simulation directory.

```python
import pandas
from pathlib import Path
import itertools
import altair as alt
import sys

sys.path.append("../src")
import figures

from sdanalysis.util import get_filename_vars
```

## Data

The thermodynamics quantities have been aggregated from the directories

- `../data/simulations/thermodynamics/output`
- `../data/simulations/dynamics/output`

for the crystal, and liquid configurations respectively.

Using the `make thermo` command these quantities
have been aggregated into
the `../data/analysis/thermodynamics.h5` file under the table name `"thermo"`.

```python
df = pandas.read_hdf("../data/analysis/thermodynamics.h5", "thermo").sort_index()
```

## Analysis

One of the key results of this analysis
is the potential energy of each configuration
which is shown below.

```python
df_U = df.loc[
    :,
    [
        ("potential_energy", "mean"),
        ("kinetic_energy", "mean"),
        ("volume", "mean"),
        ("enthalpy", "mean"),
    ],
]
df_U.columns = ["potential_energy", "kinetic_energy", "volume", "enthalpy"]
```

with the consequence of this result,
I will export it to a csv file.

```python
df_U.to_csv("../results/enthalpy.csv")
```

Taking the analysis a step further,
the energy difference between
the liquid and crystal configurations
at the melting point
is important for understanding the melting behaviour.

```python
melting_points = pandas.read_csv("../results/melting_points.csv")
melting_points
```

```python
def energy_diff(df, pressure, temperature, crystal, energy="potential_energy"):
    """Calculate the difference in energy between the crystal and liquid states.

    This uses the thermodynamic data in `df` to find the energy saved by the crystal configuration
    at a given temperature. This is a helper utility to make this task simpler.

    Args:
        df: The dataframe holding the aggregated thermodynamics data
        pressure: The desired pressure at which to find the values
        temperature: The desired temperature to use
        crystal: The crystal structure which is to be compared with the liquid.
        energy: The name of the column which the energy is stored

    """
    return (
        df.loc[(pressure, temperature, crystal), energy]
        - df.loc[(pressure, temperature, "liquid"), energy]
    )
```

The p2 crystal is the most stable,
so we can compare the free energy to that of the liquid.

```python
crystal = "p2"
for _, row in melting_points.iterrows():
    print(
        f"For {crystal} at P={row.pressure} and T={row.melting_point}",
        f"ΔE={energy_diff(df_U, row.pressure, row.melting_point, crystal):.3f},",
        f"ΔH={energy_diff(df_U, row.pressure, row.melting_point, crystal, 'enthalpy'):.3f}",
    )
```

## Figures

It is highly informative to see how these values change over time.

```python
def plot_thermo(df, quantity):
    """Helper function to plot a thermodynamic quantity as a function of temperature.

    This will plot both the mean and the standard deviation.

    Args:
        df: A dataframe containing the aggregates quantities (mean and std).
        quantity: The quantity to plot

    """
    c = alt.Chart(df[quantity].reset_index()).encode(
        x=alt.X("temperature:Q", scale=alt.Scale(zero=False)),
        y=alt.Y("mean", title=quantity, scale=alt.Scale(zero=False)),
        yError=alt.YError("std"),
        color="crystal:N",
        shape="pressure:N",
    )
    return c.mark_errorbar() + c.mark_point()
```

Using the helper function we can plot the quantities
for both pressures.
The

```python
c = plot_thermo(df.query("pressure == 13.50"), "potential_energy").transform_filter(
    alt.datum.temperature < 1.60
)
with alt.data_transformers.enable("default"):
    c.save("../figures/thermodynamics_potential_energy_P13.50.svg", webdriver="firefox")
c
```

```python
c = plot_thermo(df.query("pressure == 1.00"), "potential_energy").transform_filter(
    alt.datum.temperature < 0.55
)
with alt.data_transformers.enable("default"):
    c.save("../figures/thermodynamics_potential_energy_P1.00.svg", webdriver="firefox")
c
```
