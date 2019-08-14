---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.1
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

figures.use_my_theme()
alt.data_transformers.enable("csv")

from sdanalysis.util import get_filename_vars
```

```python
dir_crystal = Path("../data/simulations/thermodynamics/output")
dir_liquid = Path("../data/simulations/dynamics/output")

dfs = []

for filename in itertools.chain(
    dir_crystal.glob("thermo*.log"), dir_liquid.glob("thermo*.log")
):
    fvars = get_filename_vars(filename)

    df = pandas.read_csv(filename, sep="\t")
    df = df.drop_duplicates("timestep", keep="last")
    df = df.div(df.N, axis=0)

    df["pressure"] = fvars.pressure
    if fvars.crystal is not None:
        df["crystal"] = fvars.crystal
    else:
        df["crystal"] = "liquid"

    df["temperature"] = fvars.temperature
    df = df.set_index(["pressure", "temperature", "crystal"])
    dfs.append(df.mean(level=[0, 1, 2]))


df_thermo = pandas.concat(dfs)
```

```python
def energy_diff(df, pressure, temperature, crystal):
    return (
        df.loc[(pressure, temperature, crystal), "potential_energy"]
        - df.loc[(pressure, temperature, "liquid"), "potential_energy"]
    )
```

```python
energy_diff(df_thermo, "1.00", "0.36", "p2")
```

```python
energy_diff(df_thermo, "13.50", "1.35", "p2")
```

```python
alt.Chart(df_thermo.reset_index()).mark_point().encode(
    x="temperature:Q", y="potential_energy", color="crystal", shape="pressure"
)
```

```python

```
