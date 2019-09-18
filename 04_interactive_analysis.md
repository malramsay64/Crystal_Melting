---
jupyter:
  jupytext:
    formats: ipynb,md
    target_format: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.0
  kernelspec:
    display_name: crystal
    language: python
    name: crystal
---

# Dynamics Summary

This is a summary of the dynamics quantities
from a simulation presented in a way
that is simple to check that the values are sensible
before continuing with further analysis.

```python
import pandas as pd
import altair as alt

# This is for the interactive elements
import ipywidgets as widgets
from pathlib import Path
from sdanalysis.util import get_filename_vars
import matplotlib.pyplot as plt

import sys

sys.path.append("../src")
import figures
```

<!-- #region -->
The processing of the dataset
has to be performed before this notebook will display anything useful.
This processing can be performed using the command

```sh
make dynamics
```

in the root of this project directory.
<!-- #endregion -->

```python
# Reading the processed dataset which is generated by running `make dynamics`
df = pd.read_hdf("../data/analysis/dynamics_clean_agg.h5", "dynamics")

# Process the dataset which involves
# - removing times <= 0 which don't work with log scales
# - Averaging over all the instances of start index
df = (
    df.query("time > 0")
    .groupby(["temperature", "pressure", "time"])
    .mean()
    .reset_index()
)
```

Since this notebook is focused on an interactive summary of the data,
the next cell selects every 5th value from the dataset.
This behaviour can be changed by changing the `every_nth` variable,
note however, that reducing this value will increase the time for a figure to load.

```python
every_nth = 5
# Create a sub-dataset using only every 5th value
df_plot = (
    df.groupby(["temperature", "pressure"])
    .nth(list(range(0, len(df), every_nth)))
    .reset_index()
)
```

This is the part where the generation of the figure takes place.
The selection of the pressure
and the dynamic quantity is designed to be interactive,
selecting the value of interest.

```python
pressures = widgets.ToggleButtons(description="Pressure", options=df.pressure.unique())

# The metadata columns, or those we are using for the other axes of the figure
# shouldn't be available for plotting since they are non-sensical.
metadata_cols = ["temperature", "pressure", "start_index", "time"]
axes = widgets.ToggleButtons(
    description="Quantity",
    options=[col for col in df.columns if col not in metadata_cols],
)


@widgets.interact(pressure=pressures, axis=axes)
def create_chart(pressure, axis):
    c = (
        alt.Chart(df_plot.query("pressure==@pressure"), width=600, height=500)
        .mark_point(size=75)
        .encode(
            alt.X("time", axis=alt.Axis(format="e"), scale=alt.Scale(type="log")),
            alt.Y(axis, type="quantitative"),
            alt.Color("temperature:N"),
        )
    )
    # Set log log scale for displacement figures
    if axis in ["msd", "mfd", "mean_displacement"]:
        c.encoding.y.scale = alt.Scale(type="log")

    return c
```

```python
file = "../data/simulations/dynamics/output/thermo-Trimer-P1.00-T0.30.log"
discard_quantities = ["timestep", "lz", "xz", "yz", "N"]
with open(file) as src:
    thermo_quantities = [c.strip() for c in src.readline().split("\t")]
    thermo_quantities = [c for c in thermo_quantities if c not in discard_quantities]
    thermo_quantities = [c for c in thermo_quantities if "rigid_center" not in c]
```

```python
from sdanalysis.figures.interactive_config import parse_directory

dataset = parse_directory(
    Path("../data/simulations/thermodynamics/output/"), glob="thermo*.log"
)
```

```python
pressure = widgets.ToggleButtons(description="Pressure", options=list(dataset.keys()))
temperature = widgets.ToggleButtons(
    description="Temperature", options=list(dataset.get(pressure.value).keys())
)
crystals = widgets.ToggleButtons(
    description="Crystal",
    options=list(dataset.get(pressure.value).get(temperature.value).keys()),
)
quantities = widgets.ToggleButtons(description="Quantity", options=thermo_quantities)


def update_temperatures(change):
    temperature.options = list(dataset.get(change.new).keys())


def update_crystals(change):
    crystals.options = list(dataset.get(pressure.value).get(change.new).keys())


pressure.observe(update_temperatures, names="value")
temperature.observe(update_crystals, names="value")

@widgets.interact(
    pressure=pressure, temperature=temperature, crystal=crystals, quantity=quantities
)
def plot_figure(pressure, temperature, crystal, quantity):
    print(pressure, temperature, crystal)
    filename = dataset.get(pressure).get(temperature).get(crystal).get("None")
    df = pd.read_csv(
        filename, sep="\t", index_col="timestep", usecols=["timestep", quantity]
    )

    df = df[~df.index.duplicated(keep="first")].reset_index()
    df.index = pd.to_timedelta(df.timestep)
    df = df.drop(columns="timestep").resample("0.1ms").agg(["mean", "std"])
    df.columns = [col[-1] for col in df.columns.values]
    df = df.reset_index()
    df["timestep"] = df["timestep"].astype(int)
    c = alt.Chart(df).encode(
        x=alt.X("timestep", axis=alt.Axis(format="e")),
        y=alt.Y("mean", title=quantity, scale=alt.Scale(zero=False)),
        yError=alt.YError("std"),
    )
    c = c.mark_line() + c.mark_errorband()
    return c
```