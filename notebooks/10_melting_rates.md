---
jupyter:
  jupytext:
    formats: ipynb,md
    target_format: ipynb,md
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

# Understanding Methods of Computing Melting Rates

I need a method of computing the melting rate of the crystal structures
which will provide a good estimation of both
the melting rate and
the error of the melting rate.
Importantly I need an algorithm able to detect
the small melting rates at low temperatures.

```python
import numpy as np
import pandas
import altair as alt
from crystal_analysis import figures
```

## Load Data

The data on the melting rates has been pre-calculated and saved to a file.
The data stored is the simulation conditions along with the values

- fraction: The fraction of the simulation cell which is crystalline in nature
- surface_area: The perimeter (the 2D equivalent of surface area) of the crystalline cluster
- volume: The area (2D equivalent of area) of the crystalline cluster
- time: The timestep at which these values apply.

Only the data from the low pressure melting is used in this analysis
since at the time of writing the dataset is better
and it is easier to only deal with a single set of pressures.
I am also limiting the analysis to only the p2 crystal.

```python
# Read file with melting data
with pandas.HDFStore("../data/analysis/rates_rs_clean.h5") as store:
    melting_df = store.get("fractions")
```

## Volume Data

I have plotted the volume of the crystal as a function of time below.
The important point to note is the high levels of noise in the data,
which is a combination the thermal fluctuations and
the inaccuracy of the algorithm I am using for classification.

```python
chart = (
    alt.Chart(melting_df)
    .mark_line(opacity=0.7)
    .encode(
        x=alt.X(
            "time:Q",
            title="Time",
            axis=alt.Axis(format="e"),
            scale=alt.Scale(type="linear"),
        ),
        color=alt.Color("temperature:N", title="Temperature"),
        row=alt.Row("pressure:N", title="Pressure"),
        detail="iter_id",
        y=alt.Y("radius:Q", title="Estimated Radius"),
    )
)
chart
```

```python
with alt.data_transformers.enable("default"):
    chart.save("../figures/melting_radius.svg", webdriver="firefox")
```

## Calculating $\Delta V/ \Delta t$

The quantity we are interested in is rate of melting for a given surface area,
$$
\tau_M = -\frac{1}{A} \frac{\Delta V}{\Delta t}
$$
since our expectation is that $\tau_M$ is a constant value,
fitting a straight line to $\Delta V/\Delta t$ will give
a value of $\tau_M$ which is dependent on the surface area.

This documents my attempts at calculating this value with a small error.

### Averaging Instantaneous gradient

This is calculating the instantaneous gradient
$\frac{1}{A(t)} \frac{V(t+\Delta t) - V(t)}{\Delta t}$
and averaging over all values of $t$.
The errors being calculated using the standard deviation.
The gradient is computed using the `np.gradient` function which
documents the solver [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.gradient.html#numpy.gradient).

```python
with pandas.HDFStore("../data/analysis/rates_rs_clean.h5") as store:
    rates_df = store.get("rates")
```

Plotting the melting rate with the calculated errors as a function of temperature.

```python
chart = (
    alt.Chart(rates_df.assign(abs_mean=rates_df["mean"].abs()))
    .mark_point()
    .encode(
        x=alt.X("temp_norm:Q", title="T/Tₘ", scale=alt.Scale(zero=False)),
        y=alt.Y(
            "abs_mean",
            title="Crystal Melting Rate",
            axis=alt.Axis(format="e"),
            scale=alt.Scale(type="log"),
        ),
        color=alt.Color("pressure:N", title="Pressure"),
    )
    .transform_filter(alt.datum.abs_mean > 0)
)

# Add horizontal line at y = 0
# chart = figures.hline(chart, 0)

chart
```

```python

```
