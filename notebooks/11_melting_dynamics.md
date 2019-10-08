---
jupyter:
  jupytext:
    formats: ipynb,md
    target_format: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: crystal
    language: python
    name: crystal
---

# Melting Dynamics

This is a notebook to understand the dynamics of crystal melting, in particular how much of the slowdown of the melting can be attributed to the slow dynamics.

```python
# Import required modules
import pandas
import numpy as np
import altair as alt
from ipywidgets import interact, ToggleButtons, IntSlider
from uncertainties import unumpy

import sys

sys.path.append("../src")
import figures
```

## Input data

The data for this notebook is sourced from the calculations of
the relaxation times which can be run using
the Makefile in the home directory of this repository.
The command `make relaxations` will compute the relaxation values
required for this analysis.

The calculation of the melting rates is performed in
the [Crystal_Clustering](Crystal_Clustering.ipynb) notebook
from calculations of crystal fractions from running `make melting`.

```python
with pandas.HDFStore("../data/analysis/dynamics_clean_agg.h5") as src:
    mol_relax_df = src.get("molecular_relaxations")

with pandas.HDFStore("../data/analysis/dynamics_clean_agg.h5") as src:
    relax_df = src.get("relaxations")

with pandas.HDFStore("../data/analysis/rates_rs_clean.h5") as src:
    melting_df = src.get("rates")

timescales_df = mol_relax_df.merge(melting_df, on=["temperature", "pressure"])
```

## Timescale normalised rates

We want to be able to determine whether the timescale of the dynamics
is solely responsible for the drastic slowdown in melting rate that we observe.
We have the melting rate,
which is the distance the interface travels per unit time

$ \frac{d}{t} $

The other quantity is the rotational relaxation time
which is the timescale on which the rotations take place.

$ {t} $

In multiplying these two quantities together
we should get the distance the interface moves
in the characteristic timescale.

```python
c = (
    alt.Chart(timescales_df)
    .mark_point()
    .encode(
        x=alt.X("temp_norm", title="T/Tₘ", scale=alt.Scale(zero=False)),
        y=alt.Y(
            "scaling:Q",
            title="Melting Rate * Rotational Relaxation",
            axis=alt.Axis(format="e"),
        ),
        color=alt.Color("pressure:N", title="Pressure"),
    )
    .transform_calculate(scaling=alt.datum.mean * alt.datum.tau_T2_value)
)

c
```

```python
import scipy.optimize
```

```python
def fit_curve(x_vals, y_vals, errors=None, delta_E=None):
    if delta_E is None:

        def theory(x, c, d):
            result = 1 - np.exp((1 - x) * d / x)
            return c * result

    else:

        def theory(x, c):
            result = 1 - np.exp((1 - x) * delta_E / x)
            return c * result

    opt, err = scipy.optimize.curve_fit(
        theory, x_vals, y_vals, sigma=errors, maxfev=2000
    )

    return theory, opt, err
```

```python
df_1 = timescales_df.query("pressure == 1.00")


df_fit = pandas.DataFrame(
    {"temp_norm": df_1.temp_norm, "scaling": df_1["mean"] * df_1["tau_T2_value"]}
)

theory, opt, err = fit_curve(df_fit["temp_norm"], df_fit["scaling"])

df_fit["theory"] = theory(df_1.temp_norm, *opt)
```

```python
opt
```

```python
c = alt.Chart(df_fit).encode(
    x=alt.X("temp_norm", title="T/Tₘ", scale=alt.Scale(zero=False))
)

c.mark_point().encode(y="scaling") + c.mark_line().encode(y="theory")
```

```python
print(
    f"""
The above plot uses the values computed from the curve fit with;
- enthalpy: {opt[1]:.3f}, and
- constant: {opt[0]:.3f}
"""
)
```

```python
df_13 = timescales_df.query("pressure == 13.50")

df_fit = pandas.DataFrame(
    {"temp_norm": df_13.temp_norm, "scaling": df_13["mean"] * df_13["tau_T2_value"]}
)

theory, opt, err = fit_curve(df_fit["temp_norm"], df_fit["scaling"])
# theory, opt, err = fit_curve(df_13)

df_fit["theory"] = theory(df_13.temp_norm, *opt)
```

```python
c = alt.Chart(df_fit).encode(
    x=alt.X("temp_norm", title="T/Tₘ", scale=alt.Scale(zero=False))
)

c.mark_point().encode(y="scaling") + c.mark_line().encode(y="theory")
```

```python
print(
    f"""
The above plot uses the values computed from the curve fit with;
- enthalpy: {opt[1]:.3f}, and
- constant: {opt[0]:.3f}
"""
)
```
