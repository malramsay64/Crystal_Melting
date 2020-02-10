---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.3
  kernelspec:
    display_name: crystal
    language: python
    name: crystal
---

# Melting Uncertainties

This aims to estimate the uncertainty of the crystal melting rates,
combining the uncertainties of the dynamics and that of the rates calculation.

```python
import pandas
import uncertainties
import altair as alt
import numpy as np
import scipy.optimize
import functools
import uncertainties
from crystal_analysis import figures
```

## Error in Melting Rates

The error of the melting rates will be calculated
by taking the standard deviation of all the measurements

```python
with pandas.HDFStore("../data/analysis/rates_rs_clean.h5") as store:
    df_rates = store.get("rates")
```

## Aggregation of values

Taking the collection of simulations and turning them into a single point on the figure.
This is done using the mean and the standard error of the mean.

```python
melt_value = df_rates.groupby(["pressure", "temperature", "temp_norm"])["mean"].mean()
melt_err = df_rates.groupby(["pressure", "temperature", "temp_norm"])["mean"].sem()
melt_frac_err = melt_err / melt_value
df_melt = pandas.DataFrame(
    {
        "value": melt_value,
        "value_abs": melt_value.abs(),
        "error": melt_err,
        "error_abs": np.fmin(melt_err, melt_value.abs() - 1e-9),
        "error_frac": melt_frac_err,
        "error_min": melt_value.abs() - 2 * melt_err,
        "error_max": melt_value.abs() + 2 * melt_err,
    }
)
```

## Melting Point

The first figure of importance is for finding the melting point.
That is the temperature at which the melting rate nearly goes to 0.

```python
chart = alt.Chart(df_melt.reset_index()).encode(
    x=alt.X("temperature", title="Temperature", scale=alt.Scale(zero=False)),
    color=alt.Color("pressure:N", title="Pressure"),
    y=alt.Y("value", title="Melting Rate", axis=alt.Axis(format="e")),
    yError=alt.YError("error"),
)

chart = chart.mark_point() + chart.mark_errorbar()
chart = figures.hline(chart, 0)
# Only plot the points close to the melting point
chart = chart.transform_filter(alt.datum.temp_norm < 1.10)

chart_p1 = chart.transform_filter(alt.datum.pressure == 1.00)
chart_p13 = chart.transform_filter(alt.datum.pressure == 13.50)

with alt.data_transformers.enable("default"):
    chart_p1.save("../figures/melting_point_rates_P1.00.svg", webdriver="firefox")
    chart_p13.save("../figures/melting_point_rates_P13.50.svg", webdriver="firefox")

chart_p1 & chart_p13
```

## Normalised melting rates

This normalises the melting rates by the melting point,
putting the rates from both pressures on the same figure.

```python
chart = alt.Chart(df_melt.reset_index()).encode(
    x=alt.X("temp_norm", title="T/Tₘ", scale=alt.Scale(zero=False)),
    color=alt.Color("pressure:N", title="Pressure"),
    y=alt.Y("value", title="Crystal Growth Rate", axis=alt.Axis(format="e"),),
    yError=alt.YError("error"),
)

chart = chart.mark_point() + chart.mark_errorbar()
chart = figures.hline(chart, 0)

chart
```

```python
with alt.data_transformers.enable("default"):
    chart.save("../figures/melting_rates_err.svg", webdriver="firefox")
```

## Thermodynamic Quantities

The melting points dataset contains the melting point for each of the pressures studied
while the enthalpy dataset conatins the enthalpy for each crystal, the liquid
at a range of temperatures.

```python
df_melting = pandas.read_csv("../results/melting_points.csv", index_col="pressure")
df_thermo = pandas.read_csv(
    "../results/enthalpy.csv", index_col=["pressure", "temperature", "crystal"]
)
```

The thermodynamic quantity we want is the difference in chemical potential
between the liquid and the crystal $\Delta \mu(T)$
which can be approximated as

$$ \Delta \mu(T) = \frac{(T_m - T)\Delta h_m}{T_m} $$

```python
df_enthalpy_melting = (
    df_melting.join(df_thermo)
    .reset_index()
    .query("temperature == melting_point")
    .set_index(["crystal"])
    .groupby("pressure")
    .apply(lambda g: g.loc["p2", "enthalpy"] - g.loc["liquid", "enthalpy"])
).to_frame("enthalpy_melting")
```

```python
df_chemical = (
    df_melt.join(df_enthalpy_melting)
    .join(df_melting)
    .reset_index()
    .assign(
        delta_mu=lambda df: (df["melting_point"] - df["temperature"])
        * df["enthalpy_melting"]
        / df["melting_point"]
    )
)
```

## Turnbull Melting

This models the Turnbull theory of melting,
which means that plotting

$$ \frac{v(T)}{\sqrt{T} \text{vs} [1-\exp{\frac{-\Delta \mu(T)}{k_B T}] $$

should result in a straight line.

```python
df_turnbull = pandas.DataFrame(
    {
        "y": df_chemical["value"] / np.sqrt(df_chemical["temperature"]),
        "x": 1 - np.exp(-df_chemical["delta_mu"] / df_chemical["temperature"]),
        "pressure": df_chemical["pressure"],
    }
)
```

```python
chart_turnbull = (
    alt.Chart(df_turnbull)
    .mark_point()
    .encode(
        x=alt.X("x", title="1-exp[Δμ/kT]"),
        y=alt.Y("y", title="v(T)/√T"),
        color=alt.Color("pressure:N", title="Pressure"),
    )
)
with alt.data_transformers.enable("default"):
    chart_turnbull.save("../figures/melting_turnbull.svg", webdriver="firefox")
chart_turnbull
```

## Wilson--Frenkel

The rotational relaxation is the most likely transport coefficient
to be useful for understanding growth,
there are however a range of quantities which can be used.
This reads in the dynamics data which has been pre-calculated
so it can be used here.

```python
with pandas.HDFStore("../data/analysis/dynamics_clean_agg.h5") as store:
    relax_df = store.get("relaxations")
relax_df = relax_df.set_index(["pressure", "temperature"])
```

```python
set(["_".join(i.split("_")[:-1]) for i in relax_df.columns])
```

Convert the error in the transport coefficient to a fractional error
so that it can be combined with that of the melting rate.

```python
transport_coefficient = "rot2"
transport_value = relax_df[f"{transport_coefficient}_mean"]
transport_err = relax_df[f"{transport_coefficient}_std"]
transport_frac_err = transport_err / transport_value

df_transport = pandas.DataFrame(
    {"value": transport_value, "error": transport_err, "error_frac": transport_frac_err}
)
```

With the error in the transport coefficient calculated,
the values of the transport coefficient need to be merged
with those of the melting rates.

```python
df_all = (
    df_chemical.set_index(["pressure", "temperature"]).join(
        df_transport, lsuffix="_melt", rsuffix="_transport"
    )
).reset_index()
```

```python
df_wilson = pandas.DataFrame(
    {
        "y": df_all["value_melt"] * df_all["value_transport"],
        "x": 1 - np.exp(-df_all["delta_mu"] / df_all["temperature"]),
        "pressure": df_chemical["pressure"],
    }
)
```

```python
chart_wilson = (
    alt.Chart(df_wilson)
    .mark_point()
    .encode(
        x=alt.X("x", title="1-exp[Δμ/kT]"),
        y=alt.Y("y", title="v(T) × τ"),
        color=alt.Color("pressure:N", title="Pressure"),
    )
)
with alt.data_transformers.enable("default"):
    chart_wilson.save("../figures/melting_wilson.svg", webdriver="firefox")
chart_wilson
```

## Fitting to Wilson-Frenkel Theory

When fitting a curve to the data
it is required to take into account
the errors in the data.

```python
value = df_all["value_melt"] * df_all["value_transport"]
err_frac = df_all["error_frac_melt"].abs() + df_all["error_frac_transport"].abs()
error = value * err_frac

df_transport_norm = pandas.DataFrame(
    {"value": value, "error": np.abs(error * 2)}, index=df_all.index
).dropna()
```

```python
def rate_theory(x, c, delta_mu):
    result = 1 - np.exp((1 - x) * delta_mu / x)
    return c * result


def fit_curve(x_vals, y_vals, errors=None, delta_mu=None):
    rate = functools.partial(rate_theory, delta_mu=delta_mu)

    opt, err = scipy.optimize.curve_fit(rate, x_vals, y_vals, sigma=errors, maxfev=2000)

    return pandas.Series({"rate_coefficient": opt[0], "rate_error": err[0][0]})
```

```python
df_rates = (
    df_all.join(df_transport_norm)
    # Only fit to normalised temperatures less than 1.20
    #     .query("temp_norm < 1.20")
    .groupby("pressure").apply(
        lambda g: fit_curve(g["temp_norm"], g["value"], g["error"], g["delta_mu"])
    )
)
```

```python
df_rates
```

```python
df_rates.to_csv("../results/rate_constants.csv")
```

```python
df_wf_fit = (
    df_all.join(df_transport_norm)
    .set_index("pressure")
    .join(df_rates)
    .assign(
        theory=lambda df: rate_theory(
            df["temp_norm"], df["rate_coefficient"], df["delta_mu"]
        )
    )
    .reset_index()
)
```

```python
chart = alt.Chart(df_wf_fit).encode(
    x=alt.X("temp_norm", title="T/Tₘ", scale=alt.Scale(zero=False)),
    color=alt.Color("pressure:N", title="Pressure"),
    y=alt.Y("value", title="Rotational Relaxation × Melting Rate"),
    yError=alt.YError("error"),
)

chart = (
    chart.mark_point() + chart.mark_errorbar() + chart.mark_line().encode(y="theory")
)

chart = figures.hline(chart, 0.0)
chart
```

```python
with alt.data_transformers.enable("default"):
    chart.save("../figures/normalised_melting_fit.svg", webdriver="firefox")
    chart.transform_filter("datum.temp_norm < 1.2").save(
        "../figures/normalised_melting_fit_low.svg"
    )
```
