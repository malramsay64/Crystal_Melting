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

# Melting Uncertainties

This aims to estimate the uncertainty of the crystal melting rates,
combining the uncertainties of the dynamics and that of the rates calculation.

```python
import pandas
import uncertainties
import altair as alt
import numpy as np
import scipy.optimize
import uncertainties
from crystal_analysis import figures
```

## Error in Melting Rates

The error of the melting rates will be calculated
by taking the standard deviation of all the measurements

```python
with pandas.HDFStore("../data/analysis/rates_rs_clean.h5") as store:
    rates_df = store.get("rates")
```

```python
rates_df["error_min"] = rates_df["mean"] - rates_df["error"]
rates_df["error_max"] = rates_df["mean"] + rates_df["error"]
```

```python
chart = alt.Chart(rates_df).encode(
    x=alt.X("temp_norm", scale=alt.Scale(zero=False)),
    y=alt.Y("mean", scale=alt.Scale(type="linear"), axis=alt.Axis(format="e")),
    color="pressure:N",
)

chart.mark_point() + chart.mark_rule().encode(y="error_min", y2="error_max")
```

```python
melt_value = rates_df.groupby(["pressure", "temperature", "temp_norm"])["mean"].mean()
melt_err = rates_df.groupby(["pressure", "temperature", "temp_norm"])["mean"].sem()
melt_frac_err = melt_err / melt_value
melt_df = pandas.DataFrame(
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

```python
chart = alt.Chart(melt_df.reset_index()).encode(
    x=alt.X("temp_norm", title="T/Tₘ", scale=alt.Scale(zero=False)),
    color=alt.Color("pressure:N", title="Pressure"),
    y=alt.Y(
        "value_abs",
        title="Melting Rate",
        axis=alt.Axis(format="e"),
        scale=alt.Scale(type="log"),
    ),
    yError=alt.YError("error_abs"),
)

chart = chart.mark_point() + chart.mark_errorbar()

chart
```

```python
with alt.data_transformers.enable("default"):
    chart.save("../figures/growth_rates_err.svg", webdriver="firefox")
```

```python
chart = alt.Chart(melt_df.reset_index()).encode(
    x=alt.X("temperature", title="Temperature", scale=alt.Scale(zero=False)),
    color=alt.Color("pressure:N", title="Pressure"),
    y=alt.Y("value", title="Melting Rate", axis=alt.Axis(format="e")),
    yError=alt.YError("error"),
)

chart = chart.mark_point() + chart.mark_errorbar()

chart = figures.hline(chart, 0)

chart = chart.transform_filter(alt.datum.temp_norm < 1.10)

chart = chart.transform_filter(alt.datum.pressure == 1.00) & chart.transform_filter(
    alt.datum.pressure == 13.50
)
chart
```

```python
with alt.data_transformers.enable("default"):
    chart.save("../figures/melting_point_rates.svg", webdriver="firefox")
```

```python
mean = rates_df.groupby(["pressure", "temperature", "temp_norm"])["mean"].mean()
err = rates_df.groupby(["pressure", "temperature", "temp_norm"])["mean"].std()
```

## Error in Rotational Relaxation

```python
with pandas.HDFStore("../data/analysis/dynamics_clean_agg.h5") as store:
    relax_df = store.get("relaxations")
relax_df = relax_df.set_index(["pressure", "temperature"])
```

```python
rot_value = relax_df["rot2_mean"]
rot_err = relax_df["rot2_std"]
rot_frac_err = rot_err / rot_value

rot_df = pandas.DataFrame(
    {"value": rot_value, "error": rot_err, "error_frac": rot_frac_err}
)
```

```python
all_df = (
    melt_df.reset_index("temp_norm")
    .join(rot_df, lsuffix="_melt", rsuffix="_rot")
    .set_index("temp_norm", append=True)
)
all_df
```

```python
value = all_df["value_melt"] * all_df["value_rot"]
err_frac = all_df["error_frac_melt"].abs() + all_df["error_frac_rot"].abs()
error = value * err_frac

melt_values = (
    pandas.DataFrame({"value": value, "error": np.abs(error * 2)})
    .reset_index()
    .dropna()
)
```

```python
chart = alt.Chart(melt_values).encode(
    x=alt.X("temp_norm", title="T/Tₘ", scale=alt.Scale(zero=False)),
    y=alt.Y("value", title="Rotational Relaxation × Melting Rate"),
    yError=alt.YError("error"),
    color=alt.Color("pressure:N", title="Pressure"),
)

c = chart.mark_point() + chart.mark_errorbar()

c = figures.hline(c, 0)
c
```

## Fitting to Theory

When fitting a curve to the data
it is required to take into account
the errors in the data.

```python
import functools


def rate_theory(x, c, delta_E):
    result = 1 - np.exp((1 - x) * delta_E / x)
    return c * result


def fit_curve(x_vals, y_vals, errors=None, delta_E=None):
    rate = functools.partial(rate_theory, delta_E=delta_E)

    opt, err = scipy.optimize.curve_fit(rate, x_vals, y_vals, sigma=errors, maxfev=2000)

    return pandas.Series({"rate_coefficient": opt[0], "rate_error": err[0][0]})
```

```python
df_melting = pandas.read_csv("../results/melting_points.csv", index_col="pressure")
df_thermo = pandas.read_csv(
    "../results/enthalpy.csv", index_col=["pressure", "temperature", "crystal"]
)
```

```python
df_energy = (
    df_thermo.join(df_melting)
    .reset_index()
    .query("temperature == melting_point")
    .set_index("crystal")
    .groupby("pressure")
    .apply(lambda g: g.loc["liquid", "enthalpy"] - g.loc["p2", "enthalpy"])
    .to_frame("crystal_free_energy")
)
```

```python
df_rates = (
    melt_values
    # Only fit to normalised temperatures less than 1.20
    .query("temp_norm < 1.20")
    .set_index("pressure")
    .join(df_energy)
    .groupby("pressure")
    .apply(
        lambda g: fit_curve(
            g["temp_norm"], g["value"], g["error"], g["crystal_free_energy"]
        )
    )
)
```

```python
df_rates.to_csv("../results/rate_constants.csv")
```

```python
df_theory = melt_values.set_index("pressure").join(df_rates).join(df_energy)
df_theory = df_theory.assign(
    theory=rate_theory(
        df_theory["temp_norm"],
        df_theory["rate_coefficient"],
        df_theory["crystal_free_energy"],
    )
).reset_index()
```

```python
chart = alt.Chart(df_theory).encode(
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
    chart.save("../figures/normalised_melting_err.svg", webdriver="firefox")
    chart.transform_filter("datum.temp_norm < 1.2").save(
        "../figures/normalised_melting_err_low.svg"
    )
```

```python
chart = alt.Chart(
    df_theory.assign(inv_temp_norm=1 / df_theory["temp_norm"]).reset_index()
).encode(
    x=alt.X("inv_temp_norm", title="T/Tₘ", scale=alt.Scale(zero=False)),
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
    chart.save("../figures/normalised_melting_err_inv.svg", webdriver="firefox")
```
