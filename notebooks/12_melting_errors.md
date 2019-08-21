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

import sys

sys.path.append("../src")
import figures
```

## Error in Melting Rates

The error of the melting rates will be calculated
by taking the standard deviation of all the measurements

```python
with pandas.HDFStore("../data/analysis/rates_clean.h5") as store:
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
melt_value = rates_df.groupby(["temperature", "pressure", "temp_norm"])["mean"].mean()
melt_err = rates_df.groupby(["temperature", "pressure", "temp_norm"])["mean"].sem()
melt_frac_err = melt_err / melt_value
melt_df = pandas.DataFrame(
    {
        "value": melt_value,
        "error": melt_err,
        "error_frac": melt_frac_err,
        "error_min": melt_value - 2 * melt_err,
        "error_max": melt_value + 2 * melt_err,
    }
)
```

```python
chart = alt.Chart(melt_df.reset_index()).encode(
    x=alt.X("temp_norm", title="T/Tₘ", scale=alt.Scale(zero=False)),
    color=alt.Color("pressure:N", title="Pressure"),
)

chart = chart.mark_point().encode(
    y=alt.Y("value", title="Melting Rate", axis=alt.Axis(format="e"))
) + chart.mark_rule().encode(y="error_min", y2="error_max")

chart = figures.hline(chart, 0)
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
)

chart = chart.mark_point().encode(
    y=alt.Y("value", title="Melting Rate", axis=alt.Axis(format="e"))
) + chart.mark_rule().encode(y="error_min", y2="error_max")

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
mean = rates_df.groupby(["temperature", "pressure", "temp_norm"])["mean"].mean()
err = rates_df.groupby(["temperature", "pressure", "temp_norm"])["mean"].std()
```

## Error in Rotational Relaxation

```python
with pandas.HDFStore("../data/analysis/dynamics_clean_agg.h5") as store:
    relax_df = store.get("relaxations")
```

```python
rot_value = relax_df.groupby(["temperature", "pressure"])["rot2_value"].mean()
rot_err = np.maximum(
    (relax_df["rot2_value"] - relax_df["rot2_lower"]).values,
    (relax_df["rot2_upper"] - relax_df["rot2_value"]).values,
)
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
    pandas.DataFrame(
        {
            "value": value,
            "error": error,
            "error_min": value - 2 * error,
            "error_max": value + 2 * error,
        }
    )
    .reset_index()
    .dropna()
)
```

```python
chart = alt.Chart(melt_values).encode(
    x=alt.X("temp_norm", title="T/Tₘ", scale=alt.Scale(zero=False)),
    color=alt.Color("pressure:N", title="Pressure"),
)

c = chart.mark_point().encode(
    y=alt.Y("value", title="Rotational Relaxation × Melting Rate")
) + chart.mark_rule().encode(y="error_min", y2="error_max")

c = figures.hline(c, 0)
c
```

## Fitting to Theory

When fitting a curve to the data
it is required to take into account
the errors in the data.

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
x = np.arange(0.95, 2.0, 0.05)

p1_values = melt_values.query("pressure == 1.00 and 1.00 < temp_norm < 1.30")
p13_values = melt_values.query("pressure == 13.50 and 1.00 < temp_norm < 1.30")

theory1, opt1, err1 = fit_curve(
    p1_values["temp_norm"], p1_values["value"], p1_values["error"], -0.18034612159032992
)
theory13, opt13, err13 = fit_curve(
    p13_values["temp_norm"],
    p13_values["value"],
    p13_values["error"],
    -0.06561802006526474,
)
```

```python
y1 = theory1(x, *opt1)
y13 = theory13(x, *opt13)
```

```python
theory_df = melt_values
theory_df["theory"] = 0.0
mask = theory_df["pressure"] == 1.00
theory_df.loc[mask, "theory"] = theory1(theory_df["temp_norm"], *opt1)
theory_df.loc[~mask, "theory"] = theory13(theory_df["temp_norm"], *opt13)
```

```python
chart = (
    alt.Chart(theory_df)
    .encode(
        x=alt.X(
            "temp_norm", title="T/Tₘ", scale=alt.Scale(zero=False, domain=(0.95, 1.45))
        ),
        color=alt.Color("pressure:N", title="Pressure"),
    )
    .transform_filter(alt.datum.temp_norm < 1.35)
)

chart = (
    chart.mark_point().encode(
        y=alt.Y("value", title="Rotational Relaxation × Melting Rate")
    )
    + chart.mark_rule().encode(y="error_min", y2="error_max")
    + chart.mark_line().encode(y="theory")
)

chart = figures.hline(chart, 0.0)
chart
```

```python
with alt.data_transformers.enable("default"):
    chart.save("../figures/normalised_melting_err.svg", webdriver="firefox")
```

```python
p1_values_2 = melt_values.query("pressure == 1.00")
p13_values_2 = melt_values.query("pressure == 13.50")

theory1_2, opt1_2, err1_2 = fit_curve(
    p1_values_2["temp_norm"], p1_values_2["value"], p1_values_2["error"]
)
theory13_2, opt13_2, err13_2 = fit_curve(
    p13_values_2["temp_norm"], p13_values_2["value"], p13_values_2["error"]
)
```

```python
theory_df_2 = melt_values
theory_df_2["theory"] = 0.0
mask = theory_df["pressure"] == 1.00
theory_df_2.loc[mask, "theory"] = theory1_2(theory_df_2["temp_norm"], *opt1_2)
theory_df_2.loc[~mask, "theory"] = theory13_2(theory_df_2["temp_norm"], *opt13_2)
```

```python
chart = alt.Chart(theory_df_2).encode(
    x=alt.X("temp_norm", title="T/Tₘ", scale=alt.Scale(zero=False)),
    color=alt.Color("pressure:N", title="Pressure"),
)

chart = (
    chart.mark_point().encode(
        y=alt.Y("value", title="Rotational Relaxation × Melting Rate")
    )
    + chart.mark_rule().encode(y="error_min", y2="error_max")
    + chart.mark_line().encode(y="theory")
)

chart = figures.hline(chart, 0.0)
chart
```

```python
opt13_2
```

```python

```
