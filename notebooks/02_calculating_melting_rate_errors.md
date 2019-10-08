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

# Understanding Methods of Computing Melting Rates

I need a method of computing the melting rate of the crystal structures
which provides a good estimation of both
- the melting rate, and
- the error of the melting rate.
Importantly I need an algorithm
able to detect the small melting rates at low temperatures.

```python
import numpy as np
import pandas
import altair as alt
import scipy.stats
import sys

sys.path.append("../src")
import figures
```

## Load Data

The data on the melting rates has been pre-calculated
since it takes a long time to generate.
The data stored is the simulation conditions along with the values
- fraction: The fraction of the simulation cell which is crystalline in nature
- surface_area: The perimeter (the 2D equivalent of surface area) of the crystalline cluster
- volume: The area (2D equivalent of area) of the crystalline cluster
- time: The timestep at which these values apply.

Only the data from the low pressure melting is used in this analysis
since at the time of writing the dataset is better
and it is easier to only deal with a single set of pressures.
This data is limited to the most stable polymorph, the p2 crystal.

By re-sampling the dataset to times of 1ms, the

```python
# Read file with melting data
time_df = pandas.read_hdf("../data/analysis/rates_rs_clean.h5", "fractions", mode="r")
time_df = time_df.query('pressure == "1.00" and temperature < 0.8')
time_df.index = pandas.TimedeltaIndex(time_df.time)

group_bys = ["crystal", "temperature", "pressure"]
```

## Volume Data

I have plotted the volume of the crystal
as a function of time below.
The important point to note is the high levels of noise in the data,
which is a combination the thermal fluctuations
and the inaccuracy of the algorithm I am using for classification.

```python
chart = (
    alt.Chart(time_df)
    .mark_point()
    .encode(
        x=alt.X("time:Q", axis=alt.Axis(format="e")),
        color="temperature:N",
        row="crystal:N",
        y="radius:Q",
    )
)
chart
```

```python
with alt.data_transformers.enable("default"):
    chart.save("../figures/melting_Trimer-P1.00-p2.svg", webdriver="firefox")
```

## Calculating $\Delta V/ \Delta t$

The quantity we are interested in
is rate of melting for a given surface area,

$$ \tau_M = -\frac{1}{A} \frac{\Delta V}{\Delta t} $$

since our expectation is that $\tau_M$ is a constant value,
fitting a straight line to $\Delta V/\Delta t$
will end up giving a value of $\tau_M$
which is dependent on the surface area.

This documents my attempts at calculating this value with a small error.

### Averaging Instantaneous gradient

This is calculating the instantaneous gradient

$$\frac{1}{A(t)} \frac{V(t+\Delta t) - V(t)}{\Delta t}$$

and averaging the results over all $t$.
The errors being calculated as the standard deviation.
The gradient is computed using the `np.gradient` function
with the documentation found [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.gradient.html#numpy.gradient).

```python
def instantaneous_gradient(df):
    if len(df.volume > 2):
        return np.gradient(df.radius, df.time)
    return np.nan


gradient_mean = time_df.groupby(group_bys).apply(
    lambda x: np.nanmean(instantaneous_gradient(x))
)
gradient_error = time_df.groupby(group_bys).apply(
    lambda x: np.nanstd(instantaneous_gradient(x))
)

gradient1 = pandas.DataFrame(
    {"mean": gradient_mean, "error": gradient_error}, index=gradient_mean.index
)
gradient1.reset_index(inplace=True)
```

Plotting the melting rate with the calculated errors as a function of temperature.

```python
chart = (
    alt.Chart(gradient1)
    .mark_point()
    .transform_calculate(
        ymin="datum.mean-datum.error/2", ymax="datum.mean+datum.error/2"
    )
    .encode(alt.X("temperature:Q", scale=alt.Scale(zero=False)))
)

disp_chart = chart.encode(
    alt.Y("mean", axis=alt.Axis(format="e"))
) + chart.mark_rule().encode(y="ymin:Q", y2="ymax:Q")
disp_chart
```

```python
with alt.data_transformers.enable("default"):
    disp_chart.save("../figures/melting_rates_Trimer-P1.00-p2.svg", webdriver="firefox")
```

This figure corresponds to the values in the table below,
which also includes the fractional error.
A value of the fractional error > 1
would indicate that the value is indistinguishable from 0.

```python
gradient1["frac_error"] = np.abs(gradient1["error"] / gradient1["mean"])
gradient1
```

#### Standard Error of the Mean

With the errors using the standard deviation
being many orders of magnitude larger
than some of the values I am trying to calculate,
a more appropriate measure of the error
is the Standard Error of the Mean (SEM).
The SEM takes into account the number of samples
in calculating the error.
Since I have a large number of samples
this should be a more appropriate metric.

```python
gradient_mean = time_df.groupby(group_bys).apply(
    lambda x: np.nanmean(instantaneous_gradient(x))
)
gradient_error = time_df.groupby(group_bys).apply(
    lambda x: scipy.stats.sem(instantaneous_gradient(x), nan_policy="omit")
)

gradient2 = pandas.DataFrame({"mean": gradient_mean, "error": gradient_error})
gradient2.reset_index(inplace=True)
```

Plotting the data shows the errors from the SEM
are significantly reduced over the standard deviation
although while the errors are small the values are also really small.

```python
chart = (
    alt.Chart(gradient2)
    .mark_point()
    .transform_calculate(
        ymin="datum.mean-datum.error/2", ymax="datum.mean+datum.error/2"
    )
    .encode(alt.X("temperature:Q", scale=alt.Scale(zero=False)))
)

disp_chart = chart.encode(
    y=alt.Y("mean", axis=alt.Axis(format="e"))
) + chart.mark_rule().encode(y="ymin:Q", y2="ymax:Q")
disp_chart
```

```python
gradient2["frac_error"] = np.abs(gradient2["error"] / gradient2["mean"])
gradient2
```

The SEM does provide a significantly smaller estimate of the error,
although it is still larger than the smallest values.
The value for 0.38 should be clearly melting,
however in this case the error
is still half the value and so barely an indication of melting.


## What error is Acceptable

Making the assumption that the perimeter doesn't change over the course of the simulation
I can fit a straight line to the volume
to get an estimate of the melting rate.
This is the approach which will likely have the lowest error in the slope of the line,
although there is a different and possibly much larger error
in the assumption that the surface area
is constant for all temperatures above 0.38.
The main idea of this approach is to give an indication
of whether the errors are a result of the data or the methods I am using.

There are two separate values I am using for the surface area.
The initial surface area, with results in gradient4
and the mean surface area with the results in gradient5.
The use of the mean value is to better match the results from the other derivative methods.

```python
def gradient_regression_first(df):
    """Calculate the gradient using linear regression.

    The y value of the gradient is of each volume divided by the initial
    surface area.

    """
    df = df.dropna()
    slope, _, _, _, std_err = scipy.stats.linregress(x=df.time, y=df.radius)
    return slope, std_err


gradient_mean = time_df.groupby(group_bys).apply(
    lambda x: gradient_regression_first(x)[0]
)
gradient_error = time_df.groupby(group_bys).apply(
    lambda x: gradient_regression_first(x)[1]
)

gradient4 = pandas.DataFrame({"mean": gradient_mean, "error": gradient_error})
gradient4.reset_index(inplace=True)
```

```python
def gradient_regression_mean(df):
    """Calculate the gradient using linear regression.

    The y value of the gradient is of each volume divided by the mean
    of the surface area over the course of the simulation. This is an
    adjustment to bring the accuracy of the resulting slope closer to
    that calculated by other methods.

    """
    df = df.dropna()
    slope, _, _, _, std_err = scipy.stats.linregress(df.time, df.radius)
    return slope, std_err


gradient_mean = time_df.groupby(group_bys).apply(
    lambda x: gradient_regression_mean(x)[0]
)
gradient_error = time_df.groupby(group_bys).apply(
    lambda x: gradient_regression_mean(x)[1]
)

gradient5 = pandas.DataFrame({"mean": gradient_mean, "error": gradient_error})
gradient5.reset_index(inplace=True)
```

```python
chart = (
    alt.Chart(gradient4)
    .mark_point()
    .transform_calculate(
        ymin="datum.mean-datum.error/2", ymax="datum.mean+datum.error/2"
    )
    .encode(alt.X("temperature:Q", scale=alt.Scale(zero=False)))
)

(
    chart.encode(y=alt.Y("mean", axis=alt.Axis(format="e")))
    + chart.mark_rule().encode(y="ymin:Q", y2="ymax:Q")
)
```

```python
gradient4["frac_error"] = np.abs(gradient4["error"] / gradient4["mean"])
gradient4
```

The errors in this approximation of the values
is significantly smaller than any of the other methods.
Additionally the errors and the values
are more consistent throughout the range of temperatures,
with no sudden jumps between values.
I wouldn't look too much into
the volume slightly increasing at the lower temperatures,
with the values indicating an encroachment of the liquid phase
by 0.7 units of distance over the timescale of the simulation;
just over half the width of a molecule.

## Overall Assessment

Taking all the calculated values of the melting rates
and making a comparison of the different techniques.
This is to see how consistent the different methods are at calculating these derivatives.

```python
gradient1["exp"] = 1
gradient2["exp"] = 2
gradient4["exp"] = 4
gradient5["exp"] = 5
gradients_all = pandas.concat([gradient4, gradient5], axis=0, sort=True)
```

Plotting all the different approaches (labelled exp),
on the same figure it is simple to compare them.
The first approach to calculating the error
is obviously wrong so I have excluded the results for clarity.

```python
chart = (
    alt.Chart(gradients_all)
    .mark_point()
    .transform_calculate(ymin="datum.mean-datum.error", ymax="datum.mean+datum.error")
    .encode(alt.X("temperature:Q", scale=alt.Scale(zero=False)), color="exp:N")
)

(
    chart.encode(y=alt.Y("mean", axis=alt.Axis(format="e")))
    + chart.mark_rule().encode(y="ymin:Q", y2="ymax:Q")
)
```

The second and third approach give very similar results,
even when comparing the results in a table.
The values in the fourth approach can be brought much closer
to the other values taking the mean of the perimeter rather than the initial value.
Although the fifth approach still falls down at above T=0.42.

To be able to more easily compare the values a log plot would make more sense,
however since some of the values are negative
this is not possible without some manipulation of the data.
The figure below shows the data from each approach
where the lowest estimated range is above 5e-10.
Additionally all the values are now the absolute magnitude
to allow for the log plot.

```python
chart = (
    alt.Chart(gradients_all)
    .mark_point()
    .transform_calculate(
        abs_mean="abs(datum.mean)",
        ymin="datum.abs_mean-datum.error",
        ymax="datum.abs_mean+datum.error",
    )
    .transform_filter("datum.ymin > 1e-10")
    .encode(alt.X("temperature:Q", scale=alt.Scale(zero=False)), color="exp:N")
)

(
    chart.encode(y=alt.Y("abs_mean:Q", scale=alt.Scale(type="log")))
    + chart.encode(y="ymin:Q", y2="ymax:Q").mark_rule()
)
```
