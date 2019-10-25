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

# Melting behaviour of Crystals

This notebook is focused on
understanding the melting behaviour of
each of the different crystal structures.
This is a more qualitative rather than quantitative
investigation of the behaviour.

```python
import pandas
import altair as alt
import gsd.hoomd
from bokeh.io import export_svgs

from sdanalysis import HoomdFrame
from sdanalysis.figures import plot_frame, show, output_notebook

import sys

sys.path.append("../src")
import detection
import figures

output_notebook()
```

## Data Source

The data is sources from the melting_rs_clean.h5 file
generated from the data in the `interface` simulations directory.
This contains simulations over a long time period
close to the melting points of the crystals.

```python
with pandas.HDFStore("../data/analysis/melting_rs_clean.h5") as src:
    df = src.get("fractions")
df.columns
```

To get the dataset in an appropriate format for plotting
it needs to go through a restructuring,
this is done using the `melt` function.

```python
df_melt = (
    df.melt(id_vars=["crystal", "time", "temperature", "pressure"])
    .query("variable in ['P2', 'P2GG', 'PG']")
    .set_index(["pressure", "temperature", "crystal"])
    .sort_index()
)
```

## Solid State Phase Transition

The first result here is the unusual behaviour of the p2gg crystal in the solid state.
This is highlighted by the image below.
Here we have a

```python
c = (
    alt.Chart(df_melt.loc[(13.50, 1.32, "p2gg")].reset_index())
    .mark_point()
    .encode(
        x=alt.X("time", title="Timesteps", axis=alt.Axis(format="e")),
        y=alt.Y("value", title="Fraction"),
        color=alt.Color("variable", title="Crystal"),
    )
    .transform_filter(alt.datum.time < 2e8)
)
c
```


```python
with alt.data_transformers.enable("default"):
    c.save(
        "../figures/solid_state_transition-P13.50-T1.35-p2gg.svg", webdriver="firefox"
    )
```

```python
with pandas.HDFStore("../data/analysis/dynamics_clean_agg.h5") as src:
    df_dynamics = src.get("relaxations")
rot_relaxation = df_dynamics.query("temperature == 1.35 and pressure == 13.50")[
    "rot2_mean"
]
print(f"The rotational relaxation time is {rot_relaxation.values[0]:.2e}")
```

Comparing this solid state transition
with the rotational relaxation time,
which is $~5 \times 10^6$,
the transition takes place remarkably quickly,
on the order of only a few rotational relaxation times,
which is orders of magnitude faster than the melting.

Another figure which better shows the relationship
between the solid state rearrangement
and the melting rate is at a temperature of 1.46.
Here the solid state rearrangement takes place quickly,
followed by a much slower and more gradual melting.

```python
c = (
    alt.Chart(df_melt.loc[(13.50, 1.40, "p2gg")].reset_index())
    .mark_point()
    .encode(
        x=alt.X("time", title="Timesteps", axis=alt.Axis(format="e")),
        y=alt.Y("value", title="Fraction"),
        color=alt.Color("variable", title="Crystal"),
    )
    .transform_filter(alt.datum.time < 2e8)
)
c
```

```python
with alt.data_transformers.enable("default"):
    c.save(
        "../figures/solid_state_transition-P13.50-T1.40-p2gg.svg", webdriver="firefox"
    )
```

One of the features of the solid state transition
is the stepped nature of the transition,
It will proceed quickly,
halt for some time,
then proceed again quickly.
To understand how this process is taking place
it would be really useful to understand
why these unusual features are present.
To do this we need to look at
the configurations themselves.

```python
trajectory_file = (
    "../data/simulations/interface/output/dump-Trimer-P13.50-T1.40-p2gg.gsd"
)
with gsd.hoomd.open(trajectory_file) as trj:
    snap_init = HoomdFrame(trj[0])
    snap_process = HoomdFrame(trj[200])
    snap_end = HoomdFrame(trj[400])
```

```python
snap_process.timestep
```

```python
snap_end.timestep
```

```python
import joblib
import functools
from sdanalysis.order import create_ml_ordering

knn_order = create_ml_ordering("../models/knn-trimer.pkl")
```

```python
frame = plot_frame(snap_init, order_list=knn_order(snap_init), categorical_colour=True)
frame = figures.style_snapshot(frame)
show(frame)
```

```python
frame.output_backend = "svg"
export_svgs(frame, "../figures/configuration-P13.50-T1.40-p2gg_init.svg")
```

```python
frame = plot_frame(
    snap_process, order_list=knn_order(snap_process), categorical_colour=True
)
frame = figures.style_snapshot(frame)
show(frame)
```


```python
frame.output_backend = "svg"
export_svgs(frame, "../figures/configuration-P13.50-T1.40-p2gg_process.svg")
```

```python
frame = plot_frame(snap_end, order_list=knn_order(snap_end), categorical_colour=True)
frame = figures.style_snapshot(frame)
show(frame)
```

```python
frame.output_backend = "svg"
export_svgs(frame, "../figures/configuration-P13.50-T1.40-p2gg_end.svg")
```

## Melting pg Crystal

While the

```python
groupbys = list(df_melt.reset_index().columns)
groupbys.remove("variable")
groupbys.remove("value")
```

```python
df_melt_agg = df_melt.reset_index().groupby(groupbys).sum().reset_index()
```

```python
c = (
    alt.Chart(df_melt_agg.query("pressure==13.50 and temperature == 1.40"))
    .mark_point()
    .encode(
        x=alt.X("time", title="Timesteps"),
        y=alt.Y("value", title="Volume"),
        color=alt.Color("crystal", title="Crystal"),
    )
)
with alt.data_transformers.enable("default"):
    c.save("../figures/melting_crystal_comparison.svg", webdriver="firefox")
```

![](../figures/melting_crystal_comparison.svg)

```python
trajectory_file = "../data/simulations/interface/output/dump-Trimer-P13.50-T1.40-pg.gsd"
with gsd.hoomd.open(trajectory_file) as trj:
    snap_init = HoomdFrame(trj[0])
    snap_process = HoomdFrame(trj[30_000])
    snap_end = HoomdFrame(trj[60_000])
```

```python
snap_process.timestep
```

```python
frame = plot_frame(snap_init, order_list=knn_order(snap_init), categorical_colour=True)
frame = figures.style_snapshot(frame)
show(frame)
```

```python
frame.output_backend = "svg"
export_svgs(frame, "../figures/configuration-P13.50-T1.40-pg_0.svg")
```

```python
from bokeh.models import Range1d

frame.x_range = Range1d(-15, 15)
frame.y_range = Range1d(40, 65)
show(frame)
```

```python
frame.output_backend = "svg"
export_svgs(frame, "../figures/configuration-P13.50-T1.40-pg_top_0.svg")
```

```python
frame.x_range = Range1d(-15, 15)
frame.y_range = Range1d(-65, -40)
show(frame)
```

```python
frame.output_backend = "svg"
export_svgs(frame, "../figures/configuration-P13.50-T1.40-pg_bottom_0.svg")
```

```python
frame = plot_frame(
    snap_process, order_list=knn_order(snap_process), categorical_colour=True
)
frame = figures.style_snapshot(frame)
show(frame)
```

```python
frame.output_backend = "svg"
export_svgs(frame, "../figures/configuration-P13.50-T1.40-pg_1.svg")
```

```python
frame = plot_frame(snap_end, order_list=knn_order(snap_end), categorical_colour=True)
frame = figures.style_snapshot(frame)
show(frame)
```

```python
frame.output_backend = "svg"
export_svgs(frame, "../figures/configuration-P13.50-T1.40-pg_2.svg")
```

```python
frame.x_range = Range1d(-15, 15)
frame.y_range = Range1d(40, 65)
show(frame)
```

```python
frame.output_backend = "svg"
export_svgs(frame, "../figures/configuration-P13.50-T1.40-pg_top_2.svg")
```
