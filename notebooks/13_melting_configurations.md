---
jupyter:
  jupytext:
    formats: ipynb,md
    target_format: ipynb,md
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

# Melting Configurations

In the analysis of the melting
I have thrown away data from high temperatures
as a result of the melting not proceeding as expected.
This notebook provides figures
which provide the evidence for this decision.

```python
from sdanalysis.figures import configuration
from sdanalysis import HoomdFrame, order

from bokeh.io import output_notebook, show, export_svgs
from bokeh.models import Range1d
import gsd.hoomd

from crystal_analysis import figures

output_notebook()
```

```python
gsd_file = "../data/simulations/rates/output/dump-Trimer-P1.00-T0.55-p2-ID1.gsd"
with gsd.hoomd.open(gsd_file) as trj:
    snap = HoomdFrame(trj[18])
```

```python
knn_order = order.create_ml_ordering("../models/knn-trimer.pkl")
ordering = knn_order(snap)
```

```python
fig = figures.style_snapshot(configuration.plot_frame(snap, order_list=ordering))
fig.x_range = Range1d(-40, 40)
fig.y_range = Range1d(-40, 40)
show(fig)
```

```python
fig.output_backend = "svg"
export_svgs(fig, "../figures/melting_disorder_P1.00-T0.55.svg")
```

```python
gsd_file = "../data/simulations/rates/output/dump-Trimer-P1.00-T0.50-p2-ID1.gsd"
with gsd.hoomd.open(gsd_file) as trj:
    snap = HoomdFrame(trj[16])

ordering = knn_order(snap)

fig = figures.style_snapshot(configuration.plot_frame(snap, order_list=ordering))
fig.x_range = Range1d(-50, 30)
fig.y_range = Range1d(-50, 30)
show(fig)
```

```python
fig.output_backend = "svg"
export_svgs(fig, "../figures/melting_disorder_P1.00-T0.50.svg")
```
