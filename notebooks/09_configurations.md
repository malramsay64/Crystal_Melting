---
jupyter:
  jupytext:
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

```python
from sdanalysis.figures import configuration
from sdanalysis import HoomdFrame, order
import gsd.hoomd
from pathlib import Path
import joblib
import functools
from bokeh.io import show, output_notebook
output_notebook()
```

```python
data_dir = Path("../data/simulations/interface/output/")
with gsd.hoomd.open(str(data_dir / "dump-Trimer-P1.00-T0.42-p2.gsd")) as trj:
    snap = HoomdFrame(trj[0])
```

```python
model = joblib.load("../models/knn-trimer.pkl")
order_func = functools.partial(order.compute_ml_order, model)
```

```python
order.compute_ml_order(model, snap.box, snap.position, snap.orientation).astype(bool)
```

```python
show(configuration.plot_frame(snap, order_func))
```

```python

```
