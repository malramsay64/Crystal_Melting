---
jupyter:
  jupytext:
    formats: ipynb,md
    target_format: ipynb,md
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
from sdanalysis import HoomdFrame, order, read
import gsd.hoomd
from pathlib import Path
import joblib
import functools
from bokeh.io import show, output_notebook

output_notebook()
```

```python
data_dir = Path("../data/simulations/interface/output/")
trj = read.open_trajectory(data_dir / "dump-Trimer-P1.00-T0.42-p2.gsd")
snap = next(trj)
```

```python
knn_order = order.create_ml_ordering("../models/knn-trimer.pkl")
```

```python
knn_order(snap).astype(bool)
```

```python
show(configuration.plot_frame(snap, knn_order))
```

```python

```
