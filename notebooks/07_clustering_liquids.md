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

```python
from sdanalysis import open_trajectory, relative_orientations
import umap
import pandas
import altair as alt
import numpy as np
from crystal_analysis import figures
import hdbscan
```

```python
basis = []
for frame in open_trajectory("../data/simulations/dynamics/output/dump-Trimer-P1.00-T0.35.gsd", frame_interval=1000):
    basis.append(relative_orientations(frame.box, frame.position, frame.orientation, max_neighbours=6))
basis = np.concatenate(basis)
basis.sort(axis=1)
```

```python
choices = np.random.choice(len(basis), 5000)
```

```python
groups = hdbscan.HDBSCAN(min_cluster_size=50, cluster_selection_epsilon=1).fit_predict(basis[choices])
transformed = umap.UMAP().fit_transform(basis[choices])
```

```python
df = pandas.DataFrame({
    "Dim 1": transformed[:, 0],
    "Dim 2": transformed[:, 1],
    "Cluster": groups,
    
})
c = alt.Chart(df).mark_point().encode(x="Dim 1", y="Dim 2", color="Cluster:N")
with alt.data_transformers.enable("default"):
    c.save("../figures/clustering_liquid.svg")
```
