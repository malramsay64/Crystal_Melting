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

```python
from sdanalysis import open_trajectory, relative_orientations
import umap
import pandas
import altair as alt
import numpy as np
from crystal_analysis import figures
import hdbscan
from sdanalysis.figures import plot_frame, show, output_notebook

output_notebook()
```

```python
basis = []
for frame in open_trajectory(
    "../data/simulations/dynamics/output/dump-Trimer-P1.00-T0.35.gsd",
    frame_interval=1000,
):
    basis.append(
        relative_orientations(
            frame.box, frame.position, frame.orientation, max_neighbours=6
        )
    )
basis_liquid = np.concatenate(basis)
basis_liquid.sort(axis=1)
```

```python
choices = np.random.choice(len(basis_liquid), 5000)
```

```python
groups_liquid = hdbscan.HDBSCAN(
    min_cluster_size=50, cluster_selection_epsilon=1
).fit_predict(basis_liquid[choices])
transformed_liquid = umap.UMAP().fit_transform(basis_liquid[choices])
```

```python
df_liquid = pandas.DataFrame(
    {
        "Dim 1": transformed_liquid[:, 0],
        "Dim 2": transformed_liquid[:, 1],
        "Cluster": groups_liquid,
    }
)
c_liquid = (
    alt.Chart(df_liquid).mark_point().encode(x="Dim 1", y="Dim 2", color="Cluster:N")
)
with alt.data_transformers.enable("default"):
    c_liquid.save("../figures/clustering_liquid.svg")
```

```python
basis = []
import gsd.hoomd
from sdanalysis import HoomdFrame

frame = HoomdFrame(
    gsd.hoomd.open(
        "../data/simulations/interface/output/dump-Trimer-P1.00-T0.35-p2.gsd"
    )[-1]
)
basis_p2 = relative_orientations(
    frame.box, frame.position, frame.orientation, max_neighbours=6
)
basis_p2.sort(axis=1)
groups_p2 = hdbscan.HDBSCAN(
    min_cluster_size=50, cluster_selection_epsilon=0.01
).fit_predict(basis_p2)
transformed_p2 = umap.UMAP().fit_transform(basis_p2)
```

```python
show(plot_frame(frame, order_list=groups_p2, categorical_colour=True))
```

```python
df_p2 = pandas.DataFrame(
    {"Dim 1": transformed_p2[:, 0], "Dim 2": transformed_p2[:, 1], "Cluster": groups_p2}
)
c_p2 = alt.Chart(df_p2).mark_point().encode(x="Dim 1", y="Dim 2", color="Cluster:N")
with alt.data_transformers.enable("default"):
    c_p2.save("../figures/clustering_crystal_p2.svg")
c_p2
```
