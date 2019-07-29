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

Crystal Segmentation
====================

This is a test of tools to compute the properties of the interface between the liquid and the crystal.
Since I am interested in the melting rate of a sample surrounded by liquid,
as the blob melts the surface area is going to change.
This means that I will require a normalisation factor to be able to properly compute this melting.

```python
import numpy as np
import sklearn
import sklearn.cluster
import scipy
import joblib
from sdanalysis.order import compute_neighbours, relative_orientations, relative_distances, create_ml_ordering
from sdanalysis.figures import plot_frame

import warnings
warnings.filterwarnings('ignore')

# Import project tools
import sys
sys.path.append('../src')
from detection import read_all_files, neighbour_connectivity

# Configure Bokeh to output the figures to the notebook
from bokeh.io import output_notebook, show
output_notebook()
```

This creates a periodic distance algorithm using the minimum image convention
which can be used as the distance metric for the clustering algorithm.

```python
input_files = read_all_files(
    '../data/simulations/dataset/output'
)
len(input_files)
```

```python
snapshot, *_ = input_files[12]
knn_order = create_ml_ordering("../models/knn-trimer.pkl")

ordering = knn_order(snapshot)
ordering = ordering > 0
ordering = ordering.reshape(-1,1)
```

```python
connectivity = neighbour_connectivity(snapshot)
spatial_cluster = sklearn.cluster.AgglomerativeClustering(n_clusters=2, connectivity=connectivity)
y = spatial_cluster.fit_predict(ordering)
```

```python
hull = scipy.spatial.ConvexHull(snapshot.position[y==1, :2])
```

```python
hull.area
```

```python
show(plot_frame(snapshot, order_list=ordering.flatten()))
```

## Stability

```python
import gsd.hoomd
from sdanalysis import HoomdFrame
from sdanalysis.read import open_trajectory
from detection import spatial_clustering
test_file = '../data/simulations/dataset/output/dump-Trimer-P1.00-T0.50-p2.gsd'
cluster_size = []
cluster_area = []
for index, snap in enumerate(open_trajectory(test_file)):
    clusters = spatial_clustering(snapshot)
    cluster_size.append(np.sum(clusters))
    hull1 = scipy.spatial.ConvexHull(snapshot.position[y==1, :2])
    hull0 = scipy.spatial.ConvexHull(snapshot.position[y==0, :2])
    cluster_area.append(min(hull0.volume, hull1.volume))
    if len(cluster_size) > 100:
        break
```

```python
with gsd.hoomd.open(test_file) as trj:
    snapshot = HoomdFrame(trj[7])
    clusters = spatial_clustering(snapshot)
    show(plot_frame(snapshot, order_list=clusters))
```

```python
import matplotlib.pyplot as plt
```

```python
# plt.plot(cluster_size)
plt.plot(cluster_area);
```

```python

```
