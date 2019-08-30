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

# Crystal Segmentation

A classification algorithm gives local information
about the type of structures which are present.
However to measure melting,
a more global understanding is required.
I want to know the size of the central crystal,
which requires a clear delineation
of the crystal region from the liquid.
The tools I am going to use for this are
- spatial clustering, and
- a convex hull.

## Importing packages

The spatial clustering algorithms
are contained within the sklearn library,
while the convex hull algorithms
are contained within scipy.

A number of other utility packages and functions are also imported.

```python
import numpy as np
import sklearn
import sklearn.cluster
import scipy
import joblib
from sdanalysis.order import (
    compute_neighbours,
    relative_orientations,
    relative_distances,
    create_ml_ordering,
)
from sdanalysis.figures import plot_frame

# Ignore any warnings which pop up
import warnings

warnings.filterwarnings("ignore")

# Import project tools
import sys

sys.path.append("../src")
from detection import read_all_files, neighbour_connectivity

# Configure Bokeh to output the figures to the notebook
from bokeh.io import output_notebook, show

output_notebook()
```

## Input Files

The files which are being processed are from the same trajectories
as the classification algorithm was trained with,
in the directory `../data/simulations/dataset/output`.
For this analysis,
rather than using the `trajectory-*` files
which contain configurations at intervals
which grow exponentially,
I am using the `dump-*` files which have configurations
at equal intervals.

I am checking that reading the files is sensible
by finding the number of files read.

```python
input_files = read_all_files("../data/simulations/dataset/output")
len(input_files)
```

The `read_all_files` command returns a tuple,
containing the snapshot,
and the simulation information about the simulation.
So we are going to extract a single snapshot
from all the input files,
which we can use to test the spatial clustering.

```python
snapshot, *_ = input_files[12]
```

## Classification

With a snapshot chosen,
we need to find the classification
of each local configuration
using the knn classification algorithm
I developed previously.
Since I am interested in the separation
of liquid and crystalline regions,
I reduce all the crystals,
which have values 1,2, and 3,
into a single value
by checking whether the ordering is > 0.

```python
knn_order = create_ml_ordering("../models/KNN-trimer.pkl")
ordering = knn_order(snapshot)
ordering = ordering > 0
```

## Spatial Clustering

With the classification of each local configuration known,
the challenge is to now draw a boundary
which separates the bulk crystal from the bulk liquid,
ignoring any of the noise from either
misclassification, or
thermal fluctuations.

Taking the initial configuration,
there is a degree of noise in the classification,
and the boundary between the liquid and crystal
is not particularly clear.

```python
show(plot_frame(snapshot, order_list=ordering))
```

The type of clustering I am using is Agglomerative Clustering.
The reason for this is that it deals well with the periodic boundary conditions.
By passing a connectivity to the clustering,
which in this case is the nearest neighbours of a molecule,
the clustering uses the locality
of the configuration.

The Agglomerative Clustering algorithm is trying to minimise
the number of connections between each of the clusters
with a penalty for placing values in the incorrect group.

```python
connectivity = neighbour_connectivity(snapshot)
spatial_cluster = sklearn.cluster.AgglomerativeClustering(
    n_clusters=2, connectivity=connectivity
)
y = spatial_cluster.fit_predict(ordering.reshape(-1, 1))
```

The application of the agglomerative clustering
to determine the separation of a configuration
into liquid and crystal
provides a clean separation of liquid and crystal regions.

```python
show(plot_frame(snapshot, order_list=y))
```

## Calculation of Area

The properties of the crystal which I am interested in
is the area and the perimeter of the crystal region,
which we cannot get directly from the clustering algorithm.
I need an additional processing step,
which is to take the Convex Hull,
that is to find the shape which encloses all points.
By passing the points which have been classified as crystal
by the spatial clustering,
I am able to get the convex hull of the crystal.

```python
hull = scipy.spatial.ConvexHull(snapshot.position[y == 1, :2])
```

Once the convex hull has been found,
properties, including
- area (perimeter for 2D hulls), and
- volume (area for 2D hulls)
can be easily obtained.

```python
hull.area
```

An alternative method of calculating the area
involves finding the Voronoi cell for each molecule
and then adding the volumes for all crystals
which are classified as crystalline.

```python
import freud

vor = freud.voronoi.Voronoi(snapshot.freud_box(), buff=3)
vor.compute(snapshot.position)
vor.computeVolumes()
np.sum(vor.volumes[y == 1])
```

This gives a value very similar to that of the convex hull.

## Stability

One of the important reasons
for using the spatial clustering
is that it should suppress many of the fluctuations
of the interface,
so the analysis using the spatial clustering
should be smooth as the crystal melts.

This is a test to ensure that the clustering
is performing as expected
over the melting of a configuration.

```python
import gsd.hoomd
import pandas as pd
from sdanalysis import HoomdFrame
from sdanalysis.read import open_trajectory
from detection import spatial_clustering
import altair as alt
import figures

test_file = "../data/simulations/dataset/output/dump-Trimer-P1.00-T0.60-p2.gsd"
raw_data = []
for index, snapshot in enumerate(open_trajectory(test_file)):
    labels = spatial_clustering(snapshot)
    ordering = knn_order(snapshot)

    # The crystal is smaller than the liquid, so ensure the smaller cluster
    # has the index 1.
    if sum(labels) > len(labels) / 2:
        labels = -labels + 1

    vor.compute(snapshot.position, box=snapshot.box)
    vor.computeVolumes()

    hull = scipy.spatial.ConvexHull(snapshot.position[labels == 1, :2])

    vor_area = np.sum(vor.volumes[labels == 1])

    raw_data.append(
        {
            "index": index,
            "cluster_particles": np.sum(labels),
            "cluster_area": hull.volume,
            "crystal_particles": np.sum(ordering > 0),
            "voronoi_area": vor_area,
        }
    )

data = pd.DataFrame.from_records(raw_data)
```

```python
c = alt.Chart(data).mark_line().encode(x="index")

c.encode(y="cluster_particles", color=alt.value("red")) + c.encode(
    y="crystal_particles", color=alt.value("blue")
)
```

```python
c.encode(y="cluster_area", color=alt.value("red")) + c.encode(
    y="voronoi_area", color=alt.value("blue")
)
```

Any issue with the above simulations,
can be checked in the configuration below.

```python
with gsd.hoomd.open(test_file) as trj:
    snapshot = HoomdFrame(trj[9])
    clusters = spatial_clustering(snapshot)
    show(plot_frame(snapshot, order_list=clusters))
```
