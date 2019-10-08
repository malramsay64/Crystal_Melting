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
snapshot, *_ = input_files[3]
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
knn_order = create_ml_ordering("../models/knn-trimer.pkl")
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


## Check Rust Implementation

For reasons of performance, the analysis of the trajectories is now performed with rust,
using a K-Nearest Neighbours algorithm.
Since this is a slightly different algorithm,
this checks the results are consistent
with those obtained in the Python implementation.

This reads in the classifications which have been obtained
using the rust implementation
and applies those to the trajectory.

```python
test_file = "../data/simulations/interface/output/dump-Trimer-P13.50-T1.32-p2.gsd"
classified = "../data/analysis/interface/dump-Trimer-P13.50-T1.32-p2.csv"

df_classes = pd.read_csv(classified)
trj = open_trajectory(test_file)
index, group = df_classes.groupby("timestep").__iter__().__next__()

show(plot_frame(snapshot, order_list=group["class"] != "Liquid"))
```

```python

```
