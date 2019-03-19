#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Utilities for handling the trimer molecule."""

import logging
from itertools import product
from pathlib import Path
from typing import List, NamedTuple, Tuple

import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from bokeh.plotting import gridplot
from scipy.sparse import coo_matrix
from sdanalysis import HoomdFrame, util
from sdanalysis.figures import plot_frame
from sdanalysis.order import compute_ml_order, compute_neighbours

logger = logging.getLogger(__name__)


def read_files(
    index: int = 0,
    pressure: List[float] = 1.00,
    temperature: List[float] = 0.40,
    crystals: List[str] = ["p2", "p2gg", "pg"],
) -> List[HoomdFrame]:
    if isinstance(pressure, float):
        pressure = [pressure]
    if isinstance(temperature, float):
        temperature = [temperature]
    if isinstance(crystals, str):
        crystals = [crystals]

    data_dir = Path("../data/simulation/trimer")
    snapshots = []
    for press, temp, crys in product(pressure, temperature, crystals):
        fname = f"dump-Trimer-P{press:.2f}-T{temp:.2f}-{crys}.gsd"
        with gsd.hoomd.open(str(data_dir / fname), "rb") as trj:
            snapshots.append(HoomdFrame(trj[index]))

    return snapshots


class SnapshotData(NamedTuple):
    snapshot: HoomdFrame
    temperature: str
    pressure: str
    crystal: str

    @classmethod
    def from_variables(
        cls, snapshot: HoomdFrame, variables: util.variables
    ) -> "SnapshotData":
        return cls(
            snapshot=snapshot,
            temperature=variables.temperature,
            pressure=variables.pressure,
            crystal=variables.crystal,
        )


def read_all_files(
    directory: Path, index: int = 0, glob: str = "dump-*"
) -> List[Tuple[util.variables, HoomdFrame]]:
    directory = Path(directory)
    snapshots = []
    for file in directory.glob(glob):
        with gsd.hoomd.open(str(file), "rb") as trj:
            try:
                snap = HoomdFrame(trj[index])
            except IndexError:
                logger.warning(
                    "Index %d in input file %s doesn't exist, continuing...",
                    index,
                    file.name,
                )
            snapshots.append(
                SnapshotData.from_variables(snap, util.get_filename_vars(file))
            )
    return snapshots


def plot_grid(frames):
    for frame in frames:
        frame.plot_height = frame.plot_height // 3
        frame.plot_width = frame.plot_width // 3
    return gridplot(frames, ncols=3)


def plot_clustering(algorithm, X, snapshots, fit=True):
    if fit:
        clusters = algorithm.fit_predict(X)
    else:
        clusters = algorithm.predict(X)
    cluster_assignment = np.split(clusters, len(snapshots))
    fig = plot_grid(
        [
            plot_frame(snap, order_list=cluster, categorical_colour=True)
            for snap, cluster in zip(snapshots, cluster_assignment)
        ]
    )
    return fig


def plot_snapshots(snapshots):
    return plot_grid([plot_frame(snap) for snap in snapshots])


def classify_mols(snapshot, crystal, boundary_buffer=3.5, is_2d: bool = True):
    """Classify molecules as crystalline, amorphous or boundary."""
    mapping = {"liq": 0, "p2": 1, "p2gg": 2, "pg": 3, "None": 4}
    position = snapshot.position
    # This gets the details of the box from the simulation
    box = snapshot.box[:3]

    # All axes have to be True, True == 1, use product for logical and operation
    position_mat = np.abs(position) < box[:3] / 3
    if is_2d:
        is_crystal = np.product(position_mat[:, :2], axis=1).astype(bool)
    else:
        is_crystal = np.product(position_mat, axis=1).astype(bool)

    boundary = np.logical_and(
        np.product(np.abs(position) < box[:3] / 3 + boundary_buffer, axis=1),
        np.product(np.abs(position) > box[:3] / 3 - boundary_buffer, axis=1),
    )

    # Create classification array
    classification = np.zeros(len(snapshot))
    classification[is_crystal] = mapping[crystal]
    classification[boundary] = 4
    return classification


def neighbour_connectivity(snapshot, max_neighbours=6, max_radius=5):
    neighbours = compute_neighbours(
        snapshot.box, snapshot.position, max_neighbours, max_radius
    )
    sparse_values = np.ones(neighbours.shape[0] * neighbours.shape[1])
    sparse_coordinates = (
        np.repeat(np.arange(neighbours.shape[0]), neighbours.shape[1]),
        neighbours.flatten(),
    )
    connectivity = coo_matrix((sparse_values, sparse_coordinates))
    return connectivity.toarray()


def spatial_clustering(snapshot, classification: np.ndarray = None):
    if classification is None:
        KnnModel = sklearn.externals.joblib.load("../models/knn-trimer.pkl")
        classification = compute_ml_order(
            KnnModel, snapshot.box, snapshot.position, snapshot.orientation
        )

    connectivity = neighbour_connectivity(snapshot)
    agg_cluster = sklearn.cluster.AgglomerativeClustering(
        n_clusters=2, connectivity=connectivity
    )
    return agg_cluster.fit_predict((classification > 0).reshape(-1, 1))


def plot_confusion_matrix(
    cm, classes, normalize=True, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")