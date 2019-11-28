#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Functions to generate the machine learning models.

This generates the machine learning models for use in the crystal detection. It takes
the configurations from the dataset direcotry, using them to train the models.

The model generated is a K-Nearest Neighbours model, chosen for it's simplicity and
speed of execution while still having nearly the best performance.

"""

import glob
import logging
from pathlib import Path
from typing import List, Tuple

import click
import gsd.hoomd
import joblib
import numpy as np
import sklearn.neighbors
from sdanalysis import HoomdFrame
from sdanalysis.order import relative_orientations
from sdanalysis.util import Variables, get_filename_vars

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
    classification = np.zeros(len(snapshot), dtype=int)
    classification[is_crystal] = mapping[crystal]
    classification[boundary] = 4
    return classification


def read_all_files(
    pathname: Path, index: int = 0, pattern: str = "dump-Trimer-*.gsd"
) -> List[Tuple[Variables, HoomdFrame]]:
    pathname = Path(pathname)
    snapshots = []
    for filename in glob.glob(str(pathname / pattern)):
        logger.debug("Reading %s", Path(filename).stem)
        with gsd.hoomd.open(str(filename)) as trj:
            try:
                snapshots.append((get_filename_vars(filename), HoomdFrame(trj[index])))
            except IndexError:
                continue
    if not snapshots:
        logger.warning(
            "There were no files found with a configuration at index %s", index
        )
    return snapshots


def prepare_training_dataset(
    directory: Path, frame_index: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a directory of files into labelled training data.

    This functions turns a directory of trajectory files into a training dataset for use
    in a machine learning algorithm.

    """
    directory = Path(directory)

    snaps = read_all_files(directory, frame_index, pattern="trajectory-*.gsd")

    classes = np.concatenate(
        [classify_mols(snap, variables.crystal) for variables, snap in snaps]
    )
    orientations = np.concatenate(
        [relative_orientations(s.box, s.position, s.orientation) for _, s in snaps]
    )

    mask = classes < 4
    features = orientations[mask]
    labels = classes[mask]

    return features, labels


def train_knn_model(
    features: np.ndarray, labels: np.ndarray
) -> sklearn.neighbors.KNeighborsClassifier:
    knn = sklearn.neighbors.KNeighborsClassifier()
    knn.fit(features, labels)
    return knn


@click.group()
def main():
    pass


@main.command()
@click.argument(
    "directory", type=click.Path(file_okay=False, dir_okay=True, exists=True)
)
def train_models(directory):
    features, labels = prepare_training_dataset(directory, 100)
    knn = train_knn_model(features, labels)

    outdir = Path.cwd() / "models"
    outdir.mkdir(exist_ok=True)

    joblib.dump(knn, outdir / "knn-trimer.pkl")


if __name__ == "__main__":
    main()
