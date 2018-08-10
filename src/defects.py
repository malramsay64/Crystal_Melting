#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Helper functions for the creation and analysis of defects."""

import numpy as np
from bokeh.layouts import gridplot
from hoomd.data import SnapshotParticleData, make_snapshot
from sdanalysis import HoomdFrame
from sdanalysis.figures import configuration
from sdanalysis.order import compute_ml_order, knn_model
from sdrun import SimulationParams


def remove_molecule(snapshot: SnapshotParticleData, index: int) -> SnapshotParticleData:
    """Remove an arbitratry molecule from a hoomd snapshot.

    Args
    ----
        snapshot (SnapshotParticleData): Snapshot from which a particle will be removed.
        index (int): The index of the molecule to remove. This index is the zero indexed body of the
            particle. All particles with the same body index will also be removed from the snapshot.

    Returns
    -------
        SnapshotParticleData: A new snapshot with one less molecule.

    The removal of a molecule is done by creating a new snapshot with N fewer particles, where N is
    the number of particles in a molecule. This approach is simpler than modifying the existing
    snapshot, and ensures a valid configuration once the molecule has been removed.

    ... note:
        The index of molecules changes when this function is applied. All molecule ids need to be 
        contigous, so on the removal of a molecule the largeset molecule id is removed, so all molecule 
        IDs between ``index`` and the number of bodies will have shifted by 1.

    """
    mask = snapshot.particles.body != index
    num_particles = int(np.sum(mask))
    if num_particles == snapshot.particles.N:
        print("Index not in snapshot")
        return snapshot

    new_snapshot = make_snapshot(
        num_particles, snapshot.box, snapshot.particles.types, snapshot.pairs.types
    )
    # All the attributes from the old snapshot which need to be applied to the new one.
    snapshot_attributes = [
        "position",
        "angmom",
        "velocity",
        "orientation",
        "acceleration",
        "image",
        "mass",
        "moment_inertia",
        "typeid",
    ]

    for attr in snapshot_attributes:
        getattr(new_snapshot.particles, attr)[:] = getattr(snapshot.particles, attr)[
            mask
        ]
    # Remove the tag of the largest
    body_mask = snapshot.particles.body != max(snapshot.particles.body)
    new_snapshot.particles.body[:] = snapshot.particles.body[body_mask]
    return new_snapshot


def central_molecule(run_params: SimulationParams) -> int:
    """Find the molecule closest to the center of the simulation.

    This finds the unit cell halfway along each of the axes, then multiplies
    by 2 since there are 2 molecules in each unit cell.

    """
    x, y, z = run_params.cell_dimensions
    return int(x / 2 * y + y / 2) * 2


def remove_vertical(
    snapshot: SnapshotParticleData, run_params: SimulationParams, num_mols: int
) -> SnapshotParticleData:
    center = central_molecule(run_params)
    for index in range(center - 2 * int(num_mols / 2), center):
        snapshot = remove_molecule(snapshot, index)
    return snapshot


def remove_horizontal(
    snapshot: SnapshotParticleData, run_params: SimulationParams, num_mols: int
) -> SnapshotParticleData:
    center = central_molecule(run_params)
    x, y, x = run_params.cell_dimensions
    extent = 2 * int(num_mols / 4) * y

    for index in range(center - extent, center + extent, (y - 1) * 2):
        snapshot = remove_molecule(snapshot, index)
        snapshot = remove_molecule(snapshot, index + 2)
    return snapshot


def plot_snapshot(snapshot: SnapshotParticleData, order: bool = False):
    """Helper function to plot a single snapshot."""
    frame = HoomdFrame(snapshot)
    if order:
        from functools import partial

        def order_function(*args, **kwargs):
            result = compute_ml_order(knn_model(), *args, **kwargs)
            return result == "liq"

        return configuration.plot_frame(frame, order_function)
    return configuration.plot_frame(frame)


def plot_snapshots(
    snapshots, num_columns: int = 2, num_rows: int = None, order: bool = False
):
    # Length of sides to make a square
    if num_rows is None:
        num_rows = int(len(snapshots) / num_columns)
    figures = []
    for i in range(num_columns):
        row = []
        for j in range(num_rows):
            if i * num_rows + j > len(snapshots):
                figures.append(row)
                return gridplot(figures)
            fig = plot_snapshot(snapshots[i * num_rows + j], order=order)
            fig.plot_height = int(fig.plot_height / num_rows)
            fig.plot_width = int(fig.plot_width / num_rows)
            row.append(fig)
        figures.append(row)
    return gridplot(figures)
