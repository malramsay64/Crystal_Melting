#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Helper functions for the creation and analysis of defects."""

from bokeh.layouts import gridplot
from hoomd.data import SnapshotParticleData, make_snapshot
from sdanalysis import HoomdFrame
from sdanalysis.figures import configuration
from sdanalysis.order import knn_model, compute_ml_order
from sdrun import SimulationParams


def remove_molecule(snapshot: SnapshotParticleData, index: int) -> SnapshotParticleData:
    """Remove an arbitratry molecule from the simulation.

    This also ensures a valid configuration once the molecule has been removed.
    """
    mask = snapshot.particles.body != index
    if sum(mask) == snapshot.particles.N:
        print("Index not in snapshot")
        return snapshot

    new_snapshot = make_snapshot(
        snapshot.particles.N - 3,
        snapshot.box,
        snapshot.particles.types,
        snapshot.pairs.types,
    )
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
    extent = 2 * int(num_mols / 2) * y

    for index in range(center - extent, center, (y - 1) * 2):
        snapshot = remove_molecule(snapshot, index)
        snapshot = remove_molecule(snapshot, index + 2)
    return snapshot


def _plot_snapshot(snapshot: SnapshotParticleData, order: bool = False):
    """Helper function to plot a single snapshot."""
    frame = HoomdFrame(snapshot)
    if order:
        from functools import partial

        order = partial(compute_ml_order, knn_model())
        return configuration.plot_frame(frame, order)
    return configuration.plot_frame(frame)


def plot_snapshots(snapshots, num_rows: int = 2, order: bool = False):
    # Length of sides to make a square
    num_columns = int(len(snapshots) / num_rows)
    figures = []
    for i in range(num_rows):
        row = []
        for j in range(num_columns):
            if i * num_columns + j > len(snapshots):
                figures.append(row)
                return gridplot(figures)
            fig = _plot_snapshot(snapshots[i * num_columns + j], order=order)
            fig.plot_height = int(fig.plot_height / num_rows)
            fig.plot_width = int(fig.plot_width / num_rows)
            row.append(fig)
        figures.append(row)
    return gridplot(figures)
