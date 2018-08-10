#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Test the module for the creation of defects within a simulation."""

import hoomd
import numpy as np
import pytest
from sdrun import SimulationParams, TrimerP2, init_from_crystal

from ..defects import (
    central_molecule,
    remove_horizontal,
    remove_molecule,
    remove_vertical,
    remove_vertical_cell,
)


@pytest.fixture
def sim_params():
    yield SimulationParams(
        temperature=0.10,
        pressure=1.00,
        crystal=TrimerP2(),
        num_steps=100,
        cell_dimensions=(30, 42, 1),
        hoomd_args="--notice-level=0",
    )


@pytest.fixture(scope="module")
def params_snapshot():
    sim_params = SimulationParams(
        temperature=0.10,
        pressure=1.00,
        crystal=TrimerP2(),
        num_steps=100,
        cell_dimensions=(30, 42, 1),
        hoomd_args="--notice-level=0",
    )
    init_snapshot = init_from_crystal(sim_params)
    return sim_params, init_snapshot


def test_remove_molecule(params_snapshot):
    """Ensure the removal of a molecule gives sensible configuration."""
    remove_mol = 0
    sim_params, snapshot = params_snapshot

    mask = snapshot.particles.body != remove_mol
    remove_snapshot = remove_molecule(snapshot, remove_mol)
    num_removed = sim_params.molecule.num_particles
    assert snapshot.particles.N == remove_snapshot.particles.N + num_removed
    assert np.all(
        snapshot.particles.position[mask] == remove_snapshot.particles.position
    )


def test_remove_molecule_failure(params_snapshot):
    """Ensure the attempted removal of a nonexistant molecule fails."""
    remove_mol = -1
    sim_params, snapshot = params_snapshot
    with pytest.raises(IndexError):
        remove_snapshot = remove_molecule(snapshot, remove_mol)


def test_central_molecule(params_snapshot):
    """Ensure the central molecule is somewhat central."""
    sim_params, snapshot = params_snapshot
    central = central_molecule(sim_params)
    assert central > 0
    assert central in snapshot.particles.body


@pytest.mark.parametrize("mols_removed", [2, 8, 9, 21])
def test_remove_vertical(params_snapshot, mols_removed):
    """Ensure the number of molecuels removed remains consistent."""
    sim_params, snapshot = params_snapshot
    remove_snap = remove_vertical(snapshot, sim_params, mols_removed)
    num_removed = sim_params.molecule.num_particles * mols_removed
    assert snapshot.particles.N == remove_snap.particles.N + num_removed


@pytest.mark.parametrize("mols_removed", [-1])
def test_remove_vertical_failure(params_snapshot, mols_removed):
    sim_params, snapshot = params_snapshot
    with pytest.raises(ValueError):
        remove_snap = remove_vertical(snapshot, sim_params, mols_removed)


@pytest.mark.parametrize("mols_removed", [2, 8, 9, 21])
def test_remove_horizontal(params_snapshot, mols_removed):
    """Ensure the number of molecuels removed remains consistent."""
    sim_params, snapshot = params_snapshot
    remove_snap = remove_horizontal(snapshot, sim_params, mols_removed)
    # The minimum removed is 4
    actual_removed = max(mols_removed // 4 * 4, 4)
    num_removed = sim_params.molecule.num_particles * actual_removed
    assert snapshot.particles.N == remove_snap.particles.N + num_removed


@pytest.mark.parametrize("mols_removed", [-1])
def test_remove_horizontal_failure(params_snapshot, mols_removed):
    sim_params, snapshot = params_snapshot
    with pytest.raises(ValueError):
        remove_snap = remove_horizontal(snapshot, sim_params, mols_removed)


@pytest.mark.parametrize("cells_removed", [2, 8, 9, 21])
def test_remove_vert_cells(params_snapshot, cells_removed):
    """Ensure the number of unit cells removed remains consistent."""
    sim_params, snapshot = params_snapshot
    remove_snap = remove_vertical_cell(snapshot, sim_params, cells_removed)
    actual_removed = cells_removed * 2
    num_removed = sim_params.molecule.num_particles * actual_removed
    assert snapshot.particles.N == remove_snap.particles.N + num_removed


@pytest.mark.parametrize("cells_removed", [-1])
def test_remove_vert_cells_failure(params_snapshot, cells_removed):
    sim_params, snapshot = params_snapshot
    with pytest.raises(ValueError):
        remove_snap = remove_vertical_cell(snapshot, sim_params, cells_removed)
