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

from ..defects import remove_molecule


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


def test_remove_molecule(sim_params):
    remove_mol = 0
    init_snapshot = init_from_crystal(sim_params)

    mask = init_snapshot.particles.body != remove_mol
    remove_snapshot = remove_molecule(init_snapshot, remove_mol)
    num_removed = sim_params.molecule.num_particles
    assert init_snapshot.particles.N == remove_snapshot.particles.N + num_removed
    assert np.all(
        init_snapshot.particles.position[mask] == remove_snapshot.particles.position
    )
