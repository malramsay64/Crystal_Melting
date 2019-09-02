#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""A module to measure the structural fluctuations of each state.

The concept of this module is to understand how much particles within each state are
moving, to get an idea of the likelihood of transitioning from one state to anther.
States which are highly constrained will allow small amounts of motion, while states
which are more flexible will be able to rapidly change configuration.

"""

import logging

import click
import gsd
import numpy as np
import pandas as pd
from sdanalysis import order
from sdanalysis.read import open_trajectory
from sdanalysis.util import get_filename_vars

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@click.command()
@click.argument("infile")
@click.argument("outfile")
def main(infile, outfile):
    dataframes = []
    pressure, temperature, crystal, *_ = get_filename_vars(infile)
    if crystal is None:
        crystal = "liquid"
    for snap in open_trajectory(infile, progressbar=True):
        orientational_order = order.orientational_order(
            snap.box, snap.position, snap.orientation
        )
        dataframes.append(
            pd.DataFrame(
                {
                    "molecule": np.arange(snap.num_mols),
                    "orientational_order": orientational_order,
                    "temperature": temperature,
                    "pressure": pressure,
                    "crystal": crystal,
                }
            )
        )
    with pd.HDFStore(outfile) as dst:
        dst.append("ordering", pd.concat(dataframes))


if __name__ == "__main__":
    main()
