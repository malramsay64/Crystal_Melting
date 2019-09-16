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
from typing import Tuple

import click
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sdanalysis import order
from sdanalysis.read import open_trajectory
from sdanalysis.util import get_filename_vars

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@click.group()
def main():
    pass


BINS = np.linspace(-1, 1, 5001)
BIN_VALUES = (BINS[1:] + BINS[:-1]) / 2


def aggregate(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert values to a histogram of bins and values

    This takes a collection of values in the range [0,1], binning values into a
    histogram. Values are binned in increments of 1a-4, with only the bins containing values being returned.

    Args:
        values: A collection of values which will be binned into 

    Returns:
        centers: The centers of each of the bins
        counts: The count of each bin

    """
    hist = np.histogram(values, bins=BINS, density=True)[0]
    non_zero = np.nonzero(hist)
    return BIN_VALUES[non_zero], hist[non_zero]


@main.command()
@click.argument("output", type=click.Path(file_okay=True, dir_okay=False))
@click.argument(
    "infiles", nargs=-1, type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
def collate(output, infiles):
    with pd.HDFStore(output, "w") as dst:
        for file in infiles:
            print(file)
            with pd.HDFStore(file) as src:
                key = "ordering"
                df = src.get(key)
                bin_values, count = aggregate(df["orientational_order"])

                df = pd.DataFrame(
                    {
                        "temperature": float(df["temperature"].values[0]),
                        "pressure": float(df["pressure"].values[0]),
                        "crystal": df["crystal"].values[0],
                        "bins": bin_values,
                        "count": count,
                        "probability": count * (BINS[1] - BINS[0]),
                    }
                )
                df["crystal"] = df["crystal"].astype(
                    CategoricalDtype(categories=["p2", "p2gg", "pg", "liquid"])
                )
                dst.append(key, df)


@main.command()
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("outfile", type=click.Path(file_okay=True, dir_okay=False))
def analyse(infile, outfile):
    dataframes = []
    file_vars = get_filename_vars(infile)
    crystal = file_vars.crystal
    if crystal is None:
        crystal = "liquid"
    for snap in open_trajectory(infile, progressbar=True):
        orientational_order = order.orientational_order(
            snap.box, snap.position, snap.orientation, angle_factor=2.0
        )
        df = pd.DataFrame(
            {
                "molecule": np.arange(snap.num_mols),
                "orientational_order": orientational_order,
                "temperature": float(file_vars.temperature),
                "pressure": float(file_vars.pressure),
                "crystal": crystal,
            }
        )
        df["crystal"] = df["crystal"].astype("category")
        dataframes.append(df)
    with pd.HDFStore(outfile) as dst:
        dst.append("ordering", pd.concat(dataframes))


@main.command()
@click.argument("outfile", type=click.Path(file_okay=True, dir_okay=False))
@click.argument(
    "infiles", nargs=-1, type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
def thermodynamics(outfile, infiles):
    dfs = []
    for filename in infiles:
        fvars = get_filename_vars(filename)

        df = pd.read_csv(filename, sep="\t")
        # All the values are written to the same output file, so make sure there is only
        # a single trajectory worth of values.
        df = df.drop_duplicates("timestep", keep="last")
        # We want quantities for each
        df = df.div(df.N, axis=0)

        # Take the second half of the values to ensure there is no issue with
        # equilibration
        df = df.iloc[len(df) // 2 :, :]

        # Calculate Total Energy
        df["total_energy"] = df["kinetic_energy"] + df["potential_energy"]

        df["pressure"] = fvars.pressure
        if fvars.crystal is not None:
            df["crystal"] = fvars.crystal
        else:
            df["crystal"] = "liquid"
        df["temperature"] = fvars.temperature
        df = df.set_index(["pressure", "temperature", "crystal"])

        # Perform aggregations on the dataframe, making it much easier to work with.
        df = df.groupby(["pressure", "temperature", "crystal"]).agg(["mean", "std"])

        dfs.append(df)

    pd.concat(dfs).to_hdf(outfile, "thermo")


if __name__ == "__main__":
    main()
