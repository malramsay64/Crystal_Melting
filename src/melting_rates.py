#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Compute the melting rates of the crystal structures.
"""

import logging
from collections import namedtuple
from pathlib import Path
from typing import NamedTuple, Optional

import click
import freud
import joblib
import numpy as np
import pandas as pd
import scipy.stats
from pandas.api.types import CategoricalDtype
from scipy.spatial import ConvexHull
from sdanalysis import order
from sdanalysis.read import open_trajectory
from sdanalysis.util import get_filename_vars

from detection import spatial_clustering
from util import normalised_temperature

logger = logging.getLogger(__name__)

KNNModel = Path(__file__).parent / "../models/knn-trimer.pkl"


class CrystalFractions(NamedTuple):
    liquid: float = 0.0
    p2: float = 0.0
    p2gg: float = 0.0
    pg: float = 0.0

    @classmethod
    def from_ordering(cls, ordering: np.ndarray) -> "CrystalFractions":
        lookup = {0: "liquid", 1: "p2", 2: "p2gg", 3: "pg"}
        values, counts = np.unique(ordering.astype(int), return_counts=True)
        num_elements = len(ordering)
        return cls(**{lookup[v]: c / num_elements for v, c in zip(values, counts)})


def compute_crystal_growth(
    infile: Path, outfile: Path = None, skip_frames: int = 100
) -> Optional[pd.DataFrame]:
    fvars = get_filename_vars(infile)
    order_list = []
    order_dimension = 5.0

    ml_order = order.create_ml_ordering(KNNModel)
    voronoi = freud.voronoi.Voronoi(freud.box.Box(10, 10), buff=4)
    for index, snap in enumerate(open_trajectory(infile, progressbar=True)):
        if index % skip_frames != 0:
            continue

        classification = ml_order(snap)
        labels = spatial_clustering(snap, classification)

        # The crystal should always be smaller, so ensure it has label 1
        if sum(labels) > len(labels) / 2:
            labels = -labels + 1

        # Compute Voroni cells and volumes
        voronoi.compute(snap.position, box=snap.box)
        voronoi.computeVolumes()
        # Sum the volumes of the crystalline molecules
        voronoi_volume = np.sum(voronoi.volumes[labels == 1])

        # Only compute ConvexHull for at least 5 particles
        if np.sum(labels == 1) > 5:
            hull = ConvexHull(snap.position[labels == 1, :2])
        else:
            hull = namedtuple("hull", ["area", "volume"])
            hull.area = 0
            hull.volume = 0

        if fvars.iteration_id is None:
            iter_id = 1
        else:
            iter_id = fvars.iteration_id
        states = CrystalFractions.from_ordering(classification)
        df = {
            "temperature": float(fvars.temperature),
            "pressure": float(fvars.pressure),
            "crystal": fvars.crystal,
            "iter_id": int(iter_id),
            "liq": float(states.liquid),
            "p2": float(states.p2),
            "p2gg": float(states.p2gg),
            "pg": float(states.pg),
            "surface_area": float(hull.area),
            "volume": float(hull.volume),
            "voronoi_volume": float(voronoi_volume),
            "time": float(snap.timestep),
        }

        order_list.append(df)

        order_df = pd.DataFrame.from_records(order_list)
        order_df.time = order_df.time.astype(np.uint32)
        logger.debug("Value of outfile is: %s", outfile)
    if outfile is None:
        return order_df

    order_df.to_hdf(outfile, "fractions", format="table", append=True, min_itemsize=4)
    return None


def _verbosity(ctx, param, value) -> None:  # pylint: disable=unused-argument
    root_logger = logging.getLogger(__name__)
    levels = {0: "INFO", 1: "DEBUG"}
    log_level = levels.get(value, "DEBUG")
    logging.basicConfig(level=log_level)
    root_logger.setLevel(log_level)
    logger.debug("Setting log level to %s", log_level)


@click.group()
def main():
    pass


@main.command()
@click.argument("infile", type=click.Path(exists=True))
@click.argument("outfile", type=click.Path(dir_okay=False, file_okay=True))
@click.option("-s", "--skip-frames", default=1, type=int)
@click.option("-v", "--verbosity", callback=_verbosity, expose_value=False, count=True)
def melting(infile, outfile, skip_frames):
    infile = Path(infile)
    outfile = Path(outfile)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    compute_crystal_growth(infile, outfile, skip_frames)


@main.command()
@click.argument("infile", type=click.Path(exists=True))
def clean(infile):
    infile = Path(infile)
    df = pd.read_hdf(infile, "fractions")
    df = df.query("volume > 100").query("time > 0")
    df["radius"] = np.sqrt(df["volume"].values) / np.pi
    df.to_hdf(
        infile.with_name(infile.stem + "_clean" + ".h5"), "fractions", format="table"
    )


@main.command()
@click.argument("infile", type=click.Path(exists=True))
def rates(infile):
    infile = Path(infile)

    def gradient_regression_mean(df):
        """Calculate the gradient using linear regression.

        The y value of the gradient is of each volume divided by the mean
        of the surface area over the course of the simulation. This is an
        adjustment to bring the accuracy of the resulting slope closer to
        that calculated by other methods.

        """
        df = df.dropna()
        try:
            slope, _, _, _, std_err = scipy.stats.linregress(df.time, df.radius)
        except FloatingPointError:
            slope, std_err = np.nan, np.nan
        return slope, std_err

    def instantaneous_gradient(df):
        if df.shape[0] > 3:
            return np.gradient(df.radius, df.time)
        return np.nan

    df = pd.read_hdf(infile, "fractions")

    # Normalise temperature based on melting point
    df["temp_norm"] = normalised_temperature(df["temperature"], df["pressure"])

    group_bys = ["temperature", "pressure", "crystal", "temp_norm", "iter_id"]
    gradient_mean = df.groupby(group_bys).apply(
        lambda x: gradient_regression_mean(x)[0]
    )
    gradient_error = df.groupby(group_bys).apply(
        lambda x: gradient_regression_mean(x)[1]
    )

    gradient1 = pd.DataFrame(
        {"mean": gradient_mean, "error": gradient_error}, index=gradient_mean.index
    )
    gradient1.reset_index(inplace=True)

    ## Temperatures above these are too high and the melting is more complicated.
    gradient1 = gradient1.query("temp_norm < 1.7")
    gradient1.to_hdf(infile, "rates", format="table")


@main.command()
@click.argument("output", type=click.Path(file_okay=True, dir_okay=False))
@click.argument(
    "infiles", nargs=-1, type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
def collate(output, infiles):
    with pd.HDFStore(output, "w") as dst:
        for file in infiles:
            with pd.HDFStore(file) as src:
                key = "fractions"
                df = src.get(key)
                df["crystal"] = df["crystal"].astype(
                    CategoricalDtype(categories=["p2", "pg", "p2gg"])
                )
                dst.append(key, df)


if __name__ == "__main__":
    main()
