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
from itertools import islice
from pathlib import Path
from typing import NamedTuple, Optional

import click
import gsd.hoomd
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sdanalysis import SimulationParams, order
from sdanalysis.frame import HoomdFrame
from sdanalysis.read import process_gsd
from sdanalysis.util import get_filename_vars
from sklearn.externals import joblib

from detection import spatial_clustering

logger = logging.getLogger(__name__)

KNNModel = joblib.load(
    Path(__file__).parent / "../models/knn-trimer.pkl"
)  # pylint:disable=invalid-name


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

    sim_params = SimulationParams(infile=infile, linear_steps=None)

    for index, (_, snap) in enumerate(process_gsd(sim_params)):
        if index % skip_frames != 0:
            continue

        ordering = order.compute_ml_order(
            KNNModel, snap.box, snap.position, snap.orientation
        )
        labels = spatial_clustering(snap, ordering)

        if np.sum(labels == 1) > 5:
            hull0 = ConvexHull(snap.position[labels == 0, :2])
            hull1 = ConvexHull(snap.position[labels == 1, :2])
            if hull0.volume > hull1.volume:
                hull = hull1
            else:
                hull = hull0

        else:
            hull = namedtuple("hull", ["area", "volume"])
            hull.area = 0
            hull.volume = 0
        states = CrystalFractions.from_ordering(ordering)
        df = {
            "temperature": float(fvars.temperature),
            "pressure": float(fvars.pressure),
            "crystal": fvars.crystal,
            "liq": float(states.liquid),
            "p2": float(states.p2),
            "p2gg": float(states.p2gg),
            "pg": float(states.pg),
            "surface_area": float(hull.area),
            "volume": float(hull.volume),
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


@click.command()
@click.argument("infile", type=click.Path(exists=True))
@click.argument("outfile", type=click.Path(dir_okay=False, file_okay=True))
@click.option("-s", "--skip-frames", default=100, type=int)
@click.option("-v", "--verbosity", callback=_verbosity, expose_value=False, count=True)
def main(infile, outfile, skip_frames):
    infile = Path(infile)
    outfile = Path(outfile)

    oufile.parent.mkdir(parents=True, exist_ok=True)
    compute_crystal_growth(infile, outfile, skip_frames)


if __name__ == "__main__":
    main()
