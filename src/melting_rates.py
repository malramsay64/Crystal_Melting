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
import os
from collections import namedtuple
from functools import partial
from multiprocessing import Manager, Pool, Queue, cpu_count
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import click
import gsd.hoomd
import numba
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sdanalysis import order, read
from sdanalysis.frame import HoomdFrame
from sdanalysis.util import get_filename_vars
from sklearn import cluster
from sklearn.externals import joblib

from detection import spatial_clustering

logger = logging.getLogger(__name__)

KNNModel = joblib.load(Path(__file__).parent / "../models/knn-trimer.pkl")


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

    with gsd.hoomd.open(str(infile)) as traj:
        max_frames = len(traj)
        for index in range(0, max_frames, skip_frames):
            snap = HoomdFrame(traj[index])
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
        if outfile is None:
            return order_df
        order_df.to_hdf(
            outfile, "fractions", format="table", append=True, min_itemsize=4
        )


def file_writer(queue: Queue, outfile: Path):
    logger.info("Started writer process")
    with pd.HDFStore(outfile, "w") as dst:
        while True:
            group, dataset = queue.get()
            logger.info("Writing: %s to %s", dataset.head(1), group)
            if dataset is None:
                break
            dst.append(group, dataset)
        dst.flush()
    logger.info("Completed Writing")


def process_crystal_growth(queue: Queue, infile: str, skip_frames: int = 100) -> None:
    logger.info("Processing: %s", infile.name)
    melting_data = compute_crystal_growth(infile, outfile=None, skip_frames=skip_frames)
    logger.debug("Adding data to write queue\n%s", melting_data.head(1))
    queue.put(("fractions", melting_data))


def parallel_process_files(input_files: Tuple[str], outfile: Path) -> None:
    manager = Manager()
    queue = manager.Queue()

    with Pool(cpu_count() + 1) as pool:
        writer = pool.apply_async(file_writer, (queue, outfile))

        pool.starmap(process_crystal_growth, ((queue, i) for i in input_files))
        logger.info("Completed Processing")

        queue.put((None, None))
        writer.wait()
        logger.info("Completed Writing")


def _verbosity(ctx, param, value) -> None:  # pylint: disable=unused-argument
    root_logger = logging.getLogger(__name__)
    levels = {0: "INFO", 1: "DEBUG"}
    log_level = levels.get(value, "DEBUG")
    logging.basicConfig(level=log_level)
    root_logger.setLevel(log_level)
    logger.debug("Setting log level to %s", log_level)


@click.command()
@click.option(
    "-i", "--input-path", default=None, type=click.Path(exists=True, file_okay=False)
)
@click.option("-o", "--output-path", default=None)
@click.option("-s", "--skip-frames", default=100, type=int)
@click.option("-v", "--verbosity", callback=_verbosity, expose_value=False, count=True)
def main(input_path, output_path, skip_frames):
    if input_path is None:
        input_path = Path.cwd()
    if output_path is None:
        output_path = Path.cwd()
    input_path = Path(input_path)
    output_path = Path(output_path)
    file_list = list(input_path.glob("dump-*.gsd"))
    if len(file_list) == 0:
        raise FileNotFoundError(f"No gsd files found in {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    outfile = Path(output_path) / "melting.h5"
    if outfile.exists():
        res = click.prompt("File already exists, append or replace {A,r}")
        if res is "r":
            os.remove(outfile)
    parallel_process_files(tuple(file_list), outfile)


if __name__ == "__main__":
    main()