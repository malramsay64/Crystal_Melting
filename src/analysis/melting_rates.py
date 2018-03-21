#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.
"""Compute the melting rates of the crystal structures.
"""

import os
from functools import partial
from pathlib import Path

import click
import gsd.hoomd
import numba
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sdanalysis import order
from sklearn import cluster


@numba.autojit
def periodic_distance(box: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
    inv_box = 1. / box
    x = X - Y
    for j in range(len(x)):
        if box[j] > 1e-6:
            images = inv_box[j] * x[j]
            x[j] = box[j] * (images - round(images))
    return np.linalg.norm(x)


def get_vars(fname: Path):
    """Extract simulation variables from the filename."""
    flist = fname.stem.split('-')
    temp = flist[3][1:]
    press = flist[2][1:]
    crys = flist[4]
    return temp, press, crys


def compute_crystal_growth(infile: Path, outfile: Path) -> None:
    temp, pressure, crys = get_vars(infile)
    order_list = []
    order_dimension = 5.
    with gsd.hoomd.open(str(infile)) as traj:
        for snap in traj:
            try:
                num_mols = snap.particles.body.max() + 1
                ordering = order.compute_ml_order(
                    order.knn_model(),
                    snap.configuration.box,
                    snap.particles.position,
                    snap.particles.orientation,
                )
                states = pd.Series(ordering).value_counts(normalize=True)
                crystalline = ordering != 'liq'
                crystalline = np.expand_dims(crystalline, axis=1)
                clustering_matrix = np.append(
                    snap.particles.position, crystalline * order_dimension, axis=1
                )
                _, labels = cluster.dbscan(
                    clustering_matrix,
                    min_samples=4,
                    eps=3,
                    metric=partial(
                        periodic_distance,
                        np.append(snap.configuration.box[:3], order_dimension * 2),
                    ),
                )
                if np.sum(labels == 1) > 3:
                    hull = ConvexHull(snap.particles.position[labels == 1,:2])
                else:

                    def hull():
                        pass

                    hull.area = 0
                    hull.volume = 0
                df = pd.DataFrame(
                    {
                        'state': states.index,
                        'fraction': states.values,
                        'temperature': temp,
                        'pressure': pressure,
                        'crystal': crys,
                        'surface-area': hull.area,
                        'volume': hull.volume,
                        'time': snap.configuration.step,
                    }
                )
                order_list.append(df)
            except ValueError:
                # There are occasions when the molecule positions are outside the unit cell volume.
                # Rather than checking all molecules in all confiurations, instead I am just waiting
                # for the errors to come up, catch them and continue on. This is a rare occurence so
                # not a huge problem with the statistics.
                continue

        order_df = pd.concat(order_list)
        order_df.time = order_df.time.astype(np.uint32)
        order_df.to_hdf(outfile, 'fractions', format='table', append=True)


@click.command()
@click.option(
    '-i', '--input-path', default=None, type=click.Path(exists=True, file_okay=False)
)
@click.option('-o', '--output-path', default=None)
def main(input_path, output_path):
    if input_path is None:
        input_path = Path.cwd()
    if output_path is None:
        output_path = Path.cwd()
    input_path = Path(input_path)
    output_path = Path(output_path)
    file_list = list(input_path.glob('dump-*.gsd'))
    if len(file_list) == 0:
        raise FileNotFoundError(f'No gsd files found in {input_path}')

    output_path.mkdir(parents=True, exist_ok=True)
    outfile = Path(output_path) / 'melting.h5'
    if outfile.exists():
        res = click.prompt('File already exists, append or replace {A,r}')
        if res is 'r':
            os.remove(outfile)
    for infile in file_list:
        compute_crystal_growth(infile, outfile)


if __name__ == "__main__":
    main()
