#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Helper functions for the dynamics calculations."""

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sdanalysis.relaxation import series_relaxation_value

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@click.group()
def main():
    pass


@main.command()
@click.argument("infile", type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option(
    "--min-samples",
    default=10,
    type=int,
    help="Minimum number of samples for each data point.",
)
def clean(infile: Path, min_samples: int):
    infile = Path(infile)
    # Cleanup the dynamics dataset
    df = pd.read_hdf(infile, "dynamics")

    # Most of the values are plotted on a log scale for the time axis, values less than
    # or equal to 0 cause issues.
    df = df.query("time > 0")

    # We want to discard values where there are not enough to get decent statistics, in
    # this case I have chosen 10 as the magic number.
    df = df.assign(
        count=df.groupby(["time", "temperature", "pressure"])["keyframe"].transform(
            "count"
        )
    )
    df = df.query("count > @min_samples")
    #  # Don't want the count in the final dataset, just a temporary column
    df = df.drop(columns=["count"], axis=1)

    # The values where the MSD is greater than 100 are going to have issues with
    # the periodic boundary conditions so remove those columns.
    df = df.query("msd < 100")
    df = df.reset_index()

    df.to_hdf(infile.with_name(infile.stem + "_clean" + ".h5"), "dynamics")

    df_mol = pd.read_hdf(infile, "molecular_relaxations")
    df_mol = df_mol.reset_index()

    # Replace invalid values (2**32 - 1) with NaN's
    df_mol.replace(2 ** 32 - 1, np.nan, inplace=True)
    # Remove keyframes where relaxation hasn't completed,
    # that is there are NaN values present.
    df_mol = df_mol.groupby(["keyframe", "temperature", "pressure"]).filter(
        lambda x: x.isna().sum().sum() == 0
    )

    df_mol = df_mol.assign(
        count=df_mol.groupby(["temperature", "pressure"])["keyframe"].transform("count")
    )
    df_mol = df_mol.query("count > @min_samples")
    #  # Don't want the count in the final dataset, just a temporary column
    df_mol = df_mol.drop(columns=["count"], axis=1)

    df_mol.to_hdf(
        infile.with_name(infile.stem + "_clean" + ".h5"), "molecular_relaxations"
    )


@main.command()
@click.argument("infile", type=click.Path(file_okay=True, dir_okay=False, exists=True))
def bootstrap(infile):
    infile = Path(infile)
    outfile = infile.with_name(infile.stem + "_agg" + ".h5")
    df = pd.read_hdf(infile, "dynamics")

    index = ["temperature", "pressure", "time"]
    df_agg = df.groupby(index).agg(["mean", "std"])
    df_agg.columns = ["_".join(x) for x in df_agg.columns]
    df_agg = df_agg.reset_index()

    df_agg.to_hdf(outfile, "dynamics")

    df_mol = pd.read_hdf(infile, "molecular_relaxations")
    df_mol = df_mol.reset_index()

    # Taking the average over all the molecules from a single keyframe
    # makes the most sense from the perspective of computing an average,
    # since all molecules are present and not independent. One can be
    # fast because others are slow.
    df_mol = df_mol.groupby(["temperature", "pressure", "keyframe"]).mean()

    index = ["temperature", "pressure"]
    df_mol_agg = df_mol.groupby(index).agg(["mean", "std"])
    df_mol_agg.columns = ["_".join(x) for x in df_mol_agg.columns]
    df_mol_agg = df_mol_agg.reset_index()

    df_mol_agg.to_hdf(outfile, "molecular_relaxations")

    df_relax = (
        df.set_index("time")
        .groupby(["temperature", "pressure", "keyframe"])
        .agg(series_relaxation_value)
    )
    df_relax["inv_diffusion"] = 1 / df_relax["msd"]

    index = ["temperature", "pressure"]
    df_relax_agg = df_relax.groupby(index).agg(["mean", "std"])
    df_relax_agg.columns = ["_".join(x) for x in df_relax_agg.columns]
    df_relax_agg = df_relax_agg.reset_index()

    df_relax_agg.to_hdf(outfile, "relaxations")


@main.command()
@click.argument("output", type=click.Path(file_okay=True, dir_okay=False))
@click.argument(
    "infiles", nargs=-1, type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
def collate(output, infiles):
    with pd.HDFStore(output, "w") as dst:
        for file in infiles:
            with pd.HDFStore(file) as src:
                for key in ["dynamics", "molecular_relaxations"]:
                    try:
                        df = src.get(key)
                    except KeyError:
                        logger.warning(
                            "Key: %s does not exist in dataframe: %s", key, file
                        )
                    if key == "dynamics":
                        # The timestep is given the time column, so convert that here
                        df["timestep"] = df["time"]
                        # This converts the timestep to the real time
                        df["time"] = df["timestep"] * 0.005
                    elif key == "molecular_relaxations":
                        df = df.set_index(
                            ["pressure", "temperature", "keyframe", "molecule"]
                        )
                        df *= 0.005
                        df = df.reset_index()
                    df["temperature"] = df["temperature"].astype(float)
                    df["pressure"] = df["pressure"].astype(float)
                    dst.append(key, df)


if __name__ == "__main__":
    main()
