---
jupyter:
  jupytext:
    formats: ipynb,md
    target_format: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.0
  kernelspec:
    display_name: crystal
    language: python
    name: crystal
---

# Thermodynamics

A thermodynamic analysis of the simulations.
This is to develop the random initialisation of a configuration
and to confirm the existing thermodynamic behaviour.

```python
import pandas as pd
import numpy as np
import altair as alt

from ipywidgets import interact

import sys

sys.path.append("../src")
import figures
```

## Hoomd Output

I have logged many thermodynamic quantities throughout the Hoomd simulations.
Although I have separated these from the main simulation data,
I can still process the files.
The data file I am investigating is from the calculation of dynamics,
meaning all the quantities should be a equilibrium.

```python
thermo_file = "../data/simulations/interface/output/thermo-Trimer-P13.50-T1.50-p2gg.log"
df = pd.read_csv(thermo_file, sep="\t")
df["time"] = df["timestep"] * 0.005
```

```python
df.columns
```

```python
c = alt.Chart(df).mark_line().encode(x=alt.X("time", title="Time"))


@interact(quantity=list(df.columns))
def plot_thermo(quantity):
    return c.encode(y=quantity)
```

```python
print(
    f"The temperature of the simulation is {df.temperature.mean():.4f} "
    "which matches the intended temperature of 1.50"
)
```

A feature I am particularly interested in is the calculation of the kinetic energy for each simulation

```python
from sdanalysis import HoomdFrame
import gsd.hoomd
```

```python
trajectory = "../data/simulations/dynamics/output/trajectory-Trimer-P13.50-T1.50.gsd"
with gsd.hoomd.open(trajectory) as traj:
    snap = HoomdFrame(traj[0])
```

It is relatively simple to calculate the translational kinetic energy for a configuration

```python
mass = 3
trans_KE = (
    0.5
    * 3
    * np.mean(np.sum(np.square(snap.frame.particles.velocity[: snap.num_mols]), axis=1))
)
trans_KE
```

With the relationship $T=KE$,
our kinetic energy is 3 times the expected value,
and 3 times what Hoomd is calculating.
Which would mean it is using a mass of 1 for the molecules.
