---
title: Defect Creation
author: Malcolm Ramsay
date: 2018-10-24

papersize: a4
fontsize: 12pt
fignos-cleveref: True
fignos-capitalise: True
---

While defects are an important part of crystal formation
on the small scale of typical molecular dynamics simulations,
defects are a rare occurrence,
even for crystals that grow quickly.
As an alternate method of investigating defects within a crystal,
the defects can be manufactured by removing particles
from an existing crystal structure.
I am using the p2 crystal structure of the Trimer molecule
which is thought to be the most stable.

# Horizontal Defects

The horizontal defect is created by removing 21 molecules
from the center of an equilibrated crystal structure
resulting in @fig:horiz_setup.

![Starting configuration for the horizontal defect, where 21 molecules have been
removed.](../figures/06_defect_horiz_setup.png){#fig:horiz_setup}

This configuration is designed to mimic the presence of two edge dislocations,
with one at each edge of the removed particles.
After running the simulation at low temperature (0.20) for 1000 steps,
the resulting configuration (@fig:horiz_low)
has behaviour that matches a line defect,
albeit with regions of disorder at each end of the created defect.

![Horizontal defect after running for 1000 steps at a temperature of
0.20.](../figures/06_defect_horiz_thermal.png){#fig:horiz_low}

The regions of disorder are likely the molecules filling up the space,
effectively being a low pressure region.
The crystal is unable to make the sharp change to fill the corners of the defect
instead slowly shearing over 4 layers to occupy the missing space.
With the corner regions now devoid of particles,
the particles on the boundary fill it with a low density disordered phase.

When increasing the temperature to 1.30,
which is the melting point of the crystal,
the disorder at the defect is more pronounced
(@fig:horiz_high).

![Horizontal defect after running for 500 steps at a temperature of
1.30.](../figures/06_defect_horiz_thermal_1.30.png){#fig:horiz_high}

The degree of disorder here is indicative of
the particles at the edge of the defect leaning the crystal structure
before the crystal structure has had time to fill the gap.

# Vertical Defects

With the horizontal defects removing a layer of the crystal
exposes another identical layer of crystal transposed a little.
However, when removing particles vertically,
there are two layers to the crystal structure
which would require the removal of two layers
to return to the original structure.
For the vertical defects I have investigated defects
of both single and double layer.

## Single Layer

@fig:vert_single_setup shows the setup of the single layer vertical defect,
which has had 21 molecules removed from a layer of the crystal structure.

![Setup of a single layer vertical
defect.](../figures/06_defect_vertical_setup.png){#fig:vert_single_setup}

The initial configuration was run for
1000 steps at the low temperature of 0.20
resulting in @fig:vert_single_low.
In this instance the crystal has done a much better job of
adapting to fill the defect.
The spacing of the vertical layers
at the center of the crystal is slightly larger
so the vertical layers don't quite line up,
however, this goes through a smooth transition
to the perfect crystal at the outer edges.

![Single layer vertical defect after running for 1000 steps at a temperature of
0.20.](../figures/06_defect_vertical_thermal.png){#fig:vert_single_low}

Another feature of the low temperature configuration
is that while it is unable to form the p2 structure,
the molecules rotate about $50^\circ$
to form the pg structure in the place of the defect.

When running the defect close to the melting point
at a temperature of 1.30 for 500 steps,
the defect is much the same as at the lower temperature,
albeit more disordered (@fig:vert_single_high).

![Single layer vertical defect after running for 500 steps at a temperature of
1.30.](../figures/06_defect_vert_thermal_1.30.png){#fig:vert_single_high}

## Dual Layer

The setup of the dual layer defect
removes two layers of 21 molecules
resulting in the initial configuration shown in @fig:vert_dual_setup.

![Setup of a dual layer vertical
defect.](../figures/06_defect_shockley_setup.png){#fig:vert_dual_setup}

When the initial configuration is let to relax
for 1000 steps at a temperature of 0.20,
the configuration behaves in much the same way as
the single layer defect (@fig:vert_dual_low).
The crystal in the middle where the molecules were removed
has expanded to fill the space,
slowly going out of alignment with the outer crystal structure.

![Dual layer vertical defect after running for 1000 steps at a temperature of
0.20.](../figures/06_defect_shockley_thermal.png){#fig:vert_dual_low}

The most noticeable difference to the single layer
is continuation of the crystal,
in removing the two layers we have allowed for
the continuation of the layering through the defects.
Additionally at the corners of the removed particles
there is a collection of disordered particles.

When at the higher temperature of 1.30,
the structure is much the same as
for the low temperature (@fig:vert_dual_high).
There is somewhat more disorder in the structure at
the corners where the molecules were removed.

![Dual layer vertical defect after running for 500 steps at a temperature of
1.30.](../figures/06_defect_shockley_thermal_1.30.png){#fig:vert_dual_high}
