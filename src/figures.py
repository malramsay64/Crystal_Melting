#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Module containing function for theming and generating figures."""

from typing import Any, Dict

import altair.vegalite.v2 as alt


def my_theme() -> Dict[str, Any]:
    """Define an Altair theme to use in all visualisations.

    This defines a simple theme, specifying the aspect ratio of 4:6
    and removing the grid from the figure which is distracting.

    """
    return {"config": {"view": {"height": 400, "width": 600}, "axis": {"grid": False}}}


def use_my_theme():
    """Register and my custom Altair theme."""
    # register and enable the theme
    alt.themes.register("my_theme", my_theme)
    alt.themes.enable("my_theme")


def _add_line(chart: alt.Chart, value: float, line: alt.Chart) -> alt.Chart:
    data = chart.data
    if data == alt.Undefined:
        try:
            for layer in chart.layer:
                if layer.data is not alt.Undefined:
                    data = layer.data
                    break
            else:
                raise RuntimeError("No data found in Chart object")
        except AttributeError:
            raise RuntimeError("No data found in Chart object")

    return alt.layer(chart, line, data=data).transform_calculate(val=f"{value}")


def hline(chart: alt.Chart, value: float) -> alt.Chart:
    """Draw a horizontal line on an Altair chart object."""
    line = alt.Chart().mark_rule(color="grey").encode(y="val:Q")
    return _add_line(chart, value, line)


def vline(chart: alt.Chart, value: float) -> alt.Chart:
    """Draw a vertical line on an Altair chart object."""
    line = alt.Chart().mark_rule(color="grey").encode(x="val:Q")
    return _add_line(chart, value, line)
