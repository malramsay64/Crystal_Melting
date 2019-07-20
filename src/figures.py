#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Module containing function for theming and generating figures."""

from typing import Any, Dict

import altair as alt
from bokeh.plotting import Figure


def my_theme() -> Dict[str, Any]:
    """Define an Altair theme to use in all visualisations.

    This defines a simple theme, specifying the aspect ratio of 4:6
    and removing the grid from the figure which is distracting.

    """
    return {
        "config": {
            "view": {"height": 400, "width": 600},
            "legend": {"titleFontSize": 20, "labelFontSize": 16},
            "axis": {"grid": False, "labelFontSize": 16, "titleFontSize": 20},
            "header": {"titleFontSize": 22, "labelFontSize": 18},
            "background": "white",
        }
    }


def use_my_theme():
    """Register and my custom Altair theme."""
    # register and enable the theme
    alt.themes.register("my_theme", my_theme)
    alt.themes.enable("my_theme")


def style_snapshot(figure: Figure) -> Figure:
    """Style a bokeh figure as a configuration snapshot.

    This is collection of style changes to make the output of a snapshot consistent and
    nice. Primarily it removes all the extra stuff which isn't helpful in defining the
    configuration like the axes, and the interactive tools.

    """
    figure.axis.visible = False
    figure.xgrid.visible = False
    figure.ygrid.visible = False
    figure.toolbar_location = None
    figure.toolbar.logo = None

    return figure


def _add_line(chart: alt.Chart, value: float, line: alt.Chart) -> alt.Chart:
    data = chart.data
    if data is alt.Undefined:
        for layer in chart.layer:
            if layer.data is alt.Undefined:
                continue
            data = layer.data
    return alt.layer(chart, line, data=data).transform_calculate(val=f"{value}")


def hline(chart: alt.Chart, value: float) -> alt.Chart:
    """Draw a horizontal line on an Altair chart object."""
    line = alt.Chart().mark_rule(color="grey").encode(y="val:Q")
    return _add_line(chart, value, line)


def vline(chart: alt.Chart, value: float) -> alt.Chart:
    """Draw a vertical line on an Altair chart object."""
    line = alt.Chart().mark_rule(color="grey").encode(x="val:Q")
    return _add_line(chart, value, line)
