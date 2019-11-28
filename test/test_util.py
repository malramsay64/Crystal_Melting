#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

import numpy as np

from crystal_analysis import util


def test_normalised_temperature():
    temp = np.array(0.36)
    pressure = np.array(1.00)
    assert np.allclose(util.normalised_temperature(temp, pressure), 1.0)
