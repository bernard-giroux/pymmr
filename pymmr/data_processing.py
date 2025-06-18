#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for mmr data processing

@author: giroux

"""
import numpy as np
import wire
import biotsavart


def wire_field(wire_pts, meas_pts, current):
    """Function to compute B field due to current flowing in a wire.

    Parameters
    ----------
    wire_pts : array_like
        Coordinates of the path of the wire, nc x 3
    meas_pts : array_like
        Coordinates of measurement points, nm x 3
    current : float
        Current intensity (A)

    Returns
    -------
    Field components Bx, By, Bz in nT, nm x 3

    Notes
    -----
    Coordinate units are meters
    """
    w = wire.Wire(current=current, path=wire_pts, discretization_length=0.1)
    sol = biotsavart.BiotSavart(wire=w)
    return sol.CalculateB(points=meas_pts) * 1.e9  # nT


if __name__ == '__main__':

    obs_pts = np.array([[2, 0, 0], [0, 2, 0.05], [0, 0, 5], [10, 10, 10]])

    B = wire_field(np.array([[1, 0, 0], [0, 1, 0], [-0.5, 1.2, 0.1]]),
                   obs_pts, 2.5)
