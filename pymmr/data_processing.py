#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for MMR data processing.

Magnetic field of the injection cables (Biot-Savart law), to be removed from the measured field: GridMMR
models the field of the currents in the ground only.

@author: giroux

"""
import warnings

import numpy as np

MU0_4PI = 1.0e-7          # mu0 / (4 pi), T m / A
_SCALE = {'T': 1.0, 'nT': 1.0e9, 'pT': 1.0e12}


def _scale(units):
    try:
        return _SCALE[units]
    except KeyError:
        raise ValueError("units should be 'T', 'nT' or 'pT'") from None


def segment_field(a, b, pts, current=1.0, chunk=None):
    """Magnetic field of straight segments carrying a current, exact Biot-Savart solution.

    Parameters
    ----------
    a, b : array_like
        Start and end points of the segments, m x 3 (the current flows from a to b).
    pts : array_like
        Points where the field is computed, n x 3.
    current : float or array_like
        Current (A), scalar or one value per segment.
    chunk : int, optional
        Number of points processed at once (memory: chunk x m x 3 values; default: about 2e6 / m).

    Returns
    -------
    ndarray
        B (T), n x 3.  NaN at points on a segment (the field is singular there).

    Notes
    -----
    With r1 = a - p and r2 = b - p (Hanson and Hirshman, 2002, Phys. Plasmas 9, 4410),

        B = mu0 I / (4 pi) (r1 x r2) (|r1| + |r2|) / (|r1| |r2| (|r1| |r2| + r1 . r2))

    The field is 0 on the line of a segment outside of it.
    """
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    current = np.broadcast_to(np.asarray(current, dtype=float), (a.shape[0],))
    length = np.linalg.norm(b - a, axis=1)
    if chunk is None:
        chunk = max(1, 2000000 // a.shape[0])
    B = np.zeros((pts.shape[0], 3))
    for i0 in range(0, pts.shape[0], chunk):
        p = pts[i0:i0 + chunk, None, :]
        r1 = a[None, :, :] - p
        r2 = b[None, :, :] - p
        n1 = np.linalg.norm(r1, axis=2)
        n2 = np.linalg.norm(r2, axis=2)
        dot = np.sum(r1 * r2, axis=2)
        cr = np.cross(r1, r2)
        # |r1| |r2| + r1 . r2 without cancellation beside the segment (r1 . r2 < 0), using
        # (|r1| |r2|)^2 - (r1 . r2)^2 = |r1 x r2|^2
        nn = n1 * n2
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.where(dot < 0.0, np.sum(cr * cr, axis=2) / (nn - dot), nn + dot)
        den = nn * s
        num = current[None, :] * (n1 + n2)
        # on the line, inside the segment: den = 0 and r1 x r2 = 0; outside: r1 x r2 = 0, den > 0
        on = (n1 + n2) <= length[None, :] * (1.0 + 1e-12)
        with np.errstate(divide='ignore', invalid='ignore'):
            f = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), 0.0)
        Bc = np.sum(f[:, :, None] * cr, axis=1)
        Bc[np.any(on, axis=1)] = np.nan
        B[i0:i0 + chunk] = Bc
    return MU0_4PI * B


def wire_field(wire_pts, meas_pts, current, units='pT'):
    """Magnetic field of a current flowing in a cable.

    Parameters
    ----------
    wire_pts : array_like
        Surveyed points of the cable, nc x 3 (m).  The cable is straight between consecutive points, and
        the current flows from the first point to the last.
    meas_pts : array_like
        Measurement points, nm x 3 (m).
    current : float
        Current (A).
    units : str
        Units of the field: 'pT' (default, as GridMMR), 'nT' or 'T'.

    Returns
    -------
    ndarray
        Bx, By, Bz, nm x 3.  NaN at points on the cable.
    """
    w = np.atleast_2d(np.asarray(wire_pts, dtype=float))
    if w.shape[0] < 2 or w.shape[1] != 3:
        raise ValueError('wire_pts must be nc x 3 with nc >= 2')
    return _scale(units) * segment_field(w[:-1], w[1:], meas_pts, current)


def circuit_field(cable_c1, cable_c2, meas_pts, current, units='pT'):
    """Magnetic field of the cables of an injection dipole.

    Parameters
    ----------
    cable_c1, cable_c2 : array_like
        Surveyed points of the cables from the transmitter to electrodes C1 and C2, n x 3 (m).  The last
        point of each cable is the electrode.  cable_c2 can be None (no cable, e.g. not surveyed).
    meas_pts : array_like
        Measurement points, nm x 3 (m).
    current : float
        Current (A), positive when the current enters the ground at C1 (as cs in GridMMR and GridDC): it flows
        from the transmitter to C1 in cable_c1, and from C2 to the transmitter in cable_c2.
    units : str
        'pT' (default), 'nT' or 'T'.

    Returns
    -------
    ndarray
        Bx, By, Bz, nm x 3.

    Notes
    -----
    The currents in the ground are not included: they are modelled by GridMMR.
    """
    B = wire_field(cable_c1, meas_pts, current, units)
    if cable_c2 is not None:
        B -= wire_field(cable_c2, meas_pts, current, units)
    return B


def remove_wire_field(c1c2, xo, data, cs, cables, units='pT', tol=1e-3):
    """Subtract the field of the injection cables from MMR data.

    Parameters
    ----------
    c1c2 : array_like
        Electrodes of the injection dipole of each datum, n x 6 (as read by inversion.df_to_data).
    xo : array_like
        Measurement points, n x 3.
    data : array_like
        Measured Bx, By, Bz, n x 3, in units.
    cs : float or array_like
        Current of each datum (A), positive when the current enters the ground at C1.
    cables : sequence of array_like
        Surveyed cables, each from the transmitter to an electrode (last point).  The cables of each datum
        are found from the positions of its electrodes, within tol (m).
    units : str
        Units of data: 'pT' (default), 'nT' or 'T'.
    tol : float
        Tolerance on the positions of the electrodes (m).

    Returns
    -------
    ndarray
        data minus the field of the cables, n x 3.
    """
    c1c2 = np.atleast_2d(np.asarray(c1c2, dtype=float))
    xo = np.atleast_2d(np.asarray(xo, dtype=float))
    out = np.array(data, dtype=float, copy=True)
    cs = np.broadcast_to(np.asarray(cs, dtype=float), (c1c2.shape[0],))
    cables = [np.atleast_2d(np.asarray(c, dtype=float)) for c in cables]
    ends = np.array([c[-1] for c in cables])

    def find(e):
        if not np.all(np.isfinite(e)):
            return None
        d = np.linalg.norm(ends - e, axis=1)
        i = np.where(d <= tol)[0]
        if i.size == 0:
            return -1
        if i.size > 1:
            raise ValueError(f'More than one cable ends at electrode {e}')
        return i[0]

    groups = {}
    for n in range(c1c2.shape[0]):
        groups.setdefault((tuple(c1c2[n]), cs[n]), []).append(n)
    for (dip, I), rows in groups.items():
        dip = np.array(dip)
        i1, i2 = find(dip[:3]), find(dip[3:])
        if i1 is None or i1 < 0:
            raise ValueError(f'No cable ends at electrode C1 {dip[:3]}')
        if i2 is not None and i2 < 0:
            raise ValueError(f'No cable ends at electrode C2 {dip[3:]}')
        if i2 is None:
            warnings.warn(f'C2 of dipole {dip} at infinity: field of its cable not removed', stacklevel=2)
        out[rows] -= circuit_field(cables[i1], None if i2 is None else cables[i2], xo[rows], I, units)
    return out
