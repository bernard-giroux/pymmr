#!/usr/bin/env python3
"""
Tests of the magnetic field of the injection cables (pymmr.data_processing).

Run with:  python3 tests/test_wire_field.py   (or pytest)
"""
import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pymmr.data_processing import segment_field, wire_field, circuit_field, remove_wire_field  # noqa: E402
from pymmr.wire import Wire  # noqa: E402
from pymmr.biotsavart import BiotSavart  # noqa: E402
from pymmr.mmr import normal_field  # noqa: E402

MU0 = 4e-7 * np.pi


class TestSegmentField(unittest.TestCase):

    def test_finite_segment_bisector(self):
        """Point on the perpendicular bisector: B = mu0 I / (2 pi d) L / sqrt(L^2 + d^2), half-length L."""
        L, I = 7.0, 2.5
        d = np.array([0.01, 0.5, 3.0, 40.0])
        pts = np.c_[np.zeros(d.size), d, np.zeros(d.size)]
        B = segment_field([-L, 0, 0], [L, 0, 0], pts, I)
        ref = MU0 * I / (2 * np.pi * d) * L / np.sqrt(L**2 + d**2)
        # current along +x, point at +y: B along +z
        np.testing.assert_allclose(B[:, 2], ref, rtol=1e-12)
        np.testing.assert_allclose(B[:, :2], 0.0, atol=1e-25)

    def test_infinite_wire(self):
        I, d = 1.0, 2.0
        B = segment_field([0, -1e6, 0], [0, 1e6, 0], [[d, 0, 0]], I)
        # current along +y, point at +x: B along -z
        np.testing.assert_allclose(B[0], [0, 0, -MU0 * I / (2 * np.pi * d)], rtol=1e-10)

    def test_arbitrary_point(self):
        """General position: segment from z0 to z1 along z, B = mu0 I / (4 pi rho) (cos a2 - cos a1) phi."""
        I = 1.3
        p = np.array([1.2, -0.7, 0.4])
        z0, z1 = -2.0, 3.5
        B = segment_field([0, 0, z0], [0, 0, z1], p, I)[0]
        rho = np.hypot(p[0], p[1])
        s = (z1 - p[2]) / np.hypot(rho, z1 - p[2]) - (z0 - p[2]) / np.hypot(rho, z0 - p[2])
        phi = np.array([-p[1], p[0], 0]) / rho
        np.testing.assert_allclose(B, MU0 * I / (4 * np.pi * rho) * s * phi, rtol=1e-12)

    def test_square_loop_centre(self):
        """Centre of a square loop of side a: B = 2 sqrt(2) mu0 I / (pi a)."""
        a, I = 3.0, 2.0
        loop = np.array([[0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0], [0, 0, 0]], dtype=float)
        B = wire_field(loop, [[a / 2, a / 2, 0]], I, units='T')
        np.testing.assert_allclose(B[0], [0, 0, 2 * np.sqrt(2) * MU0 * I / (np.pi * a)], rtol=1e-12)

    def test_against_magwire(self):
        """Same as the discretized Biot-Savart code (pymmr.biotsavart) with short elements, away from the cable."""
        path = np.array([[-27.5, 32.5, 0], [27.5, 32.5, 0], [27.5, 150, 5]])
        rng = np.random.default_rng(1)
        pts = np.c_[rng.uniform(-50, 50, 20), rng.uniform(0, 120, 20), np.ones(20)]
        bs = BiotSavart(wire=Wire(current=2.5, path=path, discretization_length=0.01))
        B_ref = bs.CalculateB(points=pts) * 1e9
        B = wire_field(path, pts, 2.5, units='nT')
        dist = np.min(np.abs(pts[:, 1] - 32.5))
        self.assertGreater(dist, 0.5)
        np.testing.assert_allclose(B, B_ref, rtol=1e-5, atol=1e-6 * np.abs(B_ref).max())

    def test_on_the_cable(self):
        B = segment_field([0, 0, 0], [1, 0, 0], [[0.5, 0, 0], [2.0, 0, 0], [0.5, 1, 0]])
        self.assertTrue(np.all(np.isnan(B[0])))
        np.testing.assert_array_equal(B[1], 0.0)        # on the line, outside the segment
        self.assertTrue(np.all(np.isfinite(B[2])))

    def test_chunks(self):
        rng = np.random.default_rng(3)
        a, b = rng.normal(size=(7, 3)), rng.normal(size=(7, 3))
        pts = rng.normal(size=(50, 3)) * 5
        np.testing.assert_allclose(segment_field(a, b, pts, chunk=6), segment_field(a, b, pts), rtol=1e-14)

    def test_units(self):
        path, p = [[0, 0, 0], [1, 0, 0]], [[0.5, 1, 0]]
        BT = wire_field(path, p, 1.0, units='T')
        np.testing.assert_allclose(wire_field(path, p, 1.0), 1e12 * BT)
        np.testing.assert_allclose(wire_field(path, p, 1.0, units='nT'), 1e9 * BT)
        with self.assertRaises(ValueError):
            wire_field(path, p, 1.0, units='mT')


class TestCircuit(unittest.TestCase):

    def setUp(self):
        # transmitter to the north, cable 1 to the east then south to C1, cable 2 to the west then south to C2
        self.tx = [0.0, 200.0, 0.0]
        self.c1_cable = np.array([self.tx, [150, 200, 0], [150, 0, 0]])
        self.c2_cable = np.array([self.tx, [-150, 200, 0], [-150, 0, 0]])
        rng = np.random.default_rng(2)
        self.xo = np.c_[rng.uniform(-100, 100, 15), rng.uniform(20, 180, 15), np.ones(15)]

    def test_sign(self):
        """Current from the transmitter to C1 in cable 1, from C2 to the transmitter in cable 2."""
        B = circuit_field(self.c1_cable, self.c2_cable, self.xo, 2.5)
        ref = wire_field(self.c1_cable, self.xo, 2.5) + wire_field(self.c2_cable[::-1], self.xo, 2.5)
        np.testing.assert_allclose(B, ref, rtol=1e-12)
        # point inside the U, at the surface: the 2 cables give Bz of the same sign
        p = [[0, 100, 0]]
        self.assertGreater(wire_field(self.c1_cable, p, 1.0)[0, 2] * wire_field(self.c2_cable[::-1], p, 1.0)[0, 2], 0)

    def test_half_space(self):
        """Same convention as GridMMR: at the surface of a uniform half-space, the field of the ground currents
        (mmr.normal_field) is that of semi-infinite vertical wires below the electrodes, the current flowing
        down below C1 and up below C2, continuing the current of the cables."""
        xo = self.xo.copy()
        xo[:, 2] = 0.0
        c1, c2 = self.c1_cable[-1], self.c2_cable[-1]
        down = np.array([0, 0, -1e7])
        Bh = segment_field(np.array([c1, c2 + down]), np.array([c1 + down, c2]), xo, 2.5) * 1e12
        Bx, By = normal_field(np.tile(np.r_[c1, c2], (xo.shape[0], 1)), xo, 2.5)
        np.testing.assert_allclose(Bh[:, 0], Bx, rtol=1e-6)
        np.testing.assert_allclose(Bh[:, 1], By, rtol=1e-6)
        np.testing.assert_allclose(Bh[:, 2], 0.0, atol=1e-6 * np.abs(Bx).max())

    def test_remove(self):
        c1c2 = np.tile(np.r_[self.c1_cable[-1], self.c2_cable[-1]], (self.xo.shape[0], 1))
        B_ground = np.random.default_rng(4).normal(size=self.xo.shape)
        data = B_ground + circuit_field(self.c1_cable, self.c2_cable, self.xo, 2.5)
        # cables in any order, electrode positions within the tolerance
        c1c2[:, 0] += 0.0005
        out = remove_wire_field(c1c2, self.xo, data, 2.5, [self.c2_cable, self.c1_cable], tol=1e-3)
        np.testing.assert_allclose(out, B_ground, atol=1e-6 * np.abs(data).max())

    def test_remove_two_dipoles_and_errors(self):
        n = self.xo.shape[0]
        c3_cable = np.array([self.tx, [0, 0, 0]])
        c1c2 = np.r_[np.tile(np.r_[self.c1_cable[-1], self.c2_cable[-1]], (n, 1)),
                     np.tile(np.r_[c3_cable[-1], self.c2_cable[-1]], (n, 1))]
        xo = np.r_[self.xo, self.xo]
        cs = np.r_[2.5 * np.ones(n), 1.5 * np.ones(n)]
        data = np.r_[circuit_field(self.c1_cable, self.c2_cable, self.xo, 2.5),
                     circuit_field(c3_cable, self.c2_cable, self.xo, 1.5)]
        out = remove_wire_field(c1c2, xo, data, cs, [self.c1_cable, self.c2_cable, c3_cable])
        np.testing.assert_allclose(out, 0.0, atol=1e-9 * np.abs(data).max())
        with self.assertRaises(ValueError):
            remove_wire_field(c1c2, xo, data, cs, [self.c1_cable, self.c2_cable])
        # pole source: cable of C2 not removed, with a warning
        c1c2[:, 3:] = np.inf
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            remove_wire_field(c1c2, xo, data, cs, [self.c1_cable, c3_cable])
            self.assertTrue(any('infinity' in str(x.message) for x in w))


if __name__ == '__main__':
    unittest.main()
