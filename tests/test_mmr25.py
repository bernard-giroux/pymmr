#!/usr/bin/env python3
"""
Tests of 2.5D MMR forward modelling (GridMMR with a 2D grid, see docs/mmr25.md).

Run with:  python3 tests/test_mmr25.py   (or pytest)
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pymmr.finite_volume import calc_padding, has_mumps  # noqa: E402
from pymmr.mmr import GridMMR, VerticalDyke, normal_field  # noqa: E402

SIGMA = 0.01
C1C2 = np.array([-60.0, 0.0, 0.0, 60.0, 0.0, 0.0])
# MUMPS needs mpi4py to find libmpi (e.g. DYLD_LIBRARY_PATH on macOS)
SOLVER = 'mumps' if has_mumps else 'superlu'


def grid(h=5.0):
    x = np.arange(-150.0, 150.0 + h / 2, h)
    z = np.arange(-100.0, h / 2, h)
    pad = np.cumsum(calc_padding(h, 20))
    return np.r_[x[0] - pad[::-1], x, x[-1] + pad], np.r_[z[0] - pad[::-1], z]


def receivers(yr=(0.0, 10.0, 30.0, -50.0)):
    xr = np.arange(-40.0, 41.0, 10.0)
    return np.array([[a, b, 0.0] for b in yr for a in xr])


def model(c1c2=C1C2, xo=None, cs=1.0, sigma=SIGMA):
    g = GridMMR(grid())
    g.set_solver(SOLVER)
    g.set_survey_mmr(c1c2, receivers() if xo is None else xo, cs)
    return g, g.fwd_mod(sigma * np.ones((g.dc.fv.nc,)))


class TestHalfSpace(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.xo = receivers()
        cls.g, cls.d = model(xo=cls.xo)

    def test_normal_field(self):
        bx, by = normal_field(np.tile(C1C2, (self.xo.shape[0], 1)), self.xo)
        scale = np.abs(by).max()
        # discretization error (h = 5 m, receivers 20 m or more from the electrodes), see docs/mmr25.md
        self.assertLess(np.abs(self.d[:, 0] - bx).max() / scale, 0.02)
        self.assertLess(np.abs(self.d[:, 1] - by).max() / scale, 0.06)
        self.assertLess(np.abs(self.d[:, 2]).max() / scale, 1e-10)

    def test_on_line(self):
        on = self.xo[:, 1] == 0.0
        self.assertEqual(np.abs(self.d[on, 0]).max(), 0.0)
        self.assertEqual(np.abs(self.d[on, 2]).max(), 0.0)

    def test_parity(self):
        xo = receivers((20.0, -20.0))
        _, d = model(xo=xo)
        n = xo.shape[0] // 2
        np.testing.assert_allclose(d[n:, 0], -d[:n, 0], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(d[n:, 1], d[:n, 1], rtol=1e-10, atol=1e-10)

    def test_shift_along_y(self):
        dy = 17.0
        c = C1C2 + np.array([0.0, dy, 0.0, 0.0, dy, 0.0])
        xo = self.xo + np.array([0.0, dy, 0.0])
        _, d = model(c1c2=c, xo=xo)
        np.testing.assert_allclose(d, self.d, rtol=1e-10, atol=1e-10 * np.abs(self.d).max())

    def test_current(self):
        _, d = model(xo=self.xo, cs=2.5)
        np.testing.assert_allclose(d, 2.5 * self.d, rtol=1e-10, atol=1e-10)

    def test_dipole_across_strike(self):
        # electrodes at different y: each electrode has its own distance along y to the receivers
        c = np.array([0.0, -60.0, 0.0, 0.0, 60.0, 0.0])
        xo = receivers((0.0, 25.0))
        _, d = model(c1c2=c, xo=xo)
        bx, by = normal_field(np.tile(c, (xo.shape[0], 1)), xo)
        scale = np.abs(bx).max()
        self.assertLess(np.abs(d[:, 0] - bx).max() / scale, 0.06)
        self.assertLess(np.abs(d[:, 1] - by).max() / scale, 0.06)


def dyke_grid(h=5.0, htop=0.25, npad=15):
    """Grid for the dyke tests: top cells refined to htop (electrodes at depth htop / 2)."""
    x = np.arange(-200.0, 200.0 + h / 2, h)
    pad = np.cumsum(calc_padding(h, npad))
    x = np.r_[x[0] - pad[::-1], x, x[-1] + pad]
    z = np.arange(-150.0, h / 2, h)
    zt = -np.cumsum(np.r_[np.full(4, htop), htop * 1.5**np.arange(1, 20)])
    zt = zt[zt > -3 * h]
    z = np.unique(np.r_[z[z < zt[-1] - h / 2], zt[::-1], 0.0])
    return x, np.r_[z[0] - pad[::-1], z]


class TestVerticalDyke(unittest.TestCase):
    """Bz off the line of the source vs the analytic solution (Edwards et al., 1978)."""

    RHO = (100.0, 10.0, 300.0)
    D = 20.0

    @classmethod
    def setUpClass(cls):
        x, z = dyke_grid()
        cls.g = GridMMR((x, z), units='nT')
        cls.g.set_solver(SOLVER)
        xc = 0.5 * (x[1:] + x[:-1])
        rho = np.where(xc < 0.0, cls.RHO[0], np.where(xc <= cls.D, cls.RHO[1], cls.RHO[2]))
        cls.sigma = np.tile(1.0 / rho, z.size - 1)
        cls.yr = (10.0, 40.0)
        cls.xo = np.array([[a, b, 0.0] for b in cls.yr for a in np.arange(-80.0, 81.0, 5.0)])
        cls.vd = VerticalDyke(*cls.RHO, cls.D, 0.0)

    def check(self, xs, tol):
        self.g.set_survey_mmr(np.array([xs, 0.0, 0.0, np.inf, np.inf, np.inf]), self.xo, 1.0)
        d = self.g.fwd_mod(self.sigma)
        bz = self.vd.fwd_mod(self.xo, xs)
        # air at least as thick as the earth part of the grid, whatever the size of the top cells
        self.assertGreaterEqual(self.g.fv.z[-1], self.g.dc.fv.z[-1] - self.g.dc.fv.z[0])
        for b in self.yr:
            s = self.xo[:, 1] == b
            err = np.abs(d[s, 2] - bz[s]).max() / np.abs(bz[s]).max()
            self.assertLess(err, tol, msg='xs = {0:g}, y = {1:g}'.format(xs, b))

    # errors with h = 5 m and htop = 0.25 m are 0.1 to 1.2 %, see docs/mmr25.md
    def test_source_left(self):
        self.check(-50.0, 0.01)

    def test_source_in_dyke(self):
        self.check(10.0, 0.02)

    def test_source_right(self):
        self.check(60.0, 0.01)



class TestVerticalDykeAnalytic(unittest.TestCase):
    """Bz is continuous at the flanks of the dyke and vanishes far from it."""

    def test_continuity(self):
        y = np.array([3.0, 10.0, 50.0])
        for rho in [(100.0, 10.0, 100.0), (100.0, 10.0, 300.0), (50.0, 500.0, 20.0), (30.0, 3.0, 7.0)]:
            vd = VerticalDyke(*rho, 20.0, 5.0, n_max=300)
            for xs in (-40.0, 5.0, 12.0, 25.0, 60.0):
                for xf in (5.0, 25.0):
                    a = vd.fwd_mod(np.c_[np.full(3, xf - 1e-9), y, 0 * y], xs)
                    b = vd.fwd_mod(np.c_[np.full(3, xf + 1e-9), y, 0 * y], xs)
                    np.testing.assert_allclose(a, b, rtol=1e-6, atol=0.0)
                far = vd.fwd_mod(np.array([[-1e6, 10.0, 0.0], [1e6, 10.0, 0.0]]), xs)
                np.testing.assert_allclose(far, 0.0, atol=1e-6 * np.abs(a).max())

    def test_reflection(self):
        # dyke mirrored about its centre: Bz changes sign
        vd = VerticalDyke(100.0, 10.0, 300.0, 20.0, 5.0)
        vm = VerticalDyke(300.0, 10.0, 100.0, 20.0, 5.0)
        xo = np.array([[a, b, 0.0] for b in (10.0, -30.0) for a in np.arange(-50.0, 61.0, 7.0)])
        xm = np.c_[30.0 - xo[:, 0], xo[:, 1:]]
        for xs in (-20.0, 12.0, 40.0):
            np.testing.assert_allclose(vd.fwd_mod(xo, xs), -vm.fwd_mod(xm, 30.0 - xs), rtol=1e-10, atol=1e-14)


class TestPlumbing(unittest.TestCase):

    def test_grid(self):
        g = GridMMR(grid())
        self.assertEqual(g.fv.dim, 2.5)
        self.assertEqual(g.dc.fv.dim, 2.5)
        with self.assertRaises(ValueError):
            GridMMR((np.arange(3.0),))

    def test_solver_reuse(self):
        g, d = model()
        solvers = g._solvers_k[2]
        d2 = g.fwd_mod(2 * SIGMA * np.ones((g.dc.fv.nc,)), keep_solver=True)
        self.assertIs(g._solvers_k[2], solvers)        # MMR systems do not depend on sigma
        np.testing.assert_allclose(d2, d, rtol=1e-8, atol=1e-12 * np.abs(d).max())   # B indep. of sigma
        g.fwd_mod(SIGMA * np.ones((g.dc.fv.nc,)), keep_solver=False)
        self.assertIsNot(g._solvers_k[2], solvers)

    def test_new_survey(self):
        g, _ = model()
        k = g.k.copy()
        xo = receivers((0.0, 1000.0))      # 4 x 1000 m > 100 r_min
        g.set_survey_mmr(C1C2, xo, 1.0)
        d = g.fwd_mod(SIGMA * np.ones((g.dc.fv.nc,)))
        self.assertEqual(d.shape, (xo.shape[0], 3))
        self.assertLess(g.k[0], k[0])                  # longer distances: smaller wavenumbers

    def test_pole_source(self):
        # electrode at infinity: data of the pole = dipole data minus those of its second electrode
        xo = receivers((0.0, 25.0))
        inf = np.array([np.inf] * 3)
        _, d12 = model(xo=xo)
        _, d1 = model(c1c2=np.r_[C1C2[:3], inf], xo=xo)
        _, d2 = model(c1c2=np.r_[C1C2[3:], inf], xo=xo)
        np.testing.assert_allclose(d1 - d2, d12, rtol=1e-8, atol=1e-8 * np.abs(d12).max())

    def test_not_implemented(self):
        g = GridMMR(grid())
        g.set_solver(SOLVER)
        g.set_survey_mmr(C1C2, receivers(), 1.0, pod_e=(1.0, 0.0, 0.0))
        with self.assertRaises(NotImplementedError):
            g.fwd_mod(SIGMA * np.ones((g.dc.fv.nc,)))
        g.set_survey_mmr(C1C2, receivers(), 1.0)
        with self.assertRaises(NotImplementedError):
            g.fwd_mod(SIGMA * np.ones((g.dc.fv.nc,)), calc_sens=True)


if __name__ == '__main__':
    unittest.main()
