#!/usr/bin/env python3
"""
Tests of 3D DC resistivity forward modelling and sensitivity with the boundary correction (GridDC with a
rectilinear grid), against analytic solutions and finite differences.

Run with:  python3 tests/test_dc3d.py   (or pytest)
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pymmr.finite_volume as fvm  # noqa: E402
from pymmr.finite_volume import calc_padding  # noqa: E402
from pymmr.dc import GridDC  # noqa: E402
from pymmr.mmr import GridMMR, normal_field  # noqa: E402

SOLVER = 'mumps' if fvm.has_mumps else 'superlu'


def padded(core, h, n_pad, top=True):
    pad = np.cumsum(calc_padding(h, n_pad))
    if top:
        return np.r_[core[0] - pad[::-1], core, core[-1] + pad]
    return np.r_[core[0] - pad[::-1], core]


class TestLayeredEarth(unittest.TestCase):
    """Two-layer earth, dipole-dipole a = 5 m, offsets up to 45 m (image series solution)."""

    @classmethod
    def setUpClass(cls):
        cls.rho1, cls.rho2, cls.hl = 100., 10., 10.
        kk = (cls.rho2 - cls.rho1) / (cls.rho2 + cls.rho1)
        nn = np.arange(1, 600)

        def V(r):
            r = np.abs(np.asarray(r, float))[:, None]
            return cls.rho1 / (2 * np.pi) * (1 / r[:, 0] + 2 * np.sum(kk**nn / np.sqrt(r**2 + (2 * nn * cls.hl)**2),
                                                                     axis=1))
        a = 5.0
        rows = np.array([[xc, xc + a, xc + (n + 1) * a, xc + (n + 2) * a]
                         for xc in np.arange(-50., 50.1, 10.) for n in range(1, 9) if xc + (n + 2) * a <= 50])
        zr = np.zeros(len(rows))
        cls.c1c2 = np.c_[rows[:, 0], zr, zr, rows[:, 1], zr, zr]
        cls.p1p2 = np.c_[rows[:, 2], zr, zr, rows[:, 3], zr, zr]
        C1, C2, P1, P2 = rows.T
        cls.d_layered = 1e3 * (V(P1 - C1) - V(P2 - C1) - V(P1 - C2) + V(P2 - C2))
        cls.d_homog = 1e3 * cls.rho1 / (2 * np.pi) * (1 / np.abs(P1 - C1) - 1 / np.abs(P2 - C1) -
                                                      1 / np.abs(P1 - C2) + 1 / np.abs(P2 - C2))
        h = 2.5
        cls.x = padded(np.arange(-60., 60. + h / 2, h), h, 12)
        cls.y = padded(np.arange(-20., 20. + h / 2, h), h, 12)
        cls.z = padded(np.arange(-30., h / 2, h), h, 12, top=False)

    def model(self, g, layered=True):
        _, _, zc = g.fv.centre_voxels()
        if layered:
            return np.where(zc > -self.hl, 1 / self.rho1, 1 / self.rho2)
        return np.ones(g.fv.nc) / self.rho1

    def grid(self, **opts):
        g = GridDC((self.x, self.y, self.z))
        g.set_solver(SOLVER)
        g.set_survey_ert(self.c1c2, self.p1p2, cs=1.0)
        for key, val in opts.items():
            setattr(g.fv if key == 'surface_images' else g, key, val)
        return g

    def test_homogeneous_exact(self):
        g = self.grid()
        np.testing.assert_allclose(g.fwd_mod(self.model(g, False)), self.d_homog, rtol=1e-10)

    def test_layered(self):
        g = self.grid()
        np.testing.assert_allclose(g.fwd_mod(self.model(g)), self.d_layered, rtol=1e-2)

    def test_previous_correction_less_accurate(self):
        """Without surface images and exact primary field (previous correction): error of order h."""
        g = self.grid(surface_images=False, exact_primary=False)
        err = np.abs(g.fwd_mod(self.model(g)) / self.d_layered - 1)
        self.assertGreater(err.max(), 0.1)


class TestSensitivity(unittest.TestCase):

    def test_finite_differences(self):
        """d(data)/d(ln sigma) vs centred finite differences (core, electrode and padding cells)."""
        h = 1.0
        x = padded(np.arange(-6., 6.01, h), h, 8)
        y = padded(np.arange(-3., 3.01, h), h, 8)
        z = padded(np.arange(-5., 0.01, h), h, 8, top=False)
        c1c2 = np.array([[-3., 0, 0, -2, 0, 0], [-1., 0, 0, 0, 0, 0], [-3., 0, 0, -2, 0, 0]])
        p1p2 = np.array([[1., 0, 0, 2, 0, 0], [2., 0, 0, 3, 0, 0], [3., 0, 0, 5, 0, 0]])

        def grid(opts):
            g = GridDC((x, y, z))
            g.set_solver('superlu')
            g.set_survey_ert(c1c2, p1p2, cs=2.5)
            for key, val in opts.items():
                setattr(g, key, val)
            return g

        rng = np.random.default_rng(1)
        sig = 10**rng.uniform(-2.5, -1.5, grid({}).nc)
        xc, yc, zc = grid({}).fv.centre_voxels()
        cells = [int(np.argmin((xc - a)**2 + (yc - b)**2 + (zc - c)**2))
                 for a, b, c in ((-0.5, 0.5, -0.5), (-2.5, -0.5, -0.5), (-3.5, 0.5, -0.5), (8.0, -0.5, -0.5))]
        for opts in ({}, {'local_sigma0': False}, {'apply_bc': False}):
            with self.subTest(**opts):
                d0 = grid(opts).fwd_mod(sig)
                d1, S = grid(opts).fwd_mod(sig, calc_sens=True)
                np.testing.assert_allclose(d1, d0, rtol=1e-10)
                for c in cells:
                    dh = 1e-5 * sig[c]
                    s2, s3 = sig.copy(), sig.copy()
                    s2[c] += dh
                    s3[c] -= dh
                    fd = (grid(opts).fwd_mod(s2) - grid(opts).fwd_mod(s3)) / (2 * dh) * sig[c]
                    self.assertLess(np.max(np.abs(S[c] - fd)) / np.max(np.abs(fd)), 1e-4)


class TestMMR(unittest.TestCase):

    def test_mmr_without_images(self):
        """MMR uses the reference fields without surface images, whatever the setting of the DC grid."""
        h = 20.0
        x = padded(np.arange(-200., 200.1, h), h, 8)
        y = padded(np.arange(-100., 100.1, h), h, 8)
        z = padded(np.arange(-160., 0.1, h), h, 8, top=False)
        xs = np.array([[-150., 0, 0, 150., 0, 0]])
        xo = np.array([[0., 50., 0], [60., -40., 0], [-80., 70., 0]])
        B = []
        for images in (True, False):
            g = GridMMR((x, y, z))
            g.set_solver(SOLVER)
            g.set_survey_mmr(xs=xs, xo=xo, cs=1.0)
            g.dc.fv.surface_images = images
            B.append(g.fwd_mod(1e-2 * np.ones(g.dc.nc)))
            self.assertEqual(g.dc.fv.surface_images, images)   # setting restored
        np.testing.assert_allclose(B[0], B[1], rtol=1e-10, atol=1e-10 * np.abs(B[1]).max())
        Bn = normal_field(xs, xo)
        Bn = np.array(Bn)
        np.testing.assert_allclose(B[0][:, :2].T, Bn, rtol=0.15, atol=1e-3 * np.abs(Bn).max())


if __name__ == '__main__':
    unittest.main()
