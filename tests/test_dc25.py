#!/usr/bin/env python3
"""
Tests of 2.5D DC resistivity forward modelling (Grid25FV), against analytic solutions.

Run with:  python3 tests/test_dc25.py   (or pytest)
"""
import os
import sys
import unittest

import numpy as np
import scipy.sparse as sp
from scipy.special import k0
from scipy.sparse.linalg import spsolve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pymmr.finite_volume as fvm  # noqa: E402
from pymmr.finite_volume import calc_padding, Grid25FV, GridFV, Solver  # noqa: E402
from pymmr.dc import GridDC, VerticalDyke  # noqa: E402

RHO1, RHO2, RHO3 = 100.0, 1000.0, 50.0


def dyke_survey():
    """Dipole-dipole (a = 1 m, n = 1) across a 10 m wide vertical dyke at x = 0-10 m."""
    x = np.arange(-20.0, 27.0, 2.0).reshape(-1, 1)
    z = np.zeros(x.shape)
    c1c2 = np.c_[x, z, z, 1.0 + x, z, z]
    p1p2 = np.c_[2.0 + x, z, z, 3.0 + x, z, z]
    return c1c2, p1p2


def dyke_grid(h=1.0):
    x = np.arange(-30.0, 40.0 + h / 2, h)
    z = np.arange(-30.0, h / 2, h)
    pad = np.cumsum(calc_padding(h, 25))
    return np.r_[x[0] - pad[::-1], x, x[-1] + pad], np.r_[z[0] - pad[::-1], z]


def dyke_model(fv, s1, s2, s3):
    s = s1 + np.zeros((fv.nx, fv.nz))
    s[fv.xc > 0, :] = s2
    s[fv.xc > 10, :] = s3
    return s.flatten(order='F')


def make_grid(xg, z, c1c2, p1p2, solver='superlu'):
    g = GridDC((xg, z))
    g.set_solver(solver)
    g.set_survey_ert(c1c2, p1p2, cs=1.0)
    return g


class TestQuadrature(unittest.TestCase):

    def test_default_weights(self):
        fv = Grid25FV((np.arange(0., 10.1), np.arange(-5., 0.1)))
        r = np.logspace(np.log10(0.5), np.log10(20.0), 50)
        err = r * (fv.g[None, :] * k0(r[:, None] * fv.k)).sum(axis=1) - 1
        self.assertLess(np.abs(err).max(), 1e-2)

    def test_optimize_k_g(self):
        fv = Grid25FV((np.arange(0., 10.1), np.arange(-5., 0.1)))
        self.assertTrue(fv.auto_kg)
        fv.optimize_k_g(11, 1.0, 1000.0)
        self.assertFalse(fv.auto_kg)   # manual call disables automatic optimization
        r = np.logspace(0, 3, 200)
        err = r * (fv.g[None, :] * k0(r[:, None] * fv.k)).sum(axis=1) - 1
        self.assertLess(np.abs(err).max(), 1e-3)


class TestForward(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.c1c2, cls.p1p2 = dyke_survey()
        cls.xg, cls.z = dyke_grid(1.0)
        cls.v_hom = VerticalDyke(RHO1, RHO1, RHO1, 10.0).fwd_mod(cls.c1c2, cls.p1p2)
        cls.v_dyke = VerticalDyke(RHO1, RHO2, RHO3, 10.0).fwd_mod(cls.c1c2, cls.p1p2)
        # stations with all electrodes at least 2 m from the contacts at x = 0 and x = 10 m
        e = np.c_[cls.c1c2[:, [0, 3]], cls.p1p2[:, [0, 3]]]
        cls.far = np.all((np.abs(e) >= 2) & (np.abs(e - 10) >= 2), axis=1)

    def test_homogeneous(self):
        g = make_grid(self.xg, self.z, self.c1c2, self.p1p2)
        v = 1e-3 * g.fwd_mod(dyke_model(g.fv, 1 / RHO1, 1 / RHO1, 1 / RHO1))
        np.testing.assert_allclose(v, self.v_hom, rtol=1e-3)

    def test_homogeneous_correct_quadrature(self):
        g = make_grid(self.xg, self.z, self.c1c2, self.p1p2)
        g.correct_quadrature = True
        v = 1e-3 * g.fwd_mod(dyke_model(g.fv, 1 / RHO1, 1 / RHO1, 1 / RHO1))
        np.testing.assert_allclose(v, self.v_hom, rtol=1e-8)   # exact by construction

    def test_dyke(self):
        g = make_grid(self.xg, self.z, self.c1c2, self.p1p2)
        v = 1e-3 * g.fwd_mod(dyke_model(g.fv, 1 / RHO1, 1 / RHO2, 1 / RHO3))
        self.assertGreater(self.far.sum(), 15)
        np.testing.assert_allclose(v[self.far], self.v_dyke[self.far], rtol=5e-3)

    def test_solver_reuse(self):
        """Refactorization with a new model gives the same result as a new solver."""
        g = make_grid(self.xg, self.z, self.c1c2, self.p1p2)
        g.fwd_mod(dyke_model(g.fv, 1 / RHO1, 1 / RHO1, 1 / RHO1))
        v1 = g.fwd_mod(dyke_model(g.fv, 1 / RHO1, 1 / RHO2, 1 / RHO3))
        g2 = make_grid(self.xg, self.z, self.c1c2, self.p1p2)
        v2 = g2.fwd_mod(dyke_model(g2.fv, 1 / RHO1, 1 / RHO2, 1 / RHO3))
        np.testing.assert_allclose(v1, v2, rtol=1e-10)

    @unittest.skipUnless(fvm.has_mumps, 'MUMPS not available')
    def test_mumps(self):
        g = make_grid(self.xg, self.z, self.c1c2, self.p1p2, 'mumps')
        v1 = g.fwd_mod(dyke_model(g.fv, 1 / RHO1, 1 / RHO2, 1 / RHO3))
        g2 = make_grid(self.xg, self.z, self.c1c2, self.p1p2, 'superlu')
        v2 = g2.fwd_mod(dyke_model(g2.fv, 1 / RHO1, 1 / RHO2, 1 / RHO3))
        np.testing.assert_allclose(v1, v2, rtol=1e-10)

    def test_layered_long_offsets(self):
        """Two-layer earth, dipole-dipole a = 10 m, offsets up to 200 m (image series solution)."""
        rho1, rho2, hl = 100., 10., 30.
        kk = (rho2 - rho1) / (rho2 + rho1)
        nn = np.arange(1, 400)

        def V(r):
            r = np.abs(np.asarray(r, float))[:, None]
            return rho1 / (2 * np.pi) * (1 / r[:, 0] + 2 * np.sum(kk**nn / np.sqrt(r**2 + (2 * nn * hl)**2), axis=1))

        a = 10.0
        rows = [[xc, xc + a, xc + (n + 1) * a, xc + (n + 2) * a]
                for xc in np.arange(-150., 150.1, 20.) for n in range(1, 21) if xc + (n + 2) * a <= 160]
        rows = np.array(rows)
        zr = np.zeros(len(rows))
        c1c2 = np.c_[rows[:, 0], zr, zr, rows[:, 1], zr, zr]
        p1p2 = np.c_[rows[:, 2], zr, zr, rows[:, 3], zr, zr]
        C1, C2, P1, P2 = rows.T
        d_exact = V(P1 - C1) - V(P2 - C1) - V(P1 - C2) + V(P2 - C2)

        h = 2.5
        xg = np.arange(-200., 200. + h / 2, h)
        z = np.arange(-120., h / 2, h)
        pad = np.cumsum(calc_padding(h, 22))
        xg = np.r_[xg[0] - pad[::-1], xg, xg[-1] + pad]
        z = np.r_[z[0] - pad[::-1], z]
        g = make_grid(xg, z, c1c2, p1p2)
        s = np.where(np.kron(g.fv.zc, np.ones(g.fv.nx)) > -hl, 1 / rho1, 1 / rho2)
        d = 1e-3 * g.fwd_mod(s)
        np.testing.assert_allclose(d, d_exact, rtol=5e-3)

    def test_sensitivity_not_implemented(self):
        g = make_grid(self.xg, self.z, self.c1c2, self.p1p2)
        with self.assertRaises(NotImplementedError):
            g.fwd_mod(dyke_model(g.fv, 1 / RHO1, 1 / RHO1, 1 / RHO1), calc_sens=True)


@unittest.skipUnless(fvm.has_mumps, 'MUMPS not available')
class TestMumpsRefactorization(unittest.TestCase):

    def test_values_in_different_order(self):
        """A + k^2 sigma has the structure of A, but scipy may store the entries in a different order."""
        x = np.r_[-12 - np.cumsum(calc_padding(1., 6))[::-1], np.arange(-12., 12.01), 12 + np.cumsum(calc_padding(1., 6))]
        y = np.r_[-6 - np.cumsum(calc_padding(1., 6))[::-1], np.arange(-6., 6.01), 6 + np.cumsum(calc_padding(1., 6))]
        z = np.r_[-8 - np.cumsum(calc_padding(1., 6))[::-1], np.arange(-8., 0.01)]
        fv = GridFV((x, y, z))
        fv.set_solver('mumps')
        rng = np.random.default_rng(0)
        s = 10**rng.uniform(-3, -1, fv.nc)
        A, _ = fv.build_A(s)
        b = rng.standard_normal(fv.nc)
        solver = Solver(fv.get_solver_params(), A)
        A2 = A + 0.5 * sp.diags_array(s)
        solver.A = A2
        x_ref = spsolve(A2.tocsc(), b)
        np.testing.assert_allclose(solver.solve(b).ravel(), x_ref, rtol=1e-8, atol=1e-12 * np.abs(x_ref).max())


if __name__ == '__main__':
    unittest.main()
