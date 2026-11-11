#!/usr/bin/env python3
"""
Tests of 2.5D DC resistivity forward modelling (Grid25FV), against analytic solutions.

Run with:  python3 tests/test_dc25.py   (or pytest)
"""
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import vtk
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

    def test_pole_pole(self):
        """Remote electrodes at infinity (exact primary field must not give inf - inf)."""
        c = np.array([[-2., 0, 0, np.inf, np.inf, np.inf], [0., 0, 0, np.inf, np.inf, np.inf]])
        p = np.array([[3., 0, 0, np.inf, np.inf, np.inf], [8., 0, 0, np.inf, np.inf, np.inf]])
        g = make_grid(self.xg, self.z, c, p)
        v = 1e-3 * g.fwd_mod(dyke_model(g.fv, 1 / RHO1, 1 / RHO1, 1 / RHO1))
        np.testing.assert_allclose(v, RHO1 / (2 * np.pi * np.array([5., 8.])), rtol=1e-3)

    def test_elevation(self):
        """Same results with the grid and the electrodes shifted vertically (images about the grid top)."""
        c1c2 = np.r_[self.c1c2[:5], [[-6., 0, -3., 4., 0, 0]]]   # one buried electrode
        p1p2 = np.r_[self.p1p2[:5], [[0., 0, 0, 2., 0, 0]]]
        d = []
        for dz in (0.0, 100.0):
            g = make_grid(self.xg, self.z + dz, c1c2 + [0, 0, dz, 0, 0, dz], p1p2 + [0, 0, dz, 0, 0, dz])
            d.append(g.fwd_mod(dyke_model(g.fv, 1 / RHO1, 1 / RHO2, 1 / RHO3)))
        np.testing.assert_allclose(d[1], d[0], rtol=1e-8)

    def test_electrode_at_grid_edge(self):
        c = np.array([[self.xg[-1], 0, 0, -1, 0, 0]])
        p = np.array([[1., 0, 0, 2, 0, 0]])
        for bc in (True, False):
            g = make_grid(self.xg, self.z, c, p)
            g.apply_bc = bc
            v = g.fwd_mod(dyke_model(g.fv, 1 / RHO1, 1 / RHO2, 1 / RHO3))
            self.assertTrue(np.all(np.isfinite(v)))

    def test_current_density_only(self):
        """calc_J without measurement electrodes (wavenumbers from the grid)."""
        g = GridDC((self.xg, self.z))
        g.set_solver('superlu')
        g.c1c2 = self.c1c2[:2]
        g.cs = 1.0
        d, J = g.fwd_mod(dyke_model(g.fv, 1 / RHO1, 1 / RHO2, 1 / RHO3), calc_J=True)
        self.assertIsNone(d)
        self.assertEqual(J.shape, (g.fv.nf, 2))
        self.assertTrue(np.all(np.isfinite(J)))

    def test_new_survey_without_model(self):
        """Solvers are rebuilt when the wavenumbers change while the model is reused."""
        g = make_grid(self.xg, self.z, self.c1c2, self.p1p2)
        g.apply_bc = False
        s = dyke_model(g.fv, 1 / RHO1, 1 / RHO2, 1 / RHO3)
        g.fwd_mod(s)
        c = np.array([[-20., 0, 0, -10, 0, 0]])
        p = np.array([[10., 0, 0, 20, 0, 0]])
        g.set_survey_ert(c, p, cs=1.0)
        g.electrodes_sorted = False
        g.Q = None
        v1 = g.fwd_mod()
        g2 = make_grid(self.xg, self.z, c, p)
        g2.apply_bc = False
        v2 = g2.fwd_mod(s)
        np.testing.assert_allclose(v1, v2, rtol=1e-10)

    def test_sensitivity(self):
        """d(data)/d(ln sigma) vs centred finite differences, in core, electrode and padding cells."""
        x = np.arange(-10., 10.01)
        z = np.arange(-6., 0.01)
        pad = np.cumsum(calc_padding(1.0, 10))
        x = np.r_[x[0] - pad[::-1], x, x[-1] + pad]
        z = np.r_[z[0] - pad[::-1], z]
        # electrodes at nodes and at cell centres (x = -1.5, -0.5), unsorted, repeated dipole
        c1c2 = np.array([[2., 0, 0, 3, 0, 0], [-6., 0, 0, -4, 0, 0], [-1.5, 0, 0, -0.5, 0, 0], [-6., 0, 0, -4, 0, 0]])
        p1p2 = np.array([[5., 0, 0, 6, 0, 0], [0., 0, 0, 2, 0, 0], [1., 0, 0, 3.5, 0, 0], [4., 0, 0, 6, 0, 0]])
        rng = np.random.default_rng(7)

        def grid(opts):
            g = GridDC((x, z))
            g.set_solver('superlu')
            g.set_survey_ert(c1c2, p1p2, cs=2.5)
            for key, val in opts.items():
                setattr(g, key, val)
            return g

        sig = 10**rng.uniform(-2.5, -1.5, grid({}).fv.nc)
        xc, _, zc = grid({}).fv.centre_voxels()
        cells = [int(np.argmin((xc - a)**2 + (zc - b)**2))
                 for a, b in ((0.5, -0.5), (3.5, -2.5), (-1.5, -0.5), (2.5, -0.5), (15., -3.), (0.5, -15.))]
        for opts in ({}, {'local_sigma0': False}, {'apply_bc': False}):
            with self.subTest(**opts):
                d0 = grid(opts).fwd_mod(sig)
                d1, S = grid(opts).fwd_mod(sig, calc_sens=True)
                np.testing.assert_allclose(d1, d0, rtol=1e-10)
                for c in cells:
                    h = 1e-5 * sig[c]
                    s2, s3 = sig.copy(), sig.copy()
                    s2[c] += h
                    s3[c] -= h
                    fd = (grid(opts).fwd_mod(s2) - grid(opts).fwd_mod(s3)) / (2 * h) * sig[c]
                    self.assertLess(np.max(np.abs(S[c] - fd)) / np.max(np.abs(fd)), 1e-4)

    def test_sensitivity_correct_quadrature(self):
        g = make_grid(self.xg, self.z, self.c1c2, self.p1p2)
        g.correct_quadrature = True
        with self.assertRaises(NotImplementedError):
            g.fwd_mod(dyke_model(g.fv, 1 / RHO1, 1 / RHO1, 1 / RHO1), calc_sens=True)


class TestPlumbing(unittest.TestCase):
    """Input/output and code shared with 3D (VTK files, ROI, regularization, run_job)."""

    @classmethod
    def setUpClass(cls):
        cls.c1c2, cls.p1p2 = dyke_survey()
        cls.xg, cls.z = dyke_grid(1.0)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = cls.tmp.name

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_vtk_roundtrip(self):
        g = make_grid(self.xg, self.z, self.c1c2, self.p1p2)
        s = dyke_model(g.fv, 1 / RHO1, 1 / RHO2, 1 / RHO3)
        fname = os.path.join(self.dir, 'model25')
        g.fv.toVTK({'Conductivity': s}, fname)
        reader = vtk.vtkXMLRectilinearGridReader()
        reader.SetFileName(fname + '.vtr')
        reader.Update()
        self.assertEqual(reader.GetOutput().GetDimensions(), (g.fv.nx + 1, 2, g.fv.nz + 1))
        np.testing.assert_array_equal(g.fv.fromVTK('Conductivity', fname + '.vtr'), s)
        with self.assertRaises(ValueError):
            g.fv.toVTK({'Jy': np.zeros(1)}, fname, component='y')

    def test_build_from_vtk(self):
        from pymmr.run_job import build_from_vtk
        from pymmr.mmr import GridMMR
        g = make_grid(self.xg, self.z, self.c1c2, self.p1p2)
        s = dyke_model(g.fv, 1 / RHO1, 1 / RHO2, 1 / RHO3)
        fname = os.path.join(self.dir, 'model25b')
        g.fv.toVTK({'Conductivity': s}, fname)
        g2, s2 = build_from_vtk(GridDC, fname + '.vtr', return_sigma=True)
        self.assertEqual(g2.fv.dim, 2.5)
        np.testing.assert_array_equal(g2.fv.x, g.fv.x)
        np.testing.assert_array_equal(g2.fv.z, g.fv.z)
        np.testing.assert_array_equal(s2, s)
        g3 = build_from_vtk(GridMMR, fname + '.vtr')     # 2.5D MMR (docs/mmr25.md)
        self.assertEqual(g3.fv.dim, 2.5)
        self.assertEqual(g3.dc.fv.nc, g.fv.nc)

    def test_roi(self):
        g = make_grid(self.xg, self.z, self.c1c2, self.p1p2)
        g.set_roi((-10., 20., -8., 0.))
        g2 = make_grid(self.xg, self.z, self.c1c2, self.p1p2)
        g2.set_roi((-10., 20., -100., 100., -8., 0.))   # y ignored
        np.testing.assert_array_equal(g.ind_roi, g2.ind_roi)
        self.assertEqual(g.ind_roi.size, 30 * 8)
        x, y, z = g.get_roi_nodes()
        np.testing.assert_array_equal(y, [-0.5, 0.5])
        self.assertEqual((x[0], x[-1], z[0], z[-1]), (-10., 20., -8., 0.))

    def test_regularization(self):
        from pymmr.inversion import Inversion
        g = make_grid(self.xg, self.z, self.c1c2, self.p1p2)
        Gx, Gy, Gz = g.fv.extract_Gxyz()
        self.assertEqual((Gx.shape, Gy.shape, Gz.shape),
                         ((g.fv.nfx, g.fv.nc), (0, g.fv.nc), (g.fv.nfz, g.fv.nc)))
        inv = Inversion()
        act = np.arange(g.fv.nc)
        w = np.ones(g.fv.nc)
        self.assertEqual(g.calc_WtW(w, inv, act).shape, (g.fv.nc, g.fv.nc))
        m = np.log(dyke_model(g.fv, 1 / RHO1, 1 / RHO2, 1 / RHO3))
        for smooth_type in ('smooth', 'blocky'):
            inv.smooth_type = smooth_type
            WtW = g.calc_reg(w, inv, m, np.zeros(g.fv.nc), None, None, None, act)[0]
            self.assertEqual(WtW.shape, (g.fv.nc, g.fv.nc))

    def test_distance_weighting(self):
        g = make_grid(self.xg, self.z, self.c1c2, self.p1p2)
        w = g.fv.distance_weighting(np.array([[0., 0., -0.5]]), 2.0)
        self.assertEqual(w.shape, (g.fv.nc,))
        self.assertTrue(np.all(np.isfinite(w)) and np.all(w > 0))
        xc, _, zc = g.fv.centre_voxels()
        self.assertLess(np.hypot(xc[np.argmax(w)], zc[np.argmax(w)] + 0.5), 1.0)

    def test_save_current_density(self):
        g = make_grid(self.xg, self.z, self.c1c2[:2], self.p1p2[:2])
        _, J = g.fwd_mod(dyke_model(g.fv, 1 / RHO1, 1 / RHO2, 1 / RHO3), calc_J=True)
        base = os.path.join(self.dir, 'cur')
        g.save_current_density(J, base)
        self.assertTrue(os.path.isfile(base + '_Jx_dc_dip1.vtr'))
        self.assertTrue(os.path.isfile(base + '_Jz_dc_dip2.vtr'))
        self.assertFalse(os.path.isfile(base + '_Jy_dc_dip1.vtr'))

    def test_run_job_fwd_dc(self):
        """2.5D forward modelling through run_job.py and a parameter file."""
        g = make_grid(self.xg, self.z, self.c1c2, self.p1p2)
        s = dyke_model(g.fv, 1 / RHO1, 1 / RHO2, 1 / RHO3)
        v_ref = g.fwd_mod(s)
        d = self.dir
        g.fv.toVTK({'Conductivity': s}, os.path.join(d, 'job_model'))
        np.savetxt(os.path.join(d, 'job_c1c2.dat'), self.c1c2)
        np.savetxt(os.path.join(d, 'job_p1p2.dat'), self.p1p2)
        with open(os.path.join(d, 'job.par'), 'w') as f:
            f.write('fwd dc    # job,\n'
                    'job       # basename,\n'
                    'job_model.vtr    # model,\n'
                    'job_c1c2.dat     # source file,\n'
                    'job_p1p2.dat     # measurement file,\n'
                    '1.0       # source current,\n'
                    'superlu   # solver name,\n'
                    '0         # verbose,\n')
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get('PYTHONPATH', ''))
        res = subprocess.run([sys.executable, '-m', 'pymmr.run_job', 'job.par'], cwd=d, env=env,
                             capture_output=True, text=True, timeout=600)
        self.assertEqual(res.returncode, 0, res.stderr[-2000:])
        out = np.loadtxt(os.path.join(d, 'job_dc.dat'))
        np.testing.assert_allclose(out[:, 12], v_ref, rtol=1e-5)


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
