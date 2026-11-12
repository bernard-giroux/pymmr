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
class TestOffLine(unittest.TestCase):
    """Off-line measurements: warning only with offline25 = False (data computed as if at y = 0)."""

    def survey(self, y_c, y_p, pole=False, offline25=False):
        g = GridDC((np.arange(-50.0, 50.1, 5.0), np.arange(-30.0, 0.1, 5.0)))
        g.offline25 = offline25
        c1c2 = np.array([[-20.0, y_c, 0.0, 20.0, y_c, 0.0], [-10.0, y_c, 0.0, 10.0, y_c, 0.0]])
        if pole:
            c1c2[:, 3:] = np.inf
        p1p2 = np.array([[0.0, y_p, 0.0, 5.0, y_p, 0.0], [-5.0, y_c, 0.0, 0.0, y_c, 0.0]])
        g.set_survey_ert(c1c2, p1p2, 1.0)
        return g

    def test_warning(self):
        with self.assertWarnsRegex(UserWarning, r'1 of 2 data .*largest offset 7 m'):
            self.survey(0.0, 7.0)
        with self.assertWarnsRegex(UserWarning, 'off-line measurements are not modelled'):
            self.survey(0.0, -3.0, pole=True)

    def test_no_warning(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.survey(0.0, 0.0)
            self.survey(12.0, 12.0)            # all electrodes on the line y = 12 m
            self.survey(12.0, 12.0, pole=True)  # remote electrode at infinity
            # default ('auto'): off-line data are modelled
            g = self.survey(0.0, 7.0, offline25='auto')
            self.assertTrue(g._uses_offline25())
            self.assertFalse(self.survey(12.0, 12.0, offline25='auto')._uses_offline25())

    def test_offline_as_online(self):
        """offline25 = False: off-line data are computed as if all electrodes were at y = 0."""
        import warnings
        x = np.arange(-50.0, 50.1, 2.5)
        z = np.arange(-30.0, 0.1, 2.5)
        c1c2 = np.array([[0.0, 0.0, 0.0, np.inf, np.inf, np.inf]] * 3)
        xp = np.array([10.0, 20.0, 30.0])
        d = []
        for yp in (0.0, 10.0):
            g = GridDC((x, z))
            g.offline25 = False
            p1p2 = np.c_[xp, np.full(3, yp), np.zeros(3), np.full((3, 3), np.inf)]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                g.set_survey_ert(c1c2, p1p2, 1.0)
            d.append(g.fwd_mod(0.01 * np.ones(g.fv.nc)))
        np.testing.assert_allclose(d[1], d[0], rtol=1e-12)


def layered_potential(rho1, rho2, hl):
    """Potential of a unit surface current electrode over a two-layer earth (image series), at surface distance r."""
    kk = (rho2 - rho1) / (rho2 + rho1)
    nn = np.arange(1, 800)

    def V(r):
        r = np.atleast_1d(np.abs(np.asarray(r, float)))[:, None]
        return rho1 / (2 * np.pi) * (1 / r[:, 0] + 2 * np.sum(kk**nn / np.sqrt(r**2 + (2 * nn * hl)**2), axis=1))
    return V


class TestOffLineModelling(unittest.TestCase):
    """Off-line measurements modelled with the wavenumbers of KQuadrature (offline25)."""

    @classmethod
    def setUpClass(cls):
        h = 2.5
        pad = np.cumsum(calc_padding(h, 20))
        core = np.arange(-100.0, 100.0 + h / 2, h)
        cls.x = np.r_[core[0] - pad[::-1], core, core[-1] + pad]
        z = np.arange(-50.0, h / 2, h)
        cls.z = np.r_[z[0] - pad[::-1], z]
        # dipole-dipole and pole-dipole, potential dipoles along x and y, on and off the line, in random order
        rows = []
        for yp in (0.0, 10.0, 30.0):
            for xp in (10.0, 25.0, 50.0):
                rows.append([-10.0, 0, 0, 0.0, 0, 0, xp, yp, 0, xp + 5.0, yp, 0])
                rows.append([-10.0, 0, 0, np.inf, np.inf, np.inf, xp, yp, 0, xp, yp + 5.0, 0])
        rows.append([-10.0, 0, 0, np.inf, np.inf, np.inf, 30.0, 20.0, 0, np.inf, np.inf, np.inf])   # pole-pole
        rows.append([-20.0, -5.0, 0, 0.0, 15.0, 0, 20.0, 5.0, 0, 30.0, -10.0, 0])   # current electrodes at different y
        rows = np.array(rows)
        cls.rows = rows[np.random.default_rng(1).permutation(rows.shape[0])]

    def grid(self):
        g = GridDC((self.x, self.z))
        g.set_solver('mumps' if fvm.has_mumps else 'superlu')
        g.set_survey_ert(self.rows[:, :6], self.rows[:, 6:], 2.0)
        return g

    def reference(self, V, scale=False):
        """Analytic data for a potential V(r) of surface electrodes (current 2 A, units of the grid: mV).

        With scale, also returns the sum of the absolute values of the terms of each datum.
        """
        d = np.zeros(self.rows.shape[0])
        a = np.zeros(self.rows.shape[0])
        for c0, sc in ((0, 1.0), (3, -1.0)):
            for p0, sp_ in ((6, 1.0), (9, -1.0)):
                c, p = self.rows[:, c0:c0 + 3], self.rows[:, p0:p0 + 3]
                ok = np.all(np.isfinite(c), axis=1) & np.all(np.isfinite(p), axis=1)
                v = V(np.linalg.norm(p[ok] - c[ok], axis=1))
                d[ok] += sc * sp_ * v
                a[ok] += np.abs(v)
        if scale:
            return 2.0 * 1e3 * d, 2.0 * 1e3 * a
        return 2.0 * 1e3 * d

    def test_half_space(self):
        g = self.grid()
        self.assertTrue(g._uses_offline25())
        d = g.fwd_mod(0.01 * np.ones(g.fv.nc))
        np.testing.assert_allclose(d, self.reference(lambda r: 1 / (2 * np.pi * 0.01 * r)), rtol=1e-8)

    def test_layered(self):
        g = self.grid()
        zc = g.fv.centre_voxels()[-1]
        rho1, rho2, hl = 100.0, 10.0, 10.0
        d = g.fwd_mod(np.where(zc > -hl, 1 / rho1, 1 / rho2))
        ref, a = self.reference(layered_potential(rho1, rho2, hl), scale=True)
        # 2% of the datum, or 0.2% of the sum of its terms when they nearly cancel (dipoles far off the line)
        err = np.abs(d - ref) / np.maximum(np.abs(ref), 0.1 * a)
        self.assertLess(err.max(), 0.02)

    def test_online_data(self):
        """On the line, the off-line method agrees with the fitted wavenumbers."""
        on = np.abs(self.rows[:, 7]) == 0.0
        on &= ~np.isfinite(self.rows[:, 10]) | (self.rows[:, 10] == 0.0)
        rng = np.random.default_rng(0)
        zc = self.grid().fv.centre_voxels()[-1]
        sigma = np.where(zc > -10.0, 0.01, 0.1) * 10**rng.uniform(-0.2, 0.2, zc.size)
        d = []
        for off in (True, False):
            g = GridDC((self.x, self.z))
            g.set_solver('mumps' if fvm.has_mumps else 'superlu')
            g.offline25 = off
            g.set_survey_ert(self.rows[on, :6], self.rows[on, 6:], 2.0)
            d.append(g.fwd_mod(sigma))
        np.testing.assert_allclose(d[0], d[1], rtol=5e-3)

    def test_plan_parameters(self):
        """The plan is rebuilt for a new n_k_offline, the current is applied at each call."""
        g = self.grid()
        sigma = 0.01 * np.ones(g.fv.nc)
        d1 = g.fwd_mod(sigma)
        np.testing.assert_allclose(g.fwd_mod(sigma), d1, rtol=1e-12)  # cached plan (solver round-off)
        g._off25 = None
        np.testing.assert_allclose(g.fwd_mod(sigma), d1, rtol=1e-12)  # fresh plan
        cs = np.linspace(0.5, 3.0, self.rows.shape[0])
        g.cs = cs
        np.testing.assert_allclose(g.fwd_mod(sigma), d1 * cs / 2.0, rtol=1e-12)
        g.n_k_offline = 20
        g.fwd_mod(sigma)
        self.assertEqual(g._off25['k'].size, 20)

    def test_not_implemented(self):
        g = self.grid()
        with self.assertRaises(NotImplementedError):
            g.fwd_mod(0.01 * np.ones(g.fv.nc), calc_J=True)


class TestOffLineSensitivity(unittest.TestCase):
    """Sensitivity of off-line data (offline25), against finite differences."""

    h = 2.0
    rows = np.array([
        [-10., 0, 0, 0., 0, 0, 6., 8., 0, 10., 8., 0],
        [-10., 0, 0, 0., 0, 0, 4., 0., 0, 8., 0., 0],
        [-6., 0, 0, np.inf, np.inf, np.inf, 2., 5., 0, 2., 9., 0],
        [-6., 0, 0, np.inf, np.inf, np.inf, 12., 15., 0, np.inf, np.inf, np.inf],
        [8., 3., 0, 14., 3., 0, -4., 10., 0, 0., 10., 0],
        [-12., -4., 0, -2., 6., 0, 4., 2., 0, 9., -3., 0],    # current electrodes at different y
    ])

    def grid(self, rows=None, **kw):
        h = self.h
        pad = np.cumsum(calc_padding(h, 10))
        cx = np.arange(-30, 30 + h / 2, h)
        cz = np.arange(-16, h / 2, h)
        g = GridDC((np.r_[cx[0] - pad[::-1], cx, cx[-1] + pad], np.r_[cz[0] - pad[::-1], cz]))
        g.set_solver('mumps' if fvm.has_mumps else 'superlu')
        for a, v in kw.items():
            setattr(g, a, v)
        rows = self.rows if rows is None else rows
        g.set_survey_ert(rows[:, :6], rows[:, 6:], 1.5)
        return g

    def sigma(self, g):
        return 0.01 * 10**np.random.default_rng(3).uniform(-0.5, 0.5, g.fv.nc)

    def check_fd(self, g, eps, tol):
        sigma = self.sigma(g)
        d, J = g.fwd_mod(sigma, calc_sens=True)
        np.testing.assert_allclose(d, g.fwd_mod(sigma), rtol=1e-12, atol=1e-12 * np.abs(d).max())
        xc, zc = g.fv.centre_voxels()[0], g.fv.centre_voxels()[-1]
        # cells at the current electrodes (conductivity of the reference fields), between and below them
        for xe, ze in ((-10, -1), (-6, -1), (8, -1), (4, -5), (-2, -9)):
            c = int(np.argmin((xc - xe)**2 + (zc - ze)**2))
            sp_, sm = sigma.copy(), sigma.copy()
            sp_[c] *= np.exp(eps)
            sm[c] *= np.exp(-eps)
            fd = (g.fwd_mod(sp_) - g.fwd_mod(sm)) / (2 * eps)
            self.assertLess(np.abs(J[c] - fd).max() / np.abs(fd).max(), tol, msg=f'cell at ({xe}, {ze})')

    def test_fd(self):
        self.check_fd(self.grid(), 1e-4, 1e-5)

    def test_fd_gmean_sigma0(self):
        self.check_fd(self.grid(local_sigma0=False), 1e-4, 1e-5)

    def test_fd_no_boundary_correction(self):
        # larger fields near the point sources: round-off of the finite differences ~1e-5
        self.check_fd(self.grid(apply_bc=False), 1e-3, 2e-4)

    def test_roi(self):
        g = self.grid()
        sigma = self.sigma(g)
        _, J = g.fwd_mod(sigma, calc_sens=True)
        g.set_roi((-10., 20., -8., 0.))
        _, J_roi = g.fwd_mod(sigma, calc_sens=True)
        np.testing.assert_allclose(J_roi, J[g.ind_roi, :], rtol=1e-10, atol=1e-12 * np.abs(J).max())

    def test_online_agreement(self):
        """On the line, same sensitivity as the fitted wavenumbers."""
        rows = self.rows[[1]]
        rows = np.r_[rows, [[-6., 0, 0, np.inf, np.inf, np.inf, 2., 0, 0, 6., 0, 0]]]
        J = []
        for off in (True, False):
            g = self.grid(rows, offline25=off)
            J.append(g.fwd_mod(self.sigma(g), calc_sens=True)[1])
        err = np.linalg.norm(J[0] - J[1], axis=0) / np.linalg.norm(J[1], axis=0)
        self.assertLess(err.max(), 1e-2)


class TestOffLineInversion(unittest.TestCase):
    """Off-line data in run_job, Inversion.run and joint MMR-ERT inversion (2.5D)."""

    rows = TestOffLineSensitivity.rows

    def grid(self, cls=GridDC):
        g = TestOffLineSensitivity().grid()
        g2 = cls((g.fv.x, g.fv.z))
        g2.set_solver('mumps' if fvm.has_mumps else 'superlu')
        return g2

    def test_run_job_fwd_dc(self):
        g = self.grid()
        g.set_survey_ert(self.rows[:, :6], self.rows[:, 6:], 1.0)
        s = TestOffLineSensitivity().sigma(g)
        v_ref = g.fwd_mod(s)
        with tempfile.TemporaryDirectory() as d:
            g.fv.toVTK({'Conductivity': s}, os.path.join(d, 'job_model'))
            np.savetxt(os.path.join(d, 'job_c1c2.dat'), self.rows[:, :6])
            np.savetxt(os.path.join(d, 'job_p1p2.dat'), self.rows[:, 6:])
            with open(os.path.join(d, 'job.par'), 'w') as f:
                f.write('fwd dc    # job,\n'
                        'job       # basename,\n'
                        'job_model.vtr    # model,\n'
                        'job_c1c2.dat     # source file,\n'
                        'job_p1p2.dat     # measurement file,\n'
                        '1.0       # source current,\n'
                        '{0}   # solver name,\n'
                        '0         # verbose,\n'.format('mumps' if fvm.has_mumps else 'superlu'))
            root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
            env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get('PYTHONPATH', ''))
            res = subprocess.run([sys.executable, '-m', 'pymmr.run_job', 'job.par'], cwd=d, env=env,
                                 capture_output=True, text=True, timeout=600)
            self.assertEqual(res.returncode, 0, res.stderr[-2000:])
            out = np.loadtxt(os.path.join(d, 'job_dc.dat'))
        np.testing.assert_allclose(out[:, 12], v_ref, rtol=1e-5)

    def test_inversion(self):
        import warnings
        from pymmr.inversion import Inversion, DataERT
        g = self.grid()
        g.set_survey_ert(self.rows[:, :6], self.rows[:, 6:], 1.0)
        xc, zc = g.fv.centre_voxels()[0], g.fv.centre_voxels()[-1]
        s_true = np.where((np.abs(xc) < 6) & (zc < -2) & (zc > -8), 0.1, 0.01)
        dobs = g.fwd_mod(s_true)
        n = dobs.size
        data = DataERT(c1c2=self.rows[:, :6], p1p2=self.rows[:, 6:], data=dobs, wt=np.ones(n), cs=np.ones(n),
                       date=None)
        g = self.grid()
        inv = Inversion()
        inv.max_it = 3
        inv.verbose = False
        m0 = 0.01 * np.ones(g.fv.nc)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s_inv, _, rms, _, _ = inv.run(g, m0=m0, m_ref=m0.copy(), data_ert=data)
        self.assertTrue(g._uses_offline25())
        self.assertTrue(np.all(np.isfinite(rms)))
        self.assertLess(rms[-1], 0.5 * rms[0])

    def true_model(self, fv):
        xc, zc = fv.centre_voxels()[0], fv.centre_voxels()[-1]
        return np.where((np.abs(xc) < 6) & (zc < -2) & (zc > -8), 0.1, 0.01)

    def test_joint_inversion(self):
        """Joint MMR (receivers off the line) and off-line ERT data through Inversion.run."""
        import warnings
        from pymmr.inversion import Inversion, DataERT, DataMMR
        from pymmr.mmr import GridMMR
        xs = np.array([[-20.0, 0.0, 0.0, 20.0, 0.0, 0.0]])
        xo = np.array([[a, b, 0.0] for b in (4.0, 10.0) for a in np.arange(-15.0, 16.0, 5.0)])
        gm = self.grid(GridMMR)
        gm.set_survey_mmr(xs, xo, 1.0)
        s_true = self.true_model(gm.dc.fv)
        nd = xo.shape[0]
        dmmr = DataMMR(xs=np.tile(xs, (nd, 1)), xo=xo, data=gm.fwd_mod(s_true), wt=np.ones(3 * nd),
                       cs=np.ones(nd), date=None)
        g = self.grid()
        g.set_survey_ert(self.rows[:, :6], self.rows[:, 6:], 1.0)
        dobs = g.fwd_mod(s_true)
        n = dobs.size
        dert = DataERT(c1c2=self.rows[:, :6], p1p2=self.rows[:, 6:], data=dobs, wt=np.ones(n), cs=np.ones(n),
                       date=None)
        gm = self.grid(GridMMR)
        inv = Inversion()
        inv.max_it = 3
        inv.verbose = False
        m0 = 0.01 * np.ones(gm.dc.fv.nc)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s_inv, d_inv, rms, _, _ = inv.run(gm, m0=m0, m_ref=m0.copy(), data_mmr=dmmr, data_ert=dert)
        self.assertTrue(gm.dc._uses_offline25())
        self.assertEqual(d_inv[0].size, 3 * nd + n)     # MMR and ERT data
        self.assertTrue(np.all(np.isfinite(rms)) and np.all(np.isfinite(s_inv[-1])))
        self.assertLess(rms[-1], 0.5 * rms[0])

    def test_run_job_inv(self):
        """Inversion of off-line ERT data through run_job.py (data file written by a fwd dc job)."""
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        g = self.grid()
        s_true = self.true_model(g.fv)
        solver = 'mumps' if fvm.has_mumps else 'superlu'
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get('PYTHONPATH', ''))
        with tempfile.TemporaryDirectory() as d:
            g.fv.toVTK({'Conductivity': s_true}, os.path.join(d, 'true'))
            g.fv.toVTK({'Conductivity': 0.01 * np.ones(g.fv.nc)}, os.path.join(d, 'start'))
            np.savetxt(os.path.join(d, 'c1c2.dat'), self.rows[:, :6])
            np.savetxt(os.path.join(d, 'p1p2.dat'), self.rows[:, 6:])
            with open(os.path.join(d, 'fwd.par'), 'w') as f:
                f.write('fwd dc    # job,\nsyn       # basename,\ntrue.vtr  # model,\n'
                        'c1c2.dat  # source file,\np1p2.dat  # measurement file,\n1.0       # source current,\n'
                        f'{solver}   # solver name,\n0         # verbose,\n')
            with open(os.path.join(d, 'inv.par'), 'w') as f:
                f.write('inv       # job,\nrun       # basename,\nstart.vtr # model,\n'
                        'syn_dc.dat  # data dc,\n2         # inv max_it,\n'
                        f'{solver}   # solver name,\n0         # verbose,\n')
            for par in ('fwd.par', 'inv.par'):
                res = subprocess.run([sys.executable, '-m', 'pymmr.run_job', par], cwd=d, env=env,
                                     capture_output=True, text=True, timeout=600)
                self.assertEqual(res.returncode, 0, res.stderr[-2000:])
            reader = vtk.vtkXMLRectilinearGridReader()
            reader.SetFileName(os.path.join(d, 'run_inv.vtr'))
            reader.Update()
            cd = reader.GetOutput().GetCellData()
            names = [cd.GetArrayName(i) for i in range(cd.GetNumberOfArrays())]
            self.assertIn('iteration 2', names)
            s2 = vtk_to_numpy(cd.GetArray('iteration 2'))
        self.assertTrue(np.all(np.isfinite(s2)))
        # the block is recovered (higher conductivity there than around it)
        inside = s_true.ravel() > 0.05
        self.assertGreater(np.median(s2[inside]), 1.2 * np.median(s2[~inside]))

    def test_joint_sensitivity(self):
        """Joint MMR-ERT (inversion mode): the ERT columns are the off-line GridDC sensitivity."""
        from pymmr.mmr import GridMMR
        gm = self.grid(GridMMR)
        gm.in_inv = True
        gm.set_survey_mmr(np.array([[-20.0, 0.0, 0.0, 20.0, 0.0, 0.0]] * 2),
                          np.array([[-5.0, 6.0, 0.0], [10.0, 12.0, 0.0]]), 1.0)
        gm.set_survey_ert(self.rows[:, :6], self.rows[:, 6:], 1.0)
        s = TestOffLineSensitivity().sigma(gm.dc)
        d, S = gm.fwd_mod(s, calc_sens=True)
        self.assertTrue(gm.dc._uses_offline25())
        g = self.grid()
        g.set_survey_ert(self.rows[:, :6], self.rows[:, 6:], 1.0)
        d_dc, S_dc = g.fwd_mod(s, calc_sens=True)
        n = self.rows.shape[0]
        np.testing.assert_allclose(d[-n:, 0], d_dc, rtol=1e-10)
        np.testing.assert_allclose(S[:, -n:], S_dc, rtol=1e-8, atol=1e-10 * np.abs(S_dc).max())


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
