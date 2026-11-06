#!/usr/bin/env python3
"""
Tests of the regularization terms used in inversion (GridDC.calc_WtW and GridDC.calc_reg).

Run with:  python3 tests/test_regularization.py   (or pytest)
"""
import os
import sys
import unittest
import warnings

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pymmr.finite_volume import calc_padding  # noqa: E402
from pymmr.dc import GridDC  # noqa: E402
from pymmr.inversion import Inversion, DataERT  # noqa: E402

SMOOTH_TYPES = ('smooth', 'blocky', 'ekblom', 'min. support')


def grids():
    return {'3D': GridDC((np.arange(0., 8.1), np.arange(0., 6.1), np.arange(-5., 0.1))),
            '2.5D': GridDC((np.arange(0., 12.1), np.arange(-6., 0.1)))}


def layered_model(g, rng):
    """Two layers in log-conductivity, with small perturbations."""
    _, _, zc = g.fv.centre_voxels()
    return np.where(zc > -2.5, np.log(0.01), np.log(0.1)) + 0.01 * rng.standard_normal(g.fv.nc)


class TestCalcWtW(unittest.TestCase):

    def test_weights_on_squared_terms(self):
        """WtW = Gx^T diag(w) Gx + ... + als I, one weight per face (no broadcasting)."""
        rng = np.random.default_rng(0)
        for name, g in grids().items():
            with self.subTest(grid=name):
                inv = Inversion()
                Gx, Gy, Gz = g.fv.extract_Gxyz()
                wx, wy, wz = (rng.uniform(0.1, 2.0, Gi.shape[0]) for Gi in (Gx, Gy, Gz))
                ref = (inv.alx**2 * Gx.T @ sp.diags_array(wx) @ Gx +
                       inv.aly**2 * Gy.T @ sp.diags_array(wy, shape=(wy.size, wy.size)) @ Gy +
                       inv.alz**2 * Gz.T @ sp.diags_array(wz) @ Gz + inv.als * sp.eye_array(g.fv.nc))
                act = np.arange(g.fv.nc)
                w = np.ones(g.fv.nc)
                for fmt in ('1d', 'column', 'sparse'):
                    conv = {'1d': lambda a: a, 'column': lambda a: a.reshape(-1, 1),
                            'sparse': lambda a: sp.diags_array(a, shape=(a.size, a.size))}[fmt]
                    WtW = g.calc_WtW(w, inv, act, conv(wx), conv(wy), conv(wz))
                    self.assertLess(sp.linalg.norm(WtW - ref) / sp.linalg.norm(ref), 1e-12, fmt)

    def test_wrong_size(self):
        g = grids()['3D']
        with self.assertRaises(ValueError):
            g.calc_WtW(np.ones(g.fv.nc), Inversion(), np.arange(g.fv.nc), WGx=np.ones(3))

    def test_smallness_weights(self):
        g = grids()['2.5D']
        inv = Inversion()
        act = np.arange(g.fv.nc)
        ws = np.linspace(1., 2., g.fv.nc)
        d = (g.calc_WtW(np.ones(g.fv.nc), inv, act, WS=ws) - g.calc_WtW(np.ones(g.fv.nc), inv, act)).diagonal()
        np.testing.assert_allclose(d, inv.als * (ws - 1), atol=1e-14)


class TestCalcReg(unittest.TestCase):

    def test_all_types(self):
        rng = np.random.default_rng(1)
        for name, g in grids().items():
            m = layered_model(g, rng)
            for st in SMOOTH_TYPES:
                with self.subTest(grid=name, smooth_type=st):
                    inv = Inversion()
                    inv.smooth_type = st
                    WtW, WGx, WGy, WGz = g.calc_reg(np.ones(g.fv.nc), inv, m, 0.0, None, None, None,
                                                    np.arange(g.fv.nc))
                    self.assertEqual(WtW.shape, (g.fv.nc, g.fv.nc))
                    self.assertTrue(np.all(np.isfinite(WtW.data)))
                    self.assertLess(sp.linalg.norm(WtW - WtW.T), 1e-12 * sp.linalg.norm(WtW))
                    for Wi in (WGx, WGy, WGz):
                        self.assertTrue(np.all(Wi > 0))

    def test_no_compounding(self):
        """Same model -> same weights (IRLS weights are recomputed, not accumulated)."""
        rng = np.random.default_rng(2)
        g = grids()['2.5D']
        m = layered_model(g, rng)
        inv = Inversion()
        inv.smooth_type = 'blocky'
        act = np.arange(g.fv.nc)
        out1 = g.calc_reg(np.ones(g.fv.nc), inv, m, 0.0, None, None, None, act)
        out2 = g.calc_reg(np.ones(g.fv.nc), inv, m, 0.0, None, None, None, act)
        for a, b in zip(out1[1:], out2[1:]):
            np.testing.assert_array_equal(a, b)
        # base weights multiply the IRLS weights
        base = np.full(g.fv.nfx, 2.0)
        out3 = g.calc_reg(np.ones(g.fv.nc), inv, m, 0.0, base, None, None, act)
        np.testing.assert_allclose(out3[1], 2.0 * out1[1])

    def test_blocky_l1_weights(self):
        """Blocky weights ~ 1/|c| (L1); flat regions get the largest weight."""
        rng = np.random.default_rng(3)
        g = grids()['3D']
        m = layered_model(g, rng)
        m[:g.fv.nx] = m[0]   # some faces with zero gradient
        inv = Inversion()
        inv.smooth_type = 'blocky'
        _, WGx, _, WGz = g.calc_reg(np.ones(g.fv.nc), inv, m, 0.0, None, None, None, np.arange(g.fv.nc))
        Gx, _, Gz = g.fv.extract_Gxyz()
        c = Gz @ m
        inner = (WGz > 1e-6) & (WGz < 10)
        prod = WGz[inner] * np.abs(c[inner])
        np.testing.assert_allclose(prod, prod[0], rtol=1e-10)
        self.assertTrue(np.all(WGx[Gx @ m == 0.0] == 10.0))

    def test_ekblom(self):
        rng = np.random.default_rng(4)
        g = grids()['3D']
        m = layered_model(g, rng)
        act = np.arange(g.fv.nc)
        inv = Inversion()
        inv.smooth_type = 'ekblom'
        # p = 2: weights 2 for gradients and smallness -> twice the smooth regularization
        inv.ekblom_p = 2.0
        WtW = g.calc_reg(np.ones(g.fv.nc), inv, m, 0.0, None, None, None, act)[0]
        WtW_smooth = g.calc_WtW(np.ones(g.fv.nc), inv, act)
        self.assertLess(sp.linalg.norm(WtW - 2 * WtW_smooth) / sp.linalg.norm(WtW), 1e-12)
        # p = 1: robust measure, weights decrease with the gradient
        inv.ekblom_p = 1.0
        inv.minsupport_e = 1.0
        _, _, _, WGz = g.calc_reg(np.ones(g.fv.nc), inv, m, 0.0, None, None, None, act)
        Gz = g.fv.extract_Gxyz()[2]
        c = np.abs(Gz @ m)
        order = np.argsort(c)
        self.assertTrue(np.all(np.diff(WGz[order]) <= 1e-12))
        self.assertLess(WGz[order[-1]], WGz[order[0]])

    def test_min_support_forms(self):
        """Exact weights for both forms of the minimum gradient support (capped at 2)."""
        rng = np.random.default_rng(6)
        g = grids()['2.5D']
        m = layered_model(g, rng)
        mref = np.log(0.02) * np.ones(g.fv.nc)
        e = 0.3
        Gx, _, Gz = g.fv.extract_Gxyz()
        for form, f in (('portniaguine', lambda c: np.sqrt(2) * e / (c**2 + e**2)),
                        ('general', lambda c: 2 * e**2 / (c**2 + e**2)**2)):
            with self.subTest(form=form):
                inv = Inversion()
                inv.smooth_type = 'min. support'
                inv.minsupport_e = e
                inv.minsupport_form = form
                _, WGx, _, WGz = g.calc_reg(np.ones(g.fv.nc), inv, m, mref, None, None, None, np.arange(g.fv.nc))
                np.testing.assert_allclose(WGx, np.minimum(f(Gx @ (m - mref)), 2.0), rtol=1e-12)
                np.testing.assert_allclose(WGz, np.minimum(f(Gz @ (m - mref)), 2.0), rtol=1e-12)
        inv.minsupport_form = 'other'
        with self.assertRaises(ValueError):
            g.calc_reg(np.ones(g.fv.nc), inv, m, mref, None, None, None, np.arange(g.fv.nc))

    def test_ekblom_smallness(self):
        """Smallness weights computed from m - m_ref, capped by ekblom_smallness_max."""
        rng = np.random.default_rng(8)
        g = grids()['2.5D']
        act = np.arange(g.fv.nc)
        mref = np.log(0.01) * np.ones(g.fv.nc)
        m = mref + 0.5 * rng.standard_normal(g.fv.nc)
        m[:5] = mref[:5]   # cells equal to the reference
        inv = Inversion()
        inv.smooth_type = 'ekblom'
        inv.minsupport_e = 0.01
        p, e = inv.ekblom_p, inv.minsupport_e
        for ws_max in (2.0, None):
            with self.subTest(ekblom_smallness_max=ws_max):
                inv.ekblom_smallness_max = ws_max
                WtW, WGx, WGy, WGz = g.calc_reg(np.ones(g.fv.nc), inv, m, mref, None, None, None, act)
                ws = p * ((m - mref)**2 + e**2)**(p / 2 - 1)
                if ws_max is not None:
                    ws = np.minimum(ws, ws_max)
                ref = g.calc_WtW(np.ones(g.fv.nc), inv, act, WGx, WGy, WGz, WS=ws)
                self.assertLess(sp.linalg.norm(WtW - ref) / sp.linalg.norm(ref), 1e-12)
                # cells equal to the reference: p / e without limit
                self.assertAlmostEqual(ws[0], 2.0 if ws_max else p / e)

    def test_column_vectors(self):
        """Model and reference given as column vectors: same result, weights of size n_faces."""
        rng = np.random.default_rng(9)
        g = grids()['3D']
        m = layered_model(g, rng)
        mref = np.log(0.02) * np.ones(g.fv.nc)
        act = np.arange(g.fv.nc)
        for st in SMOOTH_TYPES:
            with self.subTest(smooth_type=st):
                inv = Inversion()
                inv.smooth_type = st
                out1 = g.calc_reg(np.ones(g.fv.nc), inv, m, mref, None, None, None, act)
                out2 = g.calc_reg(np.ones(g.fv.nc), inv, m.reshape(-1, 1), mref.reshape(-1, 1),
                                  None, None, None, act)
                self.assertLess(sp.linalg.norm(out1[0] - out2[0]), 1e-12 * sp.linalg.norm(out1[0]))
                for a, b in zip(out1[1:], out2[1:]):
                    self.assertEqual(np.shape(b), np.shape(a))
                    np.testing.assert_array_equal(a, b)

    def test_min_support(self):
        rng = np.random.default_rng(5)
        g = grids()['3D']
        m = layered_model(g, rng)
        inv = Inversion()
        inv.smooth_type = 'min. support'
        inv.minsupport_e = 0.5
        _, _, _, WGz = g.calc_reg(np.ones(g.fv.nc), inv, m, 0.0, None, None, None, np.arange(g.fv.nc))
        c = np.abs(g.fv.extract_Gxyz()[2] @ m)
        order = np.argsort(c)
        self.assertTrue(np.all(np.diff(WGz[order]) <= 1e-12))


class TestInversionRun(unittest.TestCase):
    """Smoke test: a few Gauss-Newton iterations for each type of smoothing (3D DC)."""

    def test_run(self):
        h = 1.0
        x = np.arange(-6., 6.01, h)
        y = np.arange(-3., 3.01, h)
        z = np.arange(-4., 0.01, h)
        pad = np.cumsum(calc_padding(h, 5))
        x = np.r_[x[0] - pad[::-1], x, x[-1] + pad]
        y = np.r_[y[0] - pad[::-1], y, y[-1] + pad]
        z = np.r_[z[0] - pad[::-1], z]
        e = np.arange(-5., 5.1, 1.)
        rows = [[e[i], e[i + 1], e[i + 1 + n], e[i + 2 + n]] for n in (1, 2, 3) for i in range(len(e) - 2 - n)]
        rows = np.array(rows)
        zr = np.zeros(len(rows))
        c1c2 = np.c_[rows[:, 0], zr, zr, rows[:, 1], zr, zr]
        p1p2 = np.c_[rows[:, 2], zr, zr, rows[:, 3], zr, zr]

        g = GridDC((x, y, z))
        g.set_solver('superlu')
        g.set_survey_ert(c1c2, p1p2, cs=1.0)
        xc, yc, zc = g.fv.centre_voxels()
        s_true = np.where((np.abs(xc) < 2) & (np.abs(yc) < 2) & (zc > -3) & (zc < -1), 0.1, 0.01)
        dobs = g.fwd_mod(s_true)
        data = DataERT(c1c2=c1c2, p1p2=p1p2, data=dobs, wt=np.ones(dobs.size), cs=np.ones(dobs.size), date=None)

        # IRLS weights are computed at the end of an iteration, they act from the 3rd model on
        models = {}
        for st in SMOOTH_TYPES:
            with self.subTest(smooth_type=st):
                g = GridDC((x, y, z))
                g.set_solver('superlu')
                inv = Inversion()
                inv.smooth_type = st
                inv.max_it = 3
                inv.verbose = False
                m0 = 0.01 * np.ones(g.fv.nc)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    sigma_inv, data_inv, rms, misfit, smy = inv.run(g, m0=m0, m_ref=m0.copy(), data_ert=data)
                self.assertEqual(len(rms), 3)
                self.assertTrue(np.all(np.isfinite(rms)))
                self.assertLess(rms[-1], rms[0])
                self.assertTrue(np.all(np.isfinite(sigma_inv[-1])))
                models[st] = sigma_inv[-1]
        for st in SMOOTH_TYPES[1:]:
            self.assertFalse(np.allclose(models[st], models['smooth']), st)

        # with reg_var 'model', the IRLS weights are computed relative to the reference model
        g = GridDC((x, y, z))
        g.set_solver('superlu')
        calls = []
        calc_reg = g.calc_reg

        def spy(Q, par, xc, xref, *args):
            calls.append(np.array(xref, copy=True))
            return calc_reg(Q, par, xc, xref, *args)

        g.calc_reg = spy
        inv = Inversion()
        inv.smooth_type = 'ekblom'
        inv.max_it = 2
        inv.verbose = False
        m_ref = 0.02 * np.ones(g.fv.nc)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            inv.run(g, m0=0.01 * np.ones(g.fv.nc), m_ref=m_ref, data_ert=data)
        self.assertGreater(len(calls), 0)
        for xref in calls:
            np.testing.assert_allclose(xref, np.log(m_ref))


if __name__ == '__main__':
    unittest.main()
