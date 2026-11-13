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
from pymmr.finite_volume import calc_padding, GridFV, Grid25FV  # noqa: E402
from pymmr.dc import GridDC, VerticalDyke  # noqa: E402
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

    def test_pole_receivers_homogeneous(self):
        """Electrodes at infinity (dipole-pole, pole-pole, pole-dipole): exact for a homogeneous model."""
        inf = np.inf
        c1c2 = np.array([[-20., 0, 0, -15, 0, 0], [-20., 0, 0, inf, inf, inf], [0., 0, 0, inf, inf, inf]])
        p1p2 = np.array([[5., 0, 0, inf, inf, inf], [10., 0, 0, inf, inf, inf], [15., 0, 0, 20, 0, 0]])
        d_ex = 1e3 * self.rho1 / (2 * np.pi) * np.array([1 / 25 - 1 / 20, 1 / 30, 1 / 15 - 1 / 20])
        g = GridDC((self.x, self.y, self.z))
        g.set_solver(SOLVER)
        g.set_survey_ert(c1c2, p1p2, cs=1.0)
        np.testing.assert_allclose(g.fwd_mod(self.model(g, False)), d_ex, rtol=1e-8)

    def test_elevation(self):
        """Same results with the grid and the electrodes shifted vertically (images about the grid top)."""
        c1c2 = np.r_[self.c1c2[:6], [[-20., 0, -4., 10., 0, 0]]]   # one buried electrode
        p1p2 = np.r_[self.p1p2[:6], [[0., 0, 0, 5., 0, 0]]]
        d = []
        for dz in (0.0, 100.0):
            g = GridDC((self.x, self.y, self.z + dz))
            g.set_solver(SOLVER)
            shift = np.array([0, 0, dz, 0, 0, dz])
            g.set_survey_ert(c1c2 + shift, p1p2 + shift, cs=1.0)
            d.append(g.fwd_mod(self.model(g) if dz == 0 else np.where(g.fv.centre_voxels()[2] > dz - self.hl,
                                                                         1 / self.rho1, 1 / self.rho2)))
        np.testing.assert_allclose(d[1], d[0], rtol=1e-8)

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
        inf = np.inf
        # dipoles, and poles (source and receivers) to check the reference potential term of the source
        c1c2 = np.array([[-3., 0, 0, -2, 0, 0], [-1., 0, 0, 0, 0, 0], [-3., 0, 0, -2, 0, 0],
                         [-4., 0, 0, inf, inf, inf], [-3., 0, 0, -2, 0, 0]])
        p1p2 = np.array([[1., 0, 0, 2, 0, 0], [2., 0, 0, 3, 0, 0], [3., 0, 0, 5, 0, 0],
                         [2., 0, 0, inf, inf, inf], [4., 0, 0, inf, inf, inf]])

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


class TestContact(unittest.TestCase):
    """Electrodes on conductivity contrasts (contact_interp)."""

    def test_flux_interp(self):
        rng = np.random.default_rng(0)
        x = np.r_[0, np.cumsum(rng.uniform(0.5, 1.5, 10))]
        y = np.r_[0, np.cumsum(rng.uniform(0.5, 1.5, 9))]
        z = -np.r_[0, np.cumsum(rng.uniform(0.5, 1.5, 7))][::-1]
        fv = GridFV((x, y, z))
        n = 12
        xp, yp = rng.uniform(fv.xc[0], fv.xc[-1], n), rng.uniform(fv.yc[0], fv.yc[-1], n)
        zp = rng.uniform(fv.zc[0], fv.zc[-1], n)
        xp[:4], yp[:2], zp[:4] = x[3:7], y[4:6], fv.zc[-1]      # on faces and edges, at the surface
        # homogeneous: linear interpolation
        Q, c = fv.flux_interp(xp, yp, zp, 0.3)
        self.assertLess(abs(Q - fv.linear_interp(xp, yp, zp)).max(), 1e-14)
        np.testing.assert_array_equal(c, 1.0)
        # model invariant along y: same as 2.5D
        f2 = Grid25FV((x, z))
        s2 = np.exp(rng.normal(0, 1, f2.nc))
        s3 = np.tile(s2.reshape(f2.nz, 1, f2.nx), (1, fv.ny, 1)).ravel()
        Q2, c2 = f2.flux_interp(xp, zp, s2)
        Q3, c3 = fv.flux_interp(xp, yp, zp, s3)
        np.testing.assert_allclose(c3, c2, rtol=1e-12)
        # derivatives (complex step) vs finite differences
        s = np.exp(rng.normal(0, 1, fv.nc))
        Q, c, cells, dW, dc = fv.flux_interp(xp, yp, zp, s, derivatives=True)
        np.testing.assert_allclose(Q.sum(axis=1), 1.0)
        eps = 1e-6
        for i in range(6):
            for j in range(8):
                sp_, sm = s.copy(), s.copy()
                sp_[cells[i, j]] *= np.exp(eps)
                sm[cells[i, j]] *= np.exp(-eps)
                Qp, cp = fv.flux_interp(xp[i:i + 1], yp[i:i + 1], zp[i:i + 1], sp_)
                Qm, cm = fv.flux_interp(xp[i:i + 1], yp[i:i + 1], zp[i:i + 1], sm)
                np.testing.assert_allclose((Qp - Qm).toarray()[0, cells[i]] / (2 * eps), dW[i, :, j], atol=1e-8)
                self.assertAlmostEqual((cp - cm)[0] / (2 * eps), dc[i, j], places=8)

    def test_reference_models_exact(self):
        """The solution is exact for the reference model of the current electrode: 4 quarter-spaces around
        a vertical edge (electrode on a node), 2 quarter-spaces (electrode on a face, or near it)."""
        h = 1.0
        x = padded(np.arange(-8., 8.01, h), h, 10)
        y = padded(np.arange(-8., 8.01, h), h, 10)
        z = padded(np.arange(-8., 0.01, h), h, 10, top=False)
        inf = np.inf
        p1p2 = np.array([[-4., -3, 0, 3, 2, 0], [2., -5, 0, inf, inf, inf], [-3., 4, 0, 5, 5, 0]])
        s4 = np.array([0.01, 0.2, 0.05, 0.003])        # quadrants (x < 0, y < 0), (x > 0, y < 0), ...
        for c1, model in (([0., 0, 0], 'edge'), ([0., 0.3, 0], 'face x'), ([-0.2, 0., 0], 'face y')):
            g = GridDC((x, y, z))
            g.set_solver(SOLVER)
            g.set_survey_ert(np.tile(c1 + [inf] * 3, (3, 1)), p1p2, 1.0)
            xc, yc, _ = g.fv.centre_voxels()
            if model == 'edge':
                sigma = s4[(xc > 0).astype(int) + 2 * (yc > 0)]
            elif model == 'face x':
                sigma = np.where(xc < 0, s4[0], s4[1])
            else:
                sigma = np.where(yc < 0, s4[0], s4[2])
            d = g.fwd_mod(sigma)

            def u(p):
                p, c = np.asarray(p, float), np.asarray(c1, float)
                if model == 'edge':
                    return 1 / (2 * np.pi * s4.mean() * np.linalg.norm(p - c))
                a = 0 if model == 'face x' else 1
                sa, sb = (s4[0], s4[1]) if a == 0 else (s4[0], s4[2])
                ss, so = (sa, sb) if c[a] <= 0 else (sb, sa)
                if (p[a] <= 0) == (c[a] <= 0):
                    img = c.copy()
                    img[a] = -c[a]
                    return (1 / np.linalg.norm(p - c) + (ss - so) / (sa + sb) / np.linalg.norm(p - img)) / (2 * np.pi * ss)
                return 1 / (np.pi * (sa + sb) * np.linalg.norm(p - c))

            ref = 1e3 * np.array([u(r[:3]) - (u(r[3:]) if np.isfinite(r[3]) else 0) for r in p1p2])
            np.testing.assert_allclose(d, ref, rtol=1e-8, err_msg=model)

    def test_dyke_contacts(self):
        """Dyke with electrodes on the contacts, on grid nodes (y = 0) and on faces (y = 0.5)."""
        h = 1.0
        x = padded(np.arange(-24., 33.01, h), h, 14)
        y = padded(np.arange(-8., 8.01, h), h, 14)
        z = padded(np.arange(-12., 0.01, h), h, 14, top=False)
        xs = np.arange(-6.0, 15.0, 2.0).reshape(-1, 1)
        zero = np.zeros(xs.shape)
        A = np.c_[xs, zero, zero, 1.0 + xs, zero, zero]
        B = np.c_[2.0 + xs, zero, zero, 3.0 + xs, zero, zero]
        rho = (100.0, 1000.0, 50.0)
        for y0 in (0.0, 0.5):
            off = np.array([0, y0, 0, 0, y0, 0])
            for name, (c1c2, p1p2) in (('direct', (A + off, B + off)), ('reciprocal', (B + off, A + off))):
                va = VerticalDyke(*rho, 10.0).fwd_mod(c1c2, p1p2)
                err = {}
                for ci in (True, False):
                    g = GridDC((x, y, z))
                    g.set_solver(SOLVER)
                    g.set_survey_ert(c1c2, p1p2, 1.0)
                    g.contact_interp = ci
                    xc = g.fv.centre_voxels()[0]
                    v = 1e-3 * g.fwd_mod(np.where(xc < 0, 1 / rho[0], np.where(xc < 10, 1 / rho[1], 1 / rho[2])))
                    err[ci] = np.abs(v / va - 1)
                with self.subTest(name, y=y0):
                    # 1 m cells: errors at the contacts divided by more than 10
                    self.assertLess(err[True].max(), 0.7)
                    self.assertGreater(err[False].max(), 4.0)

    def test_homogeneous_unchanged(self):
        """In a homogeneous model, same data as the previous method."""
        h = 1.0
        x = padded(np.arange(-6., 6.01, h), h, 8)
        z = padded(np.arange(-5., 0.01, h), h, 8, top=False)
        inf = np.inf
        c1c2 = np.array([[-3., 0, 0, -2, 0.5, 0], [-1., 0.3, 0, inf, inf, inf]])
        p1p2 = np.array([[1., 0, 0, 2, 0, 0], [2., 0.5, 0, 3, -1, 0]])
        for bc in (True, False):
            d = []
            for ci in (True, False):
                g = GridDC((x, x, z))
                g.set_solver(SOLVER)
                g.set_survey_ert(c1c2, p1p2, 1.0)
                g.contact_interp, g.apply_bc = ci, bc
                d.append(g.fwd_mod(0.01 * np.ones(g.fv.nc)))
            np.testing.assert_allclose(d[0], d[1], rtol=1e-9, err_msg=f'apply_bc={bc}')

    def test_reciprocity(self):
        h = 1.0
        x = padded(np.arange(-6., 6.01, h), h, 8)
        z = padded(np.arange(-5., 0.01, h), h, 8, top=False)
        A = np.array([[-3., 0, 0, -2, 0.5, 0], [-1., 0.3, 0, 1, 0, 0]])
        B = np.array([[1., 0, 0, 2, 0, 0], [2., 0.5, 0, 3, -1, 0]])
        rng = np.random.default_rng(2)
        sig = None
        v = []
        for c1c2, p1p2 in ((A, B), (B, A)):
            g = GridDC((x, x, z))
            g.set_solver(SOLVER)
            g.set_survey_ert(c1c2, p1p2, 1.0)
            g.apply_bc = False
            if sig is None:
                sig = 0.01 * 10**rng.uniform(-0.5, 0.5, g.fv.nc)
            v.append(g.fwd_mod(sig))
        np.testing.assert_allclose(v[0], v[1], rtol=1e-8)

    def test_sensitivity_contacts(self):
        """d(data)/d(ln sigma) vs finite differences in the cells around the electrodes, on contacts (nodes
        and faces), for several contrasts."""
        h = 1.0
        x = padded(np.arange(-5., 5.01, h), h, 6)
        y = padded(np.arange(-3., 3.01, h), h, 6)
        z = padded(np.arange(-3., 0.01, h), h, 6, top=False)
        inf = np.inf
        c1c2 = np.array([[-2., 0, 0, 2, 0.5, 0], [0.5, 1, 0, inf, inf, inf], [-3., 0.3, 0, -2, 0, 0]])
        p1p2 = np.array([[-1., 0, 0, 3.5, 0, 0], [2., 0, 0, -2, 1, 0], [2., 0.5, 0, 4, 0, 0]])
        rng = np.random.default_rng(11)

        def grid(opts):
            g = GridDC((x, y, z))
            g.set_solver(SOLVER)
            g.set_survey_ert(c1c2, p1p2, cs=1.5)
            for key, val in opts.items():
                setattr(g, key, val)
            return g

        fv = grid({}).fv
        xc, yc, zc = fv.centre_voxels()
        e = np.vstack((c1c2[:, :3], c1c2[:, 3:], p1p2[:, :3], p1p2[:, 3:]))
        e = e[np.all(np.isfinite(e), axis=1)]
        near = np.zeros(fv.nc, dtype=bool)
        for p in e:
            near |= (np.abs(xc - p[0]) < 1.1) & (np.abs(yc - p[1]) < 1.1) & (zc > -1.6)
        cells = np.where(near)[0]
        noise = 10**rng.uniform(-0.1, 0.1, fv.nc)
        inside = (xc > -2) & (xc < 2) & (yc < 1)
        cases = [('homogeneous', 0.01 * np.ones(fv.nc), {})]
        for contrast in (3.0, 25.0, 1 / 25.0):
            sig = np.where(inside, 0.05 / contrast, 0.05) * noise
            cases += [(contrast, sig, opts) for opts in ({}, {'local_sigma0': False}, {'apply_bc': False})]
        for contrast, sig, opts in cases:
            with self.subTest(contrast=contrast, **opts):
                d1, S = grid(opts).fwd_mod(sig, calc_sens=True)
                np.testing.assert_allclose(d1, grid(opts).fwd_mod(sig), rtol=1e-10)
                for c in cells:
                    eps = 1e-5
                    s2, s3 = sig.copy(), sig.copy()
                    s2[c] *= np.exp(eps)
                    s3[c] *= np.exp(-eps)
                    fd = (grid(opts).fwd_mod(s2) - grid(opts).fwd_mod(s3)) / (2 * eps)
                    self.assertLess(np.max(np.abs(S[c] - fd)) / np.max(np.abs(fd)), 1e-5, msg=f'cell {c}')


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

    def test_mmr_contact_interp(self):
        """The MMR data (fields, pod_e, sensitivity) do not depend on dc.contact_interp (previous method)."""
        h = 20.0
        x = padded(np.arange(-200., 200.1, h), h, 6)
        y = padded(np.arange(-100., 100.1, h), h, 6)
        z = padded(np.arange(-120., 0.1, h), h, 6, top=False)
        xs = np.array([[-140., 0, 0, 140., 0, 0]])
        xo = np.array([[0., 40., 0], [60., -40., 0], [-80., 60., 0]])
        res = []
        for ci in (True, False):
            g = GridMMR((x, y, z))
            g.set_solver(SOLVER)
            g.set_survey_mmr(xs=xs, xo=xo, cs=1.0, pod_e=(20.0, 20.0, -20.0))
            g.dc.contact_interp = ci
            xc = g.dc.fv.centre_voxels()[0]
            sigma = np.where(np.abs(xc) < 40, 0.1, 0.01)    # contacts at the electrodes of pod_e (x = -40, 40)
            res.append(g.fwd_mod(sigma, calc_sens=True))
            self.assertEqual(g.dc.contact_interp, ci)   # setting restored
        (B0, J0), v0 = res[0]
        (B1, J1), v1 = res[1]
        np.testing.assert_allclose(B0, B1, rtol=1e-10, atol=1e-10 * np.abs(B1).max())
        np.testing.assert_allclose(J0, J1, rtol=1e-10, atol=1e-10 * np.abs(J1).max())
        np.testing.assert_allclose(v0, v1, rtol=1e-10, atol=1e-10 * np.abs(v1).max())

    def test_air_thickness(self):
        """Air at least as thick as the earth part of the grid, whatever the size of the top cells."""
        h, htop = 40.0, 0.25
        x = padded(np.arange(-80., 80.1, h), h, 4)
        zt = -np.cumsum(np.r_[np.full(4, htop), htop * 2.0**np.arange(1, 8)])
        z = padded(np.r_[np.arange(-160., zt[-1] - h / 2, h), zt[::-1], 0.0] + 50.0, h, 6, top=False)
        g = GridMMR((x, x, z))     # surface at z = 50 m, top cells of 0.25 m: the default air is 55 m
        g.set_solver(SOLVER)
        g.set_survey_mmr(xs=np.array([[-40., 0, 50., 40., 0, 50.]]), xo=np.array([[0., 20., 50.]]), cs=1.0)
        g.fwd_mod(1e-2 * np.ones(g.dc.nc))
        self.assertGreaterEqual(g.fv.z[-1] - g.dc.fv.z[-1], g.dc.fv.z[-1] - g.dc.fv.z[0])

    def test_pod_e(self):
        """Electric dipoles at the receivers: exact in a half-space, same as GridDC in a layered earth."""
        h = 5.0
        x = padded(np.arange(-50., 50.1, h), h, 10)
        z = padded(np.arange(-40., 0.1, h), h, 10, top=False)
        xs = np.array([-25., 0, 0, 25., 0, 0])
        xr = np.array([-37., -12., 8., 33.])
        pod_e = (3.0, 3.0, -10.0)
        dls = np.diag(pod_e)

        def u(p, sigma):
            return (1 / np.linalg.norm(p - xs[:3], axis=1) - 1 / np.linalg.norm(p - xs[3:], axis=1)) / (2 * np.pi * sigma)

        for yr in (0.0, 10.0):
            xo = np.c_[xr, np.full(4, yr), np.zeros(4)]
            g = GridMMR((x, x, z))
            g.set_solver(SOLVER)
            g.set_survey_mmr(xs=xs, xo=xo, cs=1.0, pod_e=pod_e)
            zc = g.dc.fv.centre_voxels()[2]
            for layered in (False, True):
                sigma = np.where(zc > -10.0, 1e-2, 1e-1) if layered else 1e-2 * np.ones(g.dc.nc)
                B, v = g.fwd_mod(sigma)
                for n in range(3):
                    if layered:
                        # linear interpolation, as for pod_e: the lower ends are on the layer boundary, where
                        # GridDC.contact_interp changes the interpolation
                        gd = GridDC((x, x, z))
                        gd.set_solver(SOLVER)
                        gd.contact_interp = False
                        gd.set_survey_ert(np.tile(xs, (4, 1)), np.c_[xo, xo + dls[n]], 1.0)
                        ref = gd.fwd_mod(sigma)
                    else:
                        ref = g.dc._units_scaling * (u(xo, 1e-2) - u(xo + dls[n], 1e-2))
                    np.testing.assert_allclose(v[:, n], ref, rtol=1e-3 if layered else 1e-6,
                                               err_msg=f'y {yr} layered {layered} component {n}')
            # dc.exact_primary = False: same as GridDC with the same setting
            g.dc.exact_primary = False
            _, v = g.fwd_mod(sigma)
            g.dc.exact_primary = True
            gd = GridDC((x, x, z))
            gd.set_solver(SOLVER)
            gd.exact_primary = False
            gd.contact_interp = False
            gd.set_survey_ert(np.tile(xs, (4, 1)), np.c_[xo, xo + dls[2]], 1.0)
            np.testing.assert_allclose(v[:, 2], gd.fwd_mod(sigma), rtol=1e-6)
            # MMR data unchanged by the electric dipoles
            g2 = GridMMR((x, x, z))
            g2.set_solver(SOLVER)
            g2.set_survey_mmr(xs=xs, xo=xo, cs=1.0)
            np.testing.assert_allclose(B, g2.fwd_mod(sigma), rtol=1e-10, atol=1e-12 * np.abs(B).max())


if __name__ == '__main__':
    unittest.main()
