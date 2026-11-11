#!/usr/bin/env python3
"""
Tests of the wavenumber-domain vector operators of Grid25FV (2.5D MMR, see docs/mmr25.md).

Run with:  python3 tests/test_mmr25_operators.py   (or pytest)
"""
import os
import sys
import unittest

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pymmr.finite_volume import calc_padding, Grid25FV, Solver25  # noqa: E402


def random_grid(seed=0, nx=10, nz=8):
    rng = np.random.default_rng(seed)
    x = np.cumsum(np.r_[0.0, rng.uniform(0.5, 2.0, nx)])
    z = np.cumsum(np.r_[0.0, rng.uniform(0.5, 2.0, nz)])
    return Grid25FV((x, z))


# smooth test fields: one Gaussian per component, A_i = exp(-((x-a_i)^2 + (z-b_i)^2) / (2 s^2))
S = 1.0
CENTRES = {'x': (0.3, -0.2), 'y': (-0.4, 0.1), 'z': (0.2, 0.5)}


def gauss(c, x, z):
    a, b = CENTRES[c]
    return np.exp(-((x - a)**2 + (z - b)**2) / (2 * S**2))


def d_gauss(c, x, z, wrt):
    a, b = CENTRES[c]
    d = (x - a) if wrt == 'x' else (z - b)
    return -d / S**2 * gauss(c, x, z)


def lap_gauss(c, x, z):
    a, b = CENTRES[c]
    r2 = (x - a)**2 + (z - b)**2
    return (r2 / S**4 - 2 / S**2) * gauss(c, x, z)


def locations(g):
    """Coordinates of the components of vectors on faces and on edges (x fast)."""
    def mesh(xv, zv):
        X, Z = np.meshgrid(xv, zv)
        return X.flatten(), Z.flatten()
    xn, zn = g.x[1:-1], g.z[1:-1]
    faces = {'x': mesh(xn, g.zc), 'y': mesh(g.xc, g.zc), 'z': mesh(g.xc, zn)}
    edges = {'x': mesh(g.xc, zn), 'y': mesh(xn, zn), 'z': mesh(xn, g.zc)}
    return faces, edges


def uniform_grid(h, L=8.0):
    x = np.arange(-L, L + h / 2, h)
    return Grid25FV((x, x.copy()))


class TestIdentities(unittest.TestCase):

    def setUp(self):
        self.g = random_grid()
        self.k = 0.7

    def test_shapes(self):
        g, k = self.g, self.k
        self.assertEqual(g.build_C_k(k).shape, (g.nf_k, g.ne_k))
        self.assertEqual(g.build_C_k(k, to_faces=False).shape, (g.ne_k, g.nf_k))
        self.assertEqual(g.build_G_k(k).shape, (g.nf_k, g.nc))
        self.assertEqual(g.build_D_k(k).shape, (g.nc, g.nf_k))
        self.assertEqual(g.build_L_k(k).shape, (g.nf_k, g.nf_k))
        self.assertEqual(g.volume_faces_k().size, g.nf_k)
        self.assertEqual(g.volume_edges_k().size, g.ne_k)

    def test_div_curl_and_curl_grad(self):
        g, k = self.g, self.k
        self.assertLess(abs(g.build_D_k(k) @ g.build_C_k(k)).max(), 1e-12)
        self.assertLess(abs(g.build_C_k(k, to_faces=False) @ g.build_G_k(k)).max(), 1e-12)

    def test_adjoints(self):
        g, k = self.g, self.k
        Vf, Ve, Vc = (sp.diags_array(v) for v in (g.volume_faces_k(), g.volume_edges_k(), g.volume_voxels()))
        Ce, Cf = g.build_C_k(k), g.build_C_k(k, to_faces=False)
        self.assertLess(abs(Ve @ Cf - (Vf @ Ce).T).max(), 1e-12)
        D_ref = -sp.diags_array(1.0 / g.volume_voxels()) @ g.build_G_k(k).T @ Vf
        self.assertLess(abs(g.build_D_k(k) - D_ref).max(), 1e-12)
        del Vc

    def test_L_spd_and_uncoupled(self):
        g, k = self.g, self.k
        L = g.build_L_k(k)
        S_ = (sp.diags_array(g.volume_faces_k()) @ L).toarray()
        self.assertLess(abs(S_ - S_.T).max(), 1e-12)
        self.assertGreater(np.linalg.eigvalsh(S_).min(), 0.0)
        i1, i2 = g.nfx, g.nfx + g.nc
        B = L.toarray()
        for a, b in ((slice(0, i1), slice(i1, None)), (slice(i1, i2), slice(i2, None))):
            self.assertEqual(abs(B[a, b]).max(), 0.0)
            self.assertEqual(abs(B[b, a]).max(), 0.0)

    def test_y_component_is_dc_operator(self):
        # the y block of L_k is -D G + k^2, the 2.5D DC operator with sigma = 1
        g, k = self.g, self.k
        i1, i2 = g.nfx, g.nfx + g.nc
        Ly = g.build_L_k(k)[i1:i2, i1:i2]
        self.assertLess(abs(Ly - Solver25._L(-(g.D @ g.G), 1.0, k)).max(), 1e-12)

    def test_dc_operator(self):
        g, k = self.g, self.k
        rng = np.random.default_rng(1)
        s = rng.uniform(0.01, 1.0, g.nc)
        A, _ = g.build_A(s)
        L_dc = -g.build_D_k(k) @ g.build_M_k(s) @ g.build_G_k(k)
        self.assertLess(abs(L_dc - Solver25._L(A, s, k)).max(), 1e-12)

    def test_M_k_tensor(self):
        g = self.g
        s = np.linspace(0.1, 1.0, g.nc)
        M = g.build_M_k((s, 2 * s, 3 * s)).diagonal()
        np.testing.assert_allclose(M[g.nfx:g.nfx + g.nc], 2 * s)
        np.testing.assert_allclose(M[:g.nfx], g.build_M(s).diagonal()[:g.nfx])
        np.testing.assert_allclose(M[g.nfx + g.nc:], 3 * g.build_M(s).diagonal()[g.nfx:])
        with self.assertRaises(ValueError):
            g.build_M_k((s, s))


class TestAccuracy(unittest.TestCase):
    """Second-order convergence for smooth fields (uniform grids)."""

    K = 0.8

    def curl_error(self, h):
        g, k = uniform_grid(h), self.K
        f, e = locations(g)
        A = np.r_[gauss('x', *f['x']), gauss('y', *f['y']), gauss('z', *f['z'])]
        # Bx = -k Az - dAy/dz,  By = dAx/dz - dAz/dx,  Bz = dAy/dx + k Ax
        B = np.r_[-k * gauss('z', *e['x']) - d_gauss('y', *e['x'], 'z'),
                  d_gauss('x', *e['y'], 'z') - d_gauss('z', *e['y'], 'x'),
                  d_gauss('y', *e['z'], 'x') + k * gauss('x', *e['z'])]
        return np.abs(g.build_C_k(k, to_faces=False) @ A - B).max()

    def helmholtz_error(self, h):
        g, k = uniform_grid(h), self.K
        f, _ = locations(g)
        A = np.concatenate([gauss(c, *f[c]) for c in 'xyz'])
        rhs = np.concatenate([-lap_gauss(c, *f[c]) + k * k * gauss(c, *f[c]) for c in 'xyz'])
        return np.abs(spsolve(g.build_L_k(k).tocsc(), rhs) - A).max()

    def test_curl(self):
        e = [self.curl_error(h) for h in (0.4, 0.2, 0.1)]
        self.assertLess(e[-1], 5e-3)
        self.assertGreater(e[0] / e[1], 3.5)
        self.assertGreater(e[1] / e[2], 3.5)

    def test_helmholtz(self):
        e = [self.helmholtz_error(h) for h in (0.4, 0.2, 0.1)]
        self.assertLess(e[-1], 5e-3)
        self.assertGreater(e[0] / e[1], 3.5)
        self.assertGreater(e[1] / e[2], 3.5)

    def test_helmholtz_padded_grid(self):
        # non-uniform (padded) grid: the error stays small
        h, k = 0.2, self.K
        x = np.arange(-4.0, 4.0 + h / 2, h)
        pad = np.cumsum(calc_padding(h, 12))
        x = np.r_[x[0] - pad[::-1], x, x[-1] + pad]
        g = Grid25FV((x, x.copy()))
        f, _ = locations(g)
        A = np.concatenate([gauss(c, *f[c]) for c in 'xyz'])
        rhs = np.concatenate([-lap_gauss(c, *f[c]) + k * k * gauss(c, *f[c]) for c in 'xyz'])
        self.assertLess(np.abs(spsolve(g.build_L_k(k).tocsc(), rhs) - A).max(), 5e-3)


if __name__ == '__main__':
    unittest.main()
