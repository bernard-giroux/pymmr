#!/usr/bin/env python3
"""
Tests of the inverse cosine/sine transforms from fixed wavenumbers (pymmr.kquad, 2.5D MMR).

Run with:  python3 tests/test_kquad.py   (or pytest)
"""
import os
import sys
import unittest

import numpy as np
from scipy.special import k0, k1

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pymmr.kquad import KQuadrature, select_k  # noqa: E402

# analytic pairs f(rho, y) <-> F(k, rho), with the convention of docs/mmr25.md
PAIRS = {
    'even 1/r': ('even', lambda r, y: 1 / np.hypot(r, y), lambda k, r: 2 * k0(k * r)),
    'even 1/r^3': ('even', lambda r, y: 1 / np.hypot(r, y)**3, lambda k, r: 2 * k * k1(k * r) / r),
    'even r/(r^2+y^2)': ('even', lambda r, y: r / (r * r + y * y), lambda k, r: np.pi * np.exp(-k * r)),
    'odd y/r^3': ('odd', lambda r, y: y / np.hypot(r, y)**3, lambda k, r: 2 * k * k0(k * r)),
    'odd y/(r^2+y^2)': ('odd', lambda r, y: y / (r * r + y * y), lambda k, r: np.pi * np.exp(-k * r)),
}
RHO = np.array([0.5, 1.0, 3.0, 10.0, 30.0, 100.0])
DY = np.array([-50.0, 0.0, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0, 200.0])


def max_error(q, parity, f, F):
    """Largest error over RHO and DY, relative to the size of the field at each rho."""
    W = q.weights(DY, parity)
    err = 0.0
    for r in RHO:
        scale = abs(f(r, 0.0 if parity == 'even' else r))
        err = max(err, np.max(np.abs(W @ F(q.k, r) - f(r, DY))) / scale)
    return err


class TestKQuadrature(unittest.TestCase):

    def test_select_k(self):
        k = select_k(0.5, 300.0, 30)
        self.assertEqual(k.size, 30)
        self.assertAlmostEqual(k[0], 0.05 / 300.0)
        self.assertAlmostEqual(k[-1], 15.0 / 0.5)
        with self.assertRaises(ValueError):
            select_k(2.0, 1.0)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            KQuadrature([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            KQuadrature([1.0, 3.0, 2.0, 4.0])
        with self.assertRaises(ValueError):
            KQuadrature(select_k(1.0, 10.0)).weights(1.0, 'cos')

    def test_analytic_pairs(self):
        q = KQuadrature(select_k(0.5, 300.0, 30))
        for name, (parity, f, F) in PAIRS.items():
            with self.subTest(pair=name):
                self.assertLess(max_error(q, parity, f, F), 2e-3)

    def test_convergence(self):
        for name, (parity, f, F) in PAIRS.items():
            with self.subTest(pair=name):
                e20 = max_error(KQuadrature(select_k(0.5, 300.0, 20)), parity, f, F)
                e40 = max_error(KQuadrature(select_k(0.5, 300.0, 40)), parity, f, F)
                self.assertLess(e40, e20 / 10)

    def test_parity(self):
        q = KQuadrature(select_k(0.5, 300.0, 20))
        dy = np.array([0.7, 13.0, 150.0])
        np.testing.assert_array_equal(q.weights(-dy, 'even'), q.weights(dy, 'even'))
        np.testing.assert_array_equal(q.weights(-dy, 'odd'), -q.weights(dy, 'odd'))
        np.testing.assert_array_equal(q.weights(0.0, 'odd'), 0.0)

    def test_call(self):
        q = KQuadrature(select_k(0.5, 300.0, 20))
        F = np.stack([2 * k0(q.k * r) for r in RHO], axis=1)   # n_k x n_rho
        out = q(F, DY, 'even')
        self.assertEqual(out.shape, (DY.size, RHO.size))
        np.testing.assert_allclose(out, q.weights(DY, 'even') @ F)


if __name__ == '__main__':
    unittest.main()
