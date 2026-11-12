#!/usr/bin/env python3
"""Tests of inversion.cglscd (preconditioned CG for the Gauss-Newton step)."""
import os
import sys
import unittest
import warnings

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pymmr.finite_volume import calc_padding, has_mumps  # noqa: E402
from pymmr.inversion import DataMMR, Inversion, RegPreconditioner, cglscd  # noqa: E402

SOLVER = 'mumps' if has_mumps else 'superlu'


def problem(nd=30, nm=400, als=1e-2, seed=0):
    """Ill-conditioned system: smooth sensitivities, 1D smoothing + smallness, data weights over 3 decades."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, nm)
    J = np.array([np.exp(-((t - c) / w) ** 2) for c, w in zip(rng.uniform(0, 1, nd), rng.uniform(0.05, 0.3, nd))])
    J *= 10.0 ** rng.uniform(-2, 2, (nd, 1))
    D = sp.diags_array(10.0 ** rng.uniform(-3, 0, nd), format='csr')
    G = sp.diags_array([-np.ones(nm - 1), np.ones(nm - 1)], offsets=[0, 1], shape=(nm - 1, nm))
    C = sp.csr_array(G.T @ G + als * sp.eye_array(nm))
    return J, D, C, rng.standard_normal(nd), 0.1 * rng.standard_normal(nm), rng


def exact(J, D, C, b, beta, dxc, reg_var, alpha=0.0, WTWt=None):
    DJ = D @ J
    H = DJ.T @ DJ + beta * C.toarray()
    g = DJ.T @ (D @ b)
    if reg_var == 'model':
        g -= beta * (C @ dxc)
    elif reg_var == 'time-lapse':
        g -= alpha * (WTWt @ dxc)
    return np.linalg.solve(H, g)


class TestCglscd(unittest.TestCase):

    def check(self, reg_var, als=1e-2, precond=True, tol=1e-11):
        J, D, C, b, dxc, rng = problem(als=als)
        beta, alpha = 0.5, 2.0
        WTWt = sp.diags_array(rng.uniform(0.5, 1.5, C.shape[0]))
        pre = RegPreconditioner(C) if precond else None
        x, err, it = cglscd(J, np.zeros(C.shape[0]), b, beta, C, dxc, D, 20000, tol, reg_var,
                            alpha=alpha, WTWt=WTWt, precond=pre)
        xe = exact(J, D, C, b, beta, dxc, reg_var, alpha, WTWt)
        self.assertLess(np.linalg.norm(x.ravel() - xe) / np.linalg.norm(xe), 1e5 * tol, msg=reg_var)
        self.assertLessEqual(err, tol)
        return it

    def test_exact_solution(self):
        for reg_var in ('model', 'model perturbation', 'time-lapse'):
            for precond in (True, False):
                self.check(reg_var, precond=precond)

    def test_iterations(self):
        # (beta C)^-1 (J^T D^2 J + beta C) = I + rank 30: n_data + 1 iterations in exact arithmetic
        it_reg = self.check('model', tol=1e-9)
        it_jac = self.check('model', precond=False, tol=1e-9)
        self.assertLessEqual(it_reg, 2 * 30)
        self.assertGreater(it_jac, 5 * it_reg)

    def test_singular_regularization(self):
        # no smallness term: C is singular, the shift of RegPreconditioner keeps it usable
        self.assertLessEqual(self.check('model perturbation', als=0.0, tol=1e-9), 2 * 30)

    def test_default_max_it(self):
        J, D, C, b, dxc, _ = problem()
        with warnings.catch_warnings():
            warnings.simplefilter('error')    # no convergence warning
            _, err, it = cglscd(J, np.zeros(C.shape[0]), b, 0.5, C, dxc, D, None, 1e-9, 'model',
                                precond=RegPreconditioner(C))
        self.assertLessEqual(it, 2 * J.shape[0] + 100)
        with self.assertWarns(UserWarning):
            _, _, it = cglscd(J, np.zeros(C.shape[0]), b, 0.5, C, dxc, D, 10, 1e-9, 'model')
        self.assertEqual(it, 10)

    def test_fixed_cells(self):
        J, D, C, b, dxc, _ = problem()
        P = np.ones(C.shape[0])
        P[100:150] = 0.0
        for pre in (RegPreconditioner(C), None):
            x, _, _ = cglscd(J, np.ones(C.shape[0]), b, 0.5, C, dxc, D, 2000, 1e-9, 'model', P=P, precond=pre)
            self.assertEqual(np.abs(x[100:150]).max(), 0.0)

    def test_direct_solver(self):
        # a grid with an iterative solver: C is factorized with the preferred direct solver, not SuperLU
        from scipy.sparse.linalg import bicgstab
        J, D, C, b, dxc, _ = problem()
        name = RegPreconditioner.direct_solver()
        if has_mumps:
            self.assertEqual(name, 'mumps')
        pre = RegPreconditioner(C, (bicgstab, 1e-9, 1e-5, 1000, '0', False))
        self.assertFalse(callable(getattr(pre.solver, 'it_slv', None)))
        if name == 'mumps':
            self.assertIsNotNone(pre.solver.ctx)
        r = np.random.default_rng(1).standard_normal((C.shape[0], 1))
        Cs = C + RegPreconditioner.SHIFT * C.diagonal().max() * sp.eye_array(C.shape[0])
        np.testing.assert_allclose(Cs @ pre.solve(r, 2.0), r / 2.0, rtol=1e-8, atol=1e-10)

    def test_bad_reg_var(self):
        J, D, C, b, dxc, _ = problem()
        with self.assertRaises(ValueError):
            cglscd(J, np.zeros(C.shape[0]), b, 0.5, C, dxc, D, 10, 1e-9, 'foo')


class TestInversion(unittest.TestCase):
    """2.5D MMR inversion: same steps with both preconditioners when cglscd converges."""

    def run_inv(self, **opts):
        from pymmr.mmr import GridMMR
        h = 10.0
        pad = np.cumsum(calc_padding(h, 8))
        x = np.arange(-100.0, 100.0 + h / 2, h)
        x = np.r_[x[0] - pad[::-1], x, x[-1] + pad]
        z = np.arange(-60.0, h / 2, h)
        z = np.r_[z[0] - pad[::-1], z]
        X, Z = np.meshgrid(0.5 * (x[1:] + x[:-1]), 0.5 * (z[1:] + z[:-1]))
        s_true = 0.01 * np.ones(X.shape)
        s_true[(np.abs(X) < 20) & (Z < -10) & (Z > -30)] = 0.1
        rec = [[a, b, 0.0] for b in (0.0, 20.0) for a in np.arange(-40.0, 41.0, 10.0)]
        xs = np.array([[-80.0, 0, 0, 80, 0, 0]] * len(rec))
        xo = np.array(rec)
        g = GridMMR((x, z))
        g.set_solver(SOLVER)
        g.set_survey_mmr(xs, xo, 1.0)
        data = g.fwd_mod(s_true.ravel())
        dm = DataMMR(xs=xs, xo=xo, data=data, wt=np.ones(3 * data.shape[0]), cs=np.ones(data.shape[0]), date=None)
        g = GridMMR((x, z))
        g.set_solver(SOLVER)
        inv = Inversion()
        inv.max_it = 2
        inv.verbose = False
        for k, v in opts.items():
            setattr(inv, k, v)
        m0 = 0.01 * np.ones(g.dc.fv.nc)
        return inv.run(g, m0=m0, m_ref=m0, data_mmr=dm)[0][-1]

    def test_same_model(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)    # cglscd converges with the default max_it
            m_reg = self.run_inv()
        m_jac = self.run_inv(precond_cglscd='jacobi', max_it_cglscd=50000)
        np.testing.assert_allclose(np.log(m_reg), np.log(m_jac), atol=1e-5)

    def test_bad_preconditioner(self):
        with self.assertRaises(ValueError):
            self.run_inv(precond_cglscd='ilu')


if __name__ == '__main__':
    unittest.main()
