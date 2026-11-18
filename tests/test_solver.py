"""Solver: starting guess of the iterative solvers, parallel threads, AMG preconditioner."""
import contextlib
import io
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import bicgstab

from pymmr import finite_volume
from pymmr.finite_volume import Solver, calc_padding, has_mumps, has_pyamg, is_free_threading, parallel_status
from pymmr.mmr import GridMMR


def system(n=400, nrhs=6, seed=0):
    rng = np.random.default_rng(seed)
    A = sp.diags_array([-np.ones(n - 1), 4 * np.ones(n), -np.ones(n - 1)], offsets=[-1, 0, 1], format='csr')
    return A, rng.standard_normal((n, nrhs))


class TestStartingGuess(unittest.TestCase):

    def check(self, n_threads, do_perm=False):
        A, b = system()
        x_ref = sp.linalg.spsolve(A.tocsc(), b)
        slv = Solver((bicgstab, 1e-12, 1e-12, 1000, '0', do_perm), A, n_threads=n_threads)
        np.testing.assert_allclose(slv.solve(b), x_ref, rtol=1e-8, atol=1e-10)
        # each right-hand side must start from its own guess: limited to 1 iteration, bicgstab returns the
        # exact solutions only if it starts from them (from 0 or from the guess of another rhs, it is far off)
        slv1 = Solver((bicgstab, 1e-12, 1e-12, 1, '0', do_perm), A, n_threads=n_threads)
        self.assertGreater(np.abs(slv1.solve(b) - x_ref).max(), 1e-3)
        np.testing.assert_allclose(slv1.solve(b, x0=x_ref), x_ref, rtol=1e-8, atol=1e-10)

    def test_sequential(self):
        self.check(1)

    def test_permutation(self):
        self.check(1, do_perm=True)

    @unittest.skipUnless(is_free_threading(), "needs a free-threaded Python")
    def test_threads(self):
        self.check(3)
        # many short solves on 8 threads: with a starting guess shared between threads (self.x0 before the
        # fix), about 7% of the right-hand sides started from the guess of another one
        A, b = system(n=4000, nrhs=400)
        x_ref = sp.linalg.spsolve(A.tocsc(), b)
        slv = Solver((bicgstab, 1e-12, 1e-12, 1, '0', False), A, n_threads=8)
        for _ in range(3):
            np.testing.assert_allclose(slv.solve(b, x0=x_ref), x_ref, rtol=1e-8, atol=1e-10)

    def test_direct_solver_ignores_x0(self):
        A, b = system()
        slv = Solver(('superlu',), A)
        x = slv.solve(b, x0=np.zeros(b.shape))
        np.testing.assert_allclose(A @ x, b, atol=1e-10)


def decoupled_system(n=30, nrhs=4, seed=0):
    """Three uncoupled 2D Laplacians on stretched grids, not symmetric, with round-off between blocks."""
    rng = np.random.default_rng(seed)
    h = 1.1 ** np.abs(np.arange(n) - n / 2)
    D = sp.diags_array([-1 / h[:-1], 1 / h[:-1]], offsets=[0, 1], shape=(n - 1, n), format='csr')
    L = D.T @ D + 1e-3 * sp.eye_array(n)
    A2 = sp.kronsum(L, L, format='csr') @ sp.diags_array(1 + 0.2 * np.sin(np.arange(n * n)))
    A = sp.block_array([[A2 * s if i == j else None for j in range(3)] for i, s in enumerate((1.0, 2.0, 5.0))],
                       format='csr')
    E = sp.random_array(A.shape, density=1e-3, rng=rng, format='csr')
    return (A + 1e-15 * E).tocsr(), rng.standard_normal((A.shape[0], nrhs))


def n_iterations(A, b, M):
    count = [0]
    def callback(_):
        count[0] += 1
    x, info = bicgstab(A, b, rtol=1e-8, atol=0.0, maxiter=5000, M=M, callback=callback)
    assert info == 0
    return count[0], x


@unittest.skipUnless(has_pyamg, "needs pyamg")
class TestAMG(unittest.TestCase):

    def test_solution_and_iterations(self):
        A, b = decoupled_system()
        amg = Solver((bicgstab, 0.0, 1e-8, 1000, 'amg', False), A)
        diag = Solver((bicgstab, 0.0, 1e-8, 5000, 'diag', False), A)
        n_amg, x = n_iterations(A, b[:, 0], amg.Mpre)
        n_diag, _ = n_iterations(A, b[:, 0], diag.Mpre)
        self.assertLessEqual(n_amg, 10)
        self.assertGreater(n_diag, 5 * n_amg)
        np.testing.assert_allclose(A @ x, b[:, 0], atol=1e-6)
        x = amg.solve(b)
        self.assertLess((np.linalg.norm(A @ x - b, axis=0) / np.linalg.norm(b, axis=0)).max(), 1e-7)

    def test_rebuilt_with_matrix(self):
        A, b = decoupled_system()
        slv = Solver((bicgstab, 0.0, 1e-8, 1000, 'amg', False), A)
        slv.A = 3.0 * A
        self.assertLessEqual(n_iterations(3.0 * A, b[:, 0], slv.Mpre)[0], 10)

    def test_permutation(self):
        A, b = decoupled_system()
        x = Solver((bicgstab, 0.0, 1e-8, 1000, 'amg', True), A).solve(b)
        self.assertLess((np.linalg.norm(A @ x - b, axis=0) / np.linalg.norm(b, axis=0)).max(), 1e-7)

    @unittest.skipUnless(is_free_threading(), "needs a free-threaded Python")
    def test_threads(self):
        A, _ = decoupled_system()
        b = np.random.default_rng(1).standard_normal((A.shape[0], 40))
        x = Solver((bicgstab, 0.0, 1e-10, 1000, 'amg', False), A, n_threads=8).solve(b)
        self.assertLess((np.linalg.norm(A @ x - b, axis=0) / np.linalg.norm(b, axis=0)).max(), 1e-9)

    def test_mmr_against_direct(self):
        x = np.arange(-400.0, 400.1, 100.0)
        z = np.arange(-400.0, 0.1, 100.0)
        pad = np.cumsum(calc_padding(100.0, 4))
        x = np.r_[x[0] - pad[::-1], x, x[-1] + pad]
        z = np.r_[z[0] - pad[::-1], z]
        xs = np.array([[-200.0, 0, 0, 200.0, 0, 0], [0, -200.0, 0, 0, 200.0, 0]])
        xo = np.array([[50.0, 50.0, 0.0], [-100.0, 150.0, 0.0]])
        data = []
        for par in (('mumps' if has_mumps else 'superlu',), (bicgstab, 1e-14, 1e-9, 1000, 'amg', False)):
            g = GridMMR((x, x.copy(), z), n_threads=2)
            g.set_solver(par[0], *par[1:])
            g.set_survey_mmr(xs=xs, xo=xo, cs=1.0)
            data.append(g.fwd_mod(1e-3 * np.ones(g.dc.nc)))
            self.assertEqual(g.fv.solver_A.n_threads, 2)
        np.testing.assert_allclose(data[1], data[0], rtol=1e-5, atol=1e-5 * np.abs(data[0]).max())


class TestParallelReport(unittest.TestCase):

    def test_amg_needs_pyamg(self):
        A, _ = system()
        with mock.patch.object(finite_volume, 'has_pyamg', False):
            with self.assertRaisesRegex(RuntimeError, 'pyamg'):
                Solver((bicgstab, 1e-9, 1e-5, 100, 'amg', False), A)

    def test_status(self):
        with mock.patch.object(finite_volume, 'is_free_threading', return_value=True):
            self.assertIn('in parallel', parallel_status(4))
        with mock.patch.object(finite_volume, 'is_free_threading', return_value=False):
            with mock.patch('sysconfig.get_config_var', return_value=1):
                self.assertIn('PYTHON_GIL=0', parallel_status(4))
            with mock.patch('sysconfig.get_config_var', return_value=0):
                self.assertIn('not a free-threaded build', parallel_status(4))

    def report(self, solver):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            solver.print_solve_report()
        return out.getvalue()

    def test_report(self):
        A, b = system()
        slv = Solver((bicgstab, 1e-9, 1e-5, 1000, 'diag', False), A, n_threads=3)
        self.assertEqual(self.report(slv), '')   # nothing solved yet
        slv.solve(b)
        expected = 'Parallel solve: 6 right-hand sides on 3 threads' if is_free_threading() else 'Sequential solve'
        self.assertIn(expected, self.report(slv))
        self.assertEqual(slv.last_solve['n_workers'], 3 if is_free_threading() else 1)
        slv = Solver((bicgstab, 1e-9, 1e-5, 1000, 'diag', False), A)
        slv.solve(b)
        self.assertEqual(self.report(slv), '')   # one thread requested

    def test_flags_threads_not_concurrent(self):
        A, _ = system()
        slv = Solver((bicgstab, 1e-9, 1e-5, 1000, 'diag', False), A, n_threads=4)
        slv.last_solve = {'n_rhs': 8, 'n_threads': 4, 'n_workers': 4, 'wall': 10.0, 'cpu': 10.5}
        self.assertIn('WARNING', self.report(slv))

    @unittest.skipUnless(is_free_threading(), "needs a free-threaded Python")
    def test_error_in_thread_is_raised(self):
        A, b = system()
        slv = Solver((bicgstab, 1e-9, 1e-5, 1000, '0', False), A, n_threads=2)
        slv.solver = mock.Mock(side_effect=ValueError('in thread'))
        with self.assertRaisesRegex(ValueError, 'in thread'):
            slv.solve(b)


if __name__ == '__main__':
    unittest.main()
