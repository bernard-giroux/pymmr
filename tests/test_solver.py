"""Solver: starting guess of the iterative solvers, solved in parallel threads."""
import unittest

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import bicgstab

from pymmr.finite_volume import Solver, is_free_threading


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


if __name__ == '__main__':
    unittest.main()
