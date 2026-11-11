#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inverse cosine/sine transforms from a fixed set of wavenumbers (2.5D MMR).

A field even (odd) in y is represented by its cosine (sine) transform F(k), see docs/mmr25.md:

    f(dy) = 1/pi int_0^inf F(k) cos(k dy) dk      (even)
    f(dy) = 1/pi int_0^inf F(k) sin(k dy) dk      (odd)

F is known at fixed wavenumbers k_1 < ... < k_n (one 2.5D solve each).  It is interpolated by a cubic
spline in ln k, and the integrals of the spline are computed accurately for each dy, which gives
weights W(dy) such that

    f(dy) = sum_i W_i(dy) F(k_i).

The map is linear in the F(k_i), and its adjoint (for the sensitivity) is W^T.

Tails
-----
- k < k_1: even fields F = a + b ln k through (k_1, F_1), (k_2, F_2) (logarithmic singularity of the
  fields decaying as 1/r), integrated analytically; odd fields F = F_1 (the contribution is small,
  sin(k dy) ~ k dy).
- k > k_n: F = F_n exp(-(k - k_n) * decay), integrated analytically, so that the transform is not
  truncated abruptly (the oscillating tail of a truncated integral is of order F_n / dy).
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import sici

# Gauss-Legendre nodes and weights on [0, 1]
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)
_GL_X = 0.5 * (_GL_X + 1.0)
_GL_W = 0.5 * _GL_W


def select_k(r_min, r_max, n_k=30, c_min=0.05, c_max=15.0):
    """Log-spaced wavenumbers for distances between r_min and r_max.

    Parameters
    ----------
    r_min : float
        Smallest distance in the x-z plane for which the fields must be accurate (sets k_max = c_max / r_min).
    r_max : float
        Largest distance (sets k_min = c_min / r_max).
    n_k : int, optional
        Number of wavenumbers.

    Returns
    -------
    ndarray
    """
    if not 0 < r_min < r_max:
        raise ValueError('select_k: need 0 < r_min < r_max')
    return np.logspace(np.log10(c_min / r_max), np.log10(c_max / r_min), n_k)


class KQuadrature:
    """Weights of the inverse cosine and sine transforms for a fixed set of wavenumbers.

    Parameters
    ----------
    k : array_like
        Wavenumbers, strictly increasing and positive (at least 4).
    decay : float, optional
        Decay length of the exponential tail beyond k_n.  Default: 15 / k_n (the field at
        k_n is then e^-15 of its value at k = 0 for a source at the distance r_min of select_k).
    """

    def __init__(self, k, decay=None):
        k = np.asarray(k, dtype=float)
        if k.ndim != 1 or k.size < 4 or np.any(k <= 0) or np.any(np.diff(k) <= 0):
            raise ValueError('KQuadrature: k must be positive, strictly increasing, with 4 values or more')
        self.k = k
        self.u = np.log(k)
        self.decay = 15.0 / k[-1] if decay is None else float(decay)
        # cardinal cubic splines in ln k: spline(u)[..., i] is basis function i
        self._spline = CubicSpline(self.u, np.eye(k.size), bc_type='not-a-knot')

    @property
    def n_k(self):
        return self.k.size

    def weights(self, dy, parity):
        """Weights W(dy).

        Parameters
        ----------
        dy : float or array_like
            Distances along y between receiver and source (y_o - y_s), may be negative.
        parity : str
            'even' (cosine transform) or 'odd' (sine transform).

        Returns
        -------
        ndarray, n_dy x n_k
        """
        if parity not in ('even', 'odd'):
            raise ValueError("parity must be 'even' or 'odd'")
        dy = np.atleast_1d(np.asarray(dy, dtype=float))
        W = np.empty((dy.size, self.n_k))
        cache = {}
        for n, d in enumerate(dy):
            a = abs(d)
            if a not in cache:
                cache[a] = self._weights(a, parity)
            W[n] = cache[a]
            if parity == 'odd' and d < 0:
                W[n] = -W[n]
        return W

    def __call__(self, F, dy, parity):
        """Inverse transform of F (n_k x ...) at distances dy, returns n_dy x ..."""
        F = np.asarray(F)
        return np.tensordot(self.weights(dy, parity), F, axes=(1, 0))

    def _weights(self, dy, parity):
        """Weights for dy >= 0."""
        k, u = self.k, self.u
        w = np.zeros((self.n_k,))
        if parity == 'odd' and dy == 0.0:
            return w

        # --- k < k_1
        k1 = k[0]
        x = k1 * dy
        if parity == 'even':
            # int_0^k1 cos(k dy) dk = k1 sinc, int_0^k1 ln k cos(k dy) dk = k1 (ln k1 sinc - Si(x) / x)
            if x > 0:
                s = np.sin(x) / x
                si_x = sici(x)[0] / x
            else:
                s = si_x = 1.0
            I0 = k1 * s
            I1 = k1 * (u[0] * s - si_x)
            du = u[1] - u[0]
            # F = F1 + (F2 - F1) (ln k - u1) / du
            c = (I1 - u[0] * I0) / du
            w[0] += I0 - c
            w[1] += c
        else:
            # int_0^k1 sin(k dy) dk = (1 - cos x) / dy = k1 * 2 sin^2(x/2) / x
            w[0] += k1 * 2.0 * np.sin(0.5 * x)**2 / x

        # --- k_1 < k < k_n: composite Gauss-Legendre, at most a quarter of period per sub-interval
        trig = np.cos if parity == 'even' else np.sin
        edges = [k[0]]
        for i in range(self.n_k - 1):
            m = max(1, int(np.ceil((k[i + 1] - k[i]) * dy / (0.5 * np.pi))))
            edges.append(np.linspace(k[i], k[i + 1], m + 1)[1:])
        edges = np.hstack(edges)
        h = np.diff(edges)
        kq = (edges[:-1, None] + h[:, None] * _GL_X[None, :]).ravel()
        wq = (h[:, None] * _GL_W[None, :]).ravel()
        B = self._spline(np.log(kq))         # nq x n_k
        w += (wq * trig(kq * dy)) @ B

        # --- k > k_n: F = F_n exp(-(k - k_n) / L)
        alpha = 1.0 / self.decay
        kn = k[-1]
        den = alpha * alpha + dy * dy
        if parity == 'even':
            w[-1] += (alpha * np.cos(kn * dy) - dy * np.sin(kn * dy)) / den
        else:
            w[-1] += (alpha * np.sin(kn * dy) + dy * np.cos(kn * dy)) / den

        return w / np.pi
