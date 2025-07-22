# -*- coding: utf-8 -*-
"""
Legendre (and Associated) Polynomials of the boost library

see

https://www.boost.org/doc/libs/latest/libs/math/doc/html/math_toolkit/sf_poly/legendre.html

for documentation

"""

# distutils: language = c++

import numpy as np
cimport numpy as np

def p(n, x):
    """Legendre Polynomial of the first kind.

    Parameters
    ----------
    n: int
        order
    x: double
        evaluation value (must be -1 <= x <= 1)

    Returns
    -------
    double
    """

    out = np.empty((n.size, x.size))
    for i in np.arange(n.size):
        for j in np.arange(x.size):
            out[i, j] = legendre_p(n[i], x[j])
    
    return out

def p_prime(n, x):
    """Derivative of Legendre Polynomial.

    Parameters
    ----------
    n: int
        order
    x: double
        evaluation value (must be -1 <= x <= 1)

    Returns
    -------
    double

    """

    out = np.empty((n.size, x.size))
    for i in np.arange(n.size):
        for j in np.arange(x.size):
            out[i, j] = legendre_p_prime(n[i], x[j])
    
    return out