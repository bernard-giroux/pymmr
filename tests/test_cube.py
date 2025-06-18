#!/usr/bin/env python3
"""
Test mmr forward modeling code by computing response of a cube, as in Chen et al 2002
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("/Users/giroux/GitHub/pymmr")
sys.path.append("/Users/giroux/GitHub/pymumps")

from pymmr.finite_volume import calc_padding
from pymmr.mmr import GridMMR, normal_field

os.chdir('/Users/giroux/GitHub/pymmr/tests')

# %%

x = np.arange(-60.0 * 16, 60.01 * 16, 60)
y = np.arange(-60.0 * 16, 60.01 * 16, 60)
z = np.arange(-60.0 * 16, 0.01, 60)

pad = np.cumsum(calc_padding(60.0, 10))

x = np.r_[x[0] - pad[::-1], x, x[-1] + pad]
y = np.r_[y[0] - pad[::-1], y, y[-1] + pad]
z = np.r_[z[0] - pad, z]

g = GridMMR((x, y, z))

c1c2 = np.array(
    [[-600.0, 0.0, 0.0, 600.0, 0.0, 0.0]]
)

nx = 28
ny = 26

xo = np.linspace(-400, 400, nx)
yo = np.linspace(-300, 300, ny)
xo = np.c_[
    np.kron(xo.reshape(-1, 1), np.ones((yo.size, 1))),
    np.kron(np.ones((xo.size, 1)), yo.reshape(-1, 1)),
    np.zeros((xo.size * yo.size, 1)),
]

sigma = 0.001 + np.zeros((g.dc.nc,))

ix = np.where(np.logical_and(g.fv.xc > -200, g.fv.xc < 200))[0]
iy = np.where(np.logical_and(g.fv.yc > -200, g.fv.yc < 200))[0]
iz = np.where(np.logical_and(g.dc.fv.zc < -80, g.dc.fv.zc > -480))[0]
sigma[g.dc.fv.ind(ix, iy, iz)] = 0.1

Bn = normal_field(c1c2, xo)

g.verbose = True
g.set_solver("mumps")
g.set_survey_mmr(xs=c1c2, xo=xo, cs=1.0)
g.apply_bc = True

B = g.fwd_mod(sigma)

# %%

Bx = B[:, 0] - Bn[0]
By = B[:, 1] - Bn[1]
Bz = B[:, 2]

Bx = Bx.reshape((nx, ny)).T
By = By.reshape((nx, ny)).T
Bz = Bz.reshape((nx, ny)).T

# %%
xo = np.linspace(-400, 400, nx)

plt.subplot(131)
plt.pcolormesh(xo, yo, Bx)
plt.gca().set_aspect('equal', 'box')
plt.colorbar()

plt.subplot(132)
plt.pcolormesh(xo, yo, By)
plt.gca().set_aspect('equal', 'box')
plt.colorbar()

plt.subplot(133)
plt.pcolormesh(xo, yo, Bz)
plt.gca().set_aspect('equal', 'box')
plt.colorbar()

plt.tight_layout()
plt.show()
