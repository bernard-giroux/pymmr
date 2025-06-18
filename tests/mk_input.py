#!/usr/bin/env python3
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("/Users/giroux/GitHub/pymmr")
sys.path.append("/Users/giroux/GitHub/pymumps")

from pymmr.finite_volume import calc_padding
from pymmr.mmr import GridMMR

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
    [[-600.0, 0.0, 0.0, 600.0, 0.0, 0.0], [0.0, -600.0, 0.0, 0.0, 600.0, 0.0]]
)

xo = np.linspace(-400, 400, 26)
yo = np.linspace(-400, 400, 26)
xo = np.c_[
    np.kron(xo.reshape(-1, 1), np.ones((yo.size, 1))),
    np.kron(np.ones((xo.size, 1)), yo.reshape(-1, 1)),
    np.zeros((xo.size * yo.size, 1)),
]

tmp = np.kron(c1c2, np.ones((xo.shape[0], 1)))
xo = np.kron(np.ones((c1c2.shape[0], 1)), xo)
c1c2 = tmp

# remove some points
mask = np.ones((xo.shape[0],), dtype=bool)
mask[5] = False
mask[200] = False
mask[-1] = False
c1c2 = c1c2[mask, :]
xo = xo[mask, :]

# change order
ind = np.arange(xo.shape[0])
ind[7] = 322
ind[322] = 7
ind[500] = 14
ind[14] = 500
c1c2 = c1c2[ind, :]
xo = xo[ind, :]

np.savetxt('c1c2.dat', c1c2)
np.savetxt('xo.dat', xo)

sigma = 0.001 + np.zeros((g.dc.nc,))

ix = np.where(np.logical_and(g.fv.xc > -200, g.fv.xc < 200))[0]
iy = np.where(np.logical_and(g.fv.yc > -200, g.fv.yc < 200))[0]
iz = np.where(np.logical_and(g.dc.fv.zc < -80, g.dc.fv.zc > -480))[0]
sigma[g.dc.fv.ind(ix, iy, iz)] = 0.1

g.dc.fv.toVTK({'Conductivity': sigma}, 'm_fwd')
