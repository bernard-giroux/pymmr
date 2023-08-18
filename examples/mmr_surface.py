#!/usr/bin/env python3
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/Users/giroux/GitHub/pymmr')
sys.path.append('/Users/giroux/GitHub/pymumps')

from pymmr.finite_volume import calc_padding, GridFV
from pymmr.mmr import GridMMR
from pymmr.inversion import Inversion


x = np.arange(-60.0*16, 60.01*16, 60)
y = np.arange(-60.0*16, 60.01*16, 60)
z = np.arange(0.0, 60.01*16, 60)

pad = np.cumsum(calc_padding(60.0, 10))

x = np.r_[x[0] - pad[::-1], x, x[-1] + pad]
y = np.r_[y[0] - pad[::-1], y, y[-1] + pad]
z = np.r_[z, z[-1] + pad]

g = GridMMR(x, y, z)
g.set_solver('mumps')


c1c2 = np.array([[-600.0, 0.0, 0.0, 600.0, 0.0, 0.0],
                 [0.0, -600.0, 0.0, 0.0, 600.0, 0.0]])

xo = np.linspace(-400, 400, 26)
yo = np.linspace(-400, 400, 26)
xo = np.c_[np.kron(xo.reshape(-1, 1), np.ones((yo.size, 1))),
           np.kron(np.ones((xo.size, 1)), yo.reshape(-1, 1)),
           np.zeros((xo.size*yo.size, 1))]

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

g.xs = c1c2
g.xo = xo

g.cs = 1.0
g.apply_bc = True

sigma = 0.001 + np.zeros((g.gdc.nc,))

ix = np.where(np.logical_and(g.xc>-200, g.xc<200))[0]
iy = np.where(np.logical_and(g.yc>-200, g.yc<200))[0]
iz = np.where(np.logical_and(g.gdc.zc>80, g.gdc.zc<480))[0]
sigma[g.gdc.ind(ix, iy, iz)] = 0.1

dobs1 = g.fwd_mod(sigma)

# %%

dobs = dobs1 + np.random.default_rng().normal(0.0, 0.03, dobs1.shape)

data = np.c_[dobs1, c1c2, xo]


# %%
inv = Inversion()
inv.beta = 2500
inv.beta_min = 100

m_ref = 0.001 + np.zeros((g.gdc.nc,))

g.set_roi([-400, 400, -400, 400, 0, 960])
g.verbose = False

results = inv.run(g, m_ref, data_mmr=data, m_active=g.ind_roi)

S_save, rms, data_inv = results

x, y, z = g.gdc.get_roi_nodes()
g2 = GridFV(x, y, z)

fields = {}
for i in range(len(S_save)):
    name = 'iteration {0:d}'.format(i+1)
    fields[name] = S_save[i]

g2.toVTK(fields, 'mmr_surface')
