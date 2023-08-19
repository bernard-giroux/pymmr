#!/usr/bin/env python3
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/Users/giroux/GitHub/pymmr')
sys.path.append('/Users/giroux/GitHub/pymumps')

from pymmr.finite_volume import calc_padding, GridFV
from pymmr.mmr import GridMMR
from pymmr.inversion import Inversion, DataMMR, DataERT


x = np.arange(-60.0*16, 60.01*16, 60)
y = np.arange(-60.0*16, 60.01*16, 60)
z = np.arange(0.0, 60.01*16, 60)

pad = np.cumsum(calc_padding(60.0, 10))

x = np.r_[x[0] - pad[::-1], x, x[-1] + pad]
y = np.r_[y[0] - pad[::-1], y, y[-1] + pad]
z = np.r_[z, z[-1] + pad]

g = GridMMR(x, y, z)
g.set_solver('mumps')
g.verbose = False

tmp = np.loadtxt('data_mmr_j.dat')
data_mmr = DataMMR(xs=tmp[:, 3:9], xo=tmp[:, 9:12], data=tmp[:, :3], wt=np.ones((3*tmp.shape[0],)))

tmp = np.loadtxt('data_ert_j.dat')
data_ert = DataERT(c1c2=tmp[:, :6], p1p2=tmp[:, 6:12], data=1000*tmp[:, 12], wt=np.ones((tmp.shape[0],)))

# %%
inv = Inversion()
inv.max_it = 5
inv.beta = 2500
inv.beta_min = 100
inv.show_plots = True

m_ref = 0.001 + np.zeros((g.gdc.nc,))

g.set_roi([-400, 400, -400, 400, 0, 960])

results = inv.run(g, data_mmr=data_mmr, m_ref=m_ref, data_ert=data_ert, m_active=g.ind_roi)

S_save, data_inv, rms = results

x, y, z = g.gdc.get_roi_nodes()
g2 = GridFV(x, y, z)

fields = {}
for i in range(len(S_save)):
    name = 'iteration {0:d}'.format(i+1)
    fields[name] = S_save[i]

g2.toVTK(fields, 'example_joint')


# %%

plt.figure()
plt.bar(np.arange(1, 1+len(rms)), rms)
plt.xlabel('Iteration')
plt.title('Misfit')
plt.show()
