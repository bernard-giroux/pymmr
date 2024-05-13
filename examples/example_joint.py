#!/usr/bin/env python3
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('/Users/giroux/GitHub/pymmr')
sys.path.append('/Users/giroux/GitHub/pymumps')

from pymmr.finite_volume import calc_padding, GridFV
from pymmr.mmr import GridMMR
from pymmr.inversion import Inversion, df_to_data, DataMMR, DataERT


os.chdir('/Users/giroux/GitHub/pymmr/examples')

x = np.arange(-60.0*16, 60.01*16, 60)
y = np.arange(-60.0*16, 60.01*16, 60)
z = np.arange(-60.0*16, 0.01, 60)

pad = np.cumsum(calc_padding(60.0, 10))

x = np.r_[x[0] - pad[::-1], x, x[-1] + pad]
y = np.r_[y[0] - pad[::-1], y, y[-1] + pad]
z = np.r_[z[0] - pad, z]

g = GridMMR((x, y, z))
g.set_solver('mumps')
g.verbose = False

# tmp = np.loadtxt('data_mmr_j.dat', comments='%')
# for col in (5, 11):
#     tmp[:, col] = -tmp[:, col]  # convert to elevation
# data_mmr = DataMMR(xs=tmp[:, 3:9], xo=tmp[:, 9:12], data=tmp[:, :3], wt=np.ones((3*tmp.shape[0],)),
#                    cs=np.ones((tmp.shape[0],)), date=None)
#
# tmp = np.loadtxt('data_ert_j.dat', comments='%')
# for col in (2, 5, 8, 11):
#     tmp[:, col] = -tmp[:, col]  # convert to elevation
# data_ert = DataERT(c1c2=tmp[:, :6], p1p2=tmp[:, 6:12], data=1000*tmp[:, 12], wt=np.ones((tmp.shape[0],)),
#                    cs=np.ones((tmp.shape[0],)), date=None)

df_mmr = pd.read_table("data_mmr.dat", sep="\\s+", escapechar="#")
df_mmr.columns = df_mmr.columns.str.replace(" ", "")  # remove spaces in keys
data_mmr = df_to_data(df_mmr)

df_ert = pd.read_table("data_dc_surf.dat", sep="\\s+", escapechar="#")
df_ert.columns = df_ert.columns.str.replace(" ", "")
data_ert = df_to_data(df_ert)

# %%
inv = Inversion()
inv.max_it = 5
inv.beta = 2500
inv.beta_min = 100
inv.show_plots = True

m_ref = 0.001 + np.zeros((g.dc.nc,))

g.set_roi([-400, 400, -400, 400, -960, 0])

results = inv.run(g, data_mmr=data_mmr, m_ref=m_ref, data_ert=data_ert, m_active=g.ind_roi)

sigma_inv, data_inv, rms, misfit, smy = results


x, y, z = g.dc.get_roi_nodes()
g2 = GridFV(x, y, z)

fields = {}
for n, sigma in enumerate(sigma_inv):
    name = 'joint - iteration {0:d}'.format(n+1)
    fields[name] = sigma

g2.toVTK(fields, 'example_joint')


# %%

plt.figure()
plt.bar(np.arange(1, 1+len(rms)), rms)
plt.xlabel('Iteration')
plt.title('Misfit')
plt.show()
