import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/Users/giroux/GitHub/pymmr')
sys.path.append('/Users/giroux/GitHub/pymumps')

from pymmr.finite_volume import calc_padding, GridFV
from pymmr.dc import GridDC
from pymmr.inversion import Inversion, DataERT

os.chdir('/Users/giroux/GitHub/pymmr/examples')


# %%
x = np.arange(-60.0*16, 60.01*16, 60)
y = np.arange(-60.0*16, 60.01*16, 60)
z = np.arange(-60.0*16, 0.01, 60)

pad = np.cumsum(calc_padding(60.0, 10))

x = np.r_[x[0] - pad[::-1], x, x[-1] + pad]
y = np.r_[y[0] - pad[::-1], y, y[-1] + pad]
z = np.r_[z[0] - pad[::-1], z]

g = GridDC((x, y, z))
g.set_roi([-400, 400, -400, 400, -960, 0])
g.fv.set_solver('mumps')

sigma = 0.001 + np.zeros((g.nc,))

ix = np.where(np.logical_and(g.fv.xc > -200, g.fv.xc < 200))[0]
iy = np.where(np.logical_and(g.fv.yc > -200, g.fv.yc < 200))[0]
iz = np.where(np.logical_and(g.fv.zc <  -80, g.fv.zc > -480))[0]
sigma[g.fv.ind(ix, iy, iz)] = 0.1

c1c2 = np.array([[-600.0, 0.0, 0.0, 600.0, 0.0, 0.0],
                 [0.0, -600.0, 0.0, 0.0, 600.0, 0.0]])
X = np.linspace(-400, 400, 26)
Y = np.linspace(-400, 400, 26)
z_bh = np.linspace(-600, -20, 10)

p1p2 = []
for y in Y:
    for nx in range(len(X)-1):
        p1p2.append((X[nx], y, 0.0, X[nx+1], y, 0.0))
    for nx in range(len(X)-2):
        p1p2.append((X[nx], y, 0.0, X[nx+2], y, 0.0))
for x in X:
    for ny in range(len(Y)-1):
        p1p2.append((x, Y[ny], 0.0, x, Y[ny+1], 0.0))
    for ny in range(len(Y)-2):
        p1p2.append((x, Y[ny], 0.0, x, Y[ny+2], 0.0))

p1p2_surf = np.array(p1p2)

X = (-300.0, 300.0)
# downhole
for x in X:
    for y in X:
        for nz in range(len(z_bh)-1):
            p1p2.append((x, y, z_bh[nz], x, y, z_bh[nz+1]))
        for nz in range(len(z_bh)-2):
            p1p2.append((x, y, z_bh[nz], x, y, z_bh[nz+2]))
# crosshole
for nz in range(len(z_bh)):
    p1p2.append((X[0], 0.0, z_bh[nz], X[1], 0.0, z_bh[nz]))
for nz in range(len(z_bh)):
    p1p2.append((0.0, X[0], z_bh[nz], 0.0, X[1], z_bh[nz]))

p1p2 = np.array(p1p2)

tmp = np.kron(c1c2, np.ones((p1p2.shape[0], 1)))
tmp_surf = np.kron(c1c2, np.ones((p1p2_surf.shape[0], 1)))
p1p2 = np.kron(np.ones((c1c2.shape[0], 1)), p1p2)
p1p2_surf = np.kron(np.ones((c1c2.shape[0], 1)), p1p2_surf)
c1c2 = tmp
c1c2_surf = tmp_surf

# on enlève quelques points
mask = np.ones((c1c2.shape[0],), dtype=bool)
mask[50:60] = False
mask[2000:2100] = False
mask[-30:] = False
c1c2 = c1c2[mask, :]
p1p2 = p1p2[mask, :]

# on change l'ordre
ind = np.arange(c1c2.shape[0])
ind[9] = 2220
ind[2220] = 9
ind[1802] = 14
ind[14] = 1802
c1c2 = c1c2[ind, :]
p1p2 = p1p2[ind, :]

g.set_survey_ert(c1c2, p1p2, 1.0)
data, J = g.fwd_mod(sigma, calc_J=True)

g.save_current_density(J, 'example_dc_J')

m_ref = 0.001 + np.zeros((g.nc,))

data_ert = DataERT(c1c2=c1c2, p1p2=p1p2, data=data, wt=np.ones((data.shape[0],)), cs=1.0, date=None)

filename = "data_dc.dat"
header = "c1_x c1_y c1_z c2_x c2_y c2_z p1_x p1_y p1_z p2_x p2_y p2_z V cs"
np.savetxt(filename, np.c_[c1c2, p1p2, data, g.cs], header=header, fmt="%g")

g.set_survey_ert(c1c2_surf, p1p2_surf, 1.0)
data_surf = g.fwd_mod(sigma)

filename = "data_dc_surf.dat"
header = "c1_x c1_y c1_z c2_x c2_y c2_z p1_x p1_y p1_z p2_x p2_y p2_z V cs"
np.savetxt(filename, np.c_[c1c2_surf, p1p2_surf, data_surf, g.cs], header=header, fmt="%g")


# %%

inv = Inversion()
inv.max_it = 5
inv.beta = 1
inv.beta_min = 0.01
inv.beta_dw = 1.0
#inv.smooth_type = 'blocky'
inv.show_plots = True

g.verbose = False
results = inv.run(g, m_ref, data_ert=data_ert, m_active=g.ind_roi)

sigma_inv, data_inv, rms, misfit, smy = results
g.verbose = True

x, y, z = g.get_roi_nodes()
g2 = GridFV(x, y, z)

fields = {}

for n, sigma in enumerate(sigma_inv):
    name = 'dc - iteration {0:d}'.format(n+1)
    fields[name] = sigma

g2.toVTK(fields, 'example_dc')


# %%

plt.figure()
plt.bar(np.arange(1, 1+len(rms)), rms)
plt.xlabel('Iteration')
plt.title('Misfit')
plt.show()
