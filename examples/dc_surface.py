import sys
import numpy as np

sys.path.append('/Users/giroux/GitHub/pymmr')
sys.path.append('/Users/giroux/GitHub/pymumps')

from pymmr.finite_volume import calc_padding, GridFV
from pymmr.dc import GridDC
from pymmr.inversion import Inversion, DataERT

x = np.arange(-60.0*16, 60.01*16, 60)
y = np.arange(-60.0*16, 60.01*16, 60)
z = np.arange(0.0, 60.01*16, 60)

pad = np.cumsum(calc_padding(60.0, 10))

x = np.r_[x[0] - pad[::-1], x, x[-1] + pad]
y = np.r_[y[0] - pad[::-1], y, y[-1] + pad]
z = np.r_[z, z[-1] + pad]

g = GridDC(x, y, z)
g.set_roi([-400, 400, -400, 400, 0, 960])
g.set_solver('mumps')

sigma = 0.001 + np.zeros((g.nc,))

ix = np.where(np.logical_and(g.xc>-200, g.xc<200))[0]
iy = np.where(np.logical_and(g.yc>-200, g.yc<200))[0]
iz = np.where(np.logical_and(g.zc>80, g.zc<480))[0]
sigma[g.ind(ix, iy, iz)] = 0.1

c1c2 = np.array([[-600.0, 0.0, 0.0, 600.0, 0.0, 0.0],
                 [0.0, -600.0, 0.0, 0.0, 600.0, 0.0]])
X = np.linspace(-400, 400, 26)
Y = np.linspace(-400, 400, 26)

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
p1p2 = np.array(p1p2)
tmp = np.kron(c1c2, np.ones((p1p2.shape[0], 1)))
p1p2 = np.kron(np.ones((c1c2.shape[0], 1)), p1p2)
c1c2 = tmp

# on enlève quelques points
mask = np.ones((c1c2.shape[0],), dtype=bool)
mask[50:60] = False
mask[2000:2100] = False
mask[-1] = False
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

data = g.fwd_mod(sigma, c1c2, p1p2)

m_ref = 0.001 + np.zeros((g.nc,))

data_ert = DataERT(c1c2=c1c2, p1p2=p1p2, data=data, wt=np.ones((data.shape[0],)))

inv = Inversion()
inv.max_it = 3
inv.beta = 2500
inv.beta_min = 100
inv.beta_dw = 1.5
#inv.smooth_type = 'blocky'
inv.show_plots = True

g.verbose = False
results = inv.run(g, m_ref, data_ert=data_ert, m_active=g.ind_roi)
S_save3, data_inv3, rms3 = results
g.verbose = True

x, y, z = g.get_roi_nodes()
g2 = GridFV(x, y, z)

fields = {}

for i in range(len(S_save3)):
    name = 'inv - iteration {0:d}'.format(i+1)
    fields[name] = S_save3[i]

g2.toVTK(fields, 'dc_surface')
