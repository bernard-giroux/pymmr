import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/Users/giroux/GitHub/pymmr')
sys.path.append('/Users/giroux/GitHub/pymumps')

from pymmr.finite_volume import calc_padding, GridFV
from pymmr.dc import GridDC
from pymmr.inversion import Inversion, DataERT
from pymmr.run_job import build_from_vtk

g, sigma = build_from_vtk(GridDC, filename='topo.vtu', return_sigma=True)
g.verbose = True
g.fv.set_solver('mumps')

c1c2 = np.array([[-50.0, 0.0, 0.0, 50.0, 0.0, 0.0],
                 [0.0, -50.0, 0.0, 0.0, 50.0, 0.0]])

p1p2 = []
z_bh = np.linspace(-2, -10, 5)

for nz in range(len(z_bh) - 1):
    p1p2.append((0, 0, z_bh[nz], 0, 0, z_bh[nz + 1]))

p1p2 = np.array(p1p2)
tmp = np.kron(c1c2, np.ones((p1p2.shape[0], 1)))
p1p2 = np.kron(np.ones((c1c2.shape[0], 1)), p1p2)
c1c2 = tmp

g.set_survey_ert(c1c2, p1p2, 1.0)
data, J = g.fwd_mod(sigma, calc_J=True)
