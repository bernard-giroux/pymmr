import os
import sys
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

sys.path.append('/Users/giroux/GitHub/pymmr')
sys.path.append('/Users/giroux/GitHub/pymumps')

from pymmr.finite_volume import calc_padding
from pymmr.dc import GridDC, VerticalDyke

os.chdir('/Users/giroux/GitHub/pymmr/examples')


# %%

rho1 = 100.0
rho2 = 1000.0
rho3 = 50.0

# rho2 = 100.0
# rho3 = 100.0

dyke = VerticalDyke(rho1, rho2, rho3, 10.0)
x = np.arange(-20.0, 27.0, 2.0).reshape(-1, 1)
c1c2 = np.c_[x, np.zeros(x.shape), np.zeros(x.shape), 1.0 + x, np.zeros(x.shape), np.zeros(x.shape)]
p1p2 = np.c_[2.0 + x, np.zeros(x.shape), np.zeros(x.shape), 3.0 + x, np.zeros(x.shape), np.zeros(x.shape)]

va = dyke.fwd_mod(c1c2, p1p2)


# %%

x = np.arange(-30.0, 40.01)
y = np.arange(-5.0, 5.01)
z = np.r_[np.arange(-30.0, -5.0), np.arange(-5.0, 0.01)]

pad = np.cumsum(calc_padding(1.0, 25))

x = np.r_[x[0] - pad[::-1], x, x[-1] + pad]
y = np.r_[y[0] - pad[::-1], y, y[-1] + pad]
z = np.r_[z[0] - pad[::-1], z]

g = GridDC((x, y, z))
g.set_solver('mumps')
g.set_survey_ert(c1c2, p1p2, cs=1.0)


# %%
sigma = 1./rho1 + np.zeros((g.fv.nx, g.fv.ny, g.fv.nz))
ind = g.fv.xc > 0.0
sigma[ind, :, :] = 1./rho2
ind = g.fv.xc > 10.0
sigma[ind, :, :] = 1./rho3

sigma = sigma.flatten(order='F')

# g.fv.toVTK({'sigma': sigma, 'rho': 1/sigma}, 'sigma')

v3 = 0.001 * g.fwd_mod(sigma)


# %%

g2 = GridDC((x, z))
# g2.verbose = True
# g2.fv.verbose = True
g2.set_solver('umfpack')
g2.set_survey_ert(c1c2, p1p2, cs=1.0)

g2.fv.optimize_k_g(11)

sigma = 1./rho1 + np.zeros((g2.fv.nx, g2.fv.nz))
ind = g2.fv.xc > 0.0
sigma[ind, :] = 1./rho2
ind = g2.fv.xc > 10.0
sigma[ind, :] = 1./rho3

sigma = sigma.flatten(order='F')

g2.apply_bc = True
v25 = 0.001 * g2.fwd_mod(sigma)


# %%
import simpeg.utils
import simpeg.maps
from discretize import TensorMesh
from simpeg.electromagnetics.static import resistivity as dc
from simpeg.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line,
)

# Define survey line parameters
survey_type = "dipole-dipole"
dimension_type = "2D"
data_type = "volt"
end_locations = np.r_[c1c2[0, 0], c1c2[-1, 0]+4]
station_separation = 1.0
num_rx_per_src = 1

topo_2d = np.c_[x, np.zeros(x.shape)]

# Generate source list for DC survey line
source_list = generate_dcip_sources_line(
    survey_type,
    data_type,
    dimension_type,
    end_locations,
    topo_2d,
    num_rx_per_src,
    station_separation,
)

# Define survey
survey = dc.survey.Survey(source_list[::2])

mesh = TensorMesh((np.diff(x), np.diff(z)), origin=(x[0], z[0]))

sig = 1./rho1 + np.zeros((g2.fv.nz, g2.fv.nx))
ind = g2.fv.xc > 0.0
sig[:, ind] = 1./rho2
ind = g2.fv.xc > 10.0
sig[:, ind] = 1./rho3
sig = sig.flatten()

air_conductivity = 1e-8
ind_active = np.ones((mesh.n_cells,), dtype=bool)
conductivity_map = simpeg.maps.InjectActiveCells(mesh, ind_active, air_conductivity)

try:
    from pymatsolver import Pardiso as SolverSP
except ImportError:
    from simpeg import SolverLU as SolverSP

simulation = dc.simulation_2d.Simulation2DNodal(
    mesh, survey=survey, sigmaMap=conductivity_map, solver=SolverSP
)

# Predict the data by running the simulation. The data are the raw voltage in
# units of volts.
dpred_2dn = simulation.dpred(sig)


simulation = dc.simulation_2d.Simulation2DCellCentered(
    mesh, survey=survey, sigmaMap=conductivity_map, solver=SolverSP
)

# Predict the data by running the simulation. The data are the raw voltage in
# units of volts.
dpred_2dc = simulation.dpred(sig)



# %%
#
# import matplotlib as mpl
# from matplotlib.colors import LogNorm
# # Plot Conductivity Model
# fig = plt.figure(figsize=(9, 4))
#
# plotting_map = simpeg.maps.InjectActiveCells(mesh, ind_active, np.nan)
# norm = LogNorm(vmin=1e-3, vmax=1e-1)
#
# ax1 = fig.add_axes([0.14, 0.17, 0.68, 0.7])
# mesh.plot_image(
#     plotting_map * sig, ax=ax1, grid=False, pcolor_opts={"norm": norm}
# )
# ax1.set_xlim(-60, 60)
# ax1.set_ylim(-60, 0)
# ax1.set_title("Conductivity Model")
# ax1.set_xlabel("x (m)")
# ax1.set_ylabel("z (m)")
#
# ax2 = fig.add_axes([0.84, 0.17, 0.03, 0.7])
# cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation="vertical")
# cbar.set_label(r"$\sigma$ (S/m)", rotation=270, labelpad=15, size=12)
#
# plt.show()

# %%
# from discretize.utils import mkvc
#
# x_topo, y_topo = np.meshgrid(g.fv.x, g.fv.y)
# z_topo = np.zeros(x_topo.shape)
# x_topo, y_topo, z_topo = mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)
# topo_xyz = np.c_[x_topo, y_topo, z_topo]
#
# end_locations = np.r_[c1c2[0, 0], c1c2[-1, 0]+4, 0.0, 0.0]
# dimension_type = "3D"
#
# source_list = generate_dcip_sources_line(
#     survey_type,
#     data_type,
#     dimension_type,
#     end_locations,
#     topo_xyz,
#     num_rx_per_src,
#     station_separation,
# )
#
# # Define survey
# survey = dc.survey.Survey(source_list[::2])
#
# mesh = TensorMesh((g.fv.hx, g.fv.hy, g.fv.hz), origin=(g.fv.x[0], g.fv.y[0], g.fv.z[0]))
#
# sig = 1./rho1 + np.zeros((g.fv.nz, g.fv.ny, g.fv.nx))
# ind = g2.fv.xc > 0.0
# sig[:, :, ind] = 1./rho2
# ind = g2.fv.xc > 10.0
# sig[:, :, ind] = 1./rho3
# sig = sig.flatten()
#
# air_conductivity = 1e-8
# ind_active = np.ones((mesh.n_cells,), dtype=bool)
# conductivity_map = simpeg.maps.InjectActiveCells(mesh, ind_active, air_conductivity)
#
# simulation = dc.simulation.Simulation3DNodal(
#     mesh, survey=survey, sigmaMap=conductivity_map, solver=SolverSP
# )
#
# # Predict the data by running the simulation. The data are the raw voltage in
# # units of volts.
# dpred_3dn = simulation.dpred(sig)
#
#
# simulation = dc.simulation.Simulation3DCellCentred(
#     mesh, survey=survey, sigmaMap=conductivity_map, solver=SolverSP
# )
#
# # Predict the data by running the simulation. The data are the raw voltage in
# # units of volts.
# dpred_3dc = simulation.dpred(sig)
#


# %%

plt.plot(c1c2[:, 0] + 1.5, va, label='analytic')
plt.plot(c1c2[:, 0] + 1.5, v3, 'o', label='3D')
plt.plot(c1c2[:, 0] + 1.5, v25, '*', label='2.5D')
plt.plot(survey.locations_a[:, 0] + 1.5, dpred_2dn, '+', label='SimPEG 2D nodal')
plt.plot(survey.locations_a[:, 0] + 1.5, dpred_2dc, '+', label='SimPEG 2D cell')
# plt.plot(survey.locations_a[:, 0] + 1.5, dpred_3dn, '+', label='SimPEG 3D nodal')
# plt.plot(survey.locations_a[:, 0] + 1.5, dpred_3dc, '+', label='SimPEG 3D cell')
plt.legend()
plt.show()
