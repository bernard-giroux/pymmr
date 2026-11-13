# pymmr
Modelling and inversion of magnetometric resistivity (MMR) and DC resistivity (ERT) data in Python.

## Main features

- MMR (B-field of a grounded current) and ERT (voltage) forward modelling, sensitivity and inversion
- Finite-volume forward solver on rectilinear grids (3D), plus a tetrahedral mesh option for DC
- Joint MMR–ERT inversion
- 2.5D modelling and inversion (conductivity invariant along y) for ERT and MMR, with a grid defined
  by two coordinate arrays `(x, z)`, or a VTK model one cell thick along y with `run_job.py`
  - MMR: current electrodes and B-field sensors can be anywhere along y ([docs/mmr25.md](docs/mmr25.md))
  - ERT: off-line measurements, i.e. current and potential electrodes at different y such as on parallel
    lines ([docs/dc25_offline.md](docs/dc25_offline.md))
- Gauss-Newton or quasi-Newton inversion with smooth, blocky, Ekblom or minimum-support regularization.
  The Gauss-Newton step is solved with a preconditioned CG solver ([docs/cglscd.md](docs/cglscd.md))
- Direct (MUMPS, Pardiso, UMFPACK, SuperLU) or iterative (`scipy.sparse.linalg`) solvers
- Time-lapse constraints are not yet implemented

### Warning
This code is under continuous development and should not be considered mature.

## Installation

Requirements: Python ≥ 3.11 and the packages listed in `requirements.txt`: numpy, scipy, matplotlib,
vtk, discretize, pandas, cython and mpi4py.

pymmr includes a Cython extension (`pymmr/legendre.pyx`) that needs the
[Boost](https://www.boost.org) headers.  Their location is set in `setup.py`; adapt `include_dirs` to
your installation.  Then, from the root of the repository:

```
pip install -e .
```

### Optional solvers

SuperLU (from SciPy) is always available.  A faster direct solver is strongly recommended in 3D:

- **MUMPS**, through [pymumps](https://github.com/bernard-giroux/pymumps) and mpi4py.  On macOS with
  MacPorts, mpi4py may need `DYLD_LIBRARY_PATH` to point to the MPI libraries, e.g.
  `export DYLD_LIBRARY_PATH=/opt/local/lib/mpich-gcc14`
- **Pardiso**, through `pypardiso` (Intel MKL)
- **UMFPACK**, through `scikit-umfpack`

Choose the solver with `g.set_solver('mumps')` (or `'pardiso'`, `'umfpack'`, `'superlu'`, or an
iterative solver such as `scipy.sparse.linalg.bicgstab`).

## Quick start

2.5D MMR inversion of synthetic data (adapted from `examples/example_mmr25.py`):

```python
import numpy as np
from pymmr.finite_volume import calc_padding
from pymmr.mmr import GridMMR
from pymmr.inversion import Inversion, DataMMR

# grid: 2 coordinate arrays (x, z) define a 2.5D model; z is elevation
h = 10.0
pad = np.cumsum(calc_padding(h, 10))
x = np.arange(-150.0, 150.0 + h / 2, h)
x = np.r_[x[0] - pad[::-1], x, x[-1] + pad]
z = np.arange(-100.0, h / 2, h)
z = np.r_[z[0] - pad[::-1], z]

# true model: conductive block in a 0.01 S/m half-space
X, Z = np.meshgrid(0.5 * (x[1:] + x[:-1]), 0.5 * (z[1:] + z[:-1]))
s_true = 0.01 * np.ones(X.shape)
s_true[(np.abs(X) < 20) & (Z < -10) & (Z > -40)] = 0.1
s_true = s_true.ravel()

# survey: current dipoles (c1 xyz, c2 xyz) and B-field sensors (x, y, z), one row per datum
dipoles = [[-100.0, 0, 0, 100, 0, 0], [-60.0, 0, 0, 80, 0, 0]]
receivers = [[a, b, 0.0] for b in (0.0, 20.0) for a in np.arange(-50.0, 51.0, 10.0)]
xs = np.array([c for c in dipoles for r in receivers])
xo = np.array([r for c in dipoles for r in receivers])

# forward modelling: Bx, By, Bz in pT for a 1 A current
g = GridMMR((x, z))
g.set_solver('mumps')
g.set_survey_mmr(xs, xo, 1.0)
data = g.fwd_mod(s_true)

# inversion
data_mmr = DataMMR(xs=xs, xo=xo, data=data, wt=np.ones(3 * data.shape[0]),
                   cs=np.ones(data.shape[0]), date=None)
inv = Inversion()
inv.max_it = 4
inv.beta = 100.0
inv.beta_min = 1.0
m0 = 0.01 * np.ones(g.dc.fv.nc)
sigma_inv, data_inv, rms, misfit, smy = inv.run(g, m0=m0, m_ref=m0, data_mmr=data_mmr)
```

For ERT, use `pymmr.dc.GridDC`, `g.set_survey_ert(c1c2, p1p2, cs)` and `DataERT`.  For a joint
inversion, pass both `data_mmr` and `data_ert` to `Inversion.run` with a `GridMMR`.  Pass three
coordinate arrays `(x, y, z)` to the grid for a 3D model.

### Parameter files with `run_job`

Jobs (`fwd mmr`, `fwd dc`, `inv`) can also be run from a parameter file:

```
python -m pymmr.run_job job.par
```

with a parameter file such as

```
test_fwd_mmr             # basename,
fwd mmr                  # job,
m_fwd.vtr                # model,
c1c2.dat                 # source file,
xo.dat                   # measurement file,
1                        # source current,
mumps                    # solver name,
1                        # verbose,
```

The keywords and the formats of the data files are described in the docstring of
`pymmr/run_job.py`.  Models are VTK rectilinear grids (conductivity in S/m).  A grid with a single
cell along y gives a 2.5D model.

## Conventions

- Coordinates: x easting, y northing, z elevation (z is negative below the surface)
- Units: conductivity in S/m, B-field in pT (`GridMMR(..., units='nT')` for nT), voltage in mV,
  current in A, data errors (`wt`) in %

## Examples

| File | Content |
|---|---|
| `examples/example_dc_fwd.py` | 3D DC forward modelling and sensitivity |
| `examples/example_dc.py` | 3D DC inversion, surface and borehole electrodes |
| `examples/example_dc_bicgstab.py` | 3D DC inversion with an iterative solver (`bicgstab`, multithreaded) |
| `examples/example_dc_tet.py` | DC on a tetrahedral mesh with topography (`topo.vtu`, made by `mk_mesh.py`) |
| `examples/example_dc25.py` | 2.5D DC compared with 3D and with the analytic vertical dyke |
| `examples/example_dc25_regularization.ipynb` | 2.5D DC inversion, comparison of regularizations |
| `examples/example_mmr.py` | 3D MMR modelling and inversion |
| `examples/example_mmr25.py` | 2.5D MMR: dyke validation and inversion |
| `examples/example_joint.py` | Joint MMR–ERT inversion |
| `examples/example_joint_blocky.py` | Joint inversion with blocky regularization |

## Documentation

- [docs/mmr25.md](docs/mmr25.md): 2.5D MMR formulation, sensitivity, electric dipoles at the sensors, validation
- [docs/dc25_offline.md](docs/dc25_offline.md): off-line measurements in 2.5D DC
- [docs/cglscd.md](docs/cglscd.md): solving the Gauss-Newton step, preconditioner, Jacobian weighting
- Docstrings of the modules (`pymmr/run_job.py` for parameter and data file formats)

## Tests

```
python -m pytest tests
```

Some tests compare 2.5D and 3D results and take a few minutes.
