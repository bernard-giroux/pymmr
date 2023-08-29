#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main script to run forward modelling and inversion jobs.

@author: giroux

A parameter file must be provided to define input data, model, etc.

Parameter file
--------------

This is a plain ascii file and each line has the following format:

```
value          # keyword,
```

The keywords are :

- **job** : type of job, possible values are "fwd mmr", "fwd dc", "inv"
- **basename** : each output file will start with this basename
- **data mmr** : Input file holding MMR data (see format below)
- **data dc** : Input file holding DC resistivity data (see format below)
- **model** : Conductivity mode; in VTK format (RectilinearGrid)
- **source file** : Coordinates in injection dipoles, for forward modelling
- **measurement file** : Coordinates of measurement dipoles or B-field sensor, for forward modelling
- **source current** : Intensity of source current in A (file or scalar)
- **beta** : Regularization factor for computing perturbation
- **inv max_it** : Maximum number of inversion iterations
- **solver name** : Choice of solver
- **solver max_it** : Maximum number of iteration for iterative solvers
- **solver tol** : Target tolerance for iterative solvers
- **precon** : Apply preconditioning when using iterative solvers
- **permut** : Apply inverse Cuthill-McKee permutation when using iterative solvers
- **region of interest** : Extents of region of interest for inversion or sensitivity calculation
- **verbose** : Display progress messages
- **show plots** : Show plots during inversion
- **boundary correction** : Apply correction described in Pidlisecky et al. 2007
- **compute current** : Compute and save current density (forward modelling)
- **compute sensitivity** : Compute and save sensitivity (forward modelling)
- **units** : units of voltage for forward modelling, "mV" or "V"

File formats
------------


"""

from mpi4py import MPI

import importlib
import re
import sys
import warnings

import numpy as np
import pandas as pd

from pymmr.finite_volume import GridFV, build_from_vtk
from pymmr.dc import GridDC
from pymmr.mmr import GridMMR
from pymmr.inversion import Inversion, df_to_data

comm = MPI.COMM_WORLD

basename = "pymmr"
job = None
g = None
data_mmr = None
data_ert = None
model_file = None
verbose = True
roi = None
apply_bc = True
calc_J = False
calc_sens = False
units = "mV"
tol = 1.e-9
solver_name = "mumps"
max_it = 1000
precon = False
do_perm = False

kw_pattern = re.compile("\s?(.+)\s?#\s?([\w\s]+),")

inv = Inversion()

# we should have a parameter file as input
with open(sys.argv[1], "r") as f:
    for line in f:
        kw_match = kw_pattern.search(line)
        if kw_match is not None:
            value = kw_match[1].rstrip()
            keyword = kw_match[2].rstrip()

            if "job" in keyword.lower():
                job = value.lower()
            elif "data" in keyword.lower() and "mmr" in keyword.lower():
                df_mmr = pd.read_table(value, sep="\s+", escapechar="#")
                df_mmr.columns = df_mmr.columns.str.replace(" ", "")  # remove spaces in keys
                data_mmr = df_to_data(df_mmr)
            elif "data" in keyword.lower() and ("ert" in keyword.lower() or "dc" in keyword.lower()):
                df_ert = pd.read_table(value, sep="\s+", escapechar="#")
                df_ert.columns = df_ert.columns.str.replace(" ", "")
                data_ert = df_to_data(df_ert)
            elif "basename" in keyword.lower():
                basename = value
            elif "beta" in keyword.lower():
                inv.beta = float(value)
            elif "inv" in keyword.lower() and "max_it" in keyword.lower():
                inv.max_it = int(value)
            elif "model" in keyword.lower():
                model_file = value
            elif 'solver' in keyword.lower() and 'name' in keyword.lower():
                if value in ('bicgstab', 'gmres'):
                    mod = importlib.import_module('scipy.sparse.linalg')
                    solver_name = getattr(mod, value)
                else:
                    solver_name = value.lower()
            elif 'solver' in keyword.lower() and 'max_it' in keyword.lower():
                max_it = int(value)
            elif 'solver' in keyword.lower() and 'tol' in keyword.lower():
                tol = float(value)
            elif 'precon' in keyword.lower():
                precon = int(value)
            elif 'permut' in keyword.lower():
                do_perm = int(value)
            elif 'verbose' in keyword.lower():
                verbose = int(value)
            elif 'region' in keyword.lower() and 'interest' in keyword.lower():
                tmp = value.split()
                if len(tmp) != 6:
                    raise ValueError('6 values needed to define ROI (xmin xmax ymin ymax zmin zmax')
                roi = [float(x) for x in tmp]
            elif 'boundary' in keyword.lower() and 'correction' in keyword.lower():
                apply_bc = int(value)
            elif 'compute' in keyword.lower() and 'current' in keyword.lower():
                calc_J = int(value)
            elif 'compute' in keyword.lower() and 'sensitivity' in keyword.lower():
                calc_sens = int(value)
            elif 'source' in keyword.lower() and 'file' in keyword.lower():
                c1c2 = np.atleast_2d(np.loadtxt(value))
            elif 'measurement' in keyword.lower() and 'file' in keyword.lower():
                meas = np.atleast_2d(np.loadtxt(value))
            elif 'source' in keyword.lower() and 'current' in keyword.lower():
                try:
                    cs = np.loadtxt(value)
                except FileNotFoundError:
                    cs = float(value)
            elif 'units' in keyword.lower():
                units = value
            elif 'show' in keyword.lower() and 'plots' in keyword.lower():
                inv.show_plots = int(value)


# Done reading parameter file

if job is None:
    raise RuntimeError("Type of job not specified")

if model_file is None:
    raise RuntimeError("Starting/input model not provided")

if data_mmr is None and data_ert is None and "inv" in job:
    raise RuntimeError("No input data provided")

if data_mmr is not None or "mmr" in job:
    # we will have MMR data to invert of MMR data to model
    g = build_from_vtk(GridMMR, model_file, comm=comm)
    m_ref = sigma = g.fromVTK("Conductivity", model_file)
else:
    # we have ERT data only
    g = build_from_vtk(GridDC, model_file, comm=comm)
    m_ref = sigma = g.fromVTK("Conductivity", model_file)

g.verbose = verbose
g.set_solver(solver_name, tol, max_it, precon, do_perm)
if roi is not None:
    g.set_roi(roi)
g.apply_bc = apply_bc

if "fwd" in job and "mmr" in job:
    g.set_survey_mmr(c1c2, meas, cs)
    data = g.fwd_mod(sigma, calc_sens=calc_sens)

    if calc_sens:
        if verbose:
            print("Saving sensitivity ... ", end="", flush=True)
        data, sens = data
        g.save_sensivitity(sens, basename)
        if verbose:
            print("done.")
    
    if verbose:
        print("Saving modelled data ... ", end="", flush=True)
    filename = basename + "_mmr.dat"
    header = "c1_x c1_y c1_z c2_x c2_y c2_z obs_x obs_y obs_z Bx By Bz cs"
    if c1c2.shape[0] == meas.shape[0]:
        xs = c1c2
        xo = meas
        cs = g.cs.reshape(-1, 1)
    else:
        # fwd modelling was done using the following combinations
        xs = np.kron(c1c2, np.ones((meas.shape[0], 1)))
        xo = np.kron(np.ones((c1c2.shape[0], 1)), meas)
        cs = np.kron(g.cs.reshape(-1, 1), np.ones((meas.shape[0], 1)))
    np.savetxt(filename, np.c_[xs, xo, data, cs], header=header, fmt="%g")
    if verbose:
        print("done.")

elif "fwd" in job and ("dc" in job or "ert" in job):
    g.set_survey_ert(c1c2, meas, cs)
    g.units = units
    data = g.fwd_mod(sigma, calc_J=calc_J, calc_sens=calc_sens)
    
    if calc_sens:
        if verbose:
            print("Saving sensitivity ... ", end="", flush=True)
        data, sens = data
        g.save_sensivitity(sens, basename)
        if verbose:
            print("done.")
    elif calc_J:
        if verbose:
            print("Saving current density ... ", end="", flush=True)
        data, J = data
        g.save_current_density(J, basename)
        if verbose:
            print("done.")

    if verbose:
        print("Saving modelled voltages ... ", end="", flush=True)
    # save data
    filename = basename + "_dc.dat"
    data = np.c_[c1c2, meas, data, g.cs]
    header = "c1_x c1_y c1_z c2_x c2_y c2_z p1_x p1_y p1_z p2_x p2_y p2_z V cs"
    np.savetxt(filename, data, header=header, fmt="%g")
    
    if verbose:
        print("done.")

elif "inv" in job:

    m_active = None
    if roi is not None:
        m_active = g.ind_roi

    S_save, data_inv, rms = inv.run(g, m_ref, data_mmr=data_mmr, data_ert=data_ert, m_active=m_active)

    x, y, z = g.get_roi_nodes()
    g2 = GridFV(x, y, z)

    fields = {}
    for i in range(len(S_save)):
        name = "iteration {0:d}".format(i + 1)
        fields[name] = S_save[i]

    g2.toVTK(fields, basename + "inv")

else:
    raise ValueError("Job type not defined")