#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main script to run forward modelling and inversion jobs.
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
job = "fwd_dc"
g = None
data_mmr = None
data_ert = None
model_file = None
verbose = False
roi = None
apply_bc = True
calc_J = False
calc_sens = False
units = "mV"

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
                df_mmr = pd.read_table(value, sep="\s+", escapechar="%")
                df_mmr.columns = df_mmr.columns.str.replace(" ", "")  # remove spaces in keys
                data_mmr = df_to_data(df_mmr)
            elif "data" in keyword.lower() and ("ert" in keyword.lower() or "dc" in keyword.lower()):
                df_ert = pd.read_table(value, sep="\s+", escapechar="%")
                df_ert.columns = df_ert.columns.str.replace(" ", "")
                data_ert = df_to_data(df_ert)
            elif "basename" in keyword.lower():
                basename = value
            elif "beta" in keyword.lower():
                inv.beta = float(value)
            elif "max" in keyword.lower() and "it" in keyword.lower():
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
            elif 'solver' in keyword.lower() and 'tolerance' in keyword.lower():
                tol = float(value)
            elif 'precon' in keyword.lower():
                precon = int(value)
            elif 'permut' in keyword.lower():
                do_perm = int(value)
            elif 'verbose' in keyword:
                verbose = int(value)
            elif 'region' in keyword and 'interest' in keyword:
                tmp = value.split()
                if len(tmp) != 6:
                    raise ValueError('6 values needed to define ROI (xmin xmax ymin ymax zmin zmax')
                roi = [float(x) for x in tmp]
            elif 'boundary' in keyword and 'correction' in keyword:
                apply_bc = int(value)
            elif 'compute' in keyword and 'current' in keyword:
                calc_J = int(value)
            elif 'compute' in keyword and 'sensitivity' in keyword:
                calc_sens = int(value)
            elif 'source' in keyword and 'file' in keyword:
                c1c2 = np.atleast_2d(np.loadtxt(value))
            elif 'measurement' in keyword and 'file' in keyword:
                meas = np.atleast_2d(np.loadtxt(value))
            elif 'source' in keyword and 'current' in keyword:
                try:
                    cs = np.loadtxt(value)
                except FileNotFoundError:
                    cs = float(value)
            elif 'units' in keyword:
                units = value


# Done reading parameter file

if data_mmr is None and data_ert is None and "inv" in job:
    raise RuntimeError("No input data provided")

if model_file is None:
    raise RuntimeError("Starting model not provided")

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
    g.xs = c1c2
    g.xo = meas
    g.cs = cs
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
    header = "src_x src_y src_z rcv_x rcv_y rcv_z Bx By Bz"
    np.savetxt(filename, np.c_[g.xs, g.xo, data], header=header)
    if verbose:
        print("done.")

elif "fwd" in job and ("dc" in job or "ert" in job):
    g.c1c2 = c1c2
    g.p1p2 = meas
    g.cs = cs
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
    data = np.c_[c1c2, meas, data]
    header = "c1_x c1_y c1_z c2_x c2_y c2_z p1_x p1_y p1_z p2_x p2_y p2_z v"
    np.savetxt(filename, data, header=header)
    
    if verbose:
        print("done.")

elif "inv" in job:

    S_save, data_inv, rms = inv.run(g, m_ref, data_mmr=data_mmr, data_ert=data_ert)

    x, y, z = g.get_roi_nodes()
    g2 = GridFV(x, y, z)

    fields = {}
    for i in range(len(S_save)):
        name = "iteration {0:d}".format(i + 1)
        fields[name] = S_save[i]

    g2.toVTK(fields, basename + "inv")
