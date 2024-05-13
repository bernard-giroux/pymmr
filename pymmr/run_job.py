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
- **model** : Conductivity model in VTK format (RectilinearGrid).  Conductivity should be in S/m
- **source file** : Coordinates in injection dipoles, for forward modelling
- **measurement file** : Coordinates of measurement dipoles or B-field sensor, for forward modelling
- **source current** : Intensity of source current in A (file or scalar)
- **beta** : Regularization factor
- **inv max_it** : Maximum number of inversion iterations
- **solver name** : Choice of solver
- **solver max_it** : Maximum number of iteration for iterative solvers
- **solver tol** : Target tolerance for iterative solvers
- **precon** : Apply preconditioning when using iterative solvers
- **permut** : Apply inverse Cuthill-McKee permutation when using iterative solvers
- **region of interest** : Extents of region of interest for inversion or sensitivity calculation
- **verbose** : Display progress messages
- **show plots** : Show plots during inversion
- **save plots** : Save plots produced during inversion
- **boundary correction** : Apply correction described in Pidlisecky et al. 2007
- **compute current** : Compute and save current density (forward modelling)
- **compute sensitivity** : Compute and save sensitivity (forward modelling)
- **units** : units of voltage for forward modelling, "mV" or "V"

File formats
------------

Data files are plain text files with with a header line specific to the type of data.  The header line starts with #
and contains keywords for the different variables.  Keywords can be in arbitrary order.
For MMR, the full header is

```
# c1_x c1_y c1_z c2_x c2_y c2_z obs_x obs_y obs_z Bx By Bz wt_x wt_y wt_z cs date
```

with:

- `c1_x` `c1_y` `c1_z` : the coordinates of the first injection electrode (in m)
- `c2_x` `c2_y` `c2_z` : the coordinates of the second injection electrode (in m)
- `obs_x` `obs_y` `obs_z` : the coordinates of the observation point (in m)
- `Bx` `By` `Bz` : the measured components of the field (in pT)
- `wt_x` `wt_y` `wt_z` : the measurement error (in %)
- `cs` : the current source (in A)
- `date` : the date of the survey (in yyyymmdd)

Mandatory variables are `c1_x` `c1_y` `c1_z` `c2_x` `c2_y` `c2_z` `obs_x` `obs_y` `obs_z` `Bx` `By` `Bz`.
If not provided, the measurement errors are set to 1 %, the current to 1 A, and the date of the survey is set to None.

For ERT, the full header is

```
# c1_x c1_y c1_z c2_x c2_y c2_z p1_x p1_y p1_z p2_x p2_y p2_z V cs wt date
```

with the same meaning for `c?_?`, `cs`, `wt`, and `date` fields, and:

- `p1_x` `p1_y` `p1_z` : the coordinates of the first potential electrode (in m)
- `p2_x` `p2_y` `p2_z` : the coordinates of the second potential electrode (in m)
- `V` : the measured voltage (in mV)

"""

from mpi4py import MPI

import importlib
import re
import socket
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import vtk
from vtk.util.numpy_support import vtk_to_numpy

from pymmr.finite_volume import GridFV
from pymmr.dc import GridDC
from pymmr.mmr import GridMMR
from pymmr.inversion import Inversion, df_to_data


def build_from_vtk(grid_class, filename, comm=None, return_sigma=False):
    """Create grid from VTK file.

    The file must contain a rectilinear grid

    Parameters
    ----------
    grid_class : class
        Class for building grid (must be derived from GridFV)
    filename : string
        Name of VTK file
    comm : MPI Communicator, optional
        If None,  MPI_COMM_WORLD will be used
    return_sigma : bool, optional
        Returns conductivity

    Returns
    -------
    tuple:
        - instance of the grid
        - conductivity (if return_sigma is True)
    """
    if grid_class not in (GridDC, GridMMR):
        raise ValueError("Grid class '{}' not valid".format(grid_class))

    extension = filename.split(".")[-1]
    if extension == "vtr":
        # rectilinear grid
        reader = vtk.vtkXMLRectilinearGridReader()
        reader.SetFileName(filename)
        reader.Update()
        x = vtk_to_numpy(reader.GetOutput().GetXCoordinates())
        y = vtk_to_numpy(reader.GetOutput().GetYCoordinates())
        z = vtk_to_numpy(reader.GetOutput().GetZCoordinates())
        if return_sigma:
            sigma = vtk_to_numpy(reader.GetOutput().GetCellData().GetArray("Conductivity"))
            return grid_class((x, y, z), comm=comm), sigma
        else:
            return grid_class((x, y, z), comm=comm)
    elif extension == "vtu":
        #  unstructured grid
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        pts = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
        tet = []
        surface = []
        cell_array = reader.GetOutput().GetCells()
        list_pts = vtk.vtkIdList()
        cell_id = vtk_to_numpy(reader.GetOutput().GetCellData().GetArray("CellEntityIds"))

        for n in range(cell_array.GetNumberOfCells()):
            if cell_array.GetCellSize(n) == 4:
                # we have a tetrahedra
                cell_array.GetCellAtId(n, list_pts)
                tet.append([list_pts.GetId(nn) for nn in range(list_pts.GetNumberOfIds())])
            elif cell_array.GetCellSize(n) == 3 and cell_id[n] == 1:
                # we have a triangle forming the ground surface
                cell_array.GetCellAtId(n, list_pts)
                surface.append([list_pts.GetId(nn) for nn in range(list_pts.GetNumberOfIds())])
            else:
                raise ValueError('Cell size not valid')

        if return_sigma:
            sigma = vtk_to_numpy(reader.GetOutput().GetCellData().GetArray("Conductivity"))
            return grid_class((pts, tet, surface), comm=comm), sigma
        else:
            return grid_class((pts, tet, surface), comm=comm)


if __name__ == "__main__":
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

    kw_pattern = re.compile("\\s?(.+)\\s?#\\s?([\\w\\s]+),")

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
                    df_mmr = pd.read_table(value, sep="\\s+", escapechar="#")
                    df_mmr.columns = df_mmr.columns.str.replace(" ", "")  # remove spaces in keys
                    data_mmr = df_to_data(df_mmr)
                elif "data" in keyword.lower() and ("ert" in keyword.lower() or "dc" in keyword.lower()):
                    df_ert = pd.read_table(value, sep="\\s+", escapechar="#")
                    df_ert.columns = df_ert.columns.str.replace(" ", "")
                    data_ert = df_to_data(df_ert)
                elif "basename" in keyword.lower():
                    basename = value
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
                elif 'save' in keyword.lower() and 'plots' in keyword.lower():
                    inv.save_plots = int(value)
                elif "beta" in keyword.lower():
                    inv.beta = float(value)
                elif "inv" in keyword.lower() and "max_it" in keyword.lower():
                    inv.max_it = int(value)
                elif "data" in keyword.lower() and "weight" in keyword.lower():
                    inv.data_weighting = value
                elif "checkpointing" in keyword.lower():
                    inv.checkpointing = int(value)
                elif "start" in keyword.lower() and "checkpoint" in keyword.lower():
                    inv.start_from_chkpt = int(value)

    # Done reading parameter file

    if job is None:
        raise RuntimeError("Type of job not specified")

    if model_file is None:
        raise RuntimeError("Starting/input model not provided")

    if data_mmr is None and data_ert is None and "inv" in job:
        raise RuntimeError("No input data provided")

    if verbose:
        starttime = datetime.now()
        hostname = socket.gethostname()
        print(f'\nJob running on {hostname}\nStarted {starttime}')
        print(f"Parameter file: {sys.argv[1]}")

    if data_mmr is not None or "mmr" in job:
        # we will have MMR data to invert or MMR data to model
        grid_type = GridMMR
    else:
        # we have ERT data only
        grid_type = GridDC

    g, sigma = build_from_vtk(grid_type, model_file, comm=comm, return_sigma=True)
    m_ref = sigma

    g.verbose = verbose
    g.set_solver(solver_name, tol, max_it, precon, do_perm)
    if roi is not None:
        g.set_roi(roi)
    g.apply_bc = apply_bc

    if "fwd" in job and "mmr" in job:
        g.set_survey_mmr(c1c2, meas, cs)
        data = g.fwd_mod(sigma, calc_sens=calc_sens)

        if verbose:
            endtime = datetime.now()
            diff = endtime - starttime
            print(f"\nCalculations ended {endtime} ({diff} elapsed)")

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

        if verbose:
            endtime = datetime.now()
            diff = endtime - starttime
            print(f"\nCalculations ended {endtime} ({diff} elapsed)")

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

        inv.basename = basename + "_inv"
        m_active = None
        if roi is not None:
            m_active = g.ind_roi

        g.verbose = False
        g.solver_A.verbose = False
        S_save, data_inv, rms, misfit, smy = inv.run(g, m_ref, data_mmr=data_mmr, data_ert=data_ert, m_active=m_active)

        if verbose:
            endtime = datetime.now()
            diff = endtime - starttime
            print(f"\nCalculations ended {endtime} ({diff} elapsed)")

        x, y, z = g.get_roi_nodes()
        g2 = GridFV(x, y, z)

        fields = {}
        for i in range(len(S_save)):
            name = "iteration {0:d}".format(i + 1)
            fields[name] = S_save[i]

        g2.toVTK(fields, basename + "_inv")

        if inv.show_plots or inv.save_plots:
            fig = plt.figure()
            plt.bar(np.arange(1, 1+len(rms)), rms)
            plt.xlabel('Iteration')
            plt.ylabel('Weighted RMSE')
            plt.tight_layout()
            if inv.save_plots:
                filename = inv.basename + "_rms.pdf"
                fig.savefig(filename)
            if inv.show_plots:
                plt.show()

            fig = plt.figure()
            plt.bar(np.arange(1, 1+len(misfit)), misfit)
            plt.xlabel('Iteration')
            plt.ylabel('Misfit')
            plt.tight_layout()
            if inv.save_plots:
                filename = inv.basename + "_misfit.pdf"
                fig.savefig(filename)
            if inv.show_plots:
                plt.show()

            fig = plt.figure()
            plt.bar(np.arange(1, 1+len(smy)), smy)
            plt.xlabel('Iteration')
            plt.ylabel('Parameter variation function')
            plt.tight_layout()
            if inv.save_plots:
                filename = inv.basename + "_smy.pdf"
                fig.savefig(filename)
            if inv.show_plots:
                plt.show()

    else:
        raise ValueError("Job type not defined")