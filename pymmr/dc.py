#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for electrical resistivity modelling

@author: giroux

Main reference

@article{pidlisecky2007resinvm3d,
  title={RESINVM3D: A 3D resistivity inversion package},
  author={Pidlisecky, Adam and Haber, Eldad and Knight, Rosemary},
  journal={Geophysics},
  volume={72},
  number={2},
  pages={H1--H10},
  year={2007},
  publisher={Society of Exploration Geophysicists}
}

"""
import warnings

# from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from scipy.stats.mstats import gmean

try:
    from numba import jit
except ImportError:
    warnings.warn('Numba could not be imported, @jit decorator without effect', stacklevel=2)
    # numba est souvent en retard sur la version de numpy
    # on redéfini le décorateur pour ne rien faire de spécial
    import functools

    def jit(in1, nopython):
        def decorator_jit(func):
            @functools.wraps(func)
            def wrapper_jit(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper_jit
        return decorator_jit

from pymmr.finite_volume import GridFV, MeshFV, Solver

# TODO: généraliser ROI pour voxels arbitraires


# %% Some functions

@jit("boolean(boolean[:])", nopython=True)
def all(a: np.ndarray):
    return a.sum() == a.size


def sortrows(a: np.ndarray, sort_back=False):
    """
    Sort rows of 2D array

    Parameters
    ----------
    a : 2D ndarray
        Array to process.
    sort_back : bool, optional
        Returns array of indices to sort_back. For this to work, all rows in
        input array must be different.

    Returns
    -------
    b : 2D ndarray
        sorted array.
    ind_back : 1D ndarray
        array of indices (if sort_back is True)

    """
    tmp, cnt = np.unique(a, axis=0, return_counts=True)
    if sort_back:
        if tmp.shape != a.shape:
            raise ValueError('All rows must be different when sort_back is True')
        ind_back = np.empty((a.shape[0],), dtype=int)
        for n1 in np.arange(a.shape[0]):
            for n2 in np.arange(tmp.shape[0]):
                if all(a[n1, :] == tmp[n2, :]):
                    ind_back[n1] = n2
                    break
        return tmp, ind_back
    else:
        return np.repeat(tmp, cnt, axis=0)


# %%  class GridDC

class GridDC:
    """Grid for DC resistivity modelling.

    Parameters
    ----------
    param_fv : tuple
        parameters to instantiate the finite volume mesh/grid
        if tuple has 3 elements, a grid in built and the elements are
            x : array of float
                Node coordinates along x (m)
            y : array of float
                Node coordinates along y (m)
            z : array of float
                Node coordinates along z (m)
        if tuple has 2 elements, a mesh in built and the elements are
            pts : array of float
            tet: array of int
    units : str, optional
        Units of voltage at output
    comm : MPI Communicator, optional
        If None, MPI_COMM_WORLD will be used
    """

    units_scaling_factors = {'mV': 1.e3, 'V': 1.0}

    def __init__(self, param_fv, units='mV', comm=None):
        if len(param_fv) != 3:
            raise ValueError('GridDC: param_fv must have 3 elements')

        if param_fv[0].ndim == 1:
            self.fv = GridFV(param_fv, comm)
        else:
            self.fv = MeshFV(param_fv, comm)

        self._c1c2 = None
        self._cs = None
        self._p1p2 = None
        self.u0 = None
        self.u = None
        self.q = None
        self.Q = None
        self.apply_bc = True
        self.keep_c1c2 = True  # keep dipoles for forming RHS term of equation
                               # system, this will be false when computing
                               # sensitivity
        self.sort_electrodes = True
        self.electrodes_sorted = False
        self.sort_back = None
        self.roi = None
        self.ind_roi = np.arange(self.fv.nc)
        self.in_inv = False    # set to True when grid is used in inversion
        self.c1c2_u = None
        self.cs12_u = None
        self.units = units
        self.verbose = False

    @property
    def nc(self):
        return self.fv.nc

    @property
    def c1c2(self):
        """Coordinates of injection points (m)."""
        return self._c1c2

    @c1c2.setter
    def c1c2(self, val):
        if val is None:
            return
        tmp = np.array(val, dtype=np.float64)
        if tmp.ndim == 1:
            if tmp.size == 6:
                tmp = tmp.reshape((1, 6))
            else:
                raise ValueError('Size of source term must be nobs x 6')
        elif tmp.ndim == 2:
            if tmp.shape[1] != 6:
                raise ValueError('Size of source term must be nobs x 6')
        else:
            raise ValueError('Size of source term must be nobs x 6')
        for ns in range(tmp.shape[0]):
            if self.fv.is_inside(tmp[ns, 0], tmp[ns, 1], tmp[ns, 2]) is False or \
               self.fv.is_inside(tmp[ns, 3], tmp[ns, 4], tmp[ns, 5]) is False:
                raise ValueError('Source term outside grid')
        self._c1c2 = tmp
        self.electrodes_sorted = False
        self.Q = None  # reset interpolation matrix because electrodes will be sorted

    @property
    def cs(self):
        """Intensity of source current."""
        return self._cs

    @cs.setter
    def cs(self, val):
        self._cs = val

    @property
    def p1p2(self):
        """Coordinates of measurement points (m)."""
        return self._p1p2

    @p1p2.setter
    def p1p2(self, val):
        if val is None:
            return
        tmp = np.array(val, dtype=np.float64)
        if tmp.ndim == 1:
            if tmp.size == 6:
                tmp = tmp.reshape((1, tmp.size))
            else:
                raise ValueError('Measurement points should be nobs x 6')
        elif tmp.ndim == 2:
            if tmp.shape[1] == 3:
                # not a dipole survey, other electrode assumed to be at infinity
                tmp = np.c_[tmp, np.inf * np.ones(tmp.shape)]
            elif tmp.shape[1] != 6:
                raise ValueError('Measurement points should be nobs x 6')
        else:
            raise ValueError('Measurement points should be nobs x 6')
        for ns in range(tmp.shape[0]):
            if self.fv.is_inside(tmp[ns, 0], tmp[ns, 1], tmp[ns, 2]) is False or \
                (self.fv.is_inside(tmp[ns, 3], tmp[ns, 4], tmp[ns, 5]) is False
                 and not np.any(tmp[ns, 3:] == np.inf)):
                    raise ValueError('Measurement point outside grid')
        self._p1p2 = tmp
        self.electrodes_sorted = False
        self.Q = None  # reset interpolation matrix because electrodes will be sorted

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, val):
        if val not in GridDC.units_scaling_factors:
            raise ValueError('Wrong units of voltage')
        self._units = val
        self._units_scaling = GridDC.units_scaling_factors[val]

    def set_survey_ert(self, c1c2, p1p2, cs):
        """Set survey variables.

        Parameters
        ----------
        c1c2 : array_like
            Coordinates of injection points (m).
        p1p2 : array_like
            Coordinates of measurement points (m).
        cs : scalar or array_like
            Intensity of current source

        Notes
        -----
        - `c1c2` et `p1p2` must be of equal size, i.e. n_obs x 6.  The
        6 columns correspond to coordinates x, y, & z of the dipoles, in
        the order x1 y1 z1 x2 y2 z2
        - In the current implementation, current intensity values must be
        equal for each injection dipole, i.e. for c1c2 and p1p2 combinations
        involving a common set of c1c2 values.  In other words, current
        values cannot differ for a given injection dipole, but can differ
        for different injection dipoles.
        """
        self.c1c2 = c1c2
        self.p1p2 = p1p2
        self.cs = cs
        self._check_cs()

    def set_roi(self, roi):
        """Define region of interest for computing sensitivity or for inversion.

        Parameters
        ----------
        roi : tuple of float
            Tuple holding xmin, xmax, ymin, ymax, zmin, zmax
        """

        if roi is None:
            return
        self.roi = roi
        xmin, xmax, ymin, ymax, zmin, zmax = roi
        if self.fv.is_inside(xmin, ymin, zmin) is False or \
            self.fv.is_inside(xmax, ymax, zmax) is False:
                raise ValueError('Region of interest extending beyond grid')

        ind_x, = np.where(np.logical_and(self.fv.xc >= xmin, self.fv.xc < xmax))
        ind_y, = np.where(np.logical_and(self.fv.yc >= ymin, self.fv.yc < ymax))
        ind_z, = np.where(np.logical_and(self.fv.zc >= zmin, self.fv.zc < zmax))

        self.ind_roi = self.fv.ind(ind_x, ind_y, ind_z)

    def get_roi_nodes(self):
        """
        Returns coordinates of nodes inside region of interest.

        Returns
        -------
        tuple of ndarray
            x, y, z
        """
        if np.array_equal(self.ind_roi, np.arange(self.fv.nc)):
            return self.fv.x, self.fv.y, self.fv.z
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = self.roi
            # get indices of voxels
            ind_x, = np.where(np.logical_and(self.fv.xc >= xmin, self.fv.xc < xmax))
            ind_y, = np.where(np.logical_and(self.fv.yc >= ymin, self.fv.yc < ymax))
            ind_z, = np.where(np.logical_and(self.fv.zc >= zmin, self.fv.zc < zmax))

            # add node at end
            ind_x = np.r_[ind_x, ind_x[-1]+1]
            ind_y = np.r_[ind_y, ind_y[-1]+1]
            ind_z = np.r_[ind_z, ind_z[-1]+1]

            return self.fv.x[ind_x], self.fv.y[ind_y], self.fv.z[ind_z]

    def set_solver(self, name, tol=1e-9, max_it=1000, precon=False, do_perm=False, comm=None):
        """Define parameters of solver to be used during forward modelling.

        Parameters
        ----------
        name : `string` or `callable`
            If `string`: name of solver (mumps, pardiso, umfpack, or superlu)
            If `callable`: (iterative solver from scipy.sparse.linalg, e.g. bicgstab)
        tol : float, optional
            Tolerance for the iterative solver
        max_it : int, optional
            Max nbr of iteration for the iterative solver
        precon : bool, optional
            Apply preconditionning.
        do_perm : bool, optional
            Apply inverse Cuthill-McKee permutation.
        comm : MPI Communicator or None
            for mumps solver

        Notes
        -----
        `precon` et `do_perm` are used only with iterative solvers.

        """
        self.fv.set_solver(name, tol, max_it, precon, do_perm, comm)

    def fromVTK(self, fieldname, filename):
        return self.fv.fromVTK(fieldname, filename)

    def fwd_mod(self, sigma=None, calc_J=False, calc_sens=False, q=None, keep_solver=True):
        """
        Forward modelling.

        Parameters
        ----------
        sigma : array_like
            Conductivity model (S/m).
        calc_J : bool, optional
            Compute current density.
        calc_sens : bool, optional
            Calculate sensitivity matrix.
        q : array_like
            Source term (used for MMR modelling).
        keep_solver : bool
            Keep solver if already instantiated

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        - Voltage at dipoles `p1p2` (mV)
        - Current density `calc_J` if True,
        - Sensitivity matrix if `calc_sens` is True.   ( mV / S/m )

        Notes
        -----
        - It is possible do the computation for poles or dipoles at
        injection and/or measurement points.  In the case of a pole, one of
        the coordinate must be set to np.inf.

        - Sensitivity calculation currently implemented only for current
        intensity equal at all injection points.

        - It is possible to compute current density when `p1p2` is not defined;
        the returned voltage is None in such case.

        - `calc_J` & `calc_sens` are mutually exclusive.

        """

        if self.verbose:
            print('\nForward modelling - DC resistivity')
            self.print_info()
            if self.c1c2 is not None:
                print('    {0:d} combinations of dipoles'.format(self.c1c2.shape[0]))
            if self.apply_bc:
                print('    Correction at boundary: applied')
            else:
                print('    Correction at boundary: not applied')
            if self.fv.solver_A is not None:
                self.fv.solver_A.print_info()

        if self.c1c2 is None:
            if self.sort_electrodes is False and self.c1c2_u is None:
                raise ValueError('Source term undefined')

        if q is None:
            self._check_cs()

        if calc_sens is True:
            if np.any(self.cs != self.cs[0]):
                raise ValueError('Sensitivity computation implemented only for I equal at all dipoles.')

        if calc_J is False:
            if q is not None:
                # we don't need measurement points here, we need the potential
                # over the grid (for MMR), which will be stored in self.u
                pass
            # make sure we have measurement points if we don't compute current density
            elif self.p1p2 is None:
                raise ValueError('Measurement points undefined')
        else:
            if calc_sens is True:
                raise ValueError('`calc_J` & `calc_sens` are mutually exclusive.')

        self.keep_c1c2 = not calc_sens  # il faut calculer la réponse des électrodes individuellement

        if self.c1c2 is not None:
            #  c1c2 may be None if we are calling from MMR modeling
            if self.sort_electrodes:
                if self.electrodes_sorted is False:
                    self._sort_electrodes()
            elif self.in_inv is False:
                self.c1c2_u = self.c1c2.copy()

        if self.keep_c1c2:
            c1c2 = self.c1c2_u
        else:
            c1c2 = self.c12_u

        if q is not None:
            self.q = sp.csr_matrix(q)
            self.u0 = None
        elif self.apply_bc:
            self._boundary_correction(sigma)
        else:
            self._build_q()

        if sigma is not None:
            A, M = self._build_A(sigma)
            if self.fv.solver_A is None or keep_solver is False:
                self.fv.solver_A = Solver(self.fv.get_solver_params(), A, self.verbose)
            else:
                self.fv.solver_A.A = A
        elif self.fv.solver_A is None:
            raise RuntimeError('Variable sigma should be given as input.')

        self.u = np.empty((self.fv.nc, self.q.shape[1]))
        self.u[:, :c1c2.shape[0]] = self.fv.solver_A.solve(self.q[:, :c1c2.shape[0]])

        if self.p1p2 is None:
            data = None
        else:
            data = self._units_scaling * self._get_data()

        if calc_sens:
            if self.verbose:
                print('  Computing sensitivity ... ', end='', flush=True)
            # TODO : use formulation of Haber (book), when number of p1p2 > number of voxels
            self.u[:, c1c2.shape[0]:] = self.fv.solver_A.solve(self.q[:, c1c2.shape[0]:], verbose=False)

            sens = np.empty((self.ind_roi.size, self.c1c2.shape[0]))
            S = self.fv.build_M(sigma*sigma)
            Dm = sp.diags(1.0/sigma)

            if self.verbose:
                print('  Filling sensitivity matrix ... ', end='', flush=True)

            Gf = self.fv.build_G_faces()

            # items = [(n, c1c2, Dm, S, Gf) for n in range(self.c1c2.shape[0])]
            # with Pool() as pool:
            #     results = pool.starmap_async(self._fill_jacobian, items)
            #     for tmp, n in results.get():
            #         sens[:, n] = tmp
            #     pool.close()
            #     pool.join()

            for n in range(self.c1c2.shape[0]):
                if self.verbose:
                    if n == 0:
                        msg = ''
                        pre_msg = ''
                    nback = len(msg) - len(pre_msg)
                    pre_msg = ''
                    for _ in range(nback):
                        pre_msg += '\b'
                    msg = pre_msg+'  term '+str(n+1)+' of '+str(self.c1c2.shape[0])+' '
                    print(msg, end='', flush=True)

                sens[:, n], _ = self._fill_jacobian(n, c1c2, Dm, S, Gf)

            if self.sort_electrodes and self.sort_back is not None:
                sens = sens[:, self.sort_back]

            if self.verbose:
                print('done.\nEnd of modelling.')

            return data, sens

        elif calc_J:
            if self.verbose:
                print('  Computing current density ...', end='', flush=True)
            J = np.empty((self.fv.nf, self.q.shape[1]))
            for ns in range(self.q.shape[1]):
                J[:, ns] = -M @ self.fv.G @ self.u[:, ns]

            # if self.sort_electrodes and self.sort_back is not None:
            #     J = J[:, self.sort_back]

            if self.verbose:
                print('done.\nEnd of modelling.')
            return data, J
        else:
            if self.verbose:
                print('End of modelling.')
            return data

    def calc_reg(self, Q, par, xc, xref, WGx, WGy, WGz, m_active):
        """Compute regularization matrix.
        """
        Gx, Gy, Gz = self.fv.extract_Gxyz()

        if WGx is None:
            WGx = np.ones((Gx.shape[0], 1))
            WGx_save = WGx.copy()
        else:
            WGx_save = WGx.copy()
        if WGy is None:
            WGy = np.ones((Gy.shape[0], 1))
            WGy_save = WGy.copy()
        else:
            WGy_save = WGy.copy()
        if WGz is None:
            WGz = np.ones((Gz.shape[0], 1))
            WGz_save = WGz.copy()
        else:
            WGz_save = WGz.copy()

        if par.smooth_type == 'blocky':
            cx = Gx @ (xc-xref)
            cy = Gy @ (xc-xref)
            cz = Gz @ (xc-xref)

            wtx = np.sum(cx*cx, axis=0) / (np.sum(np.abs(cx), axis=0) *
                                           np.abs(cx))
            wty = np.sum(cy*cy, axis=0) / (np.sum(np.abs(cy), axis=0) *
                                           np.abs(cy))
            wtz = np.sum(cz*cz, axis=0) / (np.sum(np.abs(cz), axis=0) *
                                           np.abs(cz))

            bloc_cutoff = 1.e-6
            wtx[wtx > 10] = 10
            wtx[wtx < bloc_cutoff] = bloc_cutoff
            wty[wty > 10] = 10
            wty[wty < bloc_cutoff] = bloc_cutoff
            wtz[wtz > 10] = 10
            wtz[wtz < bloc_cutoff] = bloc_cutoff

            WGx = WGx_save * wtx
            WGy = WGy_save * wty
            WGz = WGz_save * wtz

        elif par.smooth_type == 'ekblom':
            ci = xc-xref
            cx = Gx @ ci
            cz = Gz @ ci

            wtx = par.ekblom_p * (cx*cx + par.minsupport_e**2)**-((par.ekblom_p/2-1)/2)
            wtz = par.ekblom_p * (cz*cz + par.minsupport_e**2)**-((par.ekblom_p/2-1)/2)
            wti = par.ekblom_p * (ci*ci + par.minsupport_e**2)**-((par.ekblom_p/2-1))
            wtx[wtx > 2] = 2
            wtz[wtz > 2] = 2

            WGx = wtx
            WGz = wtz
            WGy = wti

        elif par.smooth_type == 'min. support':
            cx = Gx @ xc
            cy = Gy @ xc
            cz = Gz @ xc

            wtx = (np.sqrt(2)*par.minsupport_e) / (cx*cx + par.minsupport_e**2)
            wty = (np.sqrt(2)*par.minsupport_e) / (cy*cy + par.minsupport_e**2)
            wtz = (np.sqrt(2)*par.minsupport_e) / (cz*cz + par.minsupport_e**2)
            wtx[wtx > 2] = 2
            wty[wty > 2] = 2
            wtz[wtz > 2] = 2

            WGx = wtx
            WGy = wty
            WGz = wtz

        return self.calc_WtW(Q, par, m_active, WGx, WGy, WGz), WGx, WGy, WGz

    def calc_WtW(self, wt, par, m_active, WGx=None, WGy=None, WGz=None):
        """Compute model weighting matrix.

        Parameters
        ----------
        wt : array_like
            Weight.
        par :
            Weighting parameters.
        m_active : array_like
            Cellules du modèles actives.
        WGx : array_like
            Poids de lissage des cellules en x.
        WGy : array_like
            Poids de lissage des cellules en y.
        WGz : array_like
            Poids de lissage des cellules en z.

        Returns
        -------
        output : `csr_matrix`
            Model weighting matrix.
        """

        Gx, Gy, Gz = self.fv.extract_Gxyz()

        if WGx is not None:
            Gx = sp.diags(WGx.diagonal(), shape=(self.fv.nfx, self.fv.nfx), format='csr') @ Gx
        if WGy is not None:
            Gy = sp.diags(WGy.diagonal(), shape=(self.fv.nfy, self.fv.nfy), format='csr') @ Gy
        if WGz is not None:
            Gz = sp.diags(WGz.diagonal(), shape=(self.fv.nfz, self.fv.nfz), format='csr') @ Gz

        Gs = sp.vstack((par.alx*Gx, par.aly*Gy, par.alz*Gz), format='csr')
        V = sp.diags(np.ones((self.fv.nc,)), format='csr')
        Wt = sp.diags(wt, shape=(self.fv.nc, self.fv.nc), format='csr')

        WtW = Wt.T @ (Gs.T @ Gs + par.als * V) @ Wt
        WtW = WtW[m_active, :]
        return WtW[:, m_active]

    def print_info(self, file=None):
        self.fv.print_info(file)
        if self.roi is not None:
            print('    Region of interest:', file=file)
            print('      X min: {0:e}\tX max: {1:e}'.format(self.roi[0], self.roi[1]), file=file)
            print('      Y min: {0:e}\tY max: {1:e}'.format(self.roi[2], self.roi[3]), file=file)
            print('      Z min: {0:e}\tZ max: {1:e}'.format(self.roi[4], self.roi[5]), file=file)

    def save_sensitivity(self, sens, basename):
        """Save sensitivity to VTK files.

        Parameters
        ----------
        sens : ndarray
            Sensitivity
        basename : str
            basename for output files (1 per dipole).
        """
        x, y, z = self.get_roi_nodes()
        # grille temporaire pour sauvegarder sens
        g2 = GridFV(x, y, z)
        # make 1 file for each injection dipole
        for n in range(self.n_c1c2_u):
            (ind,) = np.where(n == self.ind_c1c2)
            fields = {}
            for i in ind:
                name = "∂d/∂m - P1: " + str(self.p1p2[i, 0]) + " " + str(self.p1p2[i, 1]) + " "
                name += str(self.p1p2[i, 2]) + ", "
                name += "P2: " + str(self.p1p2[i, 3]) + " " + str(self.p1p2[i, 4]) + " " + str(self.p1p2[i, 5])
        
                fields[name] = sens[:, i]

            src = "C1: " + str(self.c1c2_u[n, 0]) + " " + str(self.c1c2_u[n, 1]) + " " + str(self.c1c2_u[n, 2])
            src += ", C2: " + str(self.c1c2_u[n, 3]) + " " + str(self.c1c2_u[n, 4]) + " " + str(self.c1c2_u[n, 5])
            metadata = {"Source dipole": src}
            filename = basename + "_dc_sens_dip" + str(n + 1)
            g2.toVTK(fields, filename, metadata=metadata)

    def save_current_density(self, J, basename):
        """Save current density to VTK files.

        Parameters
        ----------
        J : ndarray
            Current density
        basename : str
            basename for output files (1 per dipole).
        """
        # make 1 file for each injection dipole
        for n in range(self.n_c1c2_u):
            src = "C1: " + str(self.c1c2_u[n, 0]) + " " + str(self.c1c2_u[n, 1]) + " " + str(self.c1c2_u[n, 2])
            src += ", C2: " + str(self.c1c2_u[n, 3]) + " " + str(self.c1c2_u[n, 4]) + " " + str(self.c1c2_u[n, 5])
            metadata = {"Source dipole": src}

            jx, jy, jz = self.fv.extract_xyz_faces(J[:, n])
            filename = basename+'_Jx_dc_dip'+str(n+1)
            self.fv.toVTK({'Jx': jx}, filename, component='x', metadata=metadata)
            filename = basename+'_Jy_dc_dip'+str(n+1)
            self.fv.toVTK({'Jy': jy}, filename, component='y', metadata=metadata)
            filename = basename+'_Jz_dc_dip'+str(n+1)
            self.fv.toVTK({'Jz': jz}, filename, component='z', metadata=metadata)

    def _build_A(self, sigma):
        # Build LHS matrix
        M = self.fv.build_M(sigma)
        A = self.fv.D @ M @ self.fv.G
        # I, J, V = sp.find(A[0, :])
        # for jj in J:
        #     A[0, jj] = 0.0
        # A[0, 0] = 1.0/(self.fv.hx[0] * self.fv.hy[0] * self.fv.hz[0])
        A[0, 0] += 1.0/(self.fv.hx[0] * self.fv.hy[0] * self.fv.hz[0])
        
        return A, M

    def _build_Q(self):

        # make sure electrodes are at least at the depth of the first cell center
        p1p2 = self.p1p2.copy()
        p1p2[:, :3] = self.fv.process_surface_elec(p1p2[:, :3])
        p1p2[:, 3:6] = self.fv.process_surface_elec(p1p2[:, 3:6])

        self.Q = []
        for ns in range(self.n_c1c2_u):
            ind = self.ind_c1c2 == ns
            Q = self.fv.linear_interp(p1p2[ind, 0], p1p2[ind, 1], p1p2[ind, 2])
            Q -= self.fv.linear_interp(p1p2[ind, 3], p1p2[ind, 4], p1p2[ind, 5])
            self.Q.append(Q)

    def _build_q(self):

        if self.keep_c1c2:
            c1c2 = self.c1c2_u.copy()
            # make sure electrodes are at least at the depth of the first cell center
            c1c2[:, :3] = self.fv.process_surface_elec(c1c2[:, :3])
            c1c2[:, 3:6] = self.fv.process_surface_elec(c1c2[:, 3:6])

            if self.cs is None:
                warnings.warn('Current source intensity undefined, using 1 A', RuntimeWarning, stacklevel=2)
                self.cs = np.ones((c1c2.shape[0],))
            cs = self.cs12_u
        else:
            c1c2 = np.vstack((self.c12_u, self.p12_u))
            # make sure electrodes are at least at the depth of the first cell center
            c1c2[:, :3] = self.fv.process_surface_elec(c1c2[:, :3])
            cs = np.r_[self.cs_u, self.cp_u]

        # volume des voxels
        iv = 1.0 / self.fv.volume_voxels()
        q = sp.lil_matrix((self.fv.nc, c1c2.shape[0]))
        for i in range(c1c2.shape[0]):
            Q = self.fv.linear_interp(c1c2[i, 0], c1c2[i, 1], c1c2[i, 2])
            if c1c2.shape[1] == 6:
                Q -= self.fv.linear_interp(c1c2[i, 3], c1c2[i, 4], c1c2[i, 5])
            q[:, i] = -cs[i] * Q.toarray() * iv
        self.q = q.tocsr()
        self.u0 = None

    def _boundary_correction(self, ref_model):
        """Apply correction described in Pidlisecky et al. 2007"""

        if self.keep_c1c2:
            c1c2 = self.c1c2_u.copy()
            # keep track of electrodes below the surface
            below_surf_c1 = self.fv.below_surface(c1c2[:, :3])
            below_surf_c2 = self.fv.below_surface(c1c2[:, 3:6])
            # make sure electrodes are at least at the depth of the first cell center
            c1c2[:, :3] = self.fv.process_surface_elec(c1c2[:, :3])
            c1c2[:, 3:6] = self.fv.process_surface_elec(c1c2[:, 3:6])
            if self.cs is None:
                warnings.warn('Current source intensity undefined, using 1 A', RuntimeWarning, stacklevel=2)
                self.cs = np.ones((c1c2.shape[0],))
            cs = self.cs12_u
        else:
            c1c2 = np.vstack((self.c12_u, self.p12_u))
            # keep track of electrodes below the surface
            below_surf_c1 = self.fv.below_surface(c1c2[:, :3])
            # make sure electrodes are at least at the depth of the first cell center
            c1c2[:, :3] = self.fv.process_surface_elec(c1c2[:, :3])
            cs = np.r_[self.cs_u, self.cp_u]

        x, y, z = self.fv.centre_voxels()
        avg_cond = gmean(ref_model.flatten())

        self.u0 = np.empty((self.fv.nc, c1c2.shape[0]))

        # turn off warning b/c we get a divide by zero that we will fix later
        np.seterr(divide='ignore')
        if self.keep_c1c2:
            for i in range(c1c2.shape[0]):
                pve1 = np.sqrt((x - c1c2[i, 0])**2 + (y - c1c2[i, 1])**2 + (z - c1c2[i, 2])**2)
                # norm of negative current electrode and 1st potential electrode
                nve1 = np.sqrt((x - c1c2[i, 3])**2 + (y - c1c2[i, 4])**2 + (z - c1c2[i, 5])**2)
                if below_surf_c1[i] or below_surf_c2[i]:
                    # norm of imaginary positive current electrode and 1st potential electrode
                    pveimag1 = np.sqrt((x - c1c2[i, 0])**2 + (y - c1c2[i, 1])**2 + (z + c1c2[i, 2])**2)
                    nveimag1 = np.sqrt((x - c1c2[i, 3])**2 + (y - c1c2[i, 4])**2 + (z + c1c2[i, 5])**2)
                    gf = 4.0
                else:
                    pveimag1 = np.inf
                    nveimag1 = np.inf
                    gf = 2.0
                self.u0[:, i] = cs[i]/(avg_cond*gf*np.pi) * (1./pve1 - 1./nve1 + 1./pveimag1 - 1./nveimag1).flatten()
                # note: this works for electrodes at infinity, numpy recognizes that 1./np.inf is 0
        else:
            # we have just one current electrode
            for i in range(c1c2.shape[0]):
                pve1 = np.sqrt((x - c1c2[i, 0])**2 + (y - c1c2[i, 1])**2 + (z - c1c2[i, 2])**2)
                if below_surf_c1[i]:
                    # norm of imaginary positive current electrode and 1st potential electrode
                    pveimag1 = np.sqrt((x - c1c2[i, 0])**2 + (y - c1c2[i, 1])**2 + (z + c1c2[i, 2])**2)
                    gf = 4.0
                else:
                    pveimag1 = np.inf
                    gf = 2.0
                self.u0[:, i] = cs[i] / (avg_cond*gf*np.pi) * (1./pve1 + 1./pveimag1).flatten()
        np.seterr(divide='warn')

        # check for singularities due to the source being on a node
        for i in range(c1c2.shape[0]):
            ind = np.nonzero(np.isinf(self.u0[:, i]))
            if ind[0].size > 0:
                for j in range(ind[0].size):
                    ix, iy, iz = self.fv.revind(ind[0][j])
                    if iz == 0:
                        self.u0[ind[0][j], i] = np.mean([self.u0[self.fv.ind(ix+1, iy, iz), i],
                                                         self.u0[self.fv.ind(ix, iy+1, iz), i],
                                                         self.u0[self.fv.ind(ix, iy, iz+1), i],
                                                         self.u0[self.fv.ind(ix-1, iy, iz), i],
                                                         self.u0[self.fv.ind(ix, iy-1, iz), i]])
                    else:
                        self.u0[ind[0][j], i] = np.mean([self.u0[self.fv.ind(ix+1, iy, iz), i],
                                                         self.u0[self.fv.ind(ix, iy+1, iz), i],
                                                         self.u0[self.fv.ind(ix, iy, iz+1), i],
                                                         self.u0[self.fv.ind(ix-1, iy, iz), i],
                                                         self.u0[self.fv.ind(ix, iy-1, iz), i],
                                                         self.u0[self.fv.ind(ix, iy, iz-1), i]])
        M = avg_cond * sp.eye(self.fv.nf, self.fv.nf)
        A = self.fv.D @ M @ self.fv.G
        # I, J, V = sp.find(A[0, :])
        # for jj in J:
        #     A[0, jj] = 0.0
        # A[0, 0] = 1.0/(self.fv.hx[0] * self.fv.hy[0] * self.fv.hz[0])
        A[0, 0] += 1.0 / (self.fv.hx[0] * self.fv.hy[0] * self.fv.hz[0])

        self.q = sp.csr_matrix(A @ self.u0)

    def _sort_electrodes(self):

        if self.verbose:
            print('  Sorting electrodes ...')
        
        if self.p1p2 is not None:
            if self.c1c2.shape[0] != self.p1p2.shape[0]:
                raise ValueError('Number of injection and measurement dipoles must be equal.')

            nc = self.c1c2.shape[1]
            tmp, self.sort_back = sortrows(np.hstack((self.c1c2, self.p1p2)), sort_back=True)
            self.c1c2 = tmp[:, :nc]
            self.p1p2 = tmp[:, nc:]

        else:
            self.c1c2, self.sort_back = sortrows(self.c1c2, sort_back=True)

        if self.cs is None:
            warnings.warn('Current source intensity undefined, using 1 A', RuntimeWarning, stacklevel=2)
            self.cs = np.ones((self.c1c2.shape[0],))

        # dipoles d'injection

        self.c1c2_u, ind, self.ind_c1c2 = np.unique(self.c1c2, axis=0,
                                                    return_index=True,
                                                    return_inverse=True)
        self.n_c1c2_u = self.c1c2_u.shape[0]
        self.cs12_u = self.cs[ind]

        c12 = np.vstack((self.c1c2[:, :3], self.c1c2[:, 3:]))
        self.c12_u = np.unique(c12, axis=0)
        self.ind_c1 = np.empty((self.c1c2.shape[0],), dtype=int)  # indices dans c1c2
        self.ind_c2 = np.empty((self.c1c2.shape[0],), dtype=int)  # indices dans c1c2
        self.cs_u = np.empty((self.c12_u.shape[0],))
        for n1 in range(self.c1c2.shape[0]):
            for n2 in range(self.c12_u.shape[0]):
                if all(self.c12_u[n2, :3] == self.c1c2[n1, :3]):
                    self.ind_c1[n1] = n2
                    self.cs_u[n2] = self.cs[n1]
                if all(self.c12_u[n2, :3] == self.c1c2[n1, 3:]):
                    self.ind_c2[n1] = n2
                    self.cs_u[n2] = self.cs[n1]

        self.ind_c1_u = np.empty((self.c1c2_u.shape[0],), dtype=int)  # indices dans c1c2_u
        self.ind_c2_u = np.empty((self.c1c2_u.shape[0],), dtype=int)  # indices dans c1c2_u
        for n1 in range(self.c1c2_u.shape[0]):
            for n2 in range(self.c12_u.shape[0]):
                if all(self.c12_u[n2, :] == self.c1c2_u[n1, :3]):
                    self.ind_c1_u[n1] = n2
                if all(self.c12_u[n2, :] == self.c1c2_u[n1, 3:]):
                    self.ind_c2_u[n1] = n2

        self.c12_u = self.c12_u[:, :3]
    
        if self.verbose:
            print('    Detected {0:d} injection dipole(s)'.format(self.n_c1c2_u))

        # dipoles de mesure
        if self.p1p2 is not None:
            p12 = np.vstack((self.p1p2[:, :3], self.p1p2[:, 3:]))
            self.p12_u = np.unique(p12, axis=0)

            self.ind_p1 = np.empty((self.p1p2.shape[0],), dtype=int)
            self.ind_p2 = np.empty((self.p1p2.shape[0],), dtype=int)
            self.cp_u = np.empty((self.p12_u.shape[0],))
            for n1 in range(self.p1p2.shape[0]):
                for n2 in range(self.p12_u.shape[0]):
                    if all(self.p12_u[n2, :3] == self.p1p2[n1, :3]):
                        self.ind_p1[n1] = n2
                        self.cp_u[n2] = self.cs[n1]
                    if all(self.p12_u[n2, :3] == self.p1p2[n1, 3:]):
                        self.ind_p2[n1] = n2
                        self.cp_u[n2] = self.cs[n1]

            self.p12_u = self.p12_u[:, :3]

            if self.verbose:
                print('    Detected {0:d} measurement electrode(s)'.format(self.p12_u.shape[0]))

        self.electrodes_sorted = True

    def _get_data(self, u=None):
        if self.Q is None:
            self._build_Q()
        if u is None:
            u = self.u
        data = np.array([])
        if self.keep_c1c2:
            for ns in range(self.c1c2_u.shape[0]):
                data = np.r_[data, self.Q[ns] @ u[:, ns]]
        else:
            # we must combine u fields from c1 and c2
            for ns in range(self.n_c1c2_u):
                uu = u[:, self.ind_c1_u[ns]] - u[:, self.ind_c2_u[ns]]
                data = np.r_[data, self.Q[ns] @ uu]
        if self.sort_electrodes and self.sort_back is not None:
            data = data[self.sort_back]
        return data

    def _get_u(self, d):
        u = None
        if self.c1c2_u.shape[0] == 1:
            Q = self.Q
        else:
            Q = sp.vstack(self.Q)
            
        for ns in range(self.c1c2_u.shape[0]):
            ind = self.ind_c1c2 == ns
            if u is None:
                u = Q[ind, :].T @ d[ind]
            else:
                u = np.c_[u, Q[ind, :].T @ d[ind]]
        return sp.csr_matrix(u)

    def _calc_Gv(self, m, m_ref, m_active, u, v):

        v = np.atleast_2d(v)
        mp = m_ref.copy()
        mp[m_active] = m

        S = self.fv.build_M(np.exp(mp)).power(2)

        Dm = sp.csr_matrix((np.exp(-m), (np.arange(m.size), np.arange(m.size))))

        gu = np.empty((self.fv.nc, u.shape[1]))

        if v.shape[1] == u.shape[1]:
            for i in range(u.shape[1]):
                Gc = self.fv.build_G(self.fv.G @ u[:, i])
                Gc = Gc[:, m_active]

                gui = self.fv.D @ (S @ (Gc @ (Dm @ v[:, i])))
                gu[:, i] = gui
        elif v.shape[1] < u.shape[1]:
            for i in range(u.shape[1]):
                Gc = self.fv.build_G(self.fv.G @ u[:, i])
                Gc = Gc[:, m_active]

                gui = self.fv.D @ (S @ (Gc @ (Dm @ v)))
                gu[:, i] = gui.flatten()
        return sp.csr_matrix(gu)

    def _calc_Gvt(self, m, m_ref, m_active, u, v):

        mp = m_ref.copy()
        mp[m_active] = m

        S = self.fv.build_M(np.exp(mp)).power(2)

        Dm = sp.csr_matrix((np.exp(-m), (np.arange(m.size), np.arange(m.size))))

        gu = np.empty((m.size, u.shape[1]))
        for i in range(u.shape[1]):
            Gc = self.fv.build_G(self.fv.G @ u[:, i])
            Gc = Gc[:, m_active]

            gui = Dm.T @ (Gc.T @ (S.T @ (self.fv.D.T @ v[:, i])))
            gu[:, i] = gui.flatten()
        return np.sum(gu, axis=1)

    def _fill_jacobian(self, n, c1c2, Dm, S, Gf):
        u = self.u[:, self.ind_c1[n]] - self.u[:, self.ind_c2[n]]
        u_r = self.u[:, c1c2.shape[0] + self.ind_p1[n]] - self.u[:, c1c2.shape[0] + self.ind_p2[n]]
        v = sp.diags(self.fv.G @ u)
        Gc = v @ Gf
        # Gc = self.build_G(self.G @ u)
        A = Dm @ Gc.T @ S
        tmp = -self._units_scaling * A @ self.fv.G @ u_r
        return tmp[self.ind_roi], n

    def _check_cs(self):
        """Verify validity of current source intensity."""
        if self._cs is None and self.cs12_u is None:
            raise ValueError('Current source undefined')
        if self.c1c2 is not None:
            c1c2 = self.c1c2
        else:
            c1c2 = self.c1c2_u
        if self._cs is None:
            self._cs = self.cs12_u
        elif np.isscalar(self._cs):
            self._cs = self._cs + np.zeros((c1c2.shape[0],))
        elif isinstance(self._cs, np.ndarray):
            if self._cs.ndim != 1:
                self._cs = self._cs.flatten()
            if self._cs.size != c1c2.shape[0]:
                raise ValueError(f'Number of current source ({self._cs.size}) should match number of source terms ({c1c2.shape[0]})')


# %% Solutions analytiques

class VerticalDyke():
    """
    Compute voltage for a profile/sounding crossing a vertical dyke.

    Reference
    ---------

    Robert Gaige Van Nostrand and Kenneth L. Cook
    Interpretation of resistivity data
    Professional Paper 499
    USGS
    https://pubs.er.usgs.gov/publication/pp499

    DOI: 10.3133/pp499

    """
    def __init__(self, rho1, rho2, rho3, thickness, xd=0.0, n_max=100):
        """
        Parameters
        ----------
        rho1 : float
            résistivity on the "left" of the dyke.
        rho2 : float
            dyke resistivity
        rho3 : float
            résistivity on the "right" of the dyke.
        thickness : float
            dyke thickness
        xd : float
            X coordinate of the left flank of the dyke

        """
        self.rho1 = rho1
        self.rho2 = rho2
        self.rho3 = rho3
        self.b = thickness
        self.xd = xd
        self.n_max = n_max

    @property
    def n_max(self):
        return self.n[-1]

    @n_max.setter
    def n_max(self, n_max):
        self.n = np.arange(n_max+1)

    def fwd_mod(self, c1c2, p1p2, I=1.0, return_all=False):
        """
        Calcul le potentiel.

        Parameters
        ----------
        c1c2 : array_like of float
            x,y,z coordinates of injection electrodes (poles or dipoles)
        p1p2 : array_like of float
            x,y,z coordinates of potential electrodes (poles or dipoles)
        I : float
            Current intensity
        return_all : bool
            if True, Returns all 4 combinations of voltages c1p1, c1p2,
            c2p1, & c2p2, in addition to total voltage

        Returns
        -------
        ndarray : (nobs,)
            Calculated voltage


        Notes
        -----
        c1c2 & p1p2 can have 3 or 6 columns, to represent poles or  dipoles, but
        must hold the same number of rows, which correspond to the possible
        c1p1, c1p2, c2p1, & c2p2 combinations.

        The profile varies along x; y and z coordinates are ignored.

        """

        c1p1 = np.empty((c1c2.shape[0],))
        c1p2 = np.empty((c1c2.shape[0],))
        c2p1 = np.empty((c1c2.shape[0],))
        c2p2 = np.empty((c1c2.shape[0],))
        # pour chaque dipole d'injection
        for ns in range(c1c2.shape[0]):
            c1p1[ns] = self._c_p(c1c2[ns, 0], p1p2[ns, 0], I)
            if p1p2.shape[1] > 3:
                c1p2[ns] = self._c_p(c1c2[ns, 0], p1p2[ns, 3], I)
            else:
                c1p2[ns] = 0.
            if c1c2.shape[1] > 3:
                c2p1[ns] = self._c_p(c1c2[ns, 3], p1p2[ns, 0], I)
                if p1p2.shape[1] > 3:
                    c2p2[ns] = self._c_p(c1c2[ns, 3], p1p2[ns, 3], I)
                else:
                    c2p2[ns] = 0.
            else:
                c2p1[ns] = 0.
                c2p2[ns] = 0.
        data = c1p1 - c1p2 - c2p1 + c2p2
        if return_all:
            return data, c1p1, c1p2, c2p1, c2p2
        else:
            return data

    def _c_p(self, x_c, x_p, I, z_p=0.0):
        n = self.n
        b = self.b
        x = x_p - x_c        # this is z in van Nostrand & Cook
        x1 = self.xd - x_c   # this is z_1 in van Nostrand & Cook
        x2 = x1 + self.b     # this is z_2 in van Nostrand & Cook

        r = z_p                  # this is the depth of measurement point
        R = np.sqrt(x*x + r*r)   # distance from source to measurement point

        k21 = (self.rho2 - self.rho1) / (self.rho2 + self.rho1)
        k23 = (self.rho2 - self.rho3) / (self.rho2 + self.rho3)
        k_s = (k21*k23)**n
        two_nb = 2*n*b
        # check where is current electrode C
        if x_c < self.xd:
            # C is in medium 1
            a = I * self.rho1/(2. * np.pi)
            # check where is potential electrode
            if x_p < self.xd:
                # P is in medium 1
                return a * (1./R + k21 * np.sum(k_s / np.sqrt((two_nb + 2*x1 - x)**2 + r*r)) -
                                   k23 * np.sum(k_s / np.sqrt((two_nb + 2*x2 - x)**2 + r*r)))

            elif x_p > self.xd + self.b:
                # P is in medium 3
                return a * (1 + k21) * (1 - k23) * np.sum(k_s / np.sqrt((two_nb + x)**2 + r*r))
            else:
                # P is in the dike (medium 2)
                return a * (1 + k21) * (np.sum(k_s / np.sqrt((two_nb + x)**2 + r*r)) -
                                        k23 * np.sum(k_s / np.sqrt((two_nb + 2*x2 - x)**2 + r*r)))

        elif x_c > self.xd + self.b:
            # C is in medium 3
            a = I * self.rho3/(2. * np.pi)
            if x_p < self.xd:
                # P is in medium 1
                return a * (1+k21) * (1+k23) * np.sum(k_s / np.sqrt((two_nb - x)**2 + r*r))
            if x_p > self.xd + self.b:
                # P is in medium 3
                return a * (1./R + k23 * np.sum(k_s / np.sqrt((two_nb - 2*x2 + x)**2 + r*r)) -
                            k21 * np.sum(k_s / np.sqrt((two_nb - 2*x1 + x)**2 + r*r)))
            else:
                # P is in the dike (medium 2)
                return a * (1+k23) * (np.sum(k_s / np.sqrt((two_nb - x)**2 + r*r)) -
                                      k21 * np.sum(k_s / np.sqrt((two_nb - 2*x1 + x)**2 + r*r)))
        else:
            # C is in the dike (medium 2)
            a = I * self.rho2/(2. * np.pi)
            if x_p < self.xd:
                # P is in medium 1
                return a * (1 - k21) * (np.sum(k_s / np.sqrt((two_nb - x)**2 + r*r)) -
                                        k23 * np.sum(k_s / np.sqrt((two_nb + 2*x2 - x)**2 + r*r)))
                # in the above equation, van Nostrand & Cook have (1 + k12) -> typo?
            elif x_p > self.xd + self.b:
                # P is in medium 3
                return a * (1 - k23) * (np.sum(k_s / np.sqrt((two_nb + x)**2 + r*r)) -
                                        k21 * np.sum(k_s / np.sqrt((two_nb - 2*x1 + x)**2 + r*r)))
            else:
                # P is in the dike (medium 2)
                return a * (1./R + k21*k23 * np.sum(k_s / np.sqrt((two_nb - 2*x1 + 2*x2 - x)**2 + r*r)) -
                            k23 * np.sum(k_s / np.sqrt((two_nb + 2*x2 - x)**2 + r*r)) +
                            k21*k23 * np.sum(k_s / np.sqrt((two_nb - 2*x1 + 2*x2 + x)**2 + r*r)) -
                            k21 * np.sum(k_s / np.sqrt((two_nb - 2*x1 + x)**2 + r*r)))
