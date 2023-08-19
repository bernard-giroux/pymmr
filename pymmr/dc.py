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
import importlib
import re
import sys
import warnings

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

from pymmr.finite_volume import GridFV, Solver, build_from_vtk

# TODO: généraliser ROI pour voxels arbitraires
# TODO: ajouter le courant à c1c2 plutôt qu'entrer à part


# %% Some functions

@jit("boolean(boolean[:])", nopython=True)
def all(a):
    return a.sum() == a.size


def sortrows(a, sortback=False):
    """
    Sort rows of 2D array

    Parameters
    ----------
    a : 2D ndarray
        Array to process.
    sortback : bool, optional
        Returns array of indices to sortback. For this to work, all rows in
        input array must be different.

    Returns
    -------
    b : 2D ndarray
        sorted array.
    ind_back : 1D ndarray
        array of indices (if sortback is True)

    """
    tmp, cnt = np.unique(a, axis=0, return_counts=True)
    if sortback:
        if tmp.shape != a.shape:
            raise ValueError('All rows must be different when sortback is True')
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

class GridDC(GridFV):
    """Grid for DC resistivity modelling.

    Parameters
    ----------
    x : array of float
        Node coordinates along x
    y : array of float
        Node coordinates along y
    z : array of float
        Node coordinates along z
    comm : MPI Communicator or None
        If None, use MPI_COMM_WORLD
    """
    def __init__(self, x, y, z, comm=None):
        GridFV.__init__(self, x, y, z, comm)
        self._c1c2 = None
        self._cs = None
        self._p1p2 = None
        self.use_log_sigma = False
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
        self.verbose = True
        self.roi = None
        self.ind_roi = np.arange(self.nc)
        self.in_inv = False    # set to True when grid is used in inversion
        self.c1c2_u = None
        self.cs12_u = None

    @property
    def c1c2(self):
        """Coordinates of injection points."""
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
                raise ValueError('Size of source term must be nsrc x 6')
        elif tmp.ndim == 2:
            if tmp.shape[1] != 6:
                raise ValueError('Size of source term must be nsrc x 6')
        else:
            raise ValueError('Size of source term must be nsrc x 6')
        for ns in range(tmp.shape[0]):
            if self.is_inside(tmp[ns, 0], tmp[ns, 1], tmp[ns, 2]) is False or \
               self.is_inside(tmp[ns, 3], tmp[ns, 4], tmp[ns, 5]) is False:
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
        """Coordinates of measurement points."""
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
            if self.is_inside(tmp[ns, 0], tmp[ns, 1], tmp[ns, 2]) is False or \
                (self.is_inside(tmp[ns, 3], tmp[ns, 4], tmp[ns, 5]) is False
                 and not np.any(tmp[ns, 3:] == np.inf)):
                    raise ValueError('Measurement point outside grid')
        self._p1p2 = tmp
        self.electrodes_sorted = False
        self.Q = None  # reset interpolation matrix because electrodes will be sorted

    def check_cs(self):
        if self._cs is None:
            raise ValueError('Current source undefined')
        if self.c1c2 is not None:
            c1c2 = self.c1c2
        else:
            c1c2 = self.c1c2_u
        if np.isscalar(self._cs):
            self._cs = self._cs + np.zeros((c1c2.shape[0],))
        elif isinstance(self._cs, np.ndarray):
            if self._cs.ndim != 1:
                self._cs = self._cs.flatten()
            if self._cs.size != c1c2.shape[0]:
                raise ValueError('Number of current source should match number of source terms')

    def set_roi(self, roi):
        """Define region of interest for computing sentivivity or for inversion.

        Parameters
        ----------
        roi : tuple of float
            Tuple holding xmin, xmax, ymin, ymax, zmin, zmax
        """

        if roi is None:
            return
        self.roi = roi
        xmin, xmax, ymin, ymax, zmin, zmax = roi
        if self.is_inside(xmin, ymin, zmin) is False or \
            self.is_inside(xmax, ymax, zmax) is False:
                raise ValueError('Region of interest extending beyond grid')

        ind_x, = np.where(np.logical_and(self.xc >= xmin, self.xc < xmax))
        ind_y, = np.where(np.logical_and(self.yc >= ymin, self.yc < ymax))
        ind_z, = np.where(np.logical_and(self.zc >= zmin, self.zc < zmax))

        self.ind_roi = self.ind(ind_x, ind_y, ind_z)

    def get_roi_nodes(self):
        """
        Returns coordinates of nodes inside region of interest.

        Returns
        -------
        tuple of ndarray
            x, y, z
        """
        if np.array_equal(self.ind_roi, np.arange(self.nc)):
            return self.x, self.y, self.z
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = self.roi
            # get indices of voxels
            ind_x, = np.where(np.logical_and(self.xc >= xmin, self.xc < xmax))
            ind_y, = np.where(np.logical_and(self.yc >= ymin, self.yc < ymax))
            ind_z, = np.where(np.logical_and(self.zc >= zmin, self.zc < zmax))

            # add node at end
            ind_x = np.r_[ind_x, ind_x[-1]+1]
            ind_y = np.r_[ind_y, ind_y[-1]+1]
            ind_z = np.r_[ind_z, ind_z[-1]+1]

            return self.x[ind_x], self.y[ind_y], self.z[ind_z]

    def fwd_mod(self, sigma=None, c1c2=None, p1p2=None, calc_J=False,
                   calc_sens=False, q=None, cs=1.0, keep_solver=False):
        """
        Forward modelling.

        Parameters
        ----------
        sigma : array_like
            Conductivity model (S/m).
        c1c2 : array_like, optional
            Coordinates of injection points.
        p1p2 : array_like, optional
            Coordinates of measurement dipoles.
        calc_J : bool, optional
            Compute current density.
        calc_sens : bool, optional
            Calculate sensitivity matrix.
        q : array_like
            Source term (used for MMR modelling).
        cs : scalar or array_like
            Current at injection points, 1 A by default.
        keep_solver : bool
            not used, included for compatibility

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
        - `c1c2` et `p1p2` must be of equal size, i.e. n_obs x 6.  The
        6 columns correspond to coordinates x, y, & z of the dipoles, in 
        the order x1 y1 z1 x2 y2 z2

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
            if c1c2 is not None:
                print('    {0:d} combination of dipoles'.format(c1c2.shape[0]))
            self.print_solver_info()
            if self.apply_bc:
                print('    Correction at boundary: applied')
            else:
                print('    Correction at boundary: not applied')

        self.cs = cs

        if c1c2 is not None:
            self.c1c2 = c1c2
        elif self.c1c2 is None:
            if self.sort_electrodes is False and self.c1c2_u is None:
                raise ValueError('Source term undefined')

        self.check_cs()

        if calc_sens is True:
            if np.any(self.cs != self.cs[0]):
                raise ValueError('Sensitivity computation implemented only for I equal at all dipoles.')

        if calc_J is False:
            if q is not None:
                # we don't need measurement points here, we need the potential
                # over the grid (for MMR), which will be stored in self.u
                pass
            # make sure we have measurement points if we don't compute current density
            elif p1p2 is not None:
                self.p1p2 = p1p2
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
            self.solver_A = Solver(A, self.get_solver(), self.verbose)
        elif self.solver_A is None:
            raise RuntimeError('Variable sigma should be given as input.')

        self.u = np.empty((self.nc, self.q.shape[1]))
        self.u[:, :c1c2.shape[0]] = self.solver_A.solve(self.q[:, :c1c2.shape[0]])

        if self.p1p2 is None:
            data = None
        else:
            data = 1.e3 * self._get_data()  # to get mV

        if calc_sens:
            if self.verbose:
                print('  Computing sensitivity ... ', end='', flush=True)
            # TODO : use formulation of Haber (book), when number of p1p2 > number of voxels
            self.u[:, c1c2.shape[0]:] = self.solver_A.solve(self.q[:, c1c2.shape[0]:], verbose=False)

            sens = np.empty((self.ind_roi.size, self.c1c2.shape[0]))
            S = self.build_M(sigma*sigma)
            Dm = sp.diags(1.0/sigma)

            if self.verbose:
                print('  Filling sensitivity matrix ... ', end='', flush=True)

            # items = [(n, c1c2, Dm, S) for n in range(self.c1c2.shape[0])]
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

                sens[:, n], _ = self._fill_jacobian(n, c1c2, Dm, S)

            sens *= 1.e3   # we want mV for voltage units
            if self.sort_electrodes and self.sort_back is not None:
                sens = sens[:, self.sort_back]

            if self.verbose:
                print('done.\nEnd of modelling.')

            return data, sens

        elif calc_J:
            if self.verbose:
                print('  Computing current density ...', end='', flush=True)
            J = np.empty((self.nf, self.q.shape[1]))
            for ns in range(self.q.shape[1]):
                J[:, ns] = -M @ self.G @ self.u[:, ns]

            if self.sort_electrodes and self.sort_back is not None:
                J = J[:, self.sort_back]

            if self.verbose:
                print('done.\nEnd of modelling.')
            return data, J
        else:
            if self.verbose:
                print('End of modelling.')
            return data

    def calc_reg(self, Q, par, xc, xref, WGx, WGy, WGz, m_active):
        # extract Gx, Gy & Gz from G
        Gx = self.G[:self.nfx, :]
        Gy = self.G[self.nfx:(self.nfx+self.nfy), :]
        Gz = self.G[(self.nfx+self.nfy):, :]

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

    def _build_A(self, sigma):
        # Build LHS matrix
        M = self.build_M(sigma)
        A = self.D @ M @ self.G
        # I, J, V = sp.find(A[0, :])
        # for jj in J:
        #     A[0, jj] = 0.0
        # A[0, 0] = 1.0/(self.hx[0] * self.hy[0] * self.hz[0])
        A[0, 0] += 1.0/(self.hx[0] * self.hy[0] * self.hz[0])
        
        return A, M

    def _build_Q(self):

        # make sure electrodes are at least at the depth of the first cell center
        p1p2 = self.p1p2.copy()
        ind = p1p2[:, 2] < self.zc[0]
        p1p2[ind, 2] = self.zc[0]
        ind = p1p2[:, 5] < self.zc[0]
        p1p2[ind, 5] = self.zc[0]

        self.Q = []
        for ns in range(self.n_c1c2_u):
            ind = self.ind_c1c2 == ns
            Q = self.linear_interp(p1p2[ind, 0], p1p2[ind, 1], p1p2[ind, 2])
            Q -= self.linear_interp(p1p2[ind, 3], p1p2[ind, 4], p1p2[ind, 5])
            self.Q.append(Q)

    def _build_q(self):

        if self.keep_c1c2:
            c1c2 = self.c1c2_u.copy()
            # make sure electrodes are at least at the depth of the first cell center
            ind = c1c2[:, 2] < self.zc[0]
            c1c2[ind, 2] = self.zc[0]
            ind = c1c2[:, 5] < self.zc[0]
            c1c2[ind, 5] = self.zc[0]

            if self.cs is None:
                warnings.warn('Current source intensity undefined, using 1 A', RuntimeWarning, stacklevel=2)
                self.cs = np.ones((c1c2.shape[0],))
            cs = self.cs12_u
        else:
            c1c2 = np.vstack((self.c12_u, self.p12_u))
            # make sure electrodes are at least at the depth of the first cell center
            ind = c1c2[:, 2] < self.zc[0]
            c1c2[ind, 2] = self.zc[0]
            cs = np.r_[self.cs_u, self.cp_u]

        # volume des voxels
        iv = 1.0 / self.volume_voxels()
        q = sp.lil_matrix((self.nc, c1c2.shape[0]))
        for i in range(c1c2.shape[0]):
            Q = self.linear_interp(c1c2[i, 0], c1c2[i, 1], c1c2[i, 2])
            if c1c2.shape[1] == 6:
                Q -= self.linear_interp(c1c2[i, 3], c1c2[i, 4], c1c2[i, 5])
            q[:, i] = -cs[i] * Q.toarray() * iv
        self.q = q.tocsr()
        self.u0 = None

    def _boundary_correction(self, ref_model):

        if self.keep_c1c2:
            c1c2 = self.c1c2_u.copy()
            # keep track of electrodes below the surface
            below_surf_c1 = c1c2[:, 2] != self.z[0]
            below_surf_c2 = c1c2[:, 5] != self.z[0]
            # make sure electrodes are at least at the depth of the first cell center
            ind = c1c2[:, 2] < self.zc[0]
            c1c2[ind, 2] = self.zc[0]
            ind = c1c2[:, 5] < self.zc[0]
            c1c2[ind, 5] = self.zc[0]
            if self.cs is None:
                warnings.warn('Current source intensity undefined, using 1 A', RuntimeWarning, stacklevel=2)
                self.cs = np.ones((c1c2.shape[0],))
            cs = self.cs12_u
        else:
            c1c2 = np.vstack((self.c12_u, self.p12_u))
            # keep track of electrodes below the surface
            below_surf_c1 = c1c2[:, 2] != self.z[0]
            # make sure electrodes are at least at the depth of the first cell center
            ind = c1c2[:, 2] < self.zc[0]
            c1c2[ind, 2] = self.zc[0]
            cs = np.r_[self.cs_u, self.cp_u]

        z, y, x = np.meshgrid(self.zc, self.yc, self.xc, indexing='ij')
        if self.use_log_sigma:
            avg_cond = gmean(np.exp(ref_model.flatten()))
        else:
            avg_cond = gmean(ref_model.flatten())

        self.u0 = np.empty((self.nc, c1c2.shape[0]))

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
                    ix, iy, iz = self.revind(ind[0][j])
                    if iz == 0:
                        self.u0[ind[0][j], i] = np.mean([self.u0[self.ind(ix+1, iy, iz), i],
                                                         self.u0[self.ind(ix, iy+1, iz), i],
                                                         self.u0[self.ind(ix, iy, iz+1), i],
                                                         self.u0[self.ind(ix-1, iy, iz), i],
                                                         self.u0[self.ind(ix, iy-1, iz), i]])
                    else:
                        self.u0[ind[0][j], i] = np.mean([self.u0[self.ind(ix+1, iy, iz), i],
                                                         self.u0[self.ind(ix, iy+1, iz), i],
                                                         self.u0[self.ind(ix, iy, iz+1), i],
                                                         self.u0[self.ind(ix-1, iy, iz), i],
                                                         self.u0[self.ind(ix, iy-1, iz), i],
                                                         self.u0[self.ind(ix, iy, iz-1), i]])
        M = avg_cond * sp.eye(self.nf, self.nf)
        A = self.D @ M @ self.G
        # I, J, V = sp.find(A[0, :])
        # for jj in J:
        #     A[0, jj] = 0.0
        # A[0, 0] = 1.0/(self.hx[0] * self.hy[0] * self.hz[0])
        A[0, 0] += 1.0 / (self.hx[0] * self.hy[0] * self.hz[0])

        self.q = sp.csr_matrix(A @ self.u0)

    def _sort_electrodes(self):

        if self.verbose:
            print('  Sorting electrodes ...')
        
        if self.p1p2 is not None:
            if self.c1c2.shape[0] != self.p1p2.shape[0]:
                raise ValueError('Number of injection and measurement dipoles must be equal.')

            nc = self.c1c2.shape[1]
            tmp, self.sort_back = sortrows(np.hstack((self.c1c2, self.p1p2)), sortback=True)
            self.c1c2 = tmp[:, :nc]
            self.p1p2 = tmp[:, nc:]

        else:
            self.c1c2, self.sort_back = sortrows(self.c1c2, sortback=True)

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

        # extract Gx, Gy & Gz from G
        Gx = self.G[:self.nfx, :]
        Gy = self.G[self.nfx:(self.nfx+self.nfy), :]
        Gz = self.G[(self.nfx+self.nfy):, :]
        
        if WGx is not None:
            Gx = sp.diags(WGx.diagonal(), shape=(self.nfx, self.nfx), format='csr') @ Gx
        if WGy is not None:
            Gy = sp.diags(WGy.diagonal(), shape=(self.nfy, self.nfy), format='csr') @ Gy
        if WGz is not None:
            Gz = sp.diags(WGz.diagonal(), shape=(self.nfz, self.nfz), format='csr') @ Gz

        Gs = sp.vstack((par.alx*Gx, par.aly*Gy, par.alz*Gz), format='csr')
        V = sp.diags(np.ones((self.nc,)), format='csr')
        Wt = sp.diags(wt, shape=(self.nc, self.nc), format='csr')

        WtW = Wt.T @ (Gs.T @ Gs + par.als * V) @ Wt
        WtW = WtW[m_active, :]
        return WtW[:, m_active]

    def calc_WdW(self, wt, dobs, par):
        """Compute data weighting matrix.

        Parameters
        ----------
        wt : array_like
            Weight in %.
        dobs : array_like
            Observed data.
        par : 
            Weighting parameters.
        Returns
        -------
        output : `csr_matrix`
            Data weighting matrix.
        """
        dtw = 0.01 * wt.flatten() * np.abs(dobs.flatten()) + par.e
        dtw = 1. / dtw

        # normalisation
        dtw = dtw / dtw.max()

        dtw[wt.flatten() > par.max_err] = 0.0

        return sp.csr_matrix((dtw, (np.arange(dobs.size), np.arange(dobs.size))))

    def _calc_Gv(self, m, m_ref, m_active, u, v):

        v = np.atleast_2d(v)
        mp = m_ref.copy()
        mp[m_active] = m

        S = self.build_M(np.exp(mp)).power(2)

        Dm = sp.csr_matrix((np.exp(-m), (np.arange(m.size), np.arange(m.size))))

        gu = np.empty((self.nc, u.shape[1]))

        if v.shape[1] == u.shape[1]:
            for i in range(u.shape[1]):
                Gc = self.build_G(self.G @ u[:, i])
                Gc = Gc[:, m_active]

                gui = self.D @ (S @ (Gc @ (Dm @ v[:, i])))
                gu[:, i] = gui
        elif v.shape[1] < u.shape[1]:
            for i in range(u.shape[1]):
                Gc = self.build_G(self.G @ u[:, i])
                Gc = Gc[:, m_active]

                gui = self.D @ (S @ (Gc @ (Dm @ v)))
                gu[:, i] = gui.flatten()
        return sp.csr_matrix(gu)

    def _calc_Gvt(self, m, m_ref, m_active, u, v):

        mp = m_ref.copy()
        mp[m_active] = m

        S = self.build_M(np.exp(mp)).power(2)

        Dm = sp.csr_matrix((np.exp(-m), (np.arange(m.size), np.arange(m.size))))

        gu = np.empty((m.size, u.shape[1]))
        for i in range(u.shape[1]):
            Gc = self.build_G(self.G @ u[:, i])
            Gc = Gc[:, m_active]

            gui = Dm.T @ (Gc.T @ (S.T @ (self.D.T @ v[:, i])))
            gu[:, i] = gui.flatten()
        return np.sum(gu, axis=1)

    def _fill_jacobian(self, n, c1c2, Dm, S):
        u = self.u[:, self.ind_c1[n]] - self.u[:, self.ind_c2[n]]
        u_r = self.u[:, c1c2.shape[0] + self.ind_p1[n]] - self.u[:, c1c2.shape[0] + self.ind_p2[n]]
        Gc = self.build_G(self.G @ u)
        A = Dm @ Gc.T @ S
        tmp = A @ self.G @ u_r
        return tmp[self.ind_roi], n

    def print_info(self, file=None):
        print('    Grid: {0:d} x {1:d} x {2:d} voxels'.format(self.nx, self.ny, self.nz), file=file)
        print('      X min: {0:e}\tX max: {1:e}'.format(self.x[0], self.x[-1]), file=file)
        print('      Y min: {0:e}\tY max: {1:e}'.format(self.y[0], self.y[-1]), file=file)
        print('      Z min: {0:e}\tZ max: {1:e}'.format(self.z[0], self.z[-1]), file=file)
        if self.roi is not None:
            print('    Region of interest:', file=file)
            print('      X min: {0:e}\tX max: {1:e}'.format(self.roi[0], self.roi[1]), file=file)
            print('      Y min: {0:e}\tY max: {1:e}'.format(self.roi[2], self.roi[3]), file=file)
            print('      Z min: {0:e}\tZ max: {1:e}'.format(self.roi[4], self.roi[5]), file=file)

    def __getstate__(self):
        self.solver_A = None   # some solvers have attributes that are not picklable
        state = self.__dict__.copy()
        return state


# %% Solutions analytiques

class VerticalDyke():
    """
    Compute voltage for a profil/sounding crossing a vertical dyke.

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
        c1c2 : array of float
            x,y,z coordinates of injection electrodes (poles or dipoles)
        p1p2 : array of float
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

        The profil varies along x; y and z coordinates are ignored.

        """

        data = np.empty((c1c2.shape[0],))
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

# %% main


if __name__ == '__main__':

    # %% read arguments
    if len(sys.argv) > 1:

        basename = 'pymmr'
        g = None
        sigma = None
        c1c2 = None
        xo = None
        solver_name = 'umfpack'
        max_it = 1000
        tol = 1e-9
        verbose = True
        calc_J = False
        calc_sens = False
        roi = None
        include_c1c2p1p2 = False
        apply_bc = True
        precon = False
        do_perm = False
        cs = 1.0

        kw_pattern = re.compile('\s?(.+)\s?#\s?([\w\s]+),')

        # we should have a parameter file as input
        with open(sys.argv[1], 'r') as f:
            for line in f:
                kw_match = kw_pattern.search(line)
                if kw_match is not None:
                    value = kw_match[1].rstrip()
                    keyword = kw_match[2].rstrip()
                    if 'model' in keyword:
                        g = build_from_vtk(GridDC, value)
                        sigma = g.fromVTK('Conductivity', value)
                    elif 'basename' in keyword:
                        basename = value
                    elif 'solver' in keyword and 'name' in keyword:
                        if value in ('bicgstab', 'gmres'):
                            mod = importlib.import_module('scipy.sparse.linalg')
                            solver_name = getattr(mod, value)
                        else:
                            solver_name = value
                    elif 'solver' in keyword and 'max_it' in keyword:
                        max_it = int(value)
                    elif 'solver' in keyword and 'tolerance' in keyword:
                        tol = float(value)
                    elif 'precon' in keyword:
                        precon = int(value)
                    elif 'permut' in keyword:
                        do_perm = int(value)
                    elif 'source' in keyword and 'file' in keyword:
                        c1c2 = np.atleast_2d(np.loadtxt(value))
                    elif 'measurement' in keyword and 'file' in keyword:
                        p1p2 = np.atleast_2d(np.loadtxt(value))
                    elif 'source' in keyword and 'current' in keyword:
                        cs = float(value)
                    elif 'verbose' in keyword:
                        verbose = int(value)
                    elif 'compute' in keyword and 'current' in keyword:
                        calc_J = int(value)
                    elif 'compute' in keyword and 'sensitivity' in keyword:
                        calc_sens = int(value)
                    elif 'region' in keyword and 'interest' in keyword:
                        tmp = value.split()
                        if len(tmp) != 6:
                            raise ValueError('6 values needed to define ROI (xmin xmax ymin ymax zmin zmax')
                        roi = [float(x) for x in tmp]
                    elif 'boundary' in keyword and 'correction' in keyword:
                        apply_bc = int(value)

        if g is None:
            raise RuntimeError('Grid not defined, check input parameters')
            
        g.set_solver(solver_name, tol, max_it, precon, do_perm)

        g.verbose = verbose
        g.apply_bc = apply_bc

        if roi is not None:
            g.set_roi(roi)

        out = g.fwd_mod(sigma, c1c2, p1p2, calc_J=calc_J,
                           calc_sens=calc_sens, cs=cs)

        if calc_sens:
            if verbose:
                print('Saving sensitivity ... ', end='', flush=True)
            data, sens = out

            x, y, z = g.get_roi_nodes()
            # grille temporaire pour sauvegarder sens
            g2 = GridFV(x, y, z)
            # make 1 file for each injection dipole
            for n in range(g.n_c1c2_u):
                ind, = np.where(n == g.ind_c1c2)
                fields = {}
                for i in ind:
                    name = '∂d/∂m - P1: ' + str(p1p2[i, 0]) + ' ' + str(p1p2[i, 1]) + ' ' + str(p1p2[i, 2]) + ', '
                    name += 'P2: ' + str(p1p2[i, 3]) + ' ' + str(p1p2[i, 4]) + ' ' + str(p1p2[i, 5])

                    fields[name] = sens[:, i]

                fname = basename+'_dc_sens_dip'+str(n+1)
                g2.toVTK(fields, fname)
            if verbose:
                print('done.')
        elif calc_J:
            if verbose:
                print('Saving current density ... ', end='', flush=True)
            data, J = out

            # make 1 file for each injection dipole
            for n in range(g.n_c1c2_u):
                fname = basename+'_Jx_dc_dip'+str(n+1)
                g.toVTK({'Jx': J[:g.nfx, n]}, fname, component='x')
                fname = basename+'_Jy_dc_dip'+str(n+1)
                g.toVTK({'Jy': J[g.nfx:(g.nfx+g.nfy), n]}, fname, component='y')
                fname = basename+'_Jz_dc_dip'+str(n+1)
                g.toVTK({'Jz': J[(g.nfx+g.nfy):, n]}, fname, component='z')

            if verbose:
                print('done.')
        else:
            data = out

        if verbose:
            print('Saving modelled voltages ... ', end='', flush=True)
        # save data
        fname = basename+'_dc.dat'
        if include_c1c2p1p2:
            data = np.c_[c1c2, p1p2, data]
            header = 'c1_x c1_y c1_z c2_x c2_y c2_z p1_x p1_y p1_z p2_x p2_y p2_z v'
        else:
            header = 'v'
        np.savetxt(fname, data, header=header)

        if verbose:
            print('done.')
