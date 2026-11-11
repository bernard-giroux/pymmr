#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for magnetometric resistivity modelling

@author: giroux

Main reference:

@Article{chen02b,
  author       = {Chen, Jiuping and Haber, Eldad and Oldenburg, Douglas W.},
  journal      = gji,
  title        = {Three-dimensional numerical modelling and inversion of magnetometric resistivity data},
  year         = {2002},
  issn         = {1365-246X},
  number       = {3},
  pages        = {679--697},
  volume       = {149},
  creationdate = {2011-12-05T00:00:00},
  doi          = {10.1046/j.1365-246X.2002.01688.x},
  keywords     = {3-D, forward modelling, inversion, magnetometric resistivity, mineral exploration, MMR},
  owner        = {giroux},
  publisher    = {Blackwell Science Ltd},
  url          = {http://dx.doi.org/10.1046/j.1365-246X.2002.01688.x},
}

"""
import copy
from typing import Iterable

import numpy as np
import scipy.sparse as sp
from scipy.special import legendre_p_all

from scipy.stats.mstats import gmean

from pymmr.dc import GridDC
from pymmr.finite_volume import calc_padding, GridFV, Grid25FV, MeshFV, Solver, Solver25
from pymmr.kquad import KQuadrature, select_k
from pymmr.legendre import p_prime

MU0 = 4.e-7 * np.pi   # magnetic permeability of free space (H/m)

# %% class GridMMR

class GridMMR:
    """Grid for magnetometric resistivity modelling.

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
            or, if x has 2 dimensions, a tetrahedral mesh is built with
            pts : array of float
            tet: array of int
            (third element ignored)
        if tuple has 2 elements, a 2.5D grid is built (conductivity invariant along y) and the elements are
            x : array of float
                Node coordinates along x (m)
            z : array of float
                Node coordinates along z (m)
    units : str, optional
        B-field units
    comm : MPI Communicator, optional
        If None, MPI_COMM_WORLD will be used
    n_threads : int, optional

    Notes
    -----
    - It is possible to compute the DC resistivity response for injection
    dipoles that are different from the source term defined for MMR modeling.
    This has been implemented for allowing joint inversion of MMR & ERT data.

    - In 2.5D (see docs/mmr25.md), current electrodes and measurement points can be at any y; the
    fields are computed for `n_k` wavenumbers (selected for the survey) and transformed back at the
    distance along y between each electrode and each measurement point.
    """

    units_scaling_factors = {'pT': 1.e12, 'nT': 1.e9}

    def __init__(self, param_fv, units='pT', comm=None, n_threads=1):
        if len(param_fv) == 2:
            if np.ndim(param_fv[0]) != 1:
                raise ValueError('GridMMR: 2.5D not implemented for triangular meshes')
            self.fv = Grid25FV(param_fv, comm=comm, n_threads=n_threads)
        elif len(param_fv) == 3:
            if param_fv[0].ndim == 1:
                self.fv = GridFV(param_fv, comm=comm, n_threads=n_threads)
            else:
                self.fv = MeshFV(param_fv, comm=comm, n_threads=n_threads)
        else:
            raise ValueError('GridMMR: param_fv must have 2 or 3 elements')

        self.dc = GridDC(param_fv, units='mV', comm=comm, n_threads=n_threads)
        self.dc.verbose = 0
        self.acq_checked = False
        self._xs = None
        self._xo = None
        self.xs_u = None
        self.cs_u = None
        self.xo_all = None
        self.ind_back = None
        self.ind_s = None
        self.mask = None
        self.nobs_xs = None
        self.nobs_mmr = 0
        self.units = units
        self._verbose = 0
        self.pod_p1p2 = None
        self.Q_pod_e = None
        # 2.5D
        self.n_k = 30                 # number of wavenumbers
        self.k = None                 # wavenumbers (selected for the survey at the first fwd_mod)
        self.surface_images = False   # image of surface electrodes in the DC reference fields (see fwd_mod)
        self._k_range = None
        self._kq = None
        self._plan25 = None
        self._solvers_k = None

    @property
    def apply_bc(self):
        return self.dc.apply_bc

    @apply_bc.setter
    def apply_bc(self, val):
        if "dc" in self.__dict__:
            self.dc.apply_bc = val

    @property
    def in_inv(self):
        return self.dc.in_inv

    @in_inv.setter
    def in_inv(self, val):
        if 'dc' in self.__dict__:
            self.dc.in_inv = val

    @property
    def xs(self):
        """Coordinates of injection points for MMR modelling (m)."""
        return self._xs

    @xs.setter
    def xs(self, val):
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
            for e in (tmp[ns, :3], tmp[ns, 3:]):
                # an electrode with non-finite coordinates is at infinity (pole source)
                if np.all(np.isfinite(e)) and self.fv.is_inside(e[0], e[1], e[2]) is False:
                    raise ValueError('Source term outside grid')
        self._xs = tmp
        self.acq_checked = False

    @property
    def xo(self):
        """Coordinates of measurement points (m)."""
        return self._xo

    @xo.setter
    def xo(self, val):
        if val is None:
            return
        tmp = np.array(val, dtype=np.float64)
        if tmp.ndim == 1:
            if tmp.size == 3:
                tmp = tmp.reshape((1, tmp.size))
            else:
                raise ValueError('Observation points should be nobs x 3')
        elif tmp.ndim == 2:
            if tmp.shape[1] != 3:
                raise ValueError('Observation points should be nobs x 3')
        else:
            raise ValueError('Observation points should be nobs x 3')
        for ns in range(tmp.shape[0]):
            if self.fv.is_inside(tmp[ns, 0], tmp[ns, 1], tmp[ns, 2]) is False:
                raise ValueError('Observation points outside grid')
        self._xo = tmp
        self.acq_checked = False

    @property
    def cs(self):
        """Intensity of source current."""
        return self._cs

    @cs.setter
    def cs(self, val):
        self._cs = val

    @property
    def c1c2(self):
        """Coordinates of injection points for DC resistivity modelling (m)."""
        return self.dc.c1c2

    @c1c2.setter
    def c1c2(self, val):
        self.dc.c1c2 = val

    @property
    def p1p2(self):
        """Coordinates of measurement points for DC resistivity modelling (m)."""
        return self.dc.p1p2

    @p1p2.setter
    def p1p2(self, val):
        self.dc.p1p2 = val

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, val):
        if val not in GridMMR.units_scaling_factors:
            raise ValueError('Wrong units of B-field')
        self._units = val
        self._units_scaling = GridMMR.units_scaling_factors[val]

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, val):
        self._verbose = val
        self.fv.verbose = val
        if val > 2:
            self.dc.verbose = True

    def set_survey_mmr(self, xs: Iterable , xo: Iterable , cs: Iterable | float, pod_e: tuple | float = None) -> None:

        """Set survey variables.

        Parameters
        ----------
        xs : array_like
            Coordinates of injection points (m).
        xo : array_like
            Coordinates of measurement points (m).
        cs : scalar or array_like
            Intensity of current source
        pod_e : tuple
            length of electrical dipoles at measurement points (m).

        Notes
        -----
        - In the current implementation, current intensity values
        must be equal for each injection dipole, i.e. for xs and
        xo combinations involving a common set of xs values.  In
        other words, current values cannot differ for a given injection
        dipole, but can differ for different injection dipoles.
        """
        self.xs = xs
        self.xo = xo
        self.cs = cs
        self._check_cs()
        self.pod_e = pod_e

    def set_survey_ert(self, c1c2, p1p2, cs):
        self.dc.set_survey_ert(c1c2, p1p2, cs)

    def check_acquisition(self):
        """Check consistency of `xs` and `xo`."""
        if self.xs.shape[0] == self.xo.shape[0]:
            # on a des paires dipoles injection - pts de mesure
            self.xs_u, ind = np.unique(self.xs, axis=0, return_index=True)
            self.cs_u = self.cs[ind]
            self.xo_all = np.unique(self.xo, axis=0)
            # on crée l'ordre utilisé par la modélisation
            mod = np.c_[np.kron(self.xs_u, np.ones((self.xo_all.shape[0], 1))),
                        np.kron(np.ones((self.xs_u.shape[0], 1)), self.xo_all)]
            cxin = np.c_[self.xs, self.xo]
            self.ind_back = np.empty((self.xs.shape[0],), dtype=int)
            ind_s = np.empty((mod.shape[0],),
                             dtype=int)  # vecteur pour classer les données dans l'ordre de la modélisation
            self.mask = np.ones((mod.shape[0],), dtype=bool)
            self.nobs_xs = np.empty((self.xs_u.shape[0],), dtype=int)

            for n in np.arange(self.xs.shape[0]):
                i = np.where(np.all(np.tile(cxin[n, :], (mod.shape[0], 1)) == mod, axis=1))[0]
                self.ind_back[n] = i[0]
            for n in np.arange(mod.shape[0]):
                i = np.where(np.all(np.tile(mod[n, :], (cxin.shape[0], 1)) == cxin, axis=1))[0]
                if len(i) == 0:
                    self.mask[n] = False
                else:
                    ind_s[n] = i[0]
            self.ind_s = ind_s[self.mask]
            for n in np.arange(self.xs_u.shape[0]):
                i = np.where(np.all(np.tile(self.xs_u[n, :], (self.xs.shape[0], 1)) == self.xs, axis=1))[0]
                self.nobs_xs[n] = i.size

        else:
            # on va calculer toutes les combinaisons
            self.xs_u = self.xs.copy()
            self.cs_u = self.cs.copy()
            self.xo_all = self.xo.copy()
            self.ind_back = np.arange(self.xs.shape[0] * self.xo.shape[0])
            self.ind_s = np.arange(self.xs.shape[0] * self.xo.shape[0])
            self.mask = np.ones((self.xs.shape[0] * self.xo.shape[0],), dtype=bool)
            self.nobs_xs = self.xo.shape[0] + np.zeros((self.xs.shape[0],), dtype=int)
        self.nobs_mmr = np.sum(self.nobs_xs)

        if self.pod_e is not None:
            pod_e_dx, pod_e_dy, pod_e_dz = self.pod_e
            p1 = self.xo_all.copy()
            pod_p1p2 = [None, None, None]
            if pod_e_dx != 0.0:
                p2 = p1 + np.tile(np.array([[pod_e_dx, 0.0, 0.0]]), (p1.shape[0], 1))
                pod_p1p2[0] = np.c_[p1, p2]
            if pod_e_dy != 0.0:
                p2 = p1 + np.tile(np.array([[0.0, pod_e_dy, 0.0]]), (p1.shape[0], 1))
                pod_p1p2[1] = np.c_[p1, p2]
            if pod_e_dz != 0.0:
                p2 = p1 + np.tile(np.array([[0.0, 0.0, pod_e_dz]]), (p1.shape[0], 1))
                pod_p1p2[2] = np.c_[p1, p2]
            if pod_p1p2[0] is not None or pod_p1p2[1] is not None or pod_p1p2[2] is not None:
                self.pod_p1p2 = pod_p1p2

        self._plan25 = None
        self.acq_checked = True

    def set_solver(self, name, atol=1e-9, rtol=1e-5, max_it=1000, precon='0', do_perm=False):
        """Define parameters of solver to be used during forward modelling.

        Parameters
        ----------
        name : `string` or `callable`
            If `string`: name of solver (mumps, pardiso, umfpack, or superlu)
            If `callable`: (iterative solver from scipy.sparse.linalg, e.g. bicgstab)
        atol : float, optional
            Absolute tolerance for the iterative solver
        rtol : float, optional
            Relative tolerance for the iterative solver
        max_it : int, optional
            Max nbr of iteration for the iterative solver
        precon : string, optional
            Apply preconditioning, possible values are 'ilu', 'diag', or '0' for no preconditionning.
        do_perm : bool, optional
            Apply inverse Cuthill-McKee permutation.

        Notes
        -----
        `precon` et `do_perm` are used only with iterative solvers.

        """
        self.dc.fv.set_solver(name, atol, rtol, max_it, precon, do_perm)
        self.fv.set_solver(name, atol, rtol, max_it, precon, do_perm)

    def set_solver_print_level(self, level):
        self.dc.fv.set_solver_print_level(level)
        self.fv.set_solver_print_level(level)

    def set_roi(self, roi):
        """Define region of interest for computing sensitivity or for inversion.

        Parameters
        ----------
        roi : tuple of float
            Tuple holding xmin, xmax, ymin, ymax, zmin, zmax
        """
        # on applique la ROI à la grille DC car la grille MMR contient des
        # noeuds additionnels pour l'air
        self.dc.set_roi(roi)
        self.ind_roi = self.dc.ind_roi

    def data_to_obs(self, data, field_data=True):
        """Rearrange data for inversion."""
        if field_data:
            data = data[self.ind_s, :]
        else:
            data = data[self.mask, :]
        # on veut Bx, ensuite By et finalement Bz dans dobs, en séquence pour chaque src
        d = data[:self.nobs_xs[0], :3].T.reshape(-1, 1)
        i1 = self.nobs_xs[0]
        for i in np.arange(1, self.xs_u.shape[0]):
            d = np.r_[d, data[i1:i1+self.nobs_xs[i], :3].T.reshape(-1, 1)]
            i1 += self.nobs_xs[i]
        return d

    def print_info(self, file=None):
        self.dc.print_info(file)

    def fromVTK(self, fieldname, filename):
        return self.fv.fromVTK(fieldname, filename)

    def fwd_mod(self, sigma, calc_sens=False, keep_solver=False):
        """Forward modelling.

        Parameters
        ----------
        sigma : array_like
            Conductivity model (S/m).
        calc_sens : bool, optional
            Calculate sensitivity matrix.
        keep_solver : bool, optional
            Use solver instantiated in a previous `fwd_mod` run.

        Returns
        -------
        data : ndarray
            components Bx, By & Bz (pT)
        sens : ndarray
            Sensitivity matrix if `calc_sens` is True.

        Notes
        -----
        - If `xs` and `xo` have a different number of rows, the response is
        computed for all possible combinations.  If the number of rows are
        equal, the response is computed for the pairs of `xs` and `xo` rows.

        - Matrix A of the MMR system depends only on grid geometry, and it is
        thus not needed to build it at each run if the grid does not change.
        This is useful if a direct solver is used, matrix A must be factorized
        only once.  Option `keep_solver` allows this, matrix A being factorized
        when the solver is instantiated.
        """
        if self.fv.nc == self.dc.fv.nc and self.fv.dim != 2.5:   # 2.5D: see _setup25
            self._add_air()

        if self.xs is None:
            raise ValueError('Source term undefined')

        if self.xo is None:
            raise ValueError('Measurement points undefined')

        if self.verbose and self.fv.comm.rank == 0:
            print('\nForward modelling - Magnetometric resistivity')
            self.dc.print_info()
            if self.fv.solver_A is not None:
                self.fv.solver_A.print_info()

        if calc_sens and not isinstance(self.dc.fv, (GridFV, Grid25FV)):
            # tetrahedral meshes lack the face operators used for the sensitivity
            raise NotImplementedError('Sensitivity computation implemented only for rectilinear grids')

        if not self.acq_checked:
            self.check_acquisition()

        if self.fv.dim == 2.5:
            return self._fwd_mod25(sigma, keep_solver, calc_sens)

        if self.in_inv and self.dc.c1c2 is not None:
            # joint inversion
            res_dc = self.dc.fwd_mod(sigma, calc_sens=calc_sens)

        c1c2_u_save = copy.copy(self.dc.c1c2_u)
        cs12_u_save = copy.copy(self.dc.cs12_u)
        self.dc.c1c2_u = self.xs_u
        self.dc.cs12_u = self.cs_u
        self.dc.sort_electrodes = False

        if self.verbose and self.fv.comm.rank == 0:
            print('  Computing interpolation matrices ... ', end='', flush=True)
        Qx, Qy, Qz = self._build_Q(self.xo_all)
        if self.pod_p1p2 is not None:
            self.Q_pod_e = self._build_Q_pod_e()
        if self.verbose and self.fv.comm.rank == 0:
            print('done.')
        self.Q = Qx, Qy, Qz

        # get current density from forward DC modeling
        if self.verbose and self.fv.comm.rank == 0:
            print('  Computing current density ... ', end='', flush=True)
        # The image of surface electrodes (moved to the first cell centre) improves DC data, but the MMR
        # field computed from the current density is less accurate with it: not used here (DC data of a
        # joint inversion, computed above, still use it)
        # (the voltages returned by this call are not used: no analytic primary field either)
        images = getattr(self.dc.fv, 'surface_images', None)
        exact_primary = self.dc.exact_primary
        if images is not None:
            self.dc.fv.surface_images = False
        self.dc.exact_primary = False
        try:
            _, Jdc = self.dc.fwd_mod(sigma, calc_J=True)
        finally:
            if images is not None:
                self.dc.fv.surface_images = images
            self.dc.exact_primary = exact_primary
        u_dc = self.dc.u.copy()

        if self.verbose and self.fv.comm.rank == 0:
            print('done.')

        Jx = Jdc[:self.dc.fv.nfx, :]
        Jy = Jdc[self.dc.fv.nfx:(self.dc.fv.nfx+self.dc.fv.nfy), :]
        Jz = Jdc[(self.dc.fv.nfx+self.dc.fv.nfy):, :]

        if self.verbose and self.fv.comm.rank == 0:
            print('  Solving MMR system')

        # create MMR source term from Jdc (J is 0 in air)
        q = np.zeros((self.fv.nf, self.xs_u.shape[0]))
        for i in range(self.xs_u.shape[0]):
            # q[(self.fv.nfx-self.dc.fv.nfx):self.fv.nfx, i] = Jx[:, i]
            # q[(self.fv.nfx+self.fv.nfy-self.dc.fv.nfy):(self.fv.nfx+self.fv.nfy), i] = Jy[:, i]
            # q[(self.fv.nfx+self.fv.nfy+self.fv.nfz-self.dc.fv.nfz):, i] = Jz[:, i]

            q[:self.dc.fv.nfx, i] = Jx[:, i]
            q[self.fv.nfx:(self.fv.nfx+self.dc.fv.nfy), i] = Jy[:, i]
            q[(self.fv.nfx+self.fv.nfy):(self.fv.nfx+self.fv.nfy+self.dc.fv.nfz), i] = Jz[:, i]

        if self.fv.solver_A is None or keep_solver is False:
            self.fv.solver_A = Solver(self.fv.get_solver_params(), self._build_A(), verbose=(self.verbose>1))
        if self.fv.solver_A.A is None:
            self.fv.solver_A.A = self._build_A()
        if self.verbose and self.fv.comm.rank == 0:
            total_memory_bytes = self.fv.solver_A.A.data.nbytes + \
                                 self.fv.solver_A.A.indptr.nbytes + \
                                 self.fv.solver_A.A.indices.nbytes

            print("      Memory footprint of A : {0:4.2f} MB".format(float(total_memory_bytes)/1048576), flush=True)
        self.u = self.fv.solver_A.solve(q)

        B = self.C_f @ self.u
        Bx = self._units_scaling * B[:self.fv.nex, :]
        By = self._units_scaling * B[self.fv.nex:(self.fv.nex+self.fv.ney), :]
        Bz = self._units_scaling * B[(self.fv.nex+self.fv.ney):, :]

        no = self.xo_all.shape[0]
        data = np.empty((no*q.shape[1], self.xo_all.shape[1]))
        data[:, 0] = (Qx @ Bx).T.flatten()
        data[:, 1] = (Qy @ By).T.flatten()
        data[:, 2] = (Qz @ Bz).T.flatten()

        if self.Q_pod_e is not None:
            data_pod_e = np.empty((no * q.shape[1], self.xo_all.shape[1]))
            for n in range(3):
                data_pod_e[:, n] = (self.Q_pod_e[n] @ u_dc).T.flatten()
            data_pod_e = self.dc._units_scaling * data_pod_e[self.ind_back, :]
        else:
            data_pod_e = None

        if not self.in_inv:
            data = data[self.ind_back, :]
        else:
            self.dc.sort_electrodes = True
            data = self.data_to_obs(data, False)

        if calc_sens:
            if self.verbose and self.fv.comm.rank == 0:
                print('  Computing sensitivity')
                print('    Computing adjoint terms ... ', end='', flush=True)

            q = self._adj()
            M = self.dc.fv.build_M(sigma)
            q_a = self.dc.fv.D @ M @ q
            t = np.max(np.abs(q_a), axis=0)
            q_a /= np.tile(t, (q_a.shape[0], 1))

            if self.verbose and self.fv.comm.rank == 0:
                print('    Solving DC adjoint problem ... ', end='', flush=True)
            self.dc.fwd_mod(calc_J=False, q=q_a)
            if self.verbose and self.fv.comm.rank == 0:
                print('done.')

            q2 = self.dc.u * np.tile(t, (self.dc.u.shape[0], 1))

            if self.verbose and self.fv.comm.rank == 0:
                print('    Assembling matrices ... ', end='', flush=True)
            sens = np.empty((self.dc.ind_roi.size, self.nobs_mmr*3))
            # d(m_f)/d(ln sigma_i) = m_f^2 w_fi / sigma_i, times face volumes from the adjoint (see _adj)
            S = sp.diags_array(M.diagonal()**2 * self.dc.fv.volume_faces())
            Dm = sp.diags_array(1.0/sigma)

            Gf = self.dc.fv.build_G_faces()
            for ns in range(self.xs_u.shape[0]):
                self._fill_jacobian(ns, sens, u_dc, Dm, S, q, q2, Gf)

            if self.verbose and self.fv.comm.rank == 0:
                print('done.')

        if self.verbose and self.fv.comm.rank == 0:
            print('End of modelling.')

        self.dc.c1c2_u = c1c2_u_save
        self.dc.cs12_u = cs12_u_save
        if calc_sens:
            if self.in_inv and self.dc.c1c2 is not None:
                data = np.r_[data, res_dc[0].reshape(-1, 1)]
                sens = np.c_[sens, res_dc[1]]
            return data, sens
        else:
            if self.in_inv and self.dc.c1c2 is not None:
                data = np.r_[data, res_dc.reshape(-1, 1)]
            if data_pod_e is not None:
                return data, data_pod_e
            else:
                return data

    # --------------------------------------------------------------------------------------------
    # 2.5D modelling (see docs/mmr25.md)
    # --------------------------------------------------------------------------------------------

    def _fwd_mod25(self, sigma, keep_solver, calc_sens=False):
        """Forward modelling in 2.5D, returns data (and sensitivity) as fwd_mod."""
        if self.pod_p1p2 is not None:
            raise NotImplementedError('Electric dipoles at measurement points (pod_e) not available in 2.5D')
        sigma = np.asarray(sigma, dtype=float).ravel() * np.ones((self.dc.fv.nc,))

        if self.in_inv and self.dc.c1c2 is not None:
            # joint inversion: DC data computed with the wavenumbers of the DC code
            res_dc = self.dc.fwd_mod(sigma, calc_sens=calc_sens)

        verbose = self.verbose and self.fv.comm.rank == 0
        self._setup25()
        poles = self._plan25['poles']
        if verbose:
            print('  {0:d} current electrodes, {1:d} wavenumbers in [{2:g}, {3:g}] 1/m'.format(
                poles.shape[0], self.k.size, self.k[0], self.k[-1]))
            print('  Computing DC fields ... ', end='', flush=True)
        U, M_k, dc_solvers = self._dc_fields25(sigma, poles, keep_solvers=calc_sens)
        if verbose:
            print('done.\n  Solving MMR systems ... ', end='', flush=True)
        Bk = self._mmr_fields25(U, M_k, keep_solver)
        if verbose:
            print('done.')
        data = self._units_scaling * self._transform25(Bk)
        if calc_sens:
            if verbose:
                print('  Computing sensitivity ... ', end='', flush=True)
            sens = self._sens25(sigma, U, M_k, dc_solvers)
            if verbose:
                print('done.')

        if not self.in_inv:
            data = data[self.ind_back, :]
        else:
            data = self.data_to_obs(data, False)
            if self.dc.c1c2 is not None:
                if calc_sens:
                    data = np.r_[data, res_dc[0].reshape(-1, 1)]
                    sens = np.c_[sens, res_dc[1]]
                else:
                    data = np.r_[data, res_dc.reshape(-1, 1)]
        if verbose:
            print('End of modelling.')
        if calc_sens:
            return data, sens
        return data

    def _setup25(self):
        """Air layers, current electrodes, wavenumbers and inverse transform weights (once per survey)."""
        if self.fv.nc == self.dc.fv.nc:
            # A_z = 0 at the top: the air must be thick compared with the source-receiver distances,
            # whatever the size of the top cells (see docs/mmr25.md, section 9)
            self._add_air(min_thickness=self.dc.fv.z[-1] - self.dc.fv.z[0])
        if self._plan25 is not None and self._k_range == self._survey_range25():
            return

        # current electrodes: the 2D solution depends on (x, z) only; y enters through dy
        e = np.vstack((self.xs_u[:, :3], self.xs_u[:, 3:6]))
        ok = np.all(np.isfinite(e), axis=1)
        poles, inv = np.unique(e[ok][:, [0, 2]], axis=0, return_inverse=True)
        ind = -np.ones((e.shape[0],), dtype=int)
        ind[ok] = inv.ravel()
        n_s = self.xs_u.shape[0]
        ind_c = (ind[:n_s], ind[n_s:])
        poles = np.c_[poles[:, 0], np.zeros((poles.shape[0],)), poles[:, 1]]

        self._k_range = self._survey_range25()
        self.k = select_k(self._k_range[1], self._k_range[2], self._k_range[0])
        self._kq = KQuadrature(self.k)

        # data combinations, in modelling order: dipole ns, measurement point nr -> ns * no + nr
        no = self.xo_all.shape[0]
        NS = np.repeat(np.arange(n_s), no)
        NR = np.tile(np.arange(no), n_s)
        terms = []
        for sign, ic, col in ((1.0, ind_c[0], 1), (-1.0, ind_c[1], 4)):
            p = ic[NS]
            sel = p >= 0
            dy = self.xo_all[NR[sel], 1] - self.xs_u[NS[sel], col]
            terms.append({'rows': np.where(sel)[0], 'rec': NR[sel], 'pole': p[sel],
                          'w': sign * self.cs_u[NS[sel]],
                          'W_even': self._kq.weights(dy, 'even'), 'W_odd': self._kq.weights(dy, 'odd')})
        self._plan25 = {'poles': poles, 'terms': terms, 'n_data': n_s * no}
        self.Q = self._build_Q(self.xo_all)

    def _survey_range25(self):
        """(n_k, r_min, r_max) for the selection of the wavenumbers.

        r_min is the smallest distance in the x-z plane between a current electrode and a measurement
        point (at least half the smallest cell size), and r_max is 4 times the largest distance, in 3D,
        between a current electrode and a measurement point, and at least 100 r_min.
        """
        e = np.vstack((self.xs_u[:, :3], self.xs_u[:, 3:6]))
        e = e[np.all(np.isfinite(e), axis=1)]
        e[:, :] = np.c_[e[:, 0], e[:, 1], self.dc.fv.process_surface_elec(e[:, [0, 2]].copy())[:, 1]]
        xo = self.xo_all
        dx = xo[:, None, 0] - e[None, :, 0]
        dz = xo[:, None, 2] - e[None, :, 2]
        dy = xo[:, None, 1] - e[None, :, 1]
        h_min = min(self.dc.fv.hx.min(), self.dc.fv.hz.min())
        r_min = max(np.sqrt(dx**2 + dz**2).min(), 0.5 * h_min)
        r_max = max(4.0 * np.sqrt(dx**2 + dy**2 + dz**2).max(), 100.0 * r_min)
        return (int(self.n_k), float(r_min), float(r_max))

    def _dc_fields25(self, sigma, poles, keep_solvers=False):
        """DC fields U_k of unit current electrodes (n_k x nc x n_poles), M_k of the DC grid, and the
        solvers of each wavenumber (None unless keep_solvers)."""
        fv = self.dc.fv
        A, _ = fv.build_A(sigma)
        n_p = poles.shape[0]
        cs = np.ones((n_p,))
        if self.dc.apply_bc:
            # boundary correction of Pidlisecky et al. (2007) for each wavenumber, as in GridDC
            if self.dc.local_sigma0:
                sigma0 = self.dc._electrode_sigma(poles, sigma)
            else:
                sigma0 = gmean(sigma)
            images = fv.surface_images
            fv.surface_images = self.surface_images
            try:
                src, _ = fv.boundary_source(poles, sigma0, cs, correct_quadrature=False, k=self.k)
            finally:
                fv.surface_images = images
            q = src.q
        else:
            p = fv.process_surface_elec(poles.copy())
            q0 = (fv.linear_interp(p[:, 0], p[:, 1], p[:, 2]).T.toarray() /
                  fv.volume_voxels()[:, None])
            q = np.broadcast_to(q0, (self.k.size,) + q0.shape)

        U = np.empty((self.k.size, fv.nc, n_p))
        par = fv.get_solver_params()
        solvers = [] if keep_solvers else None
        for i, k in enumerate(self.k):
            s = Solver(par, Solver25._L(A, sigma, k), verbose=(self.verbose > 1),
                       comm=fv.comm, n_threads=fv.n_threads)
            U[i] = s.solve(q[i]).reshape(fv.nc, n_p)
            if keep_solvers:
                solvers.append(s)
        return U, fv.build_M_k(sigma), solvers

    def _mmr_fields25(self, U, M_k, keep_solver):
        """Transformed B (T) at the measurement points: 3 x n_k x n_obs x n_poles (Bx, By, Bz)."""
        fv, fv_dc = self.fv, self.dc.fv
        same = (self._solvers_k is not None and keep_solver and self._solvers_k[0].size == self.k.size and
                np.allclose(self._solvers_k[0], self.k, rtol=1e-12, atol=0.0) and
                self._solvers_k[1] == (fv.nx, fv.nz))
        if not same:
            par = fv.get_solver_params()
            solvers = [Solver(par, fv.build_L_k(k) / MU0, verbose=(self.verbose > 1),
                              comm=fv.comm, n_threads=fv.n_threads) for k in self.k]
            self._solvers_k = (self.k.copy(), (fv.nx, fv.nz), solvers)

        # faces of the DC grid (earth) in the grid with air: same indices, components shifted
        ind_earth = np.r_[np.arange(fv_dc.nfx), fv.nfx + np.arange(fv_dc.nc),
                          fv.nfx + fv.nc + np.arange(fv_dc.nfz)]
        Qx, Qy, Qz = self.Q
        n_p = U.shape[2]
        Bk = np.empty((3, self.k.size, self.xo_all.shape[0], n_p))
        for i, k in enumerate(self.k):
            q = np.zeros((fv.nf_k, n_p))
            q[ind_earth, :] = -(M_k @ (fv_dc.build_G_k(k) @ U[i]))   # transformed current density
            a = self._solvers_k[2][i].solve(q).reshape(fv.nf_k, n_p)
            B = fv.build_C_k(k, to_faces=False) @ a
            Bk[0, i] = Qx @ B[:fv.nex]
            Bk[1, i] = Qy @ B[fv.nex:fv.nex + fv.ney]
            Bk[2, i] = Qz @ B[fv.nex + fv.ney:]
        return Bk

    def _transform25(self, Bk):
        """Inverse transform and sum over the electrodes of the dipoles: n_data x 3 (modelling order)."""
        data = np.zeros((self._plan25['n_data'], 3))
        for t in self._plan25['terms']:
            for c, W in ((0, t['W_odd']), (1, t['W_even']), (2, t['W_odd'])):
                vals = np.einsum('dk,kd->d', W, Bk[c][:, t['rec'], t['pole']])
                data[t['rows'], c] += t['w'] * vals
        return data

    def _sens25(self, sigma, U, M_k, dc_solvers):
        """Sensitivity d(data)/d(ln sigma) in 2.5D, n_roi x (3 n_data), columns ordered as data_to_obs.

        For one wavenumber and a current electrode (unit current), with R = Q C_f the interpolation of
        B at the measurement points, J = -M G U the current density and U = L_dc^-1 q (q does not depend
        on sigma, see Grid25FV.boundary_source):

            d(R A)/d(ln sigma) = -Jm^T [ G U * Vf * (X + G L_dc^-1 D M X) ]

        where X = (L/mu0)^-1 Vf^-1 R^T on the earth faces (L^-T = Vf L^-1 Vf^-1, D^T = -Vf G Vc^-1,
        L_dc^-T = Vc L_dc^-1 Vc^-1, see docs/mmr25.md), and Jm = d(M)/d(ln sigma): m^2 w / sigma on the x
        and z faces (harmonic mean with weights w, build_G_faces), sigma for the y component.  X and the
        DC adjoint fields depend only on the measurement points: 2 solves per wavenumber for all the
        data.  The sum over wavenumbers and electrodes uses the weights of _transform25.
        """
        fv, fv_dc = self.fv, self.dc.fv
        nfx, nc = fv_dc.nfx, fv_dc.nc
        ind_earth = np.r_[np.arange(nfx), fv.nfx + np.arange(nc), fv.nfx + fv.nc + np.arange(fv_dc.nfz)]
        ind_xz = np.r_[np.arange(nfx), nfx + nc + np.arange(fv_dc.nfz)]
        Gf = fv_dc.build_G_faces()
        m2_xz = M_k.diagonal()[ind_xz]**2
        dm = 1.0 / sigma
        vf = fv_dc.volume_faces_k()
        vf_inv_air = 1.0 / fv.volume_faces_k()
        Qx, Qy, Qz = self.Q
        n_rec = Qx.shape[0]
        n_p = U.shape[2]
        plan = self._plan25
        # columns of the terms of each electrode: (rows, receivers, weights) per component
        by_pole = [[] for _ in range(n_p)]
        for t in plan['terms']:
            for p in np.unique(t['pole']):
                sel = t['pole'] == p
                by_pole[p].append((t['rows'][sel], t['rec'][sel], t['w'][sel], t['W_odd'][sel], t['W_even'][sel]))

        sens = np.zeros((nc, 3 * plan['n_data']))
        nex, ney = fv.nex, fv.ney
        for i, k in enumerate(self.k):
            Cf = fv.build_C_k(k, to_faces=False)
            R = sp.vstack((Qx @ Cf[:nex], Qy @ Cf[nex:nex + ney], Qz @ Cf[nex + ney:]))   # 3 n_rec x nf_k
            rhs = R.T.toarray() * vf_inv_air[:, None]
            X = self._solvers_k[2][i].solve(rhs).reshape(fv.nf_k, 3 * n_rec)[ind_earth]
            G_k = fv_dc.build_G_k(k)
            Phi = dc_solvers[i].solve(fv_dc.build_D_k(k) @ (M_k @ X)).reshape(nc, 3 * n_rec)
            T = (X + G_k @ Phi) * vf[:, None]
            GU = G_k @ U[i]
            for p in range(n_p):
                V = GU[:, p][:, None] * T
                g = -(dm[:, None] * (Gf.T @ (m2_xz[:, None] * V[ind_xz])) + sigma[:, None] * V[nfx:nfx + nc])
                for rows, rec, w, W_odd, W_even in by_pole[p]:
                    for c, W in ((0, W_odd), (1, W_even), (2, W_odd)):
                        sens[:, 3 * rows + c] += g[:, c * n_rec + rec] * (w * W[:, i])

        # order of data_to_obs (per dipole: Bx, By, Bz of its measurement points)
        cols = self.data_to_obs(np.arange(3 * plan['n_data']).reshape(-1, 3), False).ravel()
        return self._units_scaling * sens[np.ix_(self.dc.ind_roi, cols)]

    def calc_WtW(self, wt, par, m_active, WGx=None, WGy=None, WGz=None, WS=None):
        return self.dc.calc_WtW(wt, par, m_active, WGx, WGy, WGz, WS)

    def calc_reg(self, Q, par, xc, xref, WGx, WGy, WGz, m_active):
        return self.dc.calc_reg(Q, par, xc, xref, WGx, WGy, WGz, m_active)

    def distance_weighting(self, xo, beta):
        return self.dc.fv.distance_weighting(xo, beta)

    def get_roi_nodes(self):
        return self.dc.get_roi_nodes()

    def save_sensitivity(self, sens, basename):
        """Save sensitivity to VTK files.

        Parameters
        ----------
        sens : ndarray
            Sensitivity
        basename : str
            basename for output files (1 per dipole).
        """
        x, y, z = self.dc.get_roi_nodes()
        # grille temporaire pour sauvegarder sens
        g2 = GridFV((x, y, z))
        xo = self.xo[self.ind_s, :]

        # make 1 file for each injection dipole
        for ns in range(self.xs_u.shape[0]):
            if ns == 0:
                i = i0 = 0
            else:
                i = np.sum(self.nobs_xs[:ns])
                i0 = np.sum(self.nobs_xs[:ns] * 3)
            sens_data = {}
            for nr in range(self.nobs_xs[ns]):
                xo_x = xo[i + nr, 0]
                xo_y = xo[i + nr, 1]
                xo_z = xo[i + nr, 2]
                name = '∂d/∂m - Bx: {0:4.1f} {1:4.1f} {2:4.1f}'.format(xo_x, xo_y, xo_z)
                sens_data[name] = sens[:, i0 + nr]
                name = '∂d/∂m - By: {0:4.1f} {1:4.1f} {2:4.1f}'.format(xo_x, xo_y, xo_z)
                sens_data[name] = sens[:, i0 + nr + self.nobs_xs[ns]]
                name = '∂d/∂m - Bz: {0:4.1f} {1:4.1f} {2:4.1f}'.format(xo_x, xo_y, xo_z)
                sens_data[name] = sens[:, i0 + nr + 2 * self.nobs_xs[ns]]

            src = "C1: " + str(self.xs_u[ns, 0]) + " " + str(self.xs_u[ns, 1]) + " " + str(self.xs_u[ns, 2])
            src += ", C2: " + str(self.xs_u[ns, 3]) + " " + str(self.xs_u[ns, 4]) + " " + str(self.xs_u[ns, 5])
            metadata = {"Source dipole": src}
            filename = basename+'_mmr_sens_dip'+str(ns+1)
            g2.toVTK(sens_data, filename, metadata=metadata)

    def _add_air(self, n_const_cells=4, n_crois_cells=15, factor=1.3, min_thickness=0.0):
        """Add air layers above the DC grid.

        `n_const_cells` cells of the size of the top cell, then `n_crois_cells` cells growing by
        `factor`, with more growing cells if needed to reach a total thickness of `min_thickness`.
        """
        while True:
            dz_air = calc_padding(self.dc.fv.hz[-1], n_cells=n_crois_cells, factor=factor)
            dz_air = np.r_[self.dc.fv.hz[-1]+np.zeros((n_const_cells,)), dz_air]
            if dz_air.sum() >= min_thickness:
                break
            n_crois_cells += 1
        z_air = np.cumsum(dz_air)
        self.fv.z = np.r_[self.dc.fv.z, self.dc.fv.z[-1] + z_air]

    def _adj(self):
        Qx, Qy, Qz = self.Q
        Jx = Qx @ self.C_f[:self.fv.nex, :]
        Jy = Qy @ self.C_f[self.fv.nex:self.fv.nex+self.fv.ney, :]
        Jz = Qz @ self.C_f[self.fv.nex+self.fv.ney:, :]

        J = sp.hstack((Jx.T, Jy.T, Jz.T))

        # Adjoint of B = C_f A^-1 P Jdc.  A is not symmetric on non-uniform grids, but diag(Vf) A is,
        # so A^-T = Vf A^-1 Vf^-1.  The left factor Vf (restricted to earth faces) is applied in
        # _fill_jacobian through S.
        J = sp.diags_array(1.0 / self.fv.volume_faces()) @ J

        # earth faces, same mapping as used to build the MMR source term in fwd_mod
        ind_earth = np.r_[np.arange(self.dc.fv.nfx),
                          self.fv.nfx + np.arange(self.dc.fv.nfy),
                          self.fv.nfx + self.fv.nfy + np.arange(self.dc.fv.nfz)]

        q = self.fv.solver_A.solve(J)
        q = q[ind_earth, :]
        return q

    def _build_A(self):
        # average of permeability (scalar for now)
        M_e = M_c = 1. / (4.*np.pi*1e-7)   # ( mu_0 in H/m)

        # curl matrices
        C_e = self.fv.build_C()                # projecting from cell edges to faces
        self.C_f = self.fv.build_C(to_faces=False)  # projecting from cell faces to edges

        # assemble
        A = M_e * C_e @ self.C_f - M_c * self.fv.G @ self.fv.D

        return A

    def _build_Q(self, xo):
        Qx = []
        Qy = []
        Qz = []
        for n in range(xo.shape[0]):
            Qx.append(self.fv.linear_interp(xo[n, 0], xo[n, 1], xo[n, 2], 'ex'))
            Qy.append(self.fv.linear_interp(xo[n, 0], xo[n, 1], xo[n, 2], 'ey'))
            Qz.append(self.fv.linear_interp(xo[n, 0], xo[n, 1], xo[n, 2], 'ez'))
        Qx = sp.vstack(Qx)
        Qy = sp.vstack(Qy)
        Qz = sp.vstack(Qz)
        return Qx, Qy, Qz

    def _build_Q_pod_e(self):
        Q = [None, None, None]

        for n in range(3):
            # make sure electrodes are at least at the depth of the first cell center
            if self.pod_p1p2[n] is not None:
                p1p2 = self.pod_p1p2[n]
                p1p2[:, :3] = self.dc.fv.process_surface_elec(p1p2[:, :3])
                p1p2[:, 3:6] = self.dc.fv.process_surface_elec(p1p2[:, 3:6])

                Q[n] = self.dc.fv.linear_interp(p1p2[:, 0], p1p2[:, 1], p1p2[:, 2])
                Q[n] -= self.dc.fv.linear_interp(p1p2[:, 3], p1p2[:, 4], p1p2[:, 5])
            else:
                Q[n] = sp.csr_array((self.xo_all.shape[0], self.dc.fv.nc))
        return Q

    def _fill_jacobian(self, n, J, u_dc, Dm, S, q, q2, Gf):
        if n == 0:
            i0 = 0
        else:
            i0 = np.sum(self.nobs_xs[:n]*3)
        i1 = i0 + self.nobs_xs[n]*3

        v = sp.diags_array(self.dc.fv.G @ u_dc[:, n])
        Gc = v @ Gf
        # Gc = self.dc.build_G(self.dc.G @ u_dc[:, n])
        A = -Dm @ Gc.T @ S
        # dB/d(ln sigma) = -R A^-1 P [I + M G A_dc^-1 D] d(M G u)/d(ln sigma); transposed with
        # A_dc^-T = Vc A_dc^-1 Vc^-1 and D^T = -Vf G Vc^-1, the second term is + G A_dc^-1 D M q = + G q2
        tmp = A @ q + A @ self.dc.fv.G @ q2

        mask_xs = self.mask[n*self.xo_all.shape[0]:(n+1)*self.xo_all.shape[0]]
        mask_xs = np.r_[mask_xs, mask_xs, mask_xs]
        tmp = tmp[self.dc.ind_roi, :]
        J[:, i0:i1] = self._units_scaling * tmp[:, mask_xs]

    def _check_cs(self):
        """Verify validity of current source intensity."""
        if self.cs is None:
            raise ValueError('Current source undefined')
        if np.isscalar(self.cs):
            self.cs = self.cs + np.zeros((self.xs.shape[0],))
        elif isinstance(self.cs, np.ndarray):
            if self.cs.ndim != 1:
                self.cs = self.cs.flatten()
            if self.cs.size != self.xs.shape[0]:
                raise ValueError('Number of current source should match number of source terms')


def normal_field(c1c2, xo, cs=1.0):
    """Compute normal field at observation points for surface MMR without any topography.

    Parameters
    ----------
    c1c2 : array_like
        Coordinates of source dipoles (source first, sink second) (m).
    xo : array_like
        Coordinates of observation points (m).
    cs : float
        Current intensity (A).

    Returns
    -------
    data : ndarray
        components Bx & By (nT)

    Notes
    -----
    Bz is always 0.

    The horizontal field is azimuthal, ie the following equation applies

    2 pi r curl B_phi = mu I

    and is used to compute the horizontal components.

    """
    r1 = np.sqrt(np.sum((xo - c1c2[:, :3])**2, axis=1))
    r2 = np.sqrt(np.sum((xo - c1c2[:, 3:])**2, axis=1))
    theta1 = np.arctan2(xo[:, 1] - c1c2[:, 1], xo[:, 0] - c1c2[:, 0])
    theta2 = np.arctan2(xo[:, 1] - c1c2[:, 4], xo[:, 0] - c1c2[:, 3])
    # pT.  Returned field is -(B1 phi_1 + B2 phi_2), phi = azimuthal unit vector around each electrode.
    # Current flows along -z below the source (B_phi < 0) and along +z below the sink (B_phi > 0).
    B1 = cs * 1.e-7 / r1 * 1.e12
    B2 = -cs * 1.e-7 / r2 * 1.e12
    return B1 * np.sin(theta1) + B2 * np.sin(theta2), -B1 * np.cos(theta1) - B2 * np.cos(theta2)


class VerticalDyke:
    """
        Compute MMR anomaly for a vertical dyke.

        Reference
        ---------
        Edwards, Lee, Nabighian, 1978. On the theory of magnetometric resistivity (MMR) methods.
        Geophysics, 43(6), 1176-1203.
    """
    def __init__(self, rho1, rho2, rho3, thickness, xd=0.0, n_max=100):
        """
        Parameters
        ----------
        rho1 : float
            Resistivity of leftmost medium.
        rho2 : float
            Resistivity of the dyke.
        rho3 :
            Resistivity of rightmost medium.
        thickness :
            Thickness of the dyke.
        xd : float
            X coordinate of the left flank of the dyke.
        n_max : int
            Upper limit of summations. Default is 100.
        """
        self.rho1 = rho1
        self.rho2 = rho2
        self.rho3 = rho3
        self.d = thickness
        self.xd = xd
        self.n_max = n_max

        self.k12 = ( rho1 - rho2 ) / ( rho1 + rho2 )
        self.k13 = ( rho1 - rho3 ) / ( rho1 + rho3 )
        self.k32 = ( rho3 - rho2 ) / ( rho3 + rho2 )

    @property
    def n_max(self):
        return self.n[-1]

    @n_max.setter
    def n_max(self, n_max):
        self.n = np.arange(n_max+1)

    def fwd_mod(self, xo, xs, cs=1.0):
        """
        Forward MMR solution

        Parameters
        ----------
        xo : array_like
            X,Y,Z coordinates of observation points (m).
        xs : float
            X coordinate of current electrode (m).
        cs : float
            Current intensity (A).

        Returns
        -------
            Vertical B-field anomaly (nT).

        Notes
        -----
        Solution from pages 1187-1188 and thesis of Boggs.  The expressions are for an electrode on
        the left of the dyke or within it; an electrode on the right is handled by reflection about
        the centre of the dyke (rho1 and rho3 swapped; Bz, a component of a pseudovector, changes
        sign).
        """
        xo = np.asarray(xo, dtype=float)
        if xs - self.xd > self.d:
            xm = 2.0 * self.xd + self.d        # x -> xm - x maps the dyke onto itself
            mirror = VerticalDyke(self.rho3, self.rho2, self.rho1, self.d, self.xd, self.n_max)
            xo_m = np.c_[xm - xo[:, 0], xo[:, 1:]]
            return -mirror.fwd_mod(xo_m, xm - xs, cs)

        b = xs - self.xd         # electrode location relative to dyke
        x = xo[:, 0] - self.xd   # x coord of observation pt relative to dyke
        y = xo[:, 1]             # y coord of observation pt
        y2 = y*y

        # indices for the 3 regions
        ind1 = x < 0.0
        ind2 = np.logical_and(x >= 0.0, x <= self.d)
        ind3 = x > self.d

        mu = 12.566370614359172e-7

        tmp0 = mu * cs / (4 * np.pi * y)
        # 2 (n+1) d - b - x
        tmp1 = 2. * (np.tile(self.n, (x.size, 1))+1.)*self.d - b - np.tile(x.reshape(-1, 1), (1, self.n.size))
        # 2 (n+1) d + b - x
        tmp2 = 2. * (np.tile(self.n, (x.size, 1))+1.)*self.d + b - np.tile(x.reshape(-1, 1), (1, self.n.size))
        tmp3 = b - x
        # 2 n d + b + x
        tmp4 = 2. * np.tile(self.n, (x.size, 1)) * self.d + b + np.tile(x.reshape(-1, 1), (1, self.n.size))
        # 2 (n+1) d - b + x
        tmp5 = 2. * (np.tile(self.n, (x.size, 1)) + 1.)*self.d - b + np.tile(x.reshape(-1, 1), (1, self.n.size))
        tmp6 = -b - x
        # 2 (n+1) d + b + x
        tmp7 = 2. * (np.tile(self.n, (x.size, 1)) + 1.) * self.d + b + np.tile(x.reshape(-1, 1), (1, self.n.size))
        # 2 n d - b + x
        tmp8 = 2. * np.tile(self.n, (x.size, 1)) * self.d - b + np.tile(x.reshape(-1, 1), (1, self.n.size))

        y2t = np.tile(y2.reshape(-1, 1), (1, self.n.size))

        # case a: electrode within the dyke
        if 0.0 <= b <= self.d:
            Bz1 = tmp0 * (
                self.k13 + (1. - self.k12) * self.k32 * (
                    np.sum((self.k12 * self.k32) ** self.n * tmp1 / np.sqrt(y2t + tmp1 * tmp1), axis=1) +
                    self.k12 * np.sum((self.k12 * self.k32) ** self.n * tmp2 / np.sqrt(y2t + tmp2 * tmp2), axis=1)
                )
                - self.k12 * tmp3 / np.sqrt(y2 + tmp3 * tmp3)
            )

            Bz2 = tmp0 * (
                self.k13 + self.k32 * (
                    np.sum((self.k12 * self.k32) ** self.n * tmp1 / np.sqrt(y2t + tmp1 * tmp1), axis=1) +
                    self.k12 * np.sum((self.k12 * self.k32) ** self.n * tmp2 / np.sqrt(y2t + tmp2 * tmp2), axis=1)
                ) -
                self.k12 * (
                    np.sum((self.k12 * self.k32) ** self.n * tmp4 / np.sqrt(y2t + tmp4 * tmp4), axis=1) +
                    self.k32 * np.sum((self.k12 * self.k32) ** self.n * tmp5 / np.sqrt(y2t + tmp5 * tmp5), axis=1)
                )
            )

            Bz3 = tmp0 * (
                self.k13 - (1. - self.k32) * self.k12 * (
                    np.sum((self.k12 * self.k32) ** self.n * tmp4 / np.sqrt(y2t + tmp4 * tmp4), axis=1) +
                    self.k32 * np.sum((self.k12 * self.k32) ** self.n * tmp5 / np.sqrt(y2t + tmp5 * tmp5), axis=1)
                ) +
                self.k32 * -tmp3 / np.sqrt(y2 + tmp3 * tmp3)
            )
        else:   # electrode external to dyke
            Bz1 = tmp0 * (
                self.k13 + (1. - self.k12*self.k12) * self.k32 *
                    np.sum((self.k12 * self.k32) ** self.n * tmp1 / np.sqrt(y2t + tmp1 * tmp1), axis=1) -
                self.k12 * tmp6 / np.sqrt(y2 + tmp6 * tmp6)
            )

            Bz2 = tmp0 * (
                self.k13 + (1. + self.k12) * self.k32 * (
                    np.sum((self.k12 * self.k32) ** self.n * tmp1 / np.sqrt(y2t + tmp1 * tmp1), axis=1) -
                    self.k12 * np.sum((self.k12 * self.k32) ** self.n * tmp5 / np.sqrt(y2t + tmp5 * tmp5), axis=1)
                ) -
                self.k12 * -tmp3 / np.sqrt(y2 + tmp3 * tmp3)
            )

            # Source plus the images transmitted through the dyke, at b - 2nd.  Coefficients obtained
            # from continuity of Bz at x = d with Bz2 and Bz3 -> 0 for x -> inf (the previous
            # expression, from the thesis of Boggs, did not satisfy either); checked against the 2.5D
            # finite-volume solution (tests/test_mmr25.py).
            Bz3 = tmp0 * (
                self.k13 - tmp3 / np.sqrt(y2 + tmp3 * tmp3)
                - (1. + self.k12) * (1. - self.k32) *
                np.sum((self.k12 * self.k32) ** self.n * tmp8 / np.sqrt(y2t + tmp8 * tmp8), axis=1)
            )

        Bz1[np.logical_not(ind1)] = 0.0
        Bz2[np.logical_not(ind2)] = 0.0
        Bz3[np.logical_not(ind3)] = 0.0

        return 1.e9 * (Bz1 + Bz2 + Bz3)


class HemisphericalDepression:
    """Compute MMR anomaly for a half-sphere.

    Reference
    ---------

    """

    def __init__(self, rho0, rho1, a, b, n_max=100):
        """Initialize a HemisphericalDepression object.

        Parameters
        ----------
        rho0 : float
            Resistivity of host rock
        rho1 : float
            Resistivity of hemispherica depression
        a: float
            Radius of the hemispherical depression
        b: float
            Y coordinate of the hemispherical depression center
        n_max: int
            Upper limit of summations. Default is 100.

        Notes
        -----
        The solution given by Edwards and Nabighian considers that X-coordonate of both the current electrode and the
        center of the hemispherical depression are equal to zero, and that the current electrode is at (0, 0).
        """
        self.rho0 = rho0
        self.rho1 = rho1
        self.a = a
        self.b = b
        self.n_max = n_max

    @property
    def n_max(self):
        return self.n[-1]

    @n_max.setter
    def n_max(self, n_max):
        self.n = np.arange(1, n_max+1)

    def fwd_mod_cond(self, xo, cs=1.0):
        """Forward MMR solution for a conductive depression

        Parameters
        ----------
        xo : array_like
            X,Y,Z coordinates of observation points (m).
        cs : float
            Current intensity (A).

        Returns
        -------
            Vertical B-field anomaly (nT).

        Notes
        -----
        The current electrode is assumed to be located at (0, 0).
        """
        r = np.sqrt(xo[:, 0] ** 2 + (xo[:, 1] + self.b) ** 2)
        sin_theta = xo[:, 0] / r
        cos_theta = (xo[:, 1] + self.b) / r

        mu = 12.566370614359172e-7

        A = mu * cs * sin_theta / (4 * np.pi)

        ind_d = r <= self.a

        if self.a > np.abs(self.b):  # electrode inside depression
            raise ValueError

        # field in depression
        tmp = r/self.b
        Bz1 = A/self.b * (cos_theta - (cos_theta - tmp) /
                          np.sqrt(1 - 2*tmp*cos_theta + tmp**2))

        # field in host rock
        tmp = self.a**2/(r*self.b)
        Bz0 = A*self.a / (r*self.b) * (cos_theta - (cos_theta - tmp) /
                                       np.sqrt(1 - 2*tmp*cos_theta + tmp**2))

        Bz1[np.logical_not(ind_d)] = 0.0
        Bz0[ind_d] = 0.0

        return 1.e9 * (Bz0 + Bz1)


    def fwd_mod(self, xo, cs=1.0):
        """Forward MMR solution

        Parameters
        ----------
        xo : array_like
            X,Y,Z coordinates of observation points (m).
        cs : float
            Current intensity (A).

        Returns
        -------
            Vertical B-field anomaly (nT).

        Notes
        -----
        The current electrode is assumed to be located at (0, 0).
        """
        r = np.sqrt(xo[:, 0] ** 2 + (xo[:, 1] + self.b) ** 2)
        sin_theta = xo[:, 0] / r
        cos_theta = (xo[:, 1] + self.b) / r
        sin_theta[r==0.0] = 0.0
        cos_theta[r == 0.0] = 1.0

        Cn = (self.rho1 - self.rho0) / ((self.n + 1) * self.rho1 + self.n * self.rho0)
        Cn = np.tile(Cn.reshape(-1, 1), (1, r.size))
        mu = 12.566370614359172e-7

        tmp0 = mu * cs * sin_theta / ( 4 * np.pi * r)
        tmp0[r==0] = 0.0
        tmp1 = p_prime(self.n.astype(np.int32), cos_theta)

        ind_d = r <= self.a

        if self.a < np.abs(self.b):  # electrode outside depression
            # field in depression
            Bz1 = -tmp0 * np.sum(Cn * (np.tile(r, (self.n.size, 1))/self.b)**(np.tile(self.n.reshape(-1,1), (1, r.size)) + 1) * tmp1, axis=0)
            # field in host rock
            Bz0 = -tmp0 * np.sum(Cn * (self.a/np.tile(r, (self.n.size, 1)))**np.tile(self.n.reshape(-1,1), (1, r.size)) *
                                (self.a / self.b)**(np.tile(self.n.reshape(-1,1), (1, r.size)) + 1) * tmp1, axis=0)
        else:
            Bz1 = -tmp0 * np.sum(Cn * (np.tile(r, (self.n.size, 1))/self.a)**(np.tile(self.n.reshape(-1,1), (1, r.size)) + 1) *
                                 (self.b / self.a)**np.tile(self.n.reshape(-1,1), (1, r.size)) * tmp1, axis=0)
            Bz0 = -tmp0 * np.sum(Cn * (self.b / np.tile(r, (self.n.size, 1)))**np.tile(self.n.reshape(-1,1), (1, r.size)) * tmp1, axis=0)

        Bz1[np.logical_not(ind_d)] = 0.0
        Bz0[ind_d] = 0.0

        return 1.e9 * (Bz0 + Bz1)
