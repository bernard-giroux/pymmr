#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module pour la modélisation et l'inversion en magnétorésistivité

@author: giroux

Référence principale:

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
import importlib
import re
import sys

import numpy as np
import scipy.sparse as sp

from pymmr.dc import GridDC
from pymmr.finite_volume import calc_padding, build_from_vtk, GridFV, Solver


# %% class GridMMR

class GridMMR(GridDC):
    """Grid for magnetometric resistivity modelling.

    Parameters
    ----------
    x : array of float
        Node coordinates along x (m)
    y : array of float
        Node coordinates along y (m)
    z : array of float
        Node coordinates along z (m)
    units : str, optional
        B-field units
    comm : MPI Communicator, optional
        If None, MPI_COMM_WORLD will be used

    Notes
    -----
    - It is possible to compute the DC resistivity response for injection
    dipoles that are different from the source term defined for MMR modeling.
    This has been implemented for allowing joint inversion of MMR & ERT data.
    """

    units_scaling_factors = {'pT': 1.e12, 'nT': 1.e9}

    def __init__(self, x, y, z, units='pT', comm=None):
        GridDC.__init__(self, x, y, z, units=units, comm=comm)
        self.gdc = GridDC(x, y, z, units='mV', comm=comm)
        self.gdc.verbose = False
        self.acq_checked = False
        self._xs = None
        self._xo = None
        self.xs_u = None
        self.xo_all = None
        self.ind_back = None
        self.ind_s = None
        self.mask = None
        self.nobs_xs = None
        self.nobs_mmr = 0
        self.units = units

    @property
    def apply_bc(self):
        return self.gdc.apply_bc

    @apply_bc.setter
    def apply_bc(self, val):
        if "gdc" in self.__dict__:
            self.gdc.apply_bc = val

    @property
    def in_inv(self):
        return self.gdc.in_inv

    @in_inv.setter
    def in_inv(self, val):
        if 'gdc' in self.__dict__:
            self.gdc.in_inv = val

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
            if self.is_inside(tmp[ns, 0], tmp[ns, 1], tmp[ns, 2]) is False or \
               self.is_inside(tmp[ns, 3], tmp[ns, 4], tmp[ns, 5]) is False:
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
            if self.is_inside(tmp[ns, 0], tmp[ns, 1], tmp[ns, 2]) is False:
                raise ValueError('Observation points outside grid')
#            if tmp[ns, 2] < self.zc[0]:
#                tmp[ns, 2] = self.zc[0]
        self._xo = tmp
        self.acq_checked = False

    @property
    def c1c2(self):
        """Coordinates of injection points for DC resistivity modelling (m)."""
        return self.gdc.c1c2

    @c1c2.setter
    def c1c2(self, val):
        self.gdc.c1c2 = val

    @property
    def p1p2(self):
        """Coordinates of measurement points for DC resistivity modelling (m)."""
        return self.gdc.p1p2

    @p1p2.setter
    def p1p2(self, val):
        self.gdc.p1p2 = val

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, val):
        if val not in GridMMR.units_scaling_factors:
            raise ValueError('Wrong units of B-field')
        self._units = val
        self._units_scaling = GridMMR.units_scaling_factors[val]

    def check_acquisition(self):
        """Check consistency of `xs` and `xo`."""
        if self.xs.shape[0] == self.xo.shape[0]:
            # on a des paires dipoles injection - pts de mesure
            self.xs_u = np.unique(self.xs, axis=0)
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
            self.xo_all = self.xo.copy()
            self.ind_back = np.arange(self.xs.shape[0] * self.xo.shape[0])
            self.ind_s = np.arange(self.xs.shape[0] * self.xo.shape[0])
            self.mask = np.ones((self.xs.shape[0] * self.xo.shape[0],), dtype=bool)
            self.nobs_xs = self.xo.shape[0] + np.zeros((self.xs.shape[0],), dtype=int)
        self.nobs_mmr = np.sum(self.nobs_xs)
        self.acq_checked = True

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
        self.gdc.set_solver(name, tol, max_it, precon, do_perm, comm)
        if callable(name):
            self.solver = name
            self.tol = tol
            self.max_it = max_it
            self.want_pardiso = False
            self.want_superlu = False
            self.want_umfpack = False
            self.want_mumps = False
        elif 'superlu' in name:
            self.want_pardiso = False
            self.want_superlu = True
            self.want_umfpack = False
            self.want_mumps = False
        elif 'pardiso' in name:
            self.want_pardiso = True
            self.want_superlu = False
            self.want_umfpack = False
            self.want_mumps = False
        elif 'umfpack' in name:
            self.want_umfpack = True
            self.want_superlu = False
            self.want_pardiso = False
            self.want_mumps = False
        elif 'mumps' in name:
            self.want_pardiso = False
            self.want_superlu = False
            self.want_umfpack = False
            self.want_mumps = True
        else:
            raise RuntimeError('Solver '+name+' not implemented')

        self.solver_A = Solver((name, tol, max_it, precon, do_perm), verbose=self.verbose, comm=comm)
        self.precon = precon
        self.do_perm = do_perm
        self.comm = comm

    def set_roi(self, roi):
        """Define region of interest for computing sensitivity or for inversion.

        Parameters
        ----------
        roi : tuple of float
            Tuple holding xmin, xmax, ymin, ymax, zmin, zmax
        """
        # on applique la ROI à la grille DC car la grille MMR contient des
        # noeuds additionnels pour l'air
        self.gdc.set_roi(roi)
        self.ind_roi = self.gdc.ind_roi

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

    def fwd_mod(self, sigma, xs=None, xo=None, calc_sens=False, keep_solver=False, cs=1.0):
        """Forward modelling.

        Parameters
        ----------
        sigma : array_like
            Conductivity model (S/m).
        xs : array_like, optional
            Coordinates of injection points (m).
        xo : array_like, optional
            Coordinates of measurement points (m).
        calc_sens : bool, optional
            Calculate sensitivity matrix.
        keep_solver : bool, optional
            Use solver instantiated in a previous `fwd_mod` run.
        cs : scalar or array_like, optional
            Current intensity at injection points, 1 A by default.

        Returns
        -------
        data : ndarray
            components Bx, By & Bz (pT)
        sens : ndarray
            Sensitivity matrix if `calc_sens` is True.
            
        Notes
        -----
        - If `xs` and `xo` have a different number of rows, the response is
        computed for akk possible combinations.  If the number of rows are
        equal, the response is computed for the pairs of `xs` and `xo` rows.

        - Matrix A of the MMR system depends only on grid geometry, and it is
        thus not needed to build it at each run if the grid does not change.
        This is useful if a direct solver is used, matrix A must be factorized
        only once.  Option `keep_solver` allows this, matrix A being factorized
        when the solver is instantiated.
        """
        if self.z.size == self.gdc.z.size:
            self._add_air()

        if xs is not None:
            self.xs = xs
        elif self.xs is None:
            raise ValueError('Source term undefined')

        if xo is not None:
            self.xo = xo
        elif self.xo is None:
            raise ValueError('Measurement points undefined')

        if self.verbose:
            print('\nForward modelling - Magnetometric resistivity')
            self.gdc.print_info()
            if self.solver_A is not None:
                self.solver_A.print_info()

        if self.acq_checked is False:
            self.check_acquisition()

        if self.in_inv and self.gdc.c1c2 is not None:
            res_dc = self.gdc.fwd_mod(sigma, calc_sens=calc_sens)

        c1c2_u_save = copy.copy(self.gdc.c1c2_u)
        cs12_u_save = copy.copy(self.gdc.cs12_u)
        self.gdc.c1c2_u = self.xs_u
        if np.isscalar(cs):
            self.gdc.cs12_u = cs + np.zeros((self.xs_u.shape[0],))
        else:
            self.gdc.cs12_u = cs
        self.gdc.sort_electrodes = False

        if self.verbose:
            print('  Computing interpolation matrices... ', end='', flush=True)
        Qx, Qy, Qz = self._build_Q(self.xo_all)
        if self.verbose:
            print('done.')
        self.Q = Qx, Qy, Qz

        # get current density from forward DC modeling
        if self.verbose:
            print('  Compute current density ... ', end='', flush=True)
        _, Jdc = self.gdc.fwd_mod(sigma, calc_J=True)
        u_dc = self.gdc.u.copy()

        if self.verbose:
            print('done.')

        Jx = Jdc[:self.gdc.nfx, :]
        Jy = Jdc[self.gdc.nfx:(self.gdc.nfx+self.gdc.nfy), :]
        Jz = Jdc[(self.gdc.nfx+self.gdc.nfy):, :]

        if self.verbose:
            print('  Solving MMR system')

        # create MMR source term from Jdc (J is 0 in air)
        q = np.zeros((self.nf, self.xs_u.shape[0]))
        for i in range(self.xs_u.shape[0]):
            q[(self.nfx-self.gdc.nfx):self.nfx, i] = Jx[:, i]
            q[(self.nfx+self.nfy-self.gdc.nfy):(self.nfx+self.nfy), i] = Jy[:, i]
            q[(self.nfx+self.nfy+self.nfz-self.gdc.nfz):, i] = Jz[:, i]

        A = self._build_A()

        if self.solver_A is None or keep_solver is False:
            self.solver_A = Solver(self.get_solver_params(), A, self.verbose)
        self.u = self.solver_A.solve(q)

        B = self.C_f @ self.u
        Bx = self._units_scaling * B[:self.nex, :]
        By = self._units_scaling * B[self.nex:(self.nex+self.ney), :]
        Bz = self._units_scaling * B[(self.nex+self.ney):, :]

        no = self.xo_all.shape[0]
        data = np.empty((no*q.shape[1], self.xo_all.shape[1]))
        data[:, 0] = (Qx @ Bx).T.flatten()
        data[:, 1] = (Qy @ By).T.flatten()
        data[:, 2] = (Qz @ Bz).T.flatten()

        if self.in_inv is False:
            data = data[self.ind_back, :]
        else:
            self.gdc.sort_electrodes = True
            data = self.data_to_obs(data, False)

        if calc_sens:
            if self.verbose:
                print('  Computing sensitivity')
                print('    Computing adjoint terms ... ', end='', flush=True)

            q = self._adj()
            M = self.gdc.build_M(sigma)
            q_a = self.gdc.D @ M @ q
            t = np.max(np.abs(q_a), axis=0)
            q_a /= np.tile(t, (q_a.shape[0], 1))

            if self.verbose:
                print('    Solving DC adjoint problem ... ', end='', flush=True)
            self.gdc.fwd_mod(calc_J=False, q=q_a)
            if self.verbose:
                print('done.')

            q2 = self.gdc.u * np.tile(t, (self.gdc.u.shape[0], 1))

            if self.verbose:
                print('    Assembling matrices ... ', end='', flush=True)
            sens = np.empty((self.gdc.ind_roi.size, self.nobs_mmr*3))
            S = self.gdc.build_M(sigma*sigma)
            Dm = sp.diags(1.0/sigma)

            for ns in range(self.xs_u.shape[0]):
                self._fill_jacobian(ns, sens, u_dc, Dm, S, q, q2)

            if self.verbose:
                print('done.')

        if self.verbose:
            print('End of modelling.')

        self.gdc.c1c2_u = c1c2_u_save
        self.gdc.cs12_u = cs12_u_save
        if calc_sens:
            if self.in_inv and self.gdc.c1c2 is not None:
                data = np.r_[data, res_dc[0].reshape(-1, 1)]
                sens = np.c_[sens, res_dc[1]]
            return data, sens
        else:
            if self.in_inv and self.gdc.c1c2 is not None:
                data = np.r_[data, res_dc.reshape(-1, 1)]
            return data

    def calc_WtW(self, wt, par, m_active, WGx=None, WGy=None, WGz=None):
        return self.gdc.calc_WtW(wt, par, m_active, WGx, WGy, WGz)

    def calc_reg(self, Q, par, xc, xref, WGx, WGy, WGz, m_active):
        return self.gdc.calc_reg(Q, par, xc, xref, WGx, WGy, WGz, m_active)

    def distance_weighting(self, xo, beta):
        return self.gdc.distance_weighting(xo, beta)

    def get_roi_nodes(self):
        return self.gdc.get_roi_nodes()

    def save_sensitivity(self, sens, basename):
        """Save sensitivity to VTK files.

        Parameters
        ----------
        sens : ndarray
            Sensitivity
        basename : str
            basename for output files (1 per dipole).
        """
        x, y, z = self.gdc.get_roi_nodes()
        # grille temporaire pour sauvegarder sens
        g2 = GridFV(x, y, z)
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

    def _add_air(self, n_const_cells=4, n_crois_cells=15, factor=1.3):
        # add air layers
        dz_air = calc_padding(self.gdc.hz[0], n_cells=n_crois_cells, factor=factor)
        dz_air = np.r_[self.hz[0]+np.zeros((n_const_cells,)), dz_air]
        z_air = np.cumsum(dz_air)
        self.z = np.r_[self.gdc.z[0] - z_air[::-1], self.gdc.z]

    def _adj(self):
        Qx, Qy, Qz = self.Q
        Jx = Qx @ self.C_f[:self.nex, :]
        Jy = Qy @ self.C_f[self.nex:self.nex+self.ney, :]
        Jz = Qz @ self.C_f[self.nex+self.ney:, :]

        J = sp.hstack((Jx.T, Jy.T, Jz.T))

        ind_earth = np.r_[self.ind(np.arange(self.nx-1), np.arange(self.ny), np.where(self.zc > 0), 'fx'),
                          self.nfx+self.ind(np.arange(self.nx), np.arange(self.ny-1), np.where(self.zc > 0), 'fy'),
                          self.nfx+self.nfy+self.ind(np.arange(self.nx), np.arange(self.ny),
                                                     np.where(self.z[1:-1] > 0), 'fz')]

        q = self.solver_A.solve(J)
        q = q[ind_earth, :]
        return q

    def _build_A(self):
        # average of permeability (scalar for now)
        M_e = M_c = 1. / (4.*np.pi*1e-7)   # ( mu_0 in H/m)

        # curl matrices
        C_e = self.build_C()                # projecting from cell edges to faces
        self.C_f = self.build_C(to_faces=False)  # projecting from cell faces to edges

        # assemble
        A = M_e * C_e @ self.C_f - M_c * self.G @ self.D

        return A

    def _build_Q(self, xo):
        Qx = []
        Qy = []
        Qz = []
        for n in range(xo.shape[0]):
            Qx.append(self.linear_interp(xo[n, 0], xo[n, 1], xo[n, 2], 'ex'))
            Qy.append(self.linear_interp(xo[n, 0], xo[n, 1], xo[n, 2], 'ey'))
            Qz.append(self.linear_interp(xo[n, 0], xo[n, 1], xo[n, 2], 'ez'))
        Qx = sp.vstack(Qx)
        Qy = sp.vstack(Qy)
        Qz = sp.vstack(Qz)
        return Qx, Qy, Qz

    def _fill_jacobian(self, n, J, u_dc, Dm, S, q, q2):
        if n == 0:
            i0 = 0
        else:
            i0 = np.sum(self.nobs_xs[:n]*3)
        i1 = i0 + self.nobs_xs[n]*3

        Gc = self.gdc.build_G(self.gdc.G @ u_dc[:, n])
        A = -Dm @ Gc.T @ S
        tmp = A @ q - A @ self.gdc.G @ q2

        mask_xs = self.mask[n*self.xo_all.shape[0]:(n+1)*self.xo_all.shape[0]]
        mask_xs = np.r_[mask_xs, mask_xs, mask_xs]
        tmp = tmp[self.gdc.ind_roi, :]
        J[:, i0:i1] = self._units_scaling * tmp[:, mask_xs]


# %% main

if __name__ == '__main__':

    # %% read arguments
    if len(sys.argv) > 1:

        basename = 'pymmr'
        g = None
        sigma = None
        xs = None
        xo = None
        solver_name = 'umfpack'
        max_it = 500
        tol = 1e-8
        verbose = False
        calc_sens = False
        roi = None
        cs = 1.0
        precon = False
        do_perm = False
        units = 'pT'

        kw_pattern = re.compile('\s?(.+)\s?#\s?([\w\s]+),')

        # we should have a parameter file as input
        with open(sys.argv[1], 'r') as f:
            for line in f:
                kw_match = kw_pattern.search(line)
                if kw_match is not None:
                    value = kw_match[1].rstrip()
                    keyword = kw_match[2].rstrip()
                    if 'model' in keyword:
                        g = build_from_vtk(GridMMR, value)
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
                        xs = np.atleast_2d(np.loadtxt(value))
                    elif 'measurement' in keyword and 'file' in keyword:
                        xo = np.atleast_2d(np.loadtxt(value))
                    elif 'current' in keyword:
                        cs = float(value)
                    elif 'verbose' in keyword:
                        verbose = int(value)
                    elif 'compute' in keyword and 'sensitivity' in keyword:
                        calc_sens = bool(value)
                    elif 'region' in keyword and 'interest' in keyword:
                        tmp = value.split()
                        if len(tmp) != 6:
                            raise ValueError('6 values needed to define ROI (xmin xmax ymin ymax zmin zmax')
                        roi = [float(x) for x in tmp]
                    elif 'units' in keyword:
                        units = value

        if g is None:
            raise RuntimeError('Grid not defined, check input parameters')

        if roi is not None:
            g.set_roi(roi)
        g.units = units
        g.verbose = verbose
        g.set_solver(solver_name, tol, max_it, precon, do_perm)

        g.xs = xs
        g.xo = xo

        data = g.fwd_mod(sigma, calc_sens=calc_sens, cs=cs)

        if calc_sens:
            if verbose:
                print('Saving sensitivity ... ', end='', flush=True)
            data, sens = data

            x, y, z = g.gdc.get_roi_nodes()
            # grille temporaire pour sauvegarder sens
            g2 = GridFV(x, y, z)
            fields = ('Bx', 'By', 'Bz')
            no = xo.shape[0]

            # make 1 file for each injection dipole
            for ns in range(xs.shape[0]):
                sens_data = {}
                for nc in range(3):
                    for nr in range(no):
                        name = '∂d/∂m - {0:s}: {1:4.1f} {2:4.1f} {3:4.1f}'.format(fields[nc], xo[nr, 0], xo[nr, 1], xo[nr, 2])
                        sens_data[name] = sens[:, (ns * 3 + nc) * no + nr]

                fname = basename+'_mmr_sens_dip'+str(ns+1)
                g2.toVTK(sens_data, fname)
            if verbose:
                print('done.')

        if verbose:
            print('Saving modelled data ... ', end='', flush=True)
        fname = basename+'_mmr.dat'
        header = 'src_x src_y src_z rcv_x rcv_y rcv_z Bx By Bz'
        np.savetxt(fname, np.c_[g.xs, g.xo, data], header=header)
        if verbose:
            print('done.')
