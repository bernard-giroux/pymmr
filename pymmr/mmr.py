#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module pour la modélisation et l'inversion en magnétorésistivité

@author: giroux

Référence principale:
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

import numpy as np
import scipy.sparse as sp

from pymmr.dc import GridDC
from pymmr.finite_volume import calc_padding, GridFV, MeshFV, Solver


# %% class GridMMR

class GridMMR():
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
        if tuple has 2 elements, a mesh in built and the elements are
            pts : array of float
            tet: array of int
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

    def __init__(self, param_fv, units='pT', comm=None):
        if len(param_fv) != 3:
            raise ValueError('GridDC: param_fv must have 3 elements')

        if param_fv[0].ndim == 1:
            self.fv = GridFV(param_fv, comm=comm)
        else:
            self.fv = MeshFV(param_fv, comm=comm)

        self.dc = GridDC(param_fv, units='mV', comm=comm)
        self.dc.verbose = False
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
        self.verbose = False

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
            if self.fv.is_inside(tmp[ns, 0], tmp[ns, 1], tmp[ns, 2]) is False or \
               self.fv.is_inside(tmp[ns, 3], tmp[ns, 4], tmp[ns, 5]) is False:
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

    def set_survey_mmr(self, xs, xo, cs):
        """Set survey variables.

        Parameters
        ----------
        xs : array_like, optional
            Coordinates of injection points (m).
        xo : array_like, optional
            Coordinates of measurement points (m).
        cs : scalar or array_like
            Intensity of current source

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
        self.dc.fv.set_solver(name, tol, max_it, precon, do_perm, comm)
        self.fv.set_solver(name, tol, max_it, precon, do_perm, comm)

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
        if self.fv.nc == self.dc.fv.nc:
            self._add_air()

        if self.xs is None:
            raise ValueError('Source term undefined')

        if self.xo is None:
            raise ValueError('Measurement points undefined')

        if self.verbose:
            print('\nForward modelling - Magnetometric resistivity')
            self.dc.print_info()
            if self.fv.solver_A is not None:
                self.fv.solver_A.print_info()

        if self.acq_checked is False:
            self.check_acquisition()

        if self.in_inv and self.dc.c1c2 is not None:
            # joint inversion
            res_dc = self.dc.fwd_mod(sigma, calc_sens=calc_sens)

        c1c2_u_save = copy.copy(self.dc.c1c2_u)
        cs12_u_save = copy.copy(self.dc.cs12_u)
        self.dc.c1c2_u = self.xs_u
        self.dc.cs12_u = self.cs_u
        self.dc.sort_electrodes = False

        if self.verbose:
            print('  Computing interpolation matrices ... ', end='', flush=True)
        Qx, Qy, Qz = self._build_Q(self.xo_all)
        if self.verbose:
            print('done.')
        self.Q = Qx, Qy, Qz

        # get current density from forward DC modeling
        if self.verbose:
            print('  Compute current density ... ', end='', flush=True)
        _, Jdc = self.dc.fwd_mod(sigma, calc_J=True)
        u_dc = self.dc.u.copy()

        if self.verbose:
            print('done.')

        Jx = Jdc[:self.dc.fv.nfx, :]
        Jy = Jdc[self.dc.fv.nfx:(self.dc.fv.nfx+self.dc.fv.nfy), :]
        Jz = Jdc[(self.dc.fv.nfx+self.dc.fv.nfy):, :]

        if self.verbose:
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
            self.fv.solver_A = Solver(self.fv.get_solver_params(), self._build_A(), self.verbose)
        if self.fv.solver_A.A is None:
            self.fv.solver_A.A = self._build_A()
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

        if self.in_inv is False:
            data = data[self.ind_back, :]
        else:
            self.dc.sort_electrodes = True
            data = self.data_to_obs(data, False)

        if calc_sens:
            if self.verbose:
                print('  Computing sensitivity')
                print('    Computing adjoint terms ... ', end='', flush=True)

            q = self._adj()
            M = self.dc.fv.build_M(sigma)
            q_a = self.dc.fv.D @ M @ q
            t = np.max(np.abs(q_a), axis=0)
            q_a /= np.tile(t, (q_a.shape[0], 1))

            if self.verbose:
                print('    Solving DC adjoint problem ... ', end='', flush=True)
            self.dc.fwd_mod(calc_J=False, q=q_a)
            if self.verbose:
                print('done.')

            q2 = self.dc.u * np.tile(t, (self.dc.u.shape[0], 1))

            if self.verbose:
                print('    Assembling matrices ... ', end='', flush=True)
            sens = np.empty((self.dc.ind_roi.size, self.nobs_mmr*3))
            S = self.dc.fv.build_M(sigma*sigma)
            Dm = sp.diags(1.0/sigma)

            Gf = self.dc.fv.build_G_faces()
            for ns in range(self.xs_u.shape[0]):
                self._fill_jacobian(ns, sens, u_dc, Dm, S, q, q2, Gf)

            if self.verbose:
                print('done.')

        if self.verbose:
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
            return data

    def calc_WtW(self, wt, par, m_active, WGx=None, WGy=None, WGz=None):
        return self.dc.calc_WtW(wt, par, m_active, WGx, WGy, WGz)

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
        dz_air = calc_padding(self.dc.fv.hz[-1], n_cells=n_crois_cells, factor=factor)
        dz_air = np.r_[self.fv.hz[-1]+np.zeros((n_const_cells,)), dz_air]
        z_air = np.cumsum(dz_air)
        self.fv.z = np.r_[self.dc.fv.z, self.dc.fv.z[-1] + z_air]

    def _adj(self):
        Qx, Qy, Qz = self.Q
        Jx = Qx @ self.C_f[:self.fv.nex, :]
        Jy = Qy @ self.C_f[self.fv.nex:self.fv.nex+self.fv.ney, :]
        Jz = Qz @ self.C_f[self.fv.nex+self.fv.ney:, :]

        J = sp.hstack((Jx.T, Jy.T, Jz.T))

        ind_earth = np.r_[self.fv.ind(np.arange(self.fv.nx-1), np.arange(self.fv.ny),
                                      np.where(self.fv.zc < self.dc.fv.z[-1]), 'fx'),
                          self.fv.nfx+self.fv.ind(np.arange(self.fv.nx), np.arange(self.fv.ny-1),
                                                  np.where(self.fv.zc < self.dc.fv.z[-1]), 'fy'),
                          self.fv.nfx+self.fv.nfy+self.fv.ind(np.arange(self.fv.nx), np.arange(self.fv.ny),
                                                              np.where(self.fv.z[1:-1] < self.dc.fv.z[-1]), 'fz')]

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

    def _fill_jacobian(self, n, J, u_dc, Dm, S, q, q2, Gf):
        if n == 0:
            i0 = 0
        else:
            i0 = np.sum(self.nobs_xs[:n]*3)
        i1 = i0 + self.nobs_xs[n]*3

        v = sp.diags(self.dc.fv.G @ u_dc[:, n])
        Gc = v @ Gf
        # Gc = self.dc.build_G(self.dc.G @ u_dc[:, n])
        A = -Dm @ Gc.T @ S
        tmp = A @ q - A @ self.dc.fv.G @ q2

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



