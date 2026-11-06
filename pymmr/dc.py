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

import numpy as np
import scipy.sparse as sp
from scipy.stats.mstats import gmean

from pymmr.finite_volume import GridFV, Grid25FV, MeshFV, Solver

# TODO: généraliser ROI pour voxels arbitraires


# %% Some functions

def all(a: np.ndarray) -> bool:
    return a.sum() == a.size


def _weights(w, n):
    """Return weights as a 1D array of size n.

    w can be a 1D array, a column vector, or a (sparse) diagonal matrix.  Note that the diagonal of a
    column vector has a single element, which scipy would broadcast to all faces.
    """
    if sp.issparse(w):
        w = w.diagonal()
    else:
        w = np.asarray(w, dtype=float)
        if w.ndim == 2 and w.shape[0] == w.shape[1] and w.shape[0] > 1:
            w = np.diagonal(w)
        w = w.ravel()
    if w.size != n:
        raise ValueError(f'Weights must have {n} elements, got {w.size}')
    return w


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
    if sort_back:
        # the inverse mapping returned by np.unique gives, for each row of a,
        # the index of the matching (unique, sorted) row in tmp -- which is
        # exactly ind_back when all rows are distinct.
        tmp, inv = np.unique(a, axis=0, return_inverse=True)
        if tmp.shape != a.shape:
            raise ValueError('All rows must be different when sort_back is True')
        return tmp, inv.ravel().astype(int)
    else:
        tmp, cnt = np.unique(a, axis=0, return_counts=True)
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
    n_threads : int, optional
    """

    units_scaling_factors = {'mV': 1.e3, 'V': 1.0}

    def __init__(self, param_fv, units='mV', comm=None, n_threads=1):
        if len(param_fv) not in (2, 3):
            raise ValueError('GridDC: param_fv must have 2 or 3 elements')

        if len(param_fv) == 3:
            if param_fv[0].ndim == 1:
                self.fv = GridFV(param_fv, comm, n_threads)
            else:
                self.fv = MeshFV(param_fv, comm, n_threads)
        else:
            # we have 2
            if param_fv[0].ndim == 1:
                self.fv = Grid25FV(param_fv, comm, n_threads)
            else:
                raise ValueError('2.5D not yet implemented for triangular meshes')

        self._c1c2 = None
        self._cs = None
        self._p1p2 = None
        self.u0 = None
        self.u = None
        self.q = None
        self.Q = None
        self.apply_bc = True
        # 2.5D with apply_bc: apply the error of the inverse Fourier transform to the secondary field only.
        # Exact for homogeneous models, but less accurate when the secondary field is large (e.g. conductive
        # basement at long offsets), hence False by default
        self.correct_quadrature = False
        self.exact_primary = True        # 2.5D with apply_bc: analytic primary field at measurement electrodes
        self.local_sigma0 = True         # 2.5D with apply_bc: reference fields use conductivity at electrodes
        self._sigma0 = None              # conductivity of reference model used for boundary correction
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
        self.verbose = 0

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
                    raise ValueError('Measurement points ({0:f}, {1:f}, {2:f}) - ({3:f}, {4:f}, {5:f}) outside grid'.format(tmp[ns, 0], tmp[ns, 1], tmp[ns, 2], tmp[ns, 3], tmp[ns, 4], tmp[ns, 5]))
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
            Tuple holding xmin, xmax, ymin, ymax, zmin, zmax.
            In 2.5D, y is ignored and the tuple can also hold xmin, xmax, zmin, zmax.
        """

        if roi is None:
            return
        if self.fv.dim == 2.5:
            if len(roi) == 4:
                roi = (roi[0], roi[1], self.fv.y[0], self.fv.y[-1], roi[2], roi[3])
            elif len(roi) == 6:
                roi = (roi[0], roi[1], self.fv.y[0], self.fv.y[-1], roi[4], roi[5])
            else:
                raise ValueError('ROI must hold 4 values (xmin xmax zmin zmax) or 6 values in 2.5D')
        elif len(roi) != 6:
            raise ValueError('ROI must hold 6 values (xmin xmax ymin ymax zmin zmax)')
        self.roi = tuple(roi)
        xmin, xmax, ymin, ymax, zmin, zmax = self.roi
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
            Apply preconditioning, possible values are 'ilu', 'diag', or '0' for no preconditioning.
        do_perm : bool, optional
            Apply inverse Cuthill-McKee permutation.

        Notes
        -----
        `precon` et `do_perm` are used only with iterative solvers.

        """
        self.fv.set_solver(name, atol, rtol, max_it, precon, do_perm)

    def set_solver_print_level(self, level):
        self.fv.set_solver_print_level(level)

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

        if self.verbose and self.fv.comm.rank == 0:
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
            if not isinstance(self.fv, GridFV):
                # 2.5D needs the fields of each wavenumber; tetrahedral meshes lack the face operators
                raise NotImplementedError('Sensitivity computation implemented only for 3D rectilinear grids')
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

        if self.fv.dim == 2.5 and self.fv.auto_kg and q is None:
            self._update_k_g()

        if q is not None:
            self.q = sp.csr_array(q)
            self.u0 = None
        elif self.apply_bc:
            self._boundary_correction(sigma)
        else:
            self._build_q()

        if sigma is not None:
            A, M = self.fv.build_A(sigma)
            if self.verbose and self.fv.comm.rank == 0:
                total_memory_bytes = A.data.nbytes + \
                                     A.indptr.nbytes + \
                                     A.indices.nbytes

                print("      Memory footprint of A : {0:4.2f} MB".format(float(total_memory_bytes)/1048576), flush=True)
            self.fv.set_system(A, keep_solver, verbose=(self.verbose > 1))
        elif self.fv.solver_A is None:
            raise RuntimeError('Variable sigma should be given as input.')

        if q is not None:
            # user-supplied source terms (e.g. MMR adjoint): solve for all columns
            self.u = self.fv.solve(self.q)
        else:
            self.u = np.empty((self.fv.nc, self.q.shape[1]))
            self.u[:, :c1c2.shape[0]] = self.fv.solve(self.q[:, :c1c2.shape[0]])

        if self.p1p2 is None:
            data = None
        else:
            data = self._units_scaling * self._get_data()

        if calc_sens:
            if self.verbose and self.fv.comm.rank == 0:
                print('  Computing sensitivity ... ', end='', flush=True)
            # TODO : use formulation of Haber (book), when number of p1p2 > number of voxels
            self.u[:, c1c2.shape[0]:] = self.fv.solver_A.solve(self.q[:, c1c2.shape[0]:], verbose=False)

            sens = np.empty((self.ind_roi.size, self.c1c2.shape[0]))
            # d(m_f)/d(ln sigma_i) = m_f^2 w_fi / sigma_i  (m_f: harmonic mean on face f, w_fi: weights in Gf),
            # and A^-T = Vc A^-1 Vc^-1, D^T = -Vf G Vc^-1  ->  face volumes Vf enter the adjoint product
            S = sp.diags_array(M.diagonal()**2 * self.fv.volume_faces())
            Dm = sp.diags_array(1.0/sigma)

            if self.verbose and self.fv.comm.rank == 0:
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
                if self.verbose and self.fv.comm.rank == 0:
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

            if self.verbose and self.fv.comm.rank == 0:
                print('done.\nEnd of modelling.')

            return data, sens

        elif calc_J:
            if self.verbose and self.fv.comm.rank == 0:
                print('  Computing current density ... ', end='', flush=True)
            J = np.empty((self.fv.nf, self.q.shape[1]))
            for ns in range(self.q.shape[1]):
                J[:, ns] = -M @ self.fv.G @ self.u[:, ns]

            # if self.sort_electrodes and self.sort_back is not None:
            #     J = J[:, self.sort_back]

            if self.verbose and self.fv.comm.rank == 0:
                print('done.\nEnd of modelling.')
            return data, J
        else:
            if self.verbose and self.fv.comm.rank == 0:
                print('End of modelling.')
            return data

    def calc_reg(self, Q, par, xc, xref, WGx, WGy, WGz, m_active):
        """Compute regularization matrix with weights from iteratively reweighted least squares (IRLS).

        Parameters
        ----------
        Q : array_like
            Model weights (one value per cell).
        par :
            Inversion parameters (smooth_type, alx, aly, alz, als, minsupport_e, minsupport_form, ekblom_p,
            ekblom_smallness_max).
        xc : array_like
            Current model (one value per cell).
        xref : array_like or float
            Reference model: the weights are computed from xc - xref, the quantity regularized by the
            inversion (e.g. m - m_ref with reg_var 'model').
        WGx, WGy, WGz : array_like or None
            Base weights of the gradient terms (None: 1).  The IRLS weights computed from the current
            model multiply these base weights; they are not accumulated over iterations, i.e. the
            weights returned should not be passed back as base weights.
        m_active : array_like
            Active cells.

        Returns
        -------
        tuple
            WtW, and the weights of the gradient terms WGx, WGy, WGz (base weights x IRLS weights)

        Notes
        -----
        Weights are those of the squared terms, i.e. the diagonal of W in G^T W G (see calc_WtW):
            - 'blocky': w ~ 1/|c| (L1 measure, normalized by sum(c^2)/sum(|c|)), within [1e-6, 10]
            - 'ekblom': w = p (c^2 + e^2)^(p/2 - 1) (Ekblom measure, sum (c^2 + e^2)^(p/2)), at most 2 for
              the gradients; the smallness term is weighted the same way with c = xc - xref, at most
              par.ekblom_smallness_max (no limit if None)
            - 'min. support' (minimum gradient support, measure sum c^2 / (c^2 + e^2)), at most 2:
                - par.minsupport_form 'portniaguine': w = sqrt(2) e / (c^2 + e^2), the reweighting of
                  Portniaguine & Zhdanov (1999), W_e = (c^2 + e^2)^(-1/2) applied to the gradient
                - par.minsupport_form 'general': w = 2 e^2 / (c^2 + e^2)^2, the general IRLS weight
                  rho'(c) / c of the measure (Farquharson & Oldenburg, 1998)
        with c the gradient of xc - xref, p = par.ekblom_p and e = par.minsupport_e.
        """
        Gx, Gy, Gz = self.fv.extract_Gxyz()
        G = (Gx, Gy, Gz)
        base = [np.ones((Gi.shape[0],)) if Wi is None else _weights(Wi, Gi.shape[0])
                for Gi, Wi in zip(G, (WGx, WGy, WGz))]
        WS = None
        # 1D arrays (a column vector would broadcast to n x n arrays with the base weights)
        ci = np.asarray(xc, dtype=float).ravel() - np.asarray(xref, dtype=float).ravel()

        if par.smooth_type == 'blocky':
            wt = []
            for Gi in G:
                c = Gi @ ci
                with np.errstate(divide='ignore', invalid='ignore'):
                    w = np.sum(c*c) / (np.sum(np.abs(c)) * np.abs(c))
                # zero gradient: largest weight (w ~ 1/|c|)
                w[c == 0.0] = 10.0
                wt.append(np.clip(w, 1.e-6, 10.0))

        elif par.smooth_type == 'ekblom':
            p, e = par.ekblom_p, par.minsupport_e
            wt = [np.minimum(p * ((Gi @ ci)**2 + e**2)**(p/2 - 1), 2.0) for Gi in G]
            # smallness term (one value per cell)
            WS = p * (ci*ci + e**2)**(p/2 - 1)
            ws_max = getattr(par, 'ekblom_smallness_max', 2.0)
            if ws_max is not None:
                WS = np.minimum(WS, ws_max)

        elif par.smooth_type == 'min. support':
            e = par.minsupport_e
            form = getattr(par, 'minsupport_form', 'portniaguine')
            if form == 'portniaguine':
                wt = [np.minimum(np.sqrt(2) * e / ((Gi @ ci)**2 + e**2), 2.0) for Gi in G]
            elif form == 'general':
                wt = [np.minimum(2 * e**2 / ((Gi @ ci)**2 + e**2)**2, 2.0) for Gi in G]
            else:
                raise ValueError("minsupport_form must be 'portniaguine' or 'general'")

        else:
            wt = [np.ones((Gi.shape[0],)) for Gi in G]

        WGx, WGy, WGz = (b * w for b, w in zip(base, wt))
        return self.calc_WtW(Q, par, m_active, WGx, WGy, WGz, WS=WS), WGx, WGy, WGz

    def calc_WtW(self, wt, par, m_active, WGx=None, WGy=None, WGz=None, WS=None):
        """Compute model weighting matrix.

        Parameters
        ----------
        wt : array_like
            Weight.
        par :
            Weighting parameters.
        m_active : array_like
            Cellules du modèles actives.
        WGx : array_like, optional
            Poids de lissage des cellules en x (one value per face, 1D or column vector, or diagonal matrix).
        WGy : array_like, optional
            Poids de lissage des cellules en y.
        WGz : array_like, optional
            Poids de lissage des cellules en z.
        WS : array_like, optional
            Weights of the smallness term (one value per cell).

        Returns
        -------
        output : `csr_matrix`
            Model weighting matrix.

        Notes
        -----
        WtW = Wt^T (al_x^2 Gx^T WGx Gx + al_y^2 Gy^T WGy Gy + al_z^2 Gz^T WGz Gz + al_s WS) Wt, i.e. the
        weights multiply the squared terms (sqrt(WG) is applied to G).
        """

        Gx, Gy, Gz = self.fv.extract_Gxyz()

        if WGx is not None:
            Gx = sp.diags_array(np.sqrt(_weights(WGx, self.fv.nfx)), format='csr') @ Gx
        if WGy is not None:
            Gy = sp.diags_array(np.sqrt(_weights(WGy, self.fv.nfy)), shape=(self.fv.nfy, self.fv.nfy),
                                format='csr') @ Gy
        if WGz is not None:
            Gz = sp.diags_array(np.sqrt(_weights(WGz, self.fv.nfz)), format='csr') @ Gz

        Gs = sp.vstack((par.alx*Gx, par.aly*Gy, par.alz*Gz), format='csr')
        if WS is None:
            V = sp.diags_array(np.ones((self.fv.nc,)), format='csr')
        else:
            V = sp.diags_array(_weights(WS, self.fv.nc), format='csr')
        Wt = sp.diags_array(wt, shape=(self.fv.nc, self.fv.nc), format='csr')

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
        g2 = Grid25FV((x, z)) if self.fv.dim == 2.5 else GridFV((x, y, z))
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
            if self.fv.dim != 2.5:
                # in 2.5D, Jy = 0 in the plane y = 0
                filename = basename+'_Jy_dc_dip'+str(n+1)
                self.fv.toVTK({'Jy': jy}, filename, component='y', metadata=metadata)
            filename = basename+'_Jz_dc_dip'+str(n+1)
            self.fv.toVTK({'Jz': jz}, filename, component='z', metadata=metadata)

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
        q = sp.lil_array((self.fv.nc, c1c2.shape[0]))
        for i in range(c1c2.shape[0]):
            Q = self.fv.linear_interp(c1c2[i, 0], c1c2[i, 1], c1c2[i, 2])
            if c1c2.shape[1] == 6:
                Q -= self.fv.linear_interp(c1c2[i, 3], c1c2[i, 4], c1c2[i, 5])
            # A = -div(sigma grad), so +I at the first electrode (source) gives a positive potential around it;
            # the second electrode is the sink.  The orientation of the z axis plays no role (potential is a scalar).
            q[:, i] = cs[i] * Q.toarray() * iv
        self.q = q.tocsr()
        self.u0 = None

    def _boundary_correction(self, ref_model):
        """Apply correction described in Pidlisecky et al. 2007"""

        if self.keep_c1c2:
            c1c2 = self.c1c2_u.copy()
            if self.cs is None:
                warnings.warn('Current source intensity undefined, using 1 A', RuntimeWarning, stacklevel=2)
                self.cs = np.ones((c1c2.shape[0],))
            cs = self.cs12_u
        else:
            # sensitivity: correction applies to the injection poles only; the adjoint sources at the
            # measurement electrodes must be the plain discrete point sources (adjoint of d = Q u)
            c1c2 = self.c12_u.copy()
            cs = self.cs_u

        avg_cond = gmean(ref_model.flatten())
        self._sigma0 = avg_cond

        if self.fv.dim == 3:
            self.u0 = self.fv.compute_u_homog(c1c2, avg_cond, cs)
            A, _ = self.fv.build_A(avg_cond)
            self.q = sp.csr_array(A @ self.u0)
        else:
            # 2.5 D: correction applied for each wavenumber, q_k = L_k(sigma0) U0_k
            if self.local_sigma0:
                # reference fields with the conductivity at each current electrode (see boundary_source)
                self._sigma0 = self._electrode_sigma(c1c2, ref_model)
            self.q, self.u0 = self.fv.boundary_source(c1c2, self._sigma0, cs,
                                                      correct_quadrature=self.correct_quadrature)

        if not self.keep_c1c2:
            # append unit point sources at measurement electrodes (adjoint fields)
            p12 = self.fv.process_surface_elec(self.p12_u.copy())
            iv = 1.0 / self.fv.volume_voxels()
            q_r = sp.lil_array((self.fv.nc, p12.shape[0]))
            for i in range(p12.shape[0]):
                q_r[:, i] = self.cp_u[i] * self.fv.linear_interp(p12[i, 0], p12[i, 1], p12[i, 2]).toarray() * iv
            self.q = sp.hstack((sp.csr_array(self.q), q_r.tocsr())).tocsr()

    def _electrode_sigma(self, c1c2, sigma):
        """Conductivity interpolated at the current electrodes (n x 2 for dipoles, n for poles)."""
        sigma = np.asarray(sigma, dtype=float).ravel()
        n_e = 2 if c1c2.shape[1] == 6 else 1
        s = np.ones((c1c2.shape[0], n_e))
        for j in range(n_e):
            e = self.fv.process_surface_elec(c1c2[:, 3*j:3*j+3].copy())
            ok = np.all(np.isfinite(e), axis=1)   # electrodes at infinity: value irrelevant
            if np.any(ok):
                if sigma.size == 1:
                    s[ok, j] = sigma[0]
                else:
                    # sample within the cell centres (no extrapolation of sigma near the grid edges)
                    xe = np.clip(e[ok, 0], self.fv.xc[0], self.fv.xc[-1])
                    ze = np.clip(e[ok, 2], self.fv.zc[0], self.fv.zc[-1])
                    s[ok, j] = self.fv.linear_interp(xe, e[ok, 1], ze) @ sigma
        return s if n_e == 2 else s[:, 0]

    def _survey_distances(self):
        """Smallest distance between current and potential electrodes, and largest distance between electrodes."""
        c1c2 = self.c1c2 if self.c1c2 is not None else self.c1c2_u
        c = [c1c2[:, :3]] + ([c1c2[:, 3:6]] if c1c2.shape[1] == 6 else [])
        if self.p1p2 is None:
            raise ValueError('Measurement points undefined')
        p = [self.p1p2[:, :3]] + ([self.p1p2[:, 3:6]] if self.p1p2.shape[1] == 6 else [])
        r = []
        with np.errstate(invalid='ignore'):
            for ci in c:
                for pi in p:
                    r.append(np.sqrt((ci[:, 0] - pi[:, 0])**2 + (ci[:, 2] - pi[:, 2])**2))
        r = np.concatenate(r)
        r = r[np.isfinite(r) & (r > 0)]
        if r.size == 0:
            raise ValueError('Cannot compute electrode distances')
        e = np.vstack(c + p)
        e = e[np.all(np.isfinite(e), axis=1), :]
        extent = np.hypot(np.ptp(e[:, 0]), np.ptp(e[:, 2]))
        return r.min(), max(r.max(), extent)

    def _update_k_g(self):
        """Optimize wavenumbers for the distances spanned by the survey (2.5D).

        The transform must be accurate for the source-receiver distances, but also for the longer paths
        of the secondary field: the range spans from half the smallest source-receiver distance to
        4 times the largest distance between electrodes, and at least 2 decades (with 11 wavenumbers,
        the error of the transform is then about 1e-4 or less).

        Without measurement electrodes (e.g. current density only), the range is based on the grid:
        half the smallest cell size to the grid length.
        """
        if self.p1p2 is None:
            r_min = min(np.r_[self.fv.hx, self.fv.hz]) / 2
            r_max = max(self.fv.x[-1] - self.fv.x[0], 100 * r_min)
        else:
            r_min, r_max = self._survey_distances()
            r_min = r_min / 2
            r_max = max(4 * r_max, 100 * r_min)
        r_range = (self.fv.n_k, r_min, r_max)
        if self.fv._kg_range is None or not np.allclose(r_range, self.fv._kg_range):
            if self.verbose and self.fv.comm.rank == 0:
                print('  Optimizing {0:d} wavenumbers for r in [{1:g}, {2:g}] m'.format(*r_range))
            self.fv._optimize_k_g(*r_range)

    def _sort_electrodes(self):

        if self.verbose and self.fv.comm.rank == 0:
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

        nc1c2 = self.c1c2.shape[0]
        c12 = np.vstack((self.c1c2[:, :3], self.c1c2[:, 3:]))
        self.c12_u, inv = np.unique(c12, axis=0, return_inverse=True)
        inv = inv.ravel()
        # c12 stacks c1 (first nc1c2 rows) over c2, so the inverse indices into
        # c12_u split directly into the c1 and c2 electrode indices.
        self.ind_c1 = inv[:nc1c2]  # indices dans c12_u
        self.ind_c2 = inv[nc1c2:]  # indices dans c12_u

        # cs_u: as in the original loop, the last write wins in dipole order
        # (and c1 before c2 within a dipole).
        self.cs_u = np.empty((self.c12_u.shape[0],))
        order = np.argsort(np.concatenate((np.arange(nc1c2), np.arange(nc1c2))),
                           kind='stable')
        self.cs_u[np.concatenate((self.ind_c1, self.ind_c2))[order]] = \
            np.concatenate((self.cs, self.cs))[order]

        # ind: representative original row of each unique injection dipole
        self.ind_c1_u = self.ind_c1[ind]  # indices dans c12_u
        self.ind_c2_u = self.ind_c2[ind]  # indices dans c12_u

        self.c12_u = self.c12_u[:, :3]

        if self.verbose and self.fv.comm.rank == 0:
            print('    Detected {0:d} injection dipole(s)'.format(self.n_c1c2_u))

        # dipoles de mesure
        if self.p1p2 is not None:
            np1p2 = self.p1p2.shape[0]
            p12 = np.vstack((self.p1p2[:, :3], self.p1p2[:, 3:]))
            self.p12_u, invp = np.unique(p12, axis=0, return_inverse=True)
            invp = invp.ravel()
            self.ind_p1 = invp[:np1p2]
            self.ind_p2 = invp[np1p2:]

            # adjoint sources at the measurement electrodes have unit intensity
            # (sensitivity must scale linearly with the injected current)
            self.cp_u = np.ones((self.p12_u.shape[0],))

            self.p12_u = self.p12_u[:, :3]

            if self.verbose and self.fv.comm.rank == 0:
                print('    Detected {0:d} measurement electrode(s)'.format(self.p12_u.shape[0]))

        self.electrodes_sorted = True

    def _get_data(self, u=None):
        if self.Q is None:
            self._build_Q()
        # 2.5D with boundary correction: the analytic primary field is known exactly at the measurement
        # electrodes, interpolate only the (smooth) secondary field
        exact_primary = (u is None and self.fv.dim == 2.5 and self.keep_c1c2 and self.exact_primary and
                         self.u0 is not None and self._sigma0 is not None)
        if u is None:
            u = self.u
        data = np.array([])
        if self.keep_c1c2:
            for ns in range(self.c1c2_u.shape[0]):
                if exact_primary:
                    ind = self.ind_c1c2 == ns
                    p1p2 = self.p1p2[ind, :]
                    c1c2 = np.tile(self.c1c2_u[ns, :], (p1p2.shape[0], 1))
                    if np.isscalar(self._sigma0):
                        s0 = self._sigma0
                    else:
                        s0 = np.tile(self._sigma0[ns], (p1p2.shape[0], 1))
                    prim = self.fv.u_homog_at(c1c2, p1p2[:, :3], s0, self.cs12_u[ns])
                    if p1p2.shape[1] == 6:
                        prim -= self.fv.u_homog_at(c1c2, p1p2[:, 3:6], s0, self.cs12_u[ns])
                    data = np.r_[data, self.Q[ns] @ (u[:, ns] - self.u0[:, ns]) + prim]
                else:
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
        return sp.csr_array(u)

    def _calc_Gv(self, m, m_ref, m_active, u, v):

        v = np.atleast_2d(v)
        mp = m_ref.copy()
        mp[m_active] = m

        S = self.fv.build_M(np.exp(mp)).power(2)

        Dm = sp.csr_array((np.exp(-m), (np.arange(m.size), np.arange(m.size))))

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
        return sp.csr_array(gu)

    def _calc_Gvt(self, m, m_ref, m_active, u, v):

        mp = m_ref.copy()
        mp[m_active] = m

        S = self.fv.build_M(np.exp(mp)).power(2)

        Dm = sp.csr_array((np.exp(-m), (np.arange(m.size), np.arange(m.size))))

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
        v = sp.diags_array(self.fv.G @ u)
        Gc = v @ Gf
        # Gc = self.build_G(self.G @ u)
        A = Dm @ Gc.T @ S
        # adjoint: d(data)/d(ln sigma) = -Dm Gf^T diag(G u) diag(m^2 Vf) G u_r,
        # with u_r the field of unit sources at the receivers (S holds m^2 Vf)
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
        c1c2 & p1p2 can have 3 or 6 columns, to represent poles or dipoles, but
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
