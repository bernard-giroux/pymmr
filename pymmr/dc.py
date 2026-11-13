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
from scipy.special import k0
from scipy.stats.mstats import gmean

from pymmr.finite_volume import GridFV, Grid25FV, MeshFV, Solver, Solver25, KSource
from pymmr.kquad import KQuadrature, select_k

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
        # 2.5D: account for conductivity contrasts at the electrodes (interpolation with flux continuity at
        # the measurement electrodes, reference model with a vertical contact at the current electrodes),
        # see docs/contacts.md
        self.contact_interp = True
        self._refs = None                # reference models of the current electrodes (contact_interp)
        self._Qc = None                  # interpolation at the measurement electrodes (contact_interp)
        self._sigma_model = None         # model of the last forward modelling
        # 2.5D, current and potential electrodes at different y (off-line): 'auto' uses the wavenumbers of
        # KQuadrature (n_k_offline) when the survey has off-line data, True always, False never (off-line
        # data are then computed as if all electrodes were at y = 0)
        self.offline25 = 'auto'
        self.n_k_offline = 30
        self._off25 = None               # plan of the off-line modelling (see _plan_offline25)
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
        self._off25 = None

    @property
    def cs(self):
        """Intensity of source current."""
        return self._cs

    @cs.setter
    def cs(self, val):
        self._cs = val
        self._off25 = None

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
        self._off25 = None

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
        if self.fv.dim == 2.5:
            self._check_offline25()

    def _offline_dy(self):
        """Largest offset along y between the current and potential electrodes of each datum (2.5D).

        Returns None if the survey is not defined.
        """
        if self.c1c2 is None or self.p1p2 is None:
            return None
        c1c2 = np.atleast_2d(np.asarray(self.c1c2, dtype=float))
        p1p2 = np.atleast_2d(np.asarray(self.p1p2, dtype=float))
        if c1c2.shape[0] != p1p2.shape[0]:
            return None
        dy = np.zeros((c1c2.shape[0],))
        for ic in range(0, c1c2.shape[1], 3):
            for ip in range(0, p1p2.shape[1], 3):
                ok = np.all(np.isfinite(c1c2[:, ic:ic + 3]), axis=1) & np.all(np.isfinite(p1p2[:, ip:ip + 3]), axis=1)
                with np.errstate(invalid='ignore'):    # electrodes at infinity (masked)
                    d = np.where(ok, np.abs(p1p2[:, ip + 1] - c1c2[:, ic + 1]), 0.0)
                dy = np.maximum(dy, d)
        return dy

    def _uses_offline25(self):
        """True if the 2.5D data are computed with the off-line method (see offline25)."""
        if self.fv.dim != 2.5 or self.offline25 is False:
            return False
        if self.offline25 is True:
            return True
        dy = self._offline_dy()
        return dy is not None and bool(np.any(dy > 0.0))

    def _check_offline25(self):
        """Warn if current and potential electrodes of a datum are at different y and offline25 is False (2.5D).

        The 2.5D potential is then computed for an offset of 0 along y between the current and the potential
        electrodes: off-line measurements are not modelled (their y is ignored).
        """
        if self.offline25 is not False:
            return
        dy = self._offline_dy()
        if dy is None:
            return
        n = np.count_nonzero(dy > 0.0)
        if n > 0:
            warnings.warn("2.5D: {0:d} of {1:d} data have current and potential electrodes at different y (largest "
                          "offset {2:g} m); off-line measurements are not modelled with offline25 = False, these data "
                          "are computed as if all electrodes were at y = 0".format(n, dy.size, dy.max()), stacklevel=3)

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

        if q is None and self.p1p2 is not None and self._uses_offline25():
            if calc_J:
                raise NotImplementedError('2.5D with off-line measurements (offline25): current density not '
                                          'implemented')
            return self._fwd_offline25(sigma, calc_sens=calc_sens)

        if calc_sens is True:
            if not isinstance(self.fv, (GridFV, Grid25FV)):
                # tetrahedral meshes lack the face operators
                raise NotImplementedError('Sensitivity computation implemented only for rectilinear grids')
            if self.fv.dim == 2.5 and self.apply_bc and self.correct_quadrature:
                raise NotImplementedError('Sensitivity not implemented in 2.5D with correct_quadrature')
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

        if sigma is not None:
            self._sigma_model = np.asarray(sigma, dtype=float).ravel()
            if self._contact() or self._Qc is not None:
                self.Q = None   # interpolation at the measurement electrodes depends on the model

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
        elif calc_sens and self.fv.dim == 2.5:
            # fields of the sources and of the adjoint sources (measurement electrodes), for each wavenumber
            self.u, self.U_k = self.fv.solve(self.q, return_k=True)
        else:
            self.u = np.empty((self.fv.nc, self.q.shape[1]))
            self.u[:, :c1c2.shape[0]] = self.fv.solve(self.q[:, :c1c2.shape[0]])

        if self.p1p2 is None:
            data = None
        else:
            data = self._units_scaling * self._get_data()

        if calc_sens and self.fv.dim == 2.5:
            if self.verbose and self.fv.comm.rank == 0:
                print('  Computing sensitivity ... ', end='', flush=True)
            sens = self._sens25(sigma, M, c1c2.shape[0])
            self.U_k = None   # free memory
            if self.sort_electrodes and self.sort_back is not None:
                sens = sens[:, self.sort_back]
            if self.verbose and self.fv.comm.rank == 0:
                print('done.\nEnd of modelling.')
            return data, sens

        elif calc_sens:
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

            # terms from the reference fields (exact primary field at the measurement electrodes)
            R = self._reference_term_matrix()

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
                if R is not None:
                    sens[:, n] += self._units_scaling * self._sigma0_term(n, R, sigma)[self.ind_roi]

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
        self._Qc = [] if self._contact() else None
        for ns in range(self.n_c1c2_u):
            ind = self.ind_c1c2 == ns
            if self._contact():
                # see Grid25FV.flux_interp; linear interpolation kept for the reference fields
                Q1, c1 = self.fv.flux_interp(p1p2[ind, 0], p1p2[ind, 2], self._sigma_model)
                Q2, c2 = self.fv.flux_interp(p1p2[ind, 3], p1p2[ind, 5], self._sigma_model)
                self._Qc.append({'c1': c1, 'c2': c2,
                                 'L1': self.fv.linear_interp(p1p2[ind, 0], p1p2[ind, 1], p1p2[ind, 2]),
                                 'L2': self.fv.linear_interp(p1p2[ind, 3], p1p2[ind, 4], p1p2[ind, 5])})
                self.Q.append((Q1 - Q2).tocsr())
                continue
            Q = self.fv.linear_interp(p1p2[ind, 0], p1p2[ind, 1], p1p2[ind, 2])
            Q -= self.fv.linear_interp(p1p2[ind, 3], p1p2[ind, 4], p1p2[ind, 5])
            self.Q.append(Q)

    def _contact(self):
        """True if conductivity contrasts at the electrodes are accounted for (contact_interp, 2.5D)."""
        return self.contact_interp and isinstance(self.fv, Grid25FV) and self._sigma_model is not None

    def _elec_interp(self, x, y, z):
        """Interpolation matrix at electrodes (flux_interp with contact_interp, linear_interp otherwise)."""
        if self._contact():
            return self.fv.flux_interp(x, z, self._sigma_model)[0]
        return self.fv.linear_interp(x, y, z)

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
            # by reciprocity, the source is distributed with the interpolation weights of a receiver
            Q = self._elec_interp(c1c2[i, 0], c1c2[i, 1], c1c2[i, 2])
            if c1c2.shape[1] == 6:
                Q -= self._elec_interp(c1c2[i, 3], c1c2[i, 4], c1c2[i, 5])
            # A = -div(sigma grad), so +I at the first electrode (source) gives a positive potential around it;
            # the second electrode is the sink.  The orientation of the z axis plays no role (potential is a scalar).
            q[:, i] = cs[i] * Q.toarray() * iv
        self.q = q.tocsr()
        self.u0 = None
        self._refs = None

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
        self._sigma0_w = None
        self._refs = None

        if isinstance(self.fv, GridFV):
            if self.local_sigma0:
                # reference fields with the conductivity at each current electrode
                self._sigma0, self._sigma0_w = self._electrode_sigma(c1c2, ref_model, return_weights=True)
            self.u0 = self.fv.compute_u_homog(c1c2, self._sigma0, cs)
            # q = A(sigma0) u0(sigma0): the term -D sigma0 G u0 does not depend on sigma0 (u0 ~ 1/sigma0) and
            # is built with sigma0 = 1, but the reference potential term of A (A[0, 0], independent of sigma)
            # must act on u0, so that the solution is exact for a homogeneous model (pole receivers
            # included).  The dependence of q on sigma0 through this term is accounted for in the
            # sensitivity (see _reference_term_matrix).
            f = self.fv.compute_u_homog(c1c2, 1.0, cs)
            A, _ = self.fv.build_A(1.0)
            q = A @ f
            q[0, :] += (self.u0[0, :] - f[0, :]) / self.fv.volume_voxels()[0]
            self.q = sp.csr_array(q)
        elif self.fv.dim == 3:
            # tetrahedral mesh
            self.u0 = self.fv.compute_u_homog(c1c2, avg_cond, cs)
            A, _ = self.fv.build_A(avg_cond)
            self.q = sp.csr_array(A @ self.u0)
        elif self._contact():
            # 2.5D, reference model with a vertical contact at each current electrode (see
            # Grid25FV.contact_reference), q_k = L_k(sigma_ref) U0_k
            model = ref_model if self.local_sigma0 else avg_cond
            n_e = 2 if c1c2.shape[1] == 6 else 1
            self._refs = [self.fv.contact_reference(c1c2[:, 3*j:3*j+3], model) for j in range(n_e)]
            self._sigma0 = None
            self.q, self.u0 = self.fv.contact_source(c1c2, self._refs, cs,
                                                     correct_quadrature=self.correct_quadrature)
        else:
            # 2.5 D: correction applied for each wavenumber, q_k = L_k(sigma0) U0_k
            if self.local_sigma0:
                # reference fields with the conductivity at each current electrode (see boundary_source)
                self._sigma0, self._sigma0_w = self._electrode_sigma(c1c2, ref_model, return_weights=True)
            self.q, self.u0 = self.fv.boundary_source(c1c2, self._sigma0, cs,
                                                      correct_quadrature=self.correct_quadrature)

        if not self.keep_c1c2:
            # append unit point sources at measurement electrodes (adjoint fields)
            p12 = self.fv.process_surface_elec(self.p12_u.copy())
            iv = 1.0 / self.fv.volume_voxels()
            q_r = sp.lil_array((self.fv.nc, p12.shape[0]))
            for i in range(p12.shape[0]):
                q_r[:, i] = self.cp_u[i] * self._elec_interp(p12[i, 0], p12[i, 1], p12[i, 2]).toarray() * iv
            if isinstance(self.q, KSource):
                # same source term for all wavenumbers, no reference fields
                n_k = self.q.q.shape[0]
                q_r = np.broadcast_to(q_r.toarray(), (n_k, self.fv.nc, p12.shape[0]))
                zeros = np.zeros(q_r.shape)
                self.q = KSource(np.concatenate((self.q.q, q_r), axis=2),
                                 None if self.q.U0 is None else np.concatenate((self.q.U0, zeros), axis=2),
                                 None if self.q.u0 is None else np.concatenate((self.q.u0, zeros[0]), axis=1))
            else:
                self.q = sp.hstack((sp.csr_array(self.q), q_r.tocsr())).tocsr()

    def _electrode_sigma(self, c1c2, sigma, return_weights=False):
        """Conductivity interpolated at the current electrodes (n x 2 for dipoles, n for poles).

        If return_weights is True, also returns the interpolation matrices (one n x nc matrix per
        electrode of the dipoles), used for the sensitivity.
        """
        sigma = np.asarray(sigma, dtype=float).ravel()
        n_e = 2 if c1c2.shape[1] == 6 else 1
        s = np.ones((c1c2.shape[0], n_e))
        W = []
        for j in range(n_e):
            e = self.fv.process_surface_elec(c1c2[:, 3*j:3*j+3].copy())
            ok = np.all(np.isfinite(e), axis=1)   # electrodes at infinity: value irrelevant
            Wj = sp.csr_array((c1c2.shape[0], self.fv.nc))
            if np.any(ok):
                if sigma.size == 1:
                    s[ok, j] = sigma[0]
                else:
                    # sample within the cell centres (no extrapolation of sigma near the grid edges)
                    xe = np.clip(e[ok, 0], self.fv.xc[0], self.fv.xc[-1])
                    ye = np.clip(e[ok, 1], self.fv.yc[0], self.fv.yc[-1]) if self.fv.dim == 3 else e[ok, 1]
                    ze = np.clip(e[ok, 2], self.fv.zc[0], self.fv.zc[-1])
                    Wok = self.fv.linear_interp(xe, ye, ze)
                    s[ok, j] = Wok @ sigma
                    i_ok = np.where(ok)[0]
                    sel = sp.csr_array((np.ones(i_ok.size), (i_ok, np.arange(i_ok.size))),
                                       shape=(c1c2.shape[0], i_ok.size))
                    Wj = (sel @ Wok).tocsr()
            W.append(Wj)
        s = s if n_e == 2 else s[:, 0]
        if return_weights:
            return s, W
        return s

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

    def _plan_offline25(self):
        """Current electrodes, wavenumbers, interpolation and weights of the off-line modelling (2.5D).

        The 2D fields depend on the (x, z) position of the current electrodes only: one field per unique
        (x, z) position, for a unit current.  Each datum is the sum of the terms of its pairs of current and
        potential electrodes (up to 4), each with the weights W(dy) of KQuadrature for the offset dy along y
        between the two electrodes (see docs/dc25_offline.md).

        The plan depends on the electrodes (reset by the c1c2 and p1p2 setters) and on n_k_offline (checked
        here); the current intensity is applied in _fwd_offline25.
        """
        if self._off25 is not None and self._off25['k'].size == self.n_k_offline:
            return self._off25
        fv = self.fv
        c1c2 = np.atleast_2d(self.c1c2)
        p1p2 = np.atleast_2d(self.p1p2)
        n = c1c2.shape[0]

        def unique_xz(e):
            ok = np.all(np.isfinite(e), axis=1)
            u, inv = np.unique(e[ok][:, [0, 2]], axis=0, return_inverse=True)
            ind = -np.ones((e.shape[0],), dtype=int)
            ind[ok] = inv.ravel()
            return np.c_[u[:, 0], np.zeros((u.shape[0],)), u[:, 1]], ind

        poles, ind_c = unique_xz(np.vstack((c1c2[:, :3], c1c2[:, 3:6])))
        pts, ind_p = unique_xz(np.vstack((p1p2[:, :3], p1p2[:, 3:6])))

        pairs = []
        r_xz, r_3d = [], []
        for jc, (sc, c0) in enumerate(((1.0, 0), (-1.0, 3))):
            for jp, (sp_, p0) in enumerate(((1.0, 0), (-1.0, 3))):
                ic = ind_c[jc * n:(jc + 1) * n]
                ip = ind_p[jp * n:(jp + 1) * n]
                rows = np.where((ic >= 0) & (ip >= 0))[0]
                if rows.size == 0:
                    continue
                c = c1c2[rows, c0:c0 + 3]
                p = p1p2[rows, p0:p0 + 3]
                d = p - c
                r_xz.append(np.hypot(d[:, 0], d[:, 2]))
                r_3d.append(np.linalg.norm(d, axis=1))
                pairs.append({'rows': rows, 'pole': ic[rows], 'pt': ip[rows], 'sign': sc * sp_,
                              'c': c, 'p': p, 'dy': d[:, 1]})
        if not pairs:
            raise ValueError('2.5D off-line modelling: no pair of finite current and potential electrodes')
        r_xz = np.concatenate(r_xz)
        r_xz = r_xz[r_xz > 0]
        h_min = min(np.min(fv.hx), np.min(fv.hz))
        r_min = max(r_xz.min() if r_xz.size else h_min, h_min / 2)
        r_max = max(4 * np.concatenate(r_3d).max(), 100 * r_min)
        k = select_k(r_min, r_max, self.n_k_offline)
        kq = KQuadrature(k)
        for t in pairs:
            t['W'] = kq.weights(t['dy'], 'even')     # n x n_k
        # unique (current electrode, potential electrode) positions in the x-z plane (sensitivity)
        cp = np.vstack([np.c_[t['pole'], t['pt']] for t in pairs])
        combos, inv = np.unique(cp, axis=0, return_inverse=True)
        inv = inv.ravel()
        i0 = 0
        for t in pairs:
            t['combo'] = inv[i0:i0 + t['rows'].size]
            i0 += t['rows'].size
        pp = fv.process_surface_elec(pts.copy())
        self._off25 = {'poles': poles, 'pts': pp, 'k': k, 'pairs': pairs, 'n_data': n, 'combos': combos,
                       'Q': fv.linear_interp(pp[:, 0], pp[:, 1], pp[:, 2])}
        return self._off25

    def _fwd_offline25(self, sigma, calc_sens=False):
        """Forward modelling with current and potential electrodes at different y (2.5D).

        For a current electrode and a potential electrode at a distance dy along y, the potential is

            u = sum_i W_i(dy) Q (U_i - U0_i) + u0

        with U_i the 2D field of wavenumber k_i, W_i(dy) the even weights of KQuadrature, Q the interpolation
        in the x-z plane, U0_i the analytic fields of the boundary correction and u0 the analytic potential at
        the true positions, dy included (apply_bc and exact_primary; without them, the U_i alone).

        Sensitivity: the source terms do not depend on sigma, so that, with V_i the field of a unit source at
        the potential electrode (adjoint field, L_i^-T = Vc L_i^-1 Vc^-1 and D^T = -Vf G Vc^-1),

            d(Q U_i)/d(ln sigma) = -[Dm Gf^T (G U_i * m^2 Vf * G V_i) + k_i^2 sigma Vc U_i V_i]

        (as in _sens25, with g_i / 2 replaced by W_i(dy) for each pair of electrodes).  U0_i and u0 depend on
        sigma through the conductivity sigma0 at the current electrode (local_sigma0) and are proportional to
        1 / sigma0, which adds -R d(ln sigma0)/d(ln sigma), R being their contribution to the datum.
        The products are computed once per wavenumber and per unique pair of positions in the x-z plane.

        With contact_interp, as on the line (see _contact_data), the term of a pair is

            sum_i W_i(dy) Qf U_i + c (u0 - sum_i W_i(dy) Q U0_i)

        with Qf the interpolation with flux continuity and c the contrast factor at the potential electrode,
        and the reference fields of Grid25FV.contact_reference; the sensitivity has the terms described in
        _contact_sens_term, with W_i(dy) in place of g_i / 2.

        Returns
        -------
        data, or (data, sens) with sens of size n_roi x n_data if calc_sens is True
        """
        if sigma is None:
            raise RuntimeError('Variable sigma should be given as input.')
        sigma = np.asarray(sigma, dtype=float).ravel()
        self._sigma_model = sigma
        contact = self._contact()
        plan = self._plan_offline25()
        fv = self.fv
        poles, k = plan['poles'], plan['k']
        n_p = poles.shape[0]
        n_data = plan['n_data']
        if self.verbose and fv.comm.rank == 0:
            print('    Off-line measurements: {0:d} wavenumbers in [{1:g}, {2:g}] 1/m, {3:d} current '
                  'electrode position(s)'.format(k.size, k[0], k[-1], n_p))
        exact = self.apply_bc and self.exact_primary
        sigma0, sigma0_w = None, None
        U0 = None
        iv = 1.0 / fv.volume_voxels()
        ref = None
        if self.apply_bc:
            if contact:
                ref = fv.contact_reference(poles, sigma if self.local_sigma0 else gmean(sigma))
                src, _ = fv.contact_source(poles, [ref], np.ones((n_p,)), correct_quadrature=exact, k=k)
            else:
                if self.local_sigma0:
                    sigma0, sigma0_w = self._electrode_sigma(poles, sigma, return_weights=True)
                else:
                    sigma0 = gmean(sigma)
                src, _ = fv.boundary_source(poles, sigma0, np.ones((n_p,)), correct_quadrature=exact, k=k)
            q, U0 = src.q, src.U0
        else:
            pp = fv.process_surface_elec(poles.copy())
            q0 = self._elec_interp(pp[:, 0], pp[:, 1], pp[:, 2]).T.toarray() * iv[:, None]
            q = np.broadcast_to(q0, (k.size,) + q0.shape)
        Q = plan['Q']
        Qf = Q
        if contact:
            pts = plan['pts']
            if calc_sens:
                Qf, c_r, cells_r, dW_r, dc_r = fv.flux_interp(pts[:, 0], pts[:, 2], sigma, derivatives=True)
                if not self.apply_bc:
                    pp = fv.process_surface_elec(poles.copy())
                    _, _, cells_s, dW_s, _ = fv.flux_interp(pp[:, 0], pp[:, 2], sigma, derivatives=True)
            else:
                Qf, c_r = fv.flux_interp(pts[:, 0], pts[:, 2], sigma)
        if calc_sens:
            # unit sources at the potential electrodes (adjoint fields), the same for all wavenumbers
            q_r = Qf.T.toarray() * iv[:, None]
            Gf = fv.build_G_faces()
            M = fv.build_M(sigma)
            S = M.diagonal()**2 * fv.volume_faces()
            mass = sigma * fv.volume_voxels()
            sens = np.zeros((self.ind_roi.size, n_data))
            combos = plan['combos']
            chunk = max(1, int(2e7 // max(fv.nf, 1)))
            # contribution of the reference fields to the data, for each pair (proportional to 1 / sigma0)
            R = [np.zeros((t['rows'].size,)) for t in plan['pairs']] if exact else None
            if contact:
                # per pair: sum_i W_i U_i at the cells around the potential electrode (weights of Qf), sum_i W_i V_i
                # at the cells around the current electrode (point sources), derivatives of R and of the data
                # through q_i with respect to ln sa of the reference model
                acc_u = [np.zeros((t['rows'].size, 4)) for t in plan['pairs']]
                acc_v = [np.zeros((t['rows'].size, 4)) for t in plan['pairs']] if not self.apply_bc else None
                dRa = [np.zeros((t['rows'].size,)) for t in plan['pairs']]
                dRb = [np.zeros((t['rows'].size,)) for t in plan['pairs']]
                Tq = [np.zeros((t['rows'].size,)) for t in plan['pairs']]
                vc = fv.volume_voxels()

        A, _ = fv.build_A(sigma)
        par = fv.get_solver_params()
        cs = self.cs * np.ones((n_data,))
        data = np.zeros((n_data,))
        for i, k_i in enumerate(k):
            s = Solver(par, Solver25._L(A, sigma, k_i), verbose=(self.verbose > 1), comm=fv.comm,
                       n_threads=fv.n_threads)
            rhs = q[i] if not calc_sens else np.c_[q[i], q_r]
            F = s.solve(rhs).reshape(fv.nc, -1)
            del s
            U = F[:, :n_p]
            if contact:
                QF = Qf @ U
                if exact:
                    QF = QF - c_r[:, None] * (Q @ U0[i])
            else:
                QF = Q @ (U - U0[i] if exact else U)
            for t in plan['pairs']:
                w = t['sign'] * cs[t['rows']] * t['W'][:, i]
                np.add.at(data, t['rows'], w * QF[t['pt'], t['pole']])
            if not calc_sens:
                continue
            V = F[:, n_p:]
            if exact:
                QU0 = Q @ U0[i]
                for j, t in enumerate(plan['pairs']):
                    w = t['sign'] * cs[t['rows']] * t['W'][:, i]
                    R[j] -= w * QU0[t['pt'], t['pole']]
            if contact:
                if exact:
                    dQU0a = np.zeros((Q.shape[0], n_p))
                    dQU0b = np.zeros((Q.shape[0], n_p))
                    T = np.zeros((Q.shape[0], n_p))
                    VV = vc[:, None] * V
                    for ip in range(n_p):
                        _, dUa, dUb = fv._ref_field(poles[ip], ref, ip, 1.0, kernel=lambda r: 2. * k0(k_i * r),
                                                    derivatives=True)
                        dQU0a[:, ip] = Q @ dUa
                        dQU0b[:, ip] = Q @ dUb
                        if not ref.homogeneous:
                            T[:, ip] = VV.T @ fv.contact_source_derivative(poles[ip], ref, ip, 1.0, k_i, Gf=Gf)
                for j, t in enumerate(plan['pairs']):
                    w = t['sign'] * cs[t['rows']] * t['W'][:, i]
                    acc_u[j] += w[:, None] * U[cells_r[t['pt']], t['pole'][:, None]]
                    if acc_v is not None:
                        acc_v[j] += w[:, None] * V[cells_s[t['pole']], t['pt'][:, None]]
                    if exact:
                        dRa[j] -= w * dQU0a[t['pt'], t['pole']]
                        dRb[j] -= w * dQU0b[t['pt'], t['pole']]
                        Tq[j] += w * T[t['pt'], t['pole']]
            GU, GV = fv.G @ U, fv.G @ V
            for c0 in range(0, combos.shape[0], chunk):
                cb = combos[c0:c0 + chunk]
                P = Gf.T @ (GU[:, cb[:, 0]] * S[:, None] * GV[:, cb[:, 1]]) / sigma[:, None]
                P += (k_i * k_i) * mass[:, None] * U[:, cb[:, 0]] * V[:, cb[:, 1]]
                P = P[self.ind_roi, :]
                for t in plan['pairs']:
                    sel = (t['combo'] >= c0) & (t['combo'] < c0 + chunk)
                    if not np.any(sel):
                        continue
                    rows = t['rows'][sel]
                    w = t['sign'] * cs[rows] * t['W'][sel, i]
                    # a datum appears at most once in a pair: no repeated columns
                    sens[:, rows] -= P[:, t['combo'][sel] - c0] * w

        if contact:
            if calc_sens:
                extra = np.zeros((fv.nc, n_data))
            for j, t in enumerate(plan['pairs']):
                rows, ipt, ipo = t['rows'], t['pt'], t['pole']
                w = t['sign'] * cs[rows]
                if exact:
                    u0, du0a, du0b = (np.zeros((rows.size,)) for _ in range(3))
                    for ip in np.unique(ipo):
                        sel = ipo == ip
                        u0[sel], du0a[sel], du0b[sel] = fv._ref_field(t['c'][sel], ref, ip, 1.0, pts=t['p'][sel],
                                                                      derivatives=True, offset_y=True)
                    np.add.at(data, rows, c_r[ipt] * w * u0)
                if not calc_sens:
                    continue
                cells = cells_r[ipt]                                   # rows x 4
                # weights of Qf: sum over the cells i of the stencil of dW[i, j] sum_k W_k U_k(cell i)
                np.add.at(extra, (cells, rows[:, None]), np.einsum('ni,nij->nj', acc_u[j], dW_r[ipt]))
                if acc_v is not None:
                    np.add.at(extra, (cells_s[ipo], rows[:, None]), np.einsum('ni,nij->nj', acc_v[j], dW_s[ipo]))
                if not exact:
                    continue
                Rj = R[j] + w * u0
                np.add.at(extra, (cells, rows[:, None]), Rj[:, None] * dc_r[ipt])
                ta = Tq[j] + c_r[ipt] * (dRa[j] + w * du0a)
                tb = -Tq[j] + c_r[ipt] * (dRb[j] + w * du0b)
                if ref.homogeneous:
                    extra[:, rows] += (ta + tb) / fv.nc     # a datum appears at most once in a pair
                else:
                    np.add.at(extra, (ref.ia[ipo], rows), ta)
                    np.add.at(extra, (ref.ib[ipo], rows), tb)
            if calc_sens:
                sens += extra[self.ind_roi, :]
        elif exact:
            for j, t in enumerate(plan['pairs']):
                s0 = sigma0 if np.isscalar(sigma0) else np.asarray(sigma0).ravel()[t['pole']]
                u0 = t['sign'] * cs[t['rows']] * fv.u_homog_at(t['c'], t['p'], s0, 1.0, offset_y=True)
                np.add.at(data, t['rows'], u0)
                if calc_sens:
                    # R ~ 1 / sigma0 of the current electrode: d(R)/d(ln sigma) = -R d(ln sigma0)/d(ln sigma)
                    Rj = R[j] + u0
                    if sigma0_w is None:
                        # sigma0: geometric mean of the model
                        sens[:, t['rows']] -= Rj[None, :] / fv.nc
                    else:
                        dl = (sigma0_w[0][t['pole'], :] @ sp.diags_array(sigma)).toarray()
                        dl = dl / np.asarray(sigma0).ravel()[t['pole']][:, None]
                        sens[:, t['rows']] -= (Rj[:, None] * dl[:, self.ind_roi]).T
        self.u = None
        data = self._units_scaling * data
        if calc_sens:
            sens *= self._units_scaling
        if self.sort_electrodes and self.electrodes_sorted and self.sort_back is not None:
            # c1c2 and p1p2 were sorted by a previous call (on-line method)
            data = data[self.sort_back]
            if calc_sens:
                sens = sens[:, self.sort_back]
        if self.verbose and fv.comm.rank == 0:
            print('End of modelling.')
        if calc_sens:
            return data, sens
        return data

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
        exact_primary = u is None and self._uses_exact_primary()
        if u is None:
            u = self.u
        if self._Qc is not None:
            data = np.concatenate([self._contact_data(ns, u, exact_primary) for ns in range(self.n_c1c2_u)])
            if self.sort_electrodes and self.sort_back is not None:
                data = data[self.sort_back]
            return data
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
                d = self.Q[ns] @ uu
                if exact_primary:
                    R1, R2 = self._reference_terms(ns)
                    d = d + R1 + R2
                data = np.r_[data, d]
        if self.sort_electrodes and self.sort_back is not None:
            data = data[self.sort_back]
        return data

    def _contact_primary(self, col, pts, derivatives=False):
        """Field of the reference models of the current electrodes of column col of u at pts (contact_interp).

        Column col is the injection dipole col (keep_c1c2) or the pole col (sensitivity).  With
        derivatives (poles only), also returns the derivatives with respect to ln sa and ln sb.
        """
        fv = self.fv
        if self.keep_c1c2:
            c, cs = self.c1c2_u[col], self.cs12_u[col]
            P = fv._ref_field(c[:3], self._refs[0], col, cs, pts=pts)
            if len(self._refs) > 1:
                P = P - fv._ref_field(c[3:6], self._refs[1], col, cs, pts=pts)
            return P
        return fv._ref_field(self.c12_u[col], self._refs[0], col, self.cs_u[col], pts=pts, derivatives=derivatives)

    def _contact_data(self, ns, u, exact):
        """Data of injection dipole ns with contact_interp.

        d = Qf1 u - Qf2 u + c1 (P1 - Q1 u0) - c2 (P2 - Q2 u0), with Qf the interpolation with flux continuity,
        Q the linear interpolation, P the field of the reference models of the current electrodes at the
        measurement electrodes and u0 at the cell centres (the terms in P and u0 with exact_primary only).
        The contrast factors c (see Grid25FV.flux_interp) remove the correction of the interpolation of the
        reference field at measurement electrodes on a contact, where it does not apply.
        """
        C = self._Qc[ns]
        if self.keep_c1c2:
            cols = ((1.0, ns),)
        else:
            cols = ((1.0, self.ind_c1_u[ns]), (-1.0, self.ind_c2_u[ns]))
        uu = sum(s * u[:, col] for s, col in cols)
        d = self.Q[ns] @ uu
        if exact:
            p1p2 = self.p1p2[self.ind_c1c2 == ns]
            for sr, Ql, c, pts in ((1.0, C['L1'], C['c1'], p1p2[:, :3]), (-1.0, C['L2'], C['c2'], p1p2[:, 3:6])):
                R = sum(s * (self._contact_primary(col, pts) - Ql @ self.u0[:, col]) for s, col in cols)
                d = d + sr * c * R
        return d

    def _reference_terms(self, ns):
        """Terms of the data depending on the reference fields, for the current electrodes of dipole ns.

        In sensitivity mode (fields computed for each pole), the data of injection dipole ns are
        d = Q (u_c1 - u_c2) + R1 + R2, with R_e = +/-(P_e - Q u0_e), P_e the analytic primary field of
        electrode e at the measurement electrodes and u0_e its reference field (2.5D, exact_primary).
        R_e is proportional to 1/sigma0_e.

        Returns
        -------
        tuple of 2 ndarray
            R1, R2 for the data of dipole ns (rows where ind_c1c2 == ns)
        """
        ind = self.ind_c1c2 == ns
        p1p2 = self.p1p2[ind, :]
        terms = []
        for sign, ip in ((1.0, self.ind_c1_u[ns]), (-1.0, self.ind_c2_u[ns])):
            c = np.tile(self.c12_u[ip, :], (p1p2.shape[0], 1))
            s0 = self._sigma0 if np.isscalar(self._sigma0) else self._sigma0[ip]
            cs = self.cs_u[ip]
            P = self.fv.u_homog_at(c, p1p2[:, :3], s0, cs)
            if p1p2.shape[1] == 6:
                P = P - self.fv.u_homog_at(c, p1p2[:, 3:6], s0, cs)
            terms.append(sign * (P - self.Q[ns] @ self.u0[:, ip]))
        return terms

    def _sens25(self, sigma, M, n_poles):
        """Sensitivity d(data)/d(ln sigma) in 2.5D.

        Parameters
        ----------
        sigma : ndarray
            Conductivity model.
        M : sparse array
            Harmonic average of sigma on the faces.
        n_poles : int
            Number of current electrodes (fields of the measurement electrodes follow in U_k).

        Notes
        -----
        With L_k = -D M G + k^2 sigma (one system per wavenumber), U_k and W_k the fields of the current
        electrodes and of unit sources at the measurement electrodes, and u = 1/2 sum_k g_k U_k:

            d(data)/d(ln sigma) = -1/2 sum_k g_k [ Dm Gf^T (G U_k * m^2 Vf * G W_k) + k^2 sigma Vc U_k W_k ]

        (L_k^-T = Vc L_k^-1 Vc^-1 and D^T = -Vf G Vc^-1).  With exact_primary, the reference fields
        depend on sigma through the conductivity at the current electrodes, which adds
        -R_e d(ln sigma0_e)/d(ln sigma) (see _reference_terms).
        """
        fv = self.fv
        Gf = fv.build_G_faces()
        S = M.diagonal()**2 * fv.volume_faces()
        dm = 1.0 / sigma
        mass = sigma * fv.volume_voxels()
        U = self.U_k
        k, g = fv.solver_A.k, fv.solver_A.g
        GU = np.stack([fv.G @ U[i] for i in range(k.size)])

        n_data = self.c1c2.shape[0]
        sens = np.empty((self.ind_roi.size, n_data))

        # terms from the reference fields (exact primary field at the measurement electrodes)
        R = self._reference_term_matrix()
        contact = self._contact_sens_setup(sigma, n_poles) if self._contact() else None

        for n in range(n_data):
            i1, i2 = self.ind_c1[n], self.ind_c2[n]
            r1, r2 = n_poles + self.ind_p1[n], n_poles + self.ind_p2[n]
            tmp = np.zeros((fv.nc,))
            for i in range(k.size):
                Uk = U[i][:, i1] - U[i][:, i2]
                Wk = U[i][:, r1] - U[i][:, r2]
                GUk = GU[i][:, i1] - GU[i][:, i2]
                GWk = GU[i][:, r1] - GU[i][:, r2]
                tmp += g[i] * (dm * (Gf.T @ (GUk * S * GWk)) + k[i]**2 * mass * Uk * Wk)
            tmp *= -0.5
            if R is not None:
                tmp += self._sigma0_term(n, R, sigma)
            if contact is not None:
                tmp += self._contact_sens_term(n, contact)
            sens[:, n] = self._units_scaling * tmp[self.ind_roi]
        return sens

    def _contact_sens_setup(self, sigma, n_poles):
        """Quantities for the terms of the sensitivity specific to contact_interp (see _contact_sens_term)."""
        fv = self.fv
        exact = self._uses_exact_primary()
        p12 = fv.process_surface_elec(self.p12_u.copy())
        c = {'exact': exact, 'n_poles': n_poles}
        _, c['c_r'], c['cells_r'], c['dW_r'], c['dc_r'] = fv.flux_interp(p12[:, 0], p12[:, 2], sigma,
                                                                          derivatives=True)
        # adjoint fields (unit sources at the measurement electrodes), spatial domain
        c['w'] = self.u[:, n_poles:]
        if not exact:
            if not self.apply_bc:
                # point sources distributed with the weights of flux_interp (see _build_q)
                c12 = fv.process_surface_elec(self.c12_u.copy())
                _, _, c['cells_s'], c['dW_s'], _ = fv.flux_interp(c12[:, 0], c12[:, 2], sigma, derivatives=True)
            return c
        ref = self._refs[0]
        Lr = fv.linear_interp(p12[:, 0], p12[:, 1], p12[:, 2])
        n_r = p12.shape[0]
        # P - Q u0 (n_poles x n_r) and its derivatives with respect to ln sa and ln sb of each pole
        R = np.zeros((n_poles, n_r))
        dRa = np.zeros((n_poles, n_r))
        dRb = np.zeros((n_poles, n_r))
        # derivative of the data with respect to ln sa through the source term q_k (adjoint, A^-T = Vc A^-1 Vc^-1)
        Tq = np.zeros((n_poles, n_r))
        U = self.U_k
        k, g = fv.solver_A.k, fv.solver_A.g
        vc = fv.volume_voxels()
        Gf = fv.build_G_faces()
        for ip in range(n_poles):
            pole = self.c12_u[ip]
            if not np.all(np.isfinite(pole)):
                continue
            cs = self.cs_u[ip]
            # measurement electrodes at the position of the pole (not in the same data): no contribution
            pts = self.p12_u.copy()
            pts[np.all(pts == pole, axis=1)] = np.inf
            P, dPa, dPb = self._contact_primary(ip, pts, derivatives=True)
            u0, du0a, du0b = fv._ref_field(pole, ref, ip, cs, kernel=lambda r: 1. / r, derivatives=True)
            R[ip] = P - Lr @ u0
            dRa[ip] = dPa - Lr @ du0a
            dRb[ip] = dPb - Lr @ du0b
            if not ref.homogeneous:
                dq = fv.contact_source_derivative(pole, ref, ip, cs, k, Gf=Gf)
                for i in range(k.size):
                    Tq[ip] += 0.5 * g[i] * ((vc * dq[i]) @ U[i][:, n_poles:])
        c.update({'R': R, 'dRa': dRa, 'dRb': dRb, 'Tq': Tq})
        return c

    def _contact_sens_term(self, n, c):
        """Terms of d(data_n)/d(ln sigma) specific to contact_interp (unscaled).

        With d = Qf1 u - Qf2 u + c1 R1 - c2 R2 (see _contact_data), the interpolation weights Qf and the
        contrast factors c depend on the conductivity of the cells around the measurement electrodes, and
        R = P - Q u0 and the source terms q_k depend on the conductivity sa and sb of the reference models of
        the current electrodes (cells ia and ib).  Without boundary correction, the point sources depend on
        the conductivity around the current electrodes (reciprocal of the term of the weights Qf).
        """
        fv = self.fv
        n_poles = c['n_poles']
        tmp = np.zeros((fv.nc,))
        poles = ((1.0, self.ind_c1[n]), (-1.0, self.ind_c2[n]))
        rcv = ((1.0, self.ind_p1[n]), (-1.0, self.ind_p2[n]))
        uu = self.u[:, self.ind_c1[n]] - self.u[:, self.ind_c2[n]]
        for sr, r in rcv:
            cells = c['cells_r'][r]
            np.add.at(tmp, cells, sr * (uu[cells] @ c['dW_r'][r]))
            if c['exact']:
                Rr = sum(sp_ * c['R'][ip, r] for sp_, ip in poles)
                np.add.at(tmp, cells, sr * Rr * c['dc_r'][r])
        if 'cells_s' in c:
            ww = c['w'][:, self.ind_p1[n]] - c['w'][:, self.ind_p2[n]]
            for sp_, ip in poles:
                cells = c['cells_s'][ip]
                np.add.at(tmp, cells, sp_ * self.cs_u[ip] * (ww[cells] @ c['dW_s'][ip]))
        if not c['exact']:
            return tmp
        ref = self._refs[0]
        for sp_, ip in poles:
            if ref.ia[ip] < 0:
                continue
            for sr, r in rcv:
                ta = sp_ * sr * (c['Tq'][ip, r] + c['c_r'][r] * c['dRa'][ip, r])
                tb = sp_ * sr * (-c['Tq'][ip, r] + c['c_r'][r] * c['dRb'][ip, r])
                if ref.homogeneous:
                    # sa = sb = geometric mean of the model
                    tmp += (ta + tb) / fv.nc
                else:
                    tmp[ref.ia[ip]] += ta
                    tmp[ref.ib[ip]] += tb
        return tmp

    def _uses_exact_primary(self):
        """True if the data include the analytic primary field (rectilinear grids with boundary correction)."""
        return (isinstance(self.fv, (GridFV, Grid25FV)) and self.exact_primary and self.u0 is not None and
                (self._sigma0 is not None or self._refs is not None))

    def _reference_term_matrix(self):
        """Terms R_e of the data depending on the reference fields (see _reference_terms), n_data x 2.

        Returns None if the data do not include the analytic primary field.
        """
        if not (self.apply_bc and self._uses_exact_primary()) or self._refs is not None:
            return None
        R = np.zeros((self.c1c2.shape[0], 2))
        for ns in range(self.n_c1c2_u):
            ind = np.where(self.ind_c1c2 == ns)[0]
            R[ind, 0], R[ind, 1] = self._reference_terms(ns)
        if isinstance(self.fv, GridFV):
            # source term (3D): the reference potential term A[0, 0] acts on u0 (see _boundary_correction),
            # which contributes W_n[0] u0_e[0] to the data, W_n being the field of unit sources at the
            # measurement electrodes (A^-T = Vc A^-1 Vc^-1)
            n_poles = self.c12_u.shape[0]
            W0 = self.u[0, n_poles + self.ind_p1] - self.u[0, n_poles + self.ind_p2]
            R[:, 0] += W0 * self.u0[0, self.ind_c1]
            R[:, 1] -= W0 * self.u0[0, self.ind_c2]
        return R

    def _sigma0_term(self, n, R, sigma):
        """Contribution to d(data_n)/d(ln sigma) of the conductivity of the reference fields (unscaled).

        R_e is proportional to 1/sigma0_e, sigma0_e being the conductivity interpolated at current electrode
        e (local_sigma0), or the geometric mean of the model.
        """
        tmp = np.zeros((self.fv.nc,))
        for e, ip in ((0, self.ind_c1[n]), (1, self.ind_c2[n])):
            if self._sigma0_w is None:
                # sigma0: geometric mean of the model
                tmp -= R[n, e] / self.fv.nc
            else:
                w = self._sigma0_w[0][[ip], :].toarray().ravel()
                tmp -= R[n, e] * w * sigma / self._sigma0[ip]
        return tmp

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
