#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for finite volume modeling

@author: giroux

Notes
-----
This module assumes a cartesian coordinate system where
x : easting
y : northing
z : elevation

Sign convention: physical.  A current +I injected at the first electrode of a
dipole (source) produces a positive potential around it, and current flows
down into the ground beneath it.  With z positive upward, this current is along
-z below the source, so the azimuthal magnetic field around it is negative
(right-hand rule).  The potential, being a scalar, does not depend on the
orientation of the z axis; only vector components (J, B) do.

References
----------

@book{haber2014computational,
  title={Computational methods in geophysical electromagnetics},
  author={Haber, Eldad},
  year={2014},
  publisher={SIAM}
}

@MastersThesis{lelievre03,
  author       = {Peter George Lelièvre},
  school       = {University of British Columbia},
  title        = {Forward modeling and inversion of geophysical magnetic data},
  year         = {2003}
}

"""
import copy
import sys
import warnings
from threading import Thread

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import bicgstab, spsolve, use_solver, factorized
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.special import kv, k0
from scipy.optimize import minimize

import vtk
from vtk.util.numpy_support import vtk_to_numpy

from discretize import SimplexMesh, TensorMesh

try:
    import pypardiso
    has_pardiso = True
except ImportError:
    has_pardiso = False
except OSError as e:
    print("Warning, importing pypardiso failed:", e)
    has_pardiso = False

try:
    import scikits.umfpack as um
    has_umfpack = True
except ImportError:
    has_umfpack = False

has_mumps = False
try:
    from mpi4py import MPI  # mpi only needed with mumps

    try:
        import mumps
        has_mumps = True
    except ImportError:
        pass

except (ImportError, RuntimeError) as e:
    print("Warning, importing mpi4py failed:", e)
    import types
    MPI = types.SimpleNamespace()
    MPI.COMM_WORLD = types.SimpleNamespace()
    MPI.COMM_WORLD.rank = 0

try:
    import pypastix
    has_pastix = True
except (ImportError, OSError):
    has_pastix = False


# %% Some functions


def is_free_threading():
    if sys.version_info.minor < 13:
        return False
    else:
        return not sys._is_gil_enabled()


def calc_padding(dx, n_cells=15, factor=1.3):
    """Compute padding array to extend a grid.

    Parameters
    ----------
    dx : float
        Cell size at the start of padding.
    n_cells : int, optional
        Number of cells for padding.
    factor : float, optional
        Stretching factor.

    Returns
    -------
    ndarray
        Padding array holding cell sizes.

    Notes
    -----
    Cell size increases following

    factor**n

    where n represent n^th cell.

    """
    return dx * factor ** np.arange(1, n_cells + 1)


def _get_umf_family(A):
    """Get umfpack family string given the sparse matrix dtype.

    This function was taken in scipy.sparse.linalg file _dsolve/linsolve.py"""
    _families = {
        (np.float64, np.int32): "di",
        (np.complex128, np.int32): "zi",
        (np.float64, np.int64): "dl",
        (np.complex128, np.int64): "zl",
    }

    f_type = np.sctypeDict[A.dtype.name]
    i_type = np.sctypeDict[A.indices.dtype.name]

    try:
        family = _families[(f_type, i_type)]

    except KeyError as e:
        msg = (
            "only float64 or complex128 matrices with int32 or int64"
            " indices are supported! (got: matrix: %s, indices: %s)" % (f_type, i_type)
        )
        raise ValueError(msg) from e

    # See gh-8278. Considered converting only if
    # A.shape[0]*A.shape[1] > np.iinfo(np.int32).max,
    # but that didn't always fix the issue.
    family = family[0] + "l"
    A_new = copy.copy(A)
    A_new.indptr = np.asarray(A.indptr, dtype=np.int64)
    A_new.indices = np.asarray(A.indices, dtype=np.int64)

    return family, A_new


def _canonical_csr(A):
    """Return a copy of A in canonical CSR format (sorted indices, duplicates summed)."""
    A = sp.csr_array(A, copy=True)
    A.sum_duplicates()
    A.sort_indices()
    return A


def tetra_coord(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray):
    # Almost the same as Hugues' function,
    # except it does not involve the homogeneous coordinates.
    # v1 = b - a
    # v2 = c - a
    # v3 = d - a
    # mat = np.array((v1,v2,v3)).T
    mat = np.empty((3, 3))
    mat[:, 0] = b - a
    mat[:, 1] = c - a
    mat[:, 2] = d - a
    # mat is 3x3 here
    return np.linalg.inv(mat)


def triangle_area_2D(x1, y1, x2, y2, x3, y3):
    return (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)


def barycentric(a: np.ndarray, b: np.ndarray, c: np.ndarray, p: np.ndarray):
    ab = b - a
    ac = c - a
    m = np.cross(ab, ac)
    x = np.abs(m[0])
    y = np.abs(m[1])
    z = np.abs(m[2])
    if x >= y and x >= z:
        nu = triangle_area_2D(p[1], p[2], b[1], b[2], c[1], c[2])
        nv = triangle_area_2D(p[1], p[2], c[1], c[2], a[1], a[2])
        ood = 1. / m[0]
    elif y >= x and y >= z:
        nu = triangle_area_2D(p[0], p[2], b[0], b[2], c[0], c[2])
        nv = triangle_area_2D(p[0], p[2], c[0], c[2], a[0], a[2])
        ood = 1. / m[1]
    else:
        nu = triangle_area_2D(p[0], p[1], b[0], b[1], c[0], c[1])
        nv = triangle_area_2D(p[0], p[1], c[0], c[1], a[0], a[1])
        ood = 1. / m[2]
    u = nu * ood
    v = nv * ood
    w = 1.0 - u - v
    return np.array([u, v, w])


def inside_triangle_box(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, p: np.ndarray):
    x_min = min((v1[0], v2[0], v3[0])) - 1.e-6
    x_max = max((v1[0], v2[0], v3[0])) + 1.e-6
    y_min = min((v1[1], v2[1], v3[1])) - 1.e-6
    y_max = max((v1[1], v2[1], v3[1])) + 1.e-6
    z_min = min((v1[2], v2[2], v3[2])) - 1.e-6
    z_max = max((v1[2], v2[2], v3[2])) + 1.e-6

    if p[0] < x_min or x_max < p[0] or p[1] < y_min or y_max < p[1] or p[2] < z_min or z_max < p[2]:
        return False
    else:
        return True


def inside_tetrahedron_box(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, v4: np.ndarray, p: np.ndarray):
    x_min = min((v1[0], v2[0], v3[0], v4[0])) - 1.e-6
    x_max = max((v1[0], v2[0], v3[0], v4[0])) + 1.e-6
    y_min = min((v1[1], v2[1], v3[1], v4[1])) - 1.e-6
    y_max = max((v1[1], v2[1], v3[1], v4[1])) + 1.e-6
    z_min = min((v1[2], v2[2], v3[2], v4[2])) - 1.e-6
    z_max = max((v1[2], v2[2], v3[2], v4[2])) + 1.e-6

    if p[0] < x_min or x_max < p[0] or p[1] < y_min or y_max < p[1] or p[2] < z_min or z_max < p[2]:
        return False
    else:
        return True

# %%


class BaseFV:
    """Base class for finite volume modelling.

        Parameters
        ----------
        comm : MPI Communicator, optional
            If None, use MPI_COMM_WORLD
    """

    def __init__(self, comm=None, n_threads=1):
        if comm is None:
            comm = MPI.COMM_WORLD
        self.n_threads = n_threads
        self.comm = comm
        self.dim = 0
        self.verbose = False
        self.solver = bicgstab
        self.atol = 1e-9
        self.rtol = 1e-5
        self.max_it = 1000
        self._want_pardiso = False
        self._want_pastix = False
        self.want_superlu = False
        if has_umfpack:
            # we want UMFPACK by default if available, if not SuperLU will be used
            self._want_umfpack = True
        else:
            self._want_umfpack = False
            self.want_superlu = True
        self._want_mumps = False
        self.solver_A = None
        self.precon = '0'
        self.do_perm = False

    @property
    def A(self):
        if self.solver_A is not None:
            return self.solver_A.A
        else:
            return None

    @A.setter
    def A(self, val):
        if self.solver_A is not None:
            self.solver_A.A = val
        else:
            raise RuntimeError("Solver not defined, cannot set A")

    def set_system(self, A, keep_solver=True, verbose=False):
        """Instantiate the solver for matrix A, or update it if it exists and keep_solver is True."""
        if self.solver_A is None or keep_solver is False:
            self.solver_A = Solver(self.get_solver_params(), A, verbose=verbose)
        else:
            self.A = A

    def set_solver(self, name, atol=1e-9, rtol=1e-5, max_it=1000, precon='0', do_perm=False):
        """Define parameters of solver to be used during forward modelling.

        Parameters
        ----------
        name : `string` or `callable`
            If `string`: name of solver (mumps, pardiso, umfpack, or superlu)
            If `callable`: (iterative solver from scipy.sparse.linalg, e.g. bicgstab)
        atol : float, optional
            Absolute tolerance (atol) for the iterative solver
        rtol : float, optional
            Relative tolerance (rtol) for the iterative solver
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
        if callable(name):
            self.solver = name
            self.atol = atol
            self.rtol = rtol
            self.max_it = max_it
            self.want_pardiso = False
            self.want_pastix = False
            self.want_superlu = False
            self.want_umfpack = False
            self.want_mumps = False
        elif "superlu" in name:
            self.want_pardiso = False
            self.want_pastix = False
            self.want_superlu = True
            self.want_umfpack = False
            self.want_mumps = False
        elif "pardiso" in name:
            self.want_superlu = False
            self.want_pardiso = True
            self.want_pastix = False
            self.want_umfpack = False
            self.want_mumps = False
        elif "pastix" in name:
            self.want_superlu = False
            self.want_pastix = True
            self.want_pardiso = False
            self.want_umfpack = False
            self.want_mumps = False
        elif "umfpack" in name:
            self.want_superlu = False
            self.want_umfpack = True
            self.want_pardiso = False
            self.want_pastix = False
            self.want_mumps = False
        elif "mumps" in name:
            self.want_superlu = False
            self.want_pardiso = False
            self.want_pastix = False
            self.want_umfpack = False
            self.want_mumps = True
        else:
            raise RuntimeError("Solver " + name + " not implemented")

        # the requested solver might not be available, in which case we turned to the default but name was not changed
        if self.want_superlu:
            name = 'superlu'

        self.solver_A = Solver((name, atol, rtol, max_it, precon, do_perm), verbose=(self.verbose>1),
                               comm=self.comm, n_threads=self.n_threads)
        self.precon = precon
        self.do_perm = do_perm

    def set_solver_print_level(self, level):
        self.solver_A.set_mumps_print_level(level)

    def get_solver_params(self):
        """Return parameters needed to instantiate Solver."""
        if self.want_mumps:
            return ("mumps",)
        elif self.want_pardiso:
            return ("pardiso",)
        elif self.want_pastix:
            return ("pastix",)
        elif self.want_superlu:
            return ("superlu",)
        elif self.want_umfpack:
            return ("umfpack",)
        else:
            return self.solver, self.atol, self.rtol, self.max_it, self.precon, self.do_perm

    @property
    def want_pardiso(self):
        """Using solver pardiso (if available)."""
        return self._want_pardiso

    @want_pardiso.setter
    def want_pardiso(self, val):
        if val is True and has_pardiso is False:
            warnings.warn("Pardiso not available, default solver used.", RuntimeWarning, stacklevel=2)
            self._want_pardiso = False
            self.want_superlu = True
        else:
            self._want_pardiso = val

    @property
    def want_pastix(self):
        """Using solver pastix (if available)."""
        return self._want_pastix

    @want_pastix.setter
    def want_pastix(self, val):
        if val is True and has_pastix is False:
            warnings.warn("Pastix not available, default solver used.", RuntimeWarning, stacklevel=2)
            self._want_pastix = False
            self.want_superlu = True
        else:
            self._want_pastix = val

    @property
    def want_umfpack(self):
        """Using solver umfpack (if available)."""
        return self._want_umfpack

    @want_umfpack.setter
    def want_umfpack(self, val):
        if val is True and has_umfpack is False:
            warnings.warn("UMFPACK not available, default solver used.", RuntimeWarning, stacklevel=2)
            self._want_umfpack = False
            self.want_superlu = True
        else:
            self._want_umfpack = val

    @property
    def want_mumps(self):
        """Using solver mumps (if available)."""
        return self._want_mumps

    @want_mumps.setter
    def want_mumps(self, val):
        if val is True and has_mumps is False:
            warnings.warn("MUMPS not available, default solver used.", RuntimeWarning, stacklevel=2)
            self._want_mumps = False
            self.want_superlu = True
        else:
            self._want_mumps = val


# %% GridFV


class GridFV(BaseFV):
    """Class to manage rectilinear grids for finite volume modelling.

    Parameters
    ----------
    param: tuple
        parameters to instantiate the finite volume grid
        x : array_like
            Node coordinates along x
        y : array_like
            Node coordinates along y
        z : array_like
            Node coordinates along z
    comm : MPI Communicator, optional
        If None, use MPI_COMM_WORLD

    Notes
    -----
    Voxels are sorted column major, i.e. x is the fast axis
    (choice dictated by convention used by VTK).

    """

    def __init__(self, param, comm=None, n_threads=1):
        BaseFV.__init__(self, comm, n_threads)
        x, y, z = param
        self.x = x
        self.y = y
        self.z = z
        self.dim = 3
        self.surface_images = True   # see compute_u_homog

    def ind(self, i, j, k, component=None):
        """
        Returns the index of a voxel.

        Parameters
        ----------
        i : int or array_like of int
            indice(s) along x
        j : int or array_like of int
            indice(s) along y
        k : int or array_like of int
            indice(s) along z
        component : str, optional
            component considered, possible values are:
                - None: the method returns voxel index
                - 'ex': component x defined on an edge
                - 'ey': component y defined on an edge
                - 'ez': component z defined on an edge
                - 'fx': component x defined on a face
                - 'fy': component y defined on a face
                - 'fz': component z defined on a face
        Returns
        -------
        int or array of int
        """
        if component is None:
            nx = self.nx
            ny = self.ny
            nz = self.nz
        elif component == "ex":
            nx = self.nx
            ny = self.ny - 1
            nz = self.nz - 1
        elif component == "ey":
            nx = self.nx - 1
            ny = self.ny
            nz = self.nz - 1
        elif component == "ez":
            nx = self.nx - 1
            ny = self.ny - 1
            nz = self.nz
        elif component == "fx":
            nx = self.nx - 1
            ny = self.ny
            nz = self.nz
        elif component == "fy":
            nx = self.nx
            ny = self.ny - 1
            nz = self.nz
        elif component == "fz":
            nx = self.nx
            ny = self.ny
            nz = self.nz - 1
        else:
            raise ValueError("unknown component")

        if np.size(i) > 1:
            i = np.array(i)
            i = i.flatten()
        if np.size(j) > 1:
            j = np.array(j)
            j = j.flatten()
        if np.size(k) > 1:
            k = np.array(k)
            k = k.flatten()
        if np.any(i < 0) or np.any(i >= nx) or np.any(j < 0) or np.any(j >= ny) or np.any(k < 0) or np.any(k >= nz):
            raise IndexError("Index outside grid")
        if np.size(i) > 1 or np.size(j) > 1 or np.size(k) > 1:
            ii = np.kron(np.ones((np.size(j) * np.size(k),), dtype=np.int64), i)
            jj = np.kron(np.kron(np.ones((np.size(k),), dtype=np.int64), j), np.ones((np.size(i),), dtype=np.int64))
            kk = np.kron(k, np.ones((np.size(i) * np.size(j),), dtype=np.int64))
            return np.sort((kk * ny + jj) * nx + ii)
        else:
            return (k * ny + j) * nx + i

    def revind(self, ind, component=None):
        """
        Returns indices i, j, k of a voxel or of a component on a face
        or an edge.

        Parameters
        ----------
        ind : int
            voxel index
        component : str, optional
            component considered, possible values are:
                - None: voxel index
                - 'ex': component x defined on an edge
                - 'ey': component y defined on an edge
                - 'ez': component z defined on an edge
                - 'fx': component x defined on a face
                - 'fy': component y defined on a face
                - 'fz': component z defined on a face

        Returns
        -------
        tuple of int

        """
        if component is None:
            nx = self.nx
            ny = self.ny
        elif component == "ex":
            nx = self.nx
            ny = self.ny - 1
        elif component == "ey":
            nx = self.nx - 1
            ny = self.ny
        elif component == "ez":
            nx = self.nx - 1
            ny = self.ny - 1
        elif component == "fx":
            nx = self.nx - 1
            ny = self.ny
        elif component == "fy":
            nx = self.nx
            ny = self.ny - 1
        elif component == "fz":
            nx = self.nx
            ny = self.ny
        else:
            raise ValueError("unknown component")

        k = int(ind / (nx * ny))
        j = int((ind - k * nx * ny) / nx)
        i = ind - (k * ny + j) * nx
        return i, j, k

    def __update_n(self):
        if "_x" in self.__dict__ and "_y" in self.__dict__ and "_z" in self.__dict__:
            self.nx = len(self._x) - 1  # number of voxels along x
            self.ny = len(self._y) - 1  # number of voxels along y
            self.nz = len(self._z) - 1  # number of voxels along z
            self.nc = self.nx * self.ny * self.nz  # number of voxels
            self.nfx = (self.nx - 1) * self.ny * self.nz  # number of faces with normal vector along x
            self.nfy = self.nx * (self.ny - 1) * self.nz  # number of faces with normal vector along y
            self.nfz = self.nx * self.ny * (self.nz - 1)  # number of faces with normal vector along z
            self.nf = self.nfx + self.nfy + self.nfz  # total number of faces
            self.nex = self.nx * (self.ny - 1) * (self.nz - 1)  # number of edges with vector along x
            self.ney = (self.nx - 1) * self.ny * (self.nz - 1)  # number of edges with vector along y
            self.nez = (self.nx - 1) * (self.ny - 1) * self.nz  # number of edges with vector along z
            self.ne = self.nex + self.ney + self.nez  # total number of edges

            # Divergence & gradient matrices are precomputed because essentially always used
            self.D = self.build_D()
            self.G = self.build_G()

    @property
    def x(self):
        """Node coordinates along x."""
        return self._x

    @x.setter
    def x(self, val):
        tmp = np.sort(np.array(val, dtype=np.float64))
        if tmp.ndim != 1:
            raise ValueError("1D array needed")
        if len(tmp) < 2:
            raise ValueError("2 nodes or more needed")

        self._x = tmp
        self.hx = np.diff(tmp)
        self.xc = (tmp[1:] + tmp[:-1]) / 2
        self.dx = np.diff(self.xc)
        self.__update_n()

    @property
    def y(self):
        """Node coordinates along y."""
        return self._y

    @y.setter
    def y(self, val):
        tmp = np.sort(np.array(val, dtype=np.float64))
        if tmp.ndim != 1:
            raise ValueError("1D array needed")
        if len(tmp) < 2:
            raise ValueError("2 nodes or more needed")

        self._y = tmp
        self.hy = np.diff(tmp)
        self.yc = (tmp[1:] + tmp[:-1]) / 2
        self.dy = np.diff(self.yc)
        self.__update_n()

    @property
    def z(self):
        """Node coordinates along z."""
        return self._z

    @z.setter
    def z(self, val):
        tmp = np.sort(np.array(val, dtype=np.float64))
        if tmp.ndim != 1:
            raise ValueError("1D array needed")
        if len(tmp) < 2:
            raise ValueError("2 nodes or more needed")

        self._z = tmp
        self.hz = np.diff(tmp)
        self.zc = (tmp[1:] + tmp[:-1]) / 2
        self.dz = np.diff(self.zc)
        self.__update_n()

    def is_inside(self, x, y, z):
        """Check if point is inside grid.

        Parameters
        ----------
        x : float
            coordinate along X.
        y : float
            coordinate along Y.
        z : float
            coordinate along Z.

        Returns
        -------
        True if point is inside grid

        """
        return self.x[0] <= x <= self.x[-1] and self.y[0] <= y <= self.y[-1] and self.z[0] <= z <= self.z[-1]

    def volume_voxels(self):
        """Returns the volume of the voxels of the grid.

        Returns
        -------
        ndarray
            Volumes of voxels.

        """
        return np.kron(self.hz, np.kron(self.hy, self.hx))

    def volume_faces(self):
        """Returns the dual volume associated with the (interior) faces of the grid.

        Returns
        -------
        ndarray
            face area times distance between the centres of the two adjacent voxels,
            ordered as the faces (x, then y, then z).

        Notes
        -----
        With Vc = volume_voxels() and Vf = volume_faces(), D = -Vc^-1 G^T Vf, so that
        diag(Vc) @ A (A = -D M G) and diag(Vf) @ (C_e C_f - G D) are symmetric.
        This is used to apply A^-T with the factorization of A in the adjoint computations.
        """
        vx = np.kron(self.hz, np.kron(self.hy, self.dx))
        vy = np.kron(self.hz, np.kron(self.dy, self.hx))
        vz = np.kron(self.dz, np.kron(self.hy, self.hx))
        return np.r_[vx, vy, vz]

    def centre_voxels(self):
        """Returns coordinates at the centres of voxels.

        Returns
        -------
        tuple of 3 ndarray
            coordinates X, Y, Z

        """
        x = np.kron(np.ones((self.ny * self.nz,)), self.xc)
        y = np.kron(np.kron(np.ones((self.nz,)), self.yc), np.ones((self.nx,)))
        z = np.kron(self.zc, np.ones((self.nx * self.ny,)))
        return x, y, z

    def distance_weighting(self, xo, beta):
        """Calculate distance weighting matrix.

        Parameters
        ----------
        xo : array_like
            coordinates of observation points.
        beta : float
            Détermine weighting intensity.

        Returns
        -------
        ndarray (nc,)
            Weights.

        Notes
        -----
        Eq 19 of Li & Oldenburg 2000 Geophysics, vol 65, no 2, pp. 540-552

        """
        dV = self.volume_voxels()
        xc, yc, zc = self.centre_voxels()
        R0 = 0.5 * dV.min() ** 0.33333333333333
        Q = np.zeros((self.nc,))
        for i in np.arange(xo.shape[0]):
            R = np.sqrt((xc - xo[i, 0]) * (xc - xo[i, 0]) +
                        (yc - xo[i, 1]) * (yc - xo[i, 1]) +
                        (zc - xo[i, 2]) * (zc - xo[i, 2]))
            R = (R + R0) ** 3
            R = (dV / R) ** 2
            Q += R
        return Q ** (0.25 * beta)

    def linear_interp(self, x, y, z, component=None):
        """Calculate interpolation matrix.

        Matrix that allows interpolating a variable defined at the voxel
        centre, on a face, or on an edge, at arbitrary coordinates.

        Parameters
        ----------
        x : float or array of float
            coordinate along X.
        y : float or array of float
            coordinate along Y.
        z : float or array of float
            coordinate along Z.
        component : str, optional
            component considered, possible values are:
                - None: voxel index
                - 'ex': component x defined on une edge
                - 'ey': component y defined on une edge
                - 'ez': component z defined on une edge
                - 'fx': component x defined on une face
                - 'fy': component y defined on une face
                - 'fz': component z defined on une face

        Returns
        -------
        csr_matrix, npts x nc

        Notes
        -----
        - If any coordinate is at infinity, the corresponding matrix element
        is set to zero.
        """
        if component is None:
            nx = self.nx
            ny = self.ny
            nz = self.nz
            xc = self.xc
            yc = self.yc
            zc = self.zc
        elif component == "ex":
            nx = self.nx
            ny = self.ny - 1
            nz = self.nz - 1
            xc = self.xc
            yc = self.y[1:-1]
            zc = self.z[1:-1]
        elif component == "ey":
            nx = self.nx - 1
            ny = self.ny
            nz = self.nz - 1
            xc = self.x[1:-1]
            yc = self.yc
            zc = self.z[1:-1]
        elif component == "ez":
            nx = self.nx - 1
            ny = self.ny - 1
            nz = self.nz
            xc = self.x[1:-1]
            yc = self.y[1:-1]
            zc = self.zc
        elif component == "fx":
            nx = self.nx - 1
            ny = self.ny
            nz = self.nz
            xc = self.x[1:-1]
            yc = self.yc
            zc = self.zc
        elif component == "fy":
            nx = self.nx
            ny = self.ny - 1
            nz = self.nz
            xc = self.xc
            yc = self.y[1:-1]
            zc = self.zc
        elif component == "fz":
            nx = self.nx
            ny = self.ny
            nz = self.nz - 1
            xc = self.xc
            yc = self.yc
            zc = self.z[1:-1]
        else:
            raise ValueError("unknown component")

        nc = nx * ny * nz

        dx = [0.0, 0.0]
        dy = [0.0, 0.0]
        dz = [0.0, 0.0]
        ix = [0, 0]
        iy = [0, 0]
        iz = [0, 0]

        if np.isscalar(x):
            x = np.array([x])
            y = np.array([y])
            z = np.array([z])

        Q = sp.lil_array((len(x), nc))
        for i in range(len(x)):

            if np.any(np.array([x[i], y[i], z[i]]) == np.inf):
                # check if point at np.inf -> set Q to 0
                Q[i, 0] = 0.0
                continue

            im = np.argmin(np.abs(x[i] - xc))
            if x[i] <= xc[im] and im > 0:
                ix[0] = im - 1
                ix[1] = im
            else:
                ix[0] = im
                ix[1] = im + 1
            dx[0] = x[i] - xc[ix[0]]
            dx[1] = xc[ix[1]] - x[i]

            im = np.argmin(np.abs(y[i] - yc))
            if y[i] <= yc[im] and im > 0:
                iy[0] = im - 1
                iy[1] = im
            else:
                iy[0] = im
                iy[1] = im + 1
            dy[0] = y[i] - yc[iy[0]]
            dy[1] = yc[iy[1]] - y[i]

            im = np.argmin(np.abs(z[i] - zc))
            if z[i] <= zc[im] and im > 0:
                iz[0] = im - 1
                iz[1] = im
            else:
                iz[0] = im
                iz[1] = im + 1
            dz[0] = z[i] - zc[iz[0]]
            dz[1] = zc[iz[1]] - z[i]

            Dx = xc[ix[1]] - xc[ix[0]]
            Dy = yc[iy[1]] - yc[iy[0]]
            Dz = zc[iz[1]] - zc[iz[0]]

            Q[i, self.ind(ix[0], iy[0], iz[0], component)] = (1 - dx[0] / Dx) * (1 - dy[0] / Dy) * (1 - dz[0] / Dz)
            Q[i, self.ind(ix[1], iy[0], iz[0], component)] = (1 - dx[1] / Dx) * (1 - dy[0] / Dy) * (1 - dz[0] / Dz)
            Q[i, self.ind(ix[0], iy[1], iz[0], component)] = (1 - dx[0] / Dx) * (1 - dy[1] / Dy) * (1 - dz[0] / Dz)
            Q[i, self.ind(ix[1], iy[1], iz[0], component)] = (1 - dx[1] / Dx) * (1 - dy[1] / Dy) * (1 - dz[0] / Dz)
            Q[i, self.ind(ix[0], iy[0], iz[1], component)] = (1 - dx[0] / Dx) * (1 - dy[0] / Dy) * (1 - dz[1] / Dz)
            Q[i, self.ind(ix[1], iy[0], iz[1], component)] = (1 - dx[1] / Dx) * (1 - dy[0] / Dy) * (1 - dz[1] / Dz)
            Q[i, self.ind(ix[0], iy[1], iz[1], component)] = (1 - dx[0] / Dx) * (1 - dy[1] / Dy) * (1 - dz[1] / Dz)
            Q[i, self.ind(ix[1], iy[1], iz[1], component)] = (1 - dx[1] / Dx) * (1 - dy[1] / Dy) * (1 - dz[1] / Dz)

        return Q.tocsr()

    def build_A(self, sigma):
        """Build LHS matrix for DC resistivity forward modeling.

        Parameters
        ----------
        sigma : array_like
            conductivity model

        Returns
        -------
        tuple of 2 matrices:
            - A : LHS matrix
            - M : matrix of harmonic average of sigma
        """
        # Build LHS matrix
        if np.isscalar(sigma) is True:
            M = sigma * sp.eye_array(self.nf, self.nf)
        else:
            M = self.build_M(sigma)
        A = -self.D @ M @ self.G
        # I, J, V = sp.find(A[0, :])
        # for jj in J:
        #     A[0, jj] = 0.0
        # A[0, 0] = 1.0/(self.fv.hx[0] * self.fv.hy[0] * self.fv.hz[0])
        A[0, 0] += 1.0 / self.volume_voxels()[0]

        return A, M


    def build_D(self):
        """Construction of divergence matrix.

        Returns
        -------
        csr_matrix
            Divergence matrix

        """

        # Dx

        M = self.nx
        N = self.nx - 1
        i = np.hstack((np.arange(N), 1 + np.arange(N)))
        j = np.hstack((np.arange(N), np.arange(N)))
        nval = i.size
        ii = np.zeros((self.ny * self.nz * nval,), dtype=np.int64)
        jj = np.zeros((self.ny * self.nz * nval,), dtype=np.int64)
        for n in np.arange(self.ny * self.nz):
            ii[(n * nval) : ((n + 1) * nval)] = i + n * M
            jj[(n * nval) : ((n + 1) * nval)] = j + n * N
        s = np.tile(np.hstack((1.0 / self.hx[:-1], -1.0 / self.hx[1:])), (self.ny * self.nz,))
        Dx = sp.coo_array((s, (ii, jj)))

        # Dy

        M = self.nx * self.ny
        N = self.nx * (self.ny - 1)
        i = np.hstack((np.arange(N), self.nx + np.arange(N)))
        j = np.hstack((np.arange(N), np.arange(N)))
        nval = i.size
        ii = np.zeros((self.nz * nval,), dtype=np.int64)
        jj = np.zeros((self.nz * nval,), dtype=np.int64)
        for n in np.arange(self.nz):
            ii[(n * nval) : ((n + 1) * nval)] = i + n * M
            jj[(n * nval) : ((n + 1) * nval)] = j + n * N
        s = np.hstack(
            (np.kron(1.0 / self.hy[:-1], np.ones((self.nx,))), np.kron(-1.0 / self.hy[1:], np.ones((self.nx,))))
        )
        s = np.tile(s, (self.nz,))
        Dy = sp.coo_array((s, (ii, jj)))

        # Dz

        N = self.nx * self.ny * (self.nz - 1)
        i = np.hstack((np.arange(N), self.nx * self.ny + np.arange(N)))
        j = np.hstack((np.arange(N), np.arange(N)))
        s = np.hstack(
            (
                np.kron(1.0 / self.hz[:-1], np.ones((self.nx * self.ny,))),
                np.kron(-1.0 / self.hz[1:], np.ones((self.nx * self.ny,))),
            )
        )
        Dz = sp.coo_array((s, (i, j)))

        # assemblage

        return sp.hstack((Dx, Dy, Dz)).tocsr()

    def build_M(self, v, harmon=True):
        """Calculate harmonic or arithmetic average of a variable.

        Parameters
        ----------
        v : ndarray or tuple of 3 ndarray
            Variable defined at the centre of voxels
            if v is a tuple, medium is anisotropic, with 3 arrays
            corresponding to terms on the diagonal of the tensor
            ie
                |v_xx    0    0|
            v = |   0 v_yy    0|
                |   0    0 v_zz|

        harmon : bool, optional
            if True, harmonic average is computed, otherwise the
            arithmetic mean is computed.

        Returns
        -------
        csr_matrix
            diagonal matrix containing the average on the faces of the voxels

        """
        if type(v) is tuple:
            vx, vy, vz = v
        else:
            vx = v
            vy = v
            vz = v

        # Mx

        d = np.kron(np.ones((self.ny * self.nz,)), 2 * self.dx)
        hi = np.kron(np.ones((self.ny * self.nz,)), self.hx[1:])
        him1 = np.kron(np.ones((self.ny * self.nz,)), self.hx[:-1])
        tmp = np.ones((self.nx,), dtype=bool)
        tmp[0] = 0
        ind1 = np.kron(np.ones((self.ny * self.nz,), dtype=bool), tmp)
        tmp[0] = 1
        tmp[-1] = 0
        ind2 = np.kron(np.ones((self.ny * self.nz,), dtype=bool), tmp)
        if harmon:
            Mx = d / (hi / vx[ind1] + him1 / vx[ind2])
        else:
            Mx = (hi * vx[ind1] + him1 * vx[ind2]) / d

        # My

        d = np.kron(np.ones((self.nz,)), np.kron(2 * self.dy, np.ones((self.nx,))))
        hi = np.kron(np.ones((self.nz,)), np.kron(self.hy[1:], np.ones((self.nx,))))
        him1 = np.kron(np.ones((self.nz,)), np.kron(self.hy[:-1], np.ones((self.nx,))))

        tmp = np.ones((self.ny,), dtype=bool)
        tmp[0] = 0
        ind1 = np.kron(np.ones((self.nz,), dtype=bool), np.kron(tmp, np.ones((self.nx,), dtype=bool)))
        tmp[0] = 1
        tmp[-1] = 0
        ind2 = np.kron(np.ones((self.nz,), dtype=bool), np.kron(tmp, np.ones((self.nx,), dtype=bool)))
        if harmon:
            My = d / (hi / vy[ind1] + him1 / vy[ind2])
        else:
            My = (hi * vy[ind1] + him1 * vy[ind2]) / d

        # Mz

        d = np.kron(2 * self.dz, np.ones((self.nx * self.ny,)))
        hi = np.kron(self.hz[1:], np.ones((self.nx * self.ny,)))
        him1 = np.kron(self.hz[:-1], np.ones((self.nx * self.ny,)))
        ind1 = np.hstack(
            (np.zeros((self.nx * self.ny,), dtype=bool), np.ones((self.nx * self.ny * (self.nz - 1),), dtype=bool))
        )
        ind2 = np.hstack(
            (np.ones((self.nx * self.ny * (self.nz - 1),), dtype=bool), np.zeros((self.nx * self.ny,), dtype=bool))
        )
        if harmon:
            Mz = d / (hi / vz[ind1] + him1 / vz[ind2])
        else:
            Mz = (hi * vz[ind1] + him1 * vz[ind2]) / d

        # assemblage

        return sp.coo_array((np.hstack((Mx, My, Mz)), (np.arange(self.nf), np.arange(self.nf)))).tocsr()

    def build_G(self, v=None):
        """Construction of gradient matrix.

        Parameters
        ----------
        v : ndarray, optional
            if None, gradient for a scalar defined at the voxel centre,
            otherwise, gradient of vector v defined on the edges.
            (dVx/dx, dVy/dy, dVz/dz).

        Returns
        -------
        csr_matrix
            Gradient matrix
        """
        if v is None:
            # centers to faces
            return self._gradient_centre()
        else:
            return self._gradient_faces(v)

    def _gradient_centre(self):

        # Gx

        M = self.nx - 1
        N = self.nx
        i = np.hstack((np.arange(M), np.arange(M)))
        j = np.hstack((np.arange(M), np.arange(1, N)))
        nval = i.size
        ii = np.zeros((self.ny * self.nz * nval,), dtype=np.int64)
        jj = np.zeros((self.ny * self.nz * nval,), dtype=np.int64)
        for n in np.arange(self.ny * self.nz):
            ii[(n * nval) : ((n + 1) * nval)] = i + n * M
            jj[(n * nval) : ((n + 1) * nval)] = j + n * N
        s = np.tile(np.hstack((-1.0 / self.dx, 1.0 / self.dx)), (self.ny * self.nz,))
        Gx = sp.coo_array((s, (ii, jj)))

        # Gy

        M = self.nx * (self.ny - 1)
        N = self.nx * self.ny
        i = np.hstack((np.arange(M), np.arange(M)))
        j = np.hstack((np.arange(M), self.nx + np.arange(M)))
        nval = i.size
        ii = np.zeros((self.nz * nval,), dtype=np.int64)
        jj = np.zeros((self.nz * nval,), dtype=np.int64)
        for n in np.arange(self.nz):
            ii[(n * nval) : ((n + 1) * nval)] = i + n * M
            jj[(n * nval) : ((n + 1) * nval)] = j + n * N
        s = np.hstack((np.kron(-1.0 / self.dy, np.ones((self.nx,))), np.kron(1.0 / self.dy, np.ones((self.nx,)))))
        s = np.tile(s, (self.nz,))
        Gy = sp.coo_array((s, (ii, jj)))

        # Gz

        i = np.hstack((np.arange(self.nfz), np.arange(self.nfz)))
        j = np.hstack((np.arange(self.nfz), self.nx * self.ny + np.arange(self.nfz)))
        s = np.hstack(
            (
                np.kron(-1.0 / self.dz, np.ones((self.nx * self.ny,))),
                np.kron(1.0 / self.dz, np.ones((self.nx * self.ny,))),
            )
        )
        Gz = sp.coo_array((s, (i, j)))

        # assemblage
        return sp.vstack((Gx, Gy, Gz), format="csr")

    def _gradient_faces(self, v):

        v_x = sp.diags_array(v[: self.nfx])
        v_y = sp.diags_array(v[self.nfx : (self.nfx + self.nfy)])
        v_z = sp.diags_array(v[(self.nfx + self.nfy) :])

        # Gx

        dVf = 0.5 * (self.hx[1:] + self.hx[:-1])

        M = self.nx - 1
        N = self.nx
        i = np.hstack((np.arange(M), np.arange(M)))
        j = np.hstack((np.arange(M), np.arange(1, N)))
        nval = i.size
        ii = np.zeros((self.ny * self.nz * nval,), dtype=np.int64)
        jj = np.zeros((self.ny * self.nz * nval,), dtype=np.int64)
        for n in np.arange(self.ny * self.nz):
            ii[(n * nval) : ((n + 1) * nval)] = i + n * M
            jj[(n * nval) : ((n + 1) * nval)] = j + n * N
        s = np.tile(np.hstack((0.5 * self.hx[:-1] / dVf, 0.5 * self.hx[1:] / dVf)), (self.ny * self.nz,))
        Gx = v_x @ sp.coo_array((s, (ii, jj)))

        # Gy

        dVf = 0.5 * (self.hy[1:] + self.hy[:-1])

        M = self.nx * (self.ny - 1)
        N = self.nx * self.ny
        i = np.hstack((np.arange(M), np.arange(M)))
        j = np.hstack((np.arange(M), self.nx + np.arange(M)))
        nval = i.size
        ii = np.zeros((self.nz * nval,), dtype=np.int64)
        jj = np.zeros((self.nz * nval,), dtype=np.int64)
        for n in np.arange(self.nz):
            ii[(n * nval) : ((n + 1) * nval)] = i + n * M
            jj[(n * nval) : ((n + 1) * nval)] = j + n * N
        s = np.hstack(
            (
                np.kron(0.5 * self.hy[:-1] / dVf, np.ones((self.nx,))),
                np.kron(0.5 * self.hy[1:] / dVf, np.ones((self.nx,))),
            )
        )
        s = np.tile(s, (self.nz,))
        Gy = v_y @ sp.coo_array((s, (ii, jj)))

        # Gz

        dVf = 0.5 * (self.hz[1:] + self.hz[:-1])

        i = np.hstack((np.arange(self.nfz), np.arange(self.nfz)))
        j = np.hstack((np.arange(self.nfz), self.nx * self.ny + np.arange(self.nfz)))
        s = np.hstack(
            (
                np.kron(0.5 * self.hz[:-1] / dVf, np.ones((self.nx * self.ny,))),
                np.kron(0.5 * self.hz[1:] / dVf, np.ones((self.nx * self.ny,))),
            )
        )
        Gz = v_z @ sp.coo_array((s, (i, j)))

        # assemblage
        return sp.vstack((Gx, Gy, Gz), format="csr")

    def build_G_faces(self):
        """Construction of gradient matrix to use with vector defined on edges.

        Returns
        -------
        csr_matrix
            Gradient matrix
        """
        # Gx

        dVf = 0.5 * (self.hx[1:] + self.hx[:-1])

        M = self.nx - 1
        N = self.nx
        i = np.hstack((np.arange(M), np.arange(M)))
        j = np.hstack((np.arange(M), np.arange(1, N)))
        nval = i.size
        ii = np.zeros((self.ny * self.nz * nval,), dtype=np.int64)
        jj = np.zeros((self.ny * self.nz * nval,), dtype=np.int64)
        for n in np.arange(self.ny * self.nz):
            ii[(n * nval) : ((n + 1) * nval)] = i + n * M
            jj[(n * nval) : ((n + 1) * nval)] = j + n * N
        s = np.tile(np.hstack((0.5 * self.hx[:-1] / dVf, 0.5 * self.hx[1:] / dVf)), (self.ny * self.nz,))
        Gx = sp.coo_array((s, (ii, jj)))

        # Gy

        dVf = 0.5 * (self.hy[1:] + self.hy[:-1])

        M = self.nx * (self.ny - 1)
        N = self.nx * self.ny
        i = np.hstack((np.arange(M), np.arange(M)))
        j = np.hstack((np.arange(M), self.nx + np.arange(M)))
        nval = i.size
        ii = np.zeros((self.nz * nval,), dtype=np.int64)
        jj = np.zeros((self.nz * nval,), dtype=np.int64)
        for n in np.arange(self.nz):
            ii[(n * nval) : ((n + 1) * nval)] = i + n * M
            jj[(n * nval) : ((n + 1) * nval)] = j + n * N
        s = np.hstack(
            (
                np.kron(0.5 * self.hy[:-1] / dVf, np.ones((self.nx,))),
                np.kron(0.5 * self.hy[1:] / dVf, np.ones((self.nx,))),
            )
        )
        s = np.tile(s, (self.nz,))
        Gy = sp.coo_array((s, (ii, jj)))

        # Gz

        dVf = 0.5 * (self.hz[1:] + self.hz[:-1])

        i = np.hstack((np.arange(self.nfz), np.arange(self.nfz)))
        j = np.hstack((np.arange(self.nfz), self.nx * self.ny + np.arange(self.nfz)))
        s = np.hstack(
            (
                np.kron(0.5 * self.hz[:-1] / dVf, np.ones((self.nx * self.ny,))),
                np.kron(0.5 * self.hz[1:] / dVf, np.ones((self.nx * self.ny,))),
            )
        )
        Gz = sp.coo_array((s, (i, j)))

        return sp.vstack((Gx, Gy, Gz), format="csr")

    def build_C(self, to_faces=True):
        """Construction of curl matrix.

        Parameters
        ----------
        to_faces : bool, optional
            si True, evaluate curl on faces, for a variable defined on edges,
            otherwise evaluate curl on edges, for a variable defined on faces.

        Returns
        -------
        csr_matrix
            Curl matrix

        """
        if to_faces:
            return self._curl_faces()
        else:
            return self._curl_edges()

    def _curl_faces(self):

        # dHz / dy
        M = (self.nx - 1) * self.ny
        N = (self.nx - 1) * (self.ny - 1)
        i = np.hstack((np.arange(N), (self.nx - 1) + np.arange(N)))
        j = np.hstack((np.arange(N), np.arange(N)))
        nval = i.size
        ii = np.zeros((self.nz * nval,), dtype=np.int64)
        jj = np.zeros((self.nz * nval,), dtype=np.int64)
        for n in np.arange(self.nz):
            ii[(n * nval) : ((n + 1) * nval)] = i + n * M
            jj[(n * nval) : ((n + 1) * nval)] = j + n * N
        s = np.hstack(
            (np.kron(1.0 / self.hy[:-1], np.ones((self.nx - 1,))), np.kron(-1.0 / self.hy[1:], np.ones((self.nx - 1,))))
        )
        s = np.tile(s, (self.nz,))
        Dzy = sp.coo_array((s, (ii, jj)))

        # dHy / dz
        N = (self.nx - 1) * self.ny * (self.nz - 1)
        i = np.hstack((np.arange(N), (self.nx - 1) * self.ny + np.arange(N)))
        j = np.hstack((np.arange(N), np.arange(N)))
        s = np.hstack(
            (
                np.kron(1.0 / self.hz[:-1], np.ones(((self.nx - 1) * self.ny,))),
                np.kron(-1.0 / self.hz[1:], np.ones(((self.nx - 1) * self.ny,))),
            )
        )
        Dyz = sp.coo_array((s, (i, j)))

        # dHx / dz
        N = self.nx * (self.ny - 1) * (self.nz - 1)
        i = np.hstack((np.arange(N), self.nx * (self.ny - 1) + np.arange(N)))
        j = np.hstack((np.arange(N), np.arange(N)))
        s = np.hstack(
            (
                np.kron(1.0 / self.hz[:-1], np.ones((self.nx * (self.ny - 1),))),
                np.kron(-1.0 / self.hz[1:], np.ones((self.nx * (self.ny - 1),))),
            )
        )
        Dxz = sp.coo_array((s, (i, j)))

        # dHz / dx
        M = self.nx
        N = self.nx - 1
        i = np.hstack((np.arange(N), 1 + np.arange(N)))
        j = np.hstack((np.arange(N), np.arange(N)))
        nval = i.size
        ii = np.zeros(((self.ny - 1) * self.nz * nval,), dtype=np.int64)
        jj = np.zeros(((self.ny - 1) * self.nz * nval,), dtype=np.int64)
        for n in np.arange((self.ny - 1) * self.nz):
            ii[(n * nval) : ((n + 1) * nval)] = i + n * M
            jj[(n * nval) : ((n + 1) * nval)] = j + n * N
        s = np.tile(np.hstack((1.0 / self.hx[:-1], -1.0 / self.hx[1:])), ((self.ny - 1) * self.nz,))
        Dzx = sp.coo_array((s, (ii, jj)))

        # dHy / dx
        M = self.nx
        N = self.nx - 1
        i = np.hstack((np.arange(N), 1 + np.arange(N)))
        j = np.hstack((np.arange(N), np.arange(N)))
        nval = i.size
        ii = np.zeros((self.ny * (self.nz - 1) * nval,), dtype=np.int64)
        jj = np.zeros((self.ny * (self.nz - 1) * nval,), dtype=np.int64)
        for n in np.arange(self.ny * (self.nz - 1)):
            ii[(n * nval) : ((n + 1) * nval)] = i + n * M
            jj[(n * nval) : ((n + 1) * nval)] = j + n * N
        s = np.tile(np.hstack((1.0 / self.hx[:-1], -1.0 / self.hx[1:])), (self.ny * (self.nz - 1),))
        Dyx = sp.coo_array((s, (ii, jj)))

        # dHx / dy
        M = self.nx * self.ny
        N = self.nx * (self.ny - 1)
        i = np.hstack((np.arange(N), self.nx + np.arange(N)))
        j = np.hstack((np.arange(N), np.arange(N)))
        nval = i.size
        ii = np.zeros(((self.nz - 1) * nval,), dtype=np.int64)
        jj = np.zeros(((self.nz - 1) * nval,), dtype=np.int64)
        for n in np.arange(self.nz - 1):
            ii[(n * nval) : ((n + 1) * nval)] = i + n * M
            jj[(n * nval) : ((n + 1) * nval)] = j + n * N
        s = np.hstack(
            (np.kron(1.0 / self.hy[:-1], np.ones((self.nx,))), np.kron(-1.0 / self.hy[1:], np.ones((self.nx,))))
        )
        s = np.tile(s, (self.nz - 1,))
        Dxy = sp.coo_array((s, (ii, jj)))

        return sp.vstack(
            (
                sp.hstack((sp.csr_array((self.nfx, self.nex)), -Dyz, Dzy), format="csr"),
                sp.hstack((Dxz, sp.csr_array((self.nfy, self.ney)), -Dzx), format="csr"),
                sp.hstack((-Dxy, Dyx, sp.csr_array((self.nfz, self.nez))), format="csr"),
            )
        )

    def _curl_edges(self):

        # dAz / dy
        M = self.nx * (self.ny - 1)
        N = self.nx * self.ny
        i = np.hstack((np.arange(M), np.arange(M)))
        j = np.hstack((np.arange(M), self.nx + np.arange(M)))
        nval = i.size
        ii = np.zeros(((self.nz - 1) * nval,), dtype=np.int64)
        jj = np.zeros(((self.nz - 1) * nval,), dtype=np.int64)
        for n in np.arange(self.nz - 1):
            ii[(n * nval) : ((n + 1) * nval)] = i + n * M
            jj[(n * nval) : ((n + 1) * nval)] = j + n * N
        s = np.hstack((np.kron(-1.0 / self.dy, np.ones((self.nx,))), np.kron(1.0 / self.dy, np.ones((self.nx,)))))
        s = np.tile(s, (self.nz - 1,))
        Dzy = sp.coo_array((s, (ii, jj)))

        # dAy / dz
        N = self.nx * (self.ny - 1) * (self.nz - 1)
        i = np.hstack((np.arange(N), np.arange(N)))
        j = np.hstack((np.arange(N), self.nx * (self.ny - 1) + np.arange(N)))
        s = np.hstack(
            (
                np.kron(-1.0 / self.dz, np.ones((self.nx * (self.ny - 1),))),
                np.kron(1.0 / self.dz, np.ones((self.nx * (self.ny - 1),))),
            )
        )
        Dyz = sp.coo_array((s, (i, j)))

        # dAx / dz
        N = (self.nx - 1) * self.ny * (self.nz - 1)
        i = np.hstack((np.arange(N), np.arange(N)))
        j = np.hstack((np.arange(N), (self.nx - 1) * self.ny + np.arange(N)))
        s = np.hstack(
            (
                np.kron(-1.0 / self.dz, np.ones(((self.nx - 1) * self.ny,))),
                np.kron(1.0 / self.dz, np.ones(((self.nx - 1) * self.ny,))),
            )
        )
        Dxz = sp.coo_array((s, (i, j)))

        # dAz / dx
        M = self.nx - 1
        N = self.nx
        i = np.hstack((np.arange(M), np.arange(M)))
        j = np.hstack((np.arange(M), np.arange(1, N)))
        nval = i.size
        ii = np.zeros((self.ny * (self.nz - 1) * nval,), dtype=np.int64)
        jj = np.zeros((self.ny * (self.nz - 1) * nval,), dtype=np.int64)
        for n in np.arange(self.ny * (self.nz - 1)):
            ii[(n * nval) : ((n + 1) * nval)] = i + n * M
            jj[(n * nval) : ((n + 1) * nval)] = j + n * N
        s = np.tile(np.hstack((-1.0 / self.dx, 1.0 / self.dx)), (self.ny * (self.nz - 1),))
        Dzx = sp.coo_array((s, (ii, jj)))

        # dAy / dx
        M = self.nx - 1
        N = self.nx
        i = np.hstack((np.arange(M), np.arange(M)))
        j = np.hstack((np.arange(M), np.arange(1, N)))
        nval = i.size
        ii = np.zeros(((self.ny - 1) * self.nz * nval,), dtype=np.int64)
        jj = np.zeros(((self.ny - 1) * self.nz * nval,), dtype=np.int64)
        for n in np.arange((self.ny - 1) * self.nz):
            ii[(n * nval) : ((n + 1) * nval)] = i + n * M
            jj[(n * nval) : ((n + 1) * nval)] = j + n * N
        s = np.tile(np.hstack((-1.0 / self.dx, 1.0 / self.dx)), ((self.ny - 1) * self.nz,))
        Dyx = sp.coo_array((s, (ii, jj)))

        # dAx / dy
        M = (self.nx - 1) * (self.ny - 1)
        N = (self.nx - 1) * self.ny
        i = np.hstack((np.arange(M), np.arange(M)))
        j = np.hstack((np.arange(M), (self.nx - 1) + np.arange(M)))
        nval = i.size
        ii = np.zeros((self.nz * nval,), dtype=np.int64)
        jj = np.zeros((self.nz * nval,), dtype=np.int64)
        for n in np.arange(self.nz):
            ii[(n * nval) : ((n + 1) * nval)] = i + n * M
            jj[(n * nval) : ((n + 1) * nval)] = j + n * N
        s = np.hstack(
            (np.kron(-1.0 / self.dy, np.ones((self.nx - 1,))), np.kron(1.0 / self.dy, np.ones((self.nx - 1,))))
        )
        s = np.tile(s, (self.nz,))
        Dxy = sp.coo_array((s, (ii, jj)))

        return sp.vstack(
            (
                sp.hstack((sp.csr_array((self.nex, self.nfx)), -Dyz, Dzy), format="csr"),
                sp.hstack((Dxz, sp.csr_array((self.ney, self.nfy)), -Dzx), format="csr"),
                sp.hstack((-Dxy, Dyx, sp.csr_array((self.nez, self.nfz))), format="csr"),
            )
        )

    def extract_Gxyz(self):
        """Extract Gx, Gy & Gz from G.
        """
        Gx = self.G[:self.nfx, :]
        Gy = self.G[self.nfx:(self.nfx + self.nfy), :]
        Gz = self.G[(self.nfx + self.nfy):, :]
        return Gx, Gy, Gz

    def extract_xyz_faces(self, v):
        if v.ndim == 1:
            v = v.reshape(-1, 1)
        return v[:self.nfx, :], v[self.nfx:(self.nfx+self.nfy), :], v[(self.nfx+self.nfy):, :]

    def below_surface(self, pts):
        """Return indices of electrodes that are not at the surface."""
        return pts[:, 2] != self.z[-1]

    def process_surface_elec(self, elec):
        """Process surface Tx coordinate to make sure electrodes are at least at the depth of the first cell center.

        Parameters
        ----------
        elec : ndarray
            coordinates of electrodes, n x 3

        Returns
        -------
        ndarray
            coordinates after correction

        Note
        ----
        Vertical axis is elevation, ie positive upwards.
        """
        ind = elec[:, 2] > self.zc[-1]
        elec[ind, 2] = self.zc[-1]
        return elec

    def solve(self, b):
        return self.solver_A.solve(b)

    def compute_u_homog(self, c1c2, sigma, cs):
        """Potential of a homogeneous half-space at the centre of the voxels.

        Parameters
        ----------
        c1c2 : ndarray
            Injection electrodes, n x 6 (dipoles) or n x 3 (poles).
        sigma : float or ndarray
            Conductivity, scalar, or one value per current electrode (n x 2 for dipoles, n for poles).
        cs : ndarray
            Current intensity, n.

        Notes
        -----
        First electrode is the source (+I) and second is the sink; the potential is positive around the source.

        The field of a dipole is the difference of the fields of its two electrodes, each with its own
        treatment of the singularity (as when the fields are computed per electrode, for the sensitivity).

        Surface electrodes are moved to the depth of the first cell centre.  If surface_images is True,
        their field includes the image above the surface, so that it satisfies the no-flux condition at
        z = 0 as the discrete solution does (see Grid25FV._with_images).
        """
        c1c2 = np.array(c1c2, dtype=float)
        cs = cs * np.ones((c1c2.shape[0],))
        s1, s2 = Grid25FV._electrode_sigma(sigma, c1c2.shape[0], c1c2.shape[1] == 6)
        if c1c2.shape[1] == 6:
            return self.compute_u_homog(c1c2[:, :3], s1, cs) - self.compute_u_homog(c1c2[:, 3:6], s2, cs)
        elif c1c2.shape[1] != 3:
            raise ValueError("c1c2 must be either n x 3 or n x 6")

        x, y, z = self.centre_voxels()
        images = self._with_images(c1c2)
        c = self.process_surface_elec(c1c2.copy())
        u0 = np.empty((self.nc, c.shape[0]))
        with np.errstate(divide='ignore'):
            for i in range(c.shape[0]):
                r = np.sqrt((x - c[i, 0])**2 + (y - c[i, 1])**2 + (z - c[i, 2])**2)
                if images[i]:
                    ri = np.sqrt((x - c[i, 0])**2 + (y - c[i, 1])**2 + (z + c[i, 2] - 2 * self.z[-1])**2)
                    u0[:, i] = cs[i] / (s1[i] * 4.0 * np.pi) * (1. / r + 1. / ri)
                else:
                    u0[:, i] = cs[i] / (s1[i] * 2.0 * np.pi) / r
        # note: this works for electrodes at infinity, 1/np.inf is 0

        # singularity due to a source at the centre of a voxel: mean of the finite values of the
        # neighbouring voxels
        for i in range(c.shape[0]):
            ind = np.nonzero(np.isinf(u0[:, i]))[0]
            new_values = []
            for j in ind:
                ix, iy, iz = self.revind(j)
                nb = [self.ind(ix + dx, iy + dy, iz + dz)
                      for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
                      if 0 <= ix + dx < self.nx and 0 <= iy + dy < self.ny and 0 <= iz + dz < self.nz]
                v = u0[nb, i]
                new_values.append(np.mean(v[np.isfinite(v)]))
            u0[ind, i] = new_values

        return u0

    def _with_images(self, pts):
        """Return True for electrodes whose analytic field must include the image source (see Grid25FV)."""
        if self.surface_images:
            return np.ones((pts.shape[0],), dtype=bool)
        return self.below_surface(pts)

    def u_homog_at(self, c1c2, pts, sigma, cs):
        """3D potential of a homogeneous half-space, source of row i evaluated at point of row i.

        Exact solution at the true positions of the electrodes and points (surface electrodes have no
        image), used as primary field at the measurement electrodes (see Grid25FV.u_homog_at).

        Parameters
        ----------
        c1c2 : ndarray
            Injection electrodes, n x 6 (dipoles) or n x 3 (poles).
        pts : ndarray
            Evaluation points, n x 3.
        sigma : float or ndarray
            Conductivity, scalar or one value per current electrode (n x 2 for dipoles, n for poles).
        cs : ndarray
            Current intensity, n.
        """
        c1c2 = np.atleast_2d(np.array(c1c2, dtype=float))
        pts = np.atleast_2d(np.array(pts, dtype=float))
        cs = cs * np.ones((c1c2.shape[0],))
        s1, s2 = Grid25FV._electrode_sigma(sigma, c1c2.shape[0], c1c2.shape[1] == 6)

        def pot(c, s):
            below = self.below_surface(c)
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.sqrt(np.sum((pts - c)**2, axis=1))
                ri = np.sqrt((pts[:, 0] - c[:, 0])**2 + (pts[:, 1] - c[:, 1])**2 + (pts[:, 2] + c[:, 2] - 2 * self.z[-1])**2)
                u = np.where(below, (1. / r + 1. / ri) / 4.0, 1. / (2.0 * r)) / s
            # electrode or point at infinity: no contribution (avoids inf - inf)
            remote = np.logical_not(np.logical_and(np.all(np.isfinite(c), axis=1),
                                                   np.all(np.isfinite(pts), axis=1)))
            u[remote] = 0.0
            return u

        u = pot(c1c2[:, :3], s1)
        if c1c2.shape[1] == 6:
            u = u - pot(c1c2[:, 3:6], s2)
        u = cs / np.pi * u
        if np.any(np.isinf(u)):
            raise ValueError('Evaluation point at the position of a current electrode')
        return u

    def toVTK(self, fields, filename, component="s", on_face=True, metadata=None):
        """
        Save a field in a vtk file.

        Parameters
        ----------
        fields : dict[str, ndarray]
            dict holding the name to assign to the data (key)
            and the data array (value)
        filename : string
            name of file without extension
        component : string, optional
            - 's' : field is a scalar (defined at centre of voxels)
            - 'x' : X component of the field
            - 'y' : Y component of the field
            - 'z' : Z component of the field
        on_face : bool, optional
            field component is defined on faces (True), or on edges (False)
        metadata : dict[str, str], optional
            metadata to store in field data
        """

        if not isinstance(fields, dict):
            raise ValueError("'fields' must be a dict[str, ndarray]")

        # double precision, so that grids read back from the file are identical
        xCoords = vtk.vtkDoubleArray()
        yCoords = vtk.vtkDoubleArray()
        zCoords = vtk.vtkDoubleArray()

        if component == "s":
            x = self.x
            y = self.y
            z = self.z
        elif component == "x":
            if on_face:
                x = self.x[1:-1]
                y = self.yc
                z = self.zc
            else:
                x = self.xc
                y = self.y[1:-1]
                z = self.z[1:-1]
        elif component == "y":
            if on_face:
                x = self.xc
                y = self.y[1:-1]
                z = self.zc
            else:
                x = self.x[1:-1]
                y = self.yc
                z = self.z[1:-1]
        elif component == "z":
            if on_face:
                x = self.xc
                y = self.yc
                z = self.z[1:-1]
            else:
                x = self.x[1:-1]
                y = self.y[1:-1]
                z = self.zc
        else:
            raise ValueError("unknown component")

        for i in x:
            xCoords.InsertNextValue(i)

        for i in y:
            yCoords.InsertNextValue(i)

        for i in z:
            zCoords.InsertNextValue(i)

        rgrid = vtk.vtkRectilinearGrid()
        rgrid.SetDimensions(x.size, y.size, z.size)
        rgrid.SetXCoordinates(xCoords)
        rgrid.SetYCoordinates(yCoords)
        rgrid.SetZCoordinates(zCoords)

        for fn in fields:
            field = np.asarray(fields[fn]).ravel()   # fields may be column vectors (e.g. extract_xyz_faces)
            data = vtk.vtkDoubleArray()
            for i in field:
                data.InsertNextValue(i)
            data.SetName(fn)
            if component == "s":
                rgrid.GetCellData().AddArray(data)
            else:
                rgrid.GetPointData().AddArray(data)

        if metadata is not None:
            string_array = vtk.vtkStringArray()
            string_array.SetNumberOfTuples(len(metadata))
            for nm, key in enumerate(metadata):
                string_array.SetValue(nm, metadata[key])
                string_array.SetName(key)
            rgrid.GetFieldData().AddArray(string_array)

        writer = vtk.vtkXMLRectilinearGridWriter()
        writer.SetInputData(rgrid)
        writer.SetFileName(filename + ".vtr")
        writer.Write()

    def fromVTK(self, fieldname, filename):
        """Extract scalar field from a vtk file.

        Parameters
        ----------
        fieldname : string
            Name of the field to extract.
        filename : string
            Name of vtk file.

        Returns
        -------
        ndarray (or None if field not in file)

        """
        reader = vtk.vtkXMLRectilinearGridReader()
        reader.SetFileName(filename)
        reader.Update()
        dims = reader.GetOutput().GetDimensions()
        if dims[0] != self.nx + 1 or dims[1] != self.ny + 1 or dims[2] != self.nz + 1:
            raise ValueError("Grid size incompatible")
        data = reader.GetOutput().GetCellData().GetArray(fieldname)
        if data is None:
            return data
        else:
            return vtk_to_numpy(data)

    def print_info(self, file=None):
        print('    Grid: {0:d} x {1:d} x {2:d} voxels'.format(self.nx, self.ny, self.nz), file=file)
        print('      X min: {0:e}\tX max: {1:e}'.format(self.x[0], self.x[-1]), file=file)
        print('      Y min: {0:e}\tY max: {1:e}'.format(self.y[0], self.y[-1]), file=file)
        print('      Z min: {0:e}\tZ max: {1:e}'.format(self.z[0], self.z[-1]), file=file)


# %% MeshFV

class MeshFV(BaseFV, SimplexMesh):
    """Class to manage tetrahedral meshes for finite volume modelling.

    Parameters
    ----------
    param: tuple
        parameters to instantiate the finite volume mesh
        pts : array_like
            Coordinates of the nodes making the mesh
        tet : array_like of int
            Indices of the nodes forming the tetrahedra
        tri_surf : array_like of int
            Indices of the nodes forming the triangulated surface of the ground
    """

    def __init__(self, param, comm=None, n_threads=1):
        BaseFV.__init__(self, comm, n_threads)
        pts, tet, tri_surf = param
        SimplexMesh.__init__(self, pts, tet)
        self.tri_surf = np.array(tri_surf)
        self.dim = 3
        self.D = self.face_divergence
        self.G = self.stencil_cell_gradient

    @property
    def nc(self):
        return self.n_cells

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, dim):
        self._dim = dim

    @property
    def nf(self):
        return self.n_faces

    def is_inside(self, x, y, z):
        """Check if point is inside mesh.

        Parameters
        ----------
        x : float
            coordinate along X.
        y : float
            coordinate along Y.
        z : float
            coordinate along Z.

        Returns
        -------
        True if point is inside grid

        """
        p = np.array([x, y, z])
        for i in range(self.n_cells):
            if self._inside_tet(i, p):
                return True
        else:
            return False

    def below_surface(self, pts):
        """Return indices of points that are not at the surface."""
        ind = np.zeros((pts.shape[0],), dtype=bool)
        for npt in range(pts.shape[0]):
            p = pts[npt, :]
            for tri_no in range(len(self.tri_surf)):
                if self._inside_tri(tri_no, p):
                    ind[npt] = True
                    break
        return ind

    def process_surface_elec(self, elec, ind):
        """Process surface Tx coordinate to make sure electrodes are at least at the depth of the first cell center.

        Parameters
        ----------
        elec : ndarray
            coordinates of electrodes, n x 3
        ind : ndarray of bool
            True for coordinates to process

        Returns
        -------
        ndarray
            coordinates after correction

        Notes
        -----
        Vertical axis is elevation, ie positive upwards.
        """
        for n, i in enumerate(ind):
            if i is True:
                p = elec[n, :]
                # find tetrahedron we are in
                for t in range(self.n_cells):
                    if self._inside_tet(t, p):
                        # center of tet
                        ve = 0.25 * (self.nodes[self.simplices[t, 0]] +
                                     self.nodes[self.simplices[t, 1]] +
                                     self.nodes[self.simplices[t, 2]] +
                                     self.nodes[self.simplices[t, 3]])
                        # distance between point and center
                        v = ve - p
                        # add 1/3 of distance to point
                        elec[n, :] += 0.333 * v

                        break
        return elec

    def centre_voxels(self):
        """Returns coordinates at the centres of voxels.

        Returns
        -------
        tuple of 3 ndarray
            coordinates X, Y, Z

        """
        return self.cell_centers[:, 0], self.cell_centers[:, 1], self.cell_centers[:, 2]

    def volume_voxels(self):
        return self._cell_volumes

    def build_C(self, to_faces=True):
        if to_faces:
            return self._face_curl()
        else:
            return self.edge_curl

    def _face_curl(self):
        """faces to edges"""

        # Calculation of curl, from faces to edges
        # https://en.wikipedia.org/wiki/Curl_(mathematics)

        # For each edge, we have to
        # - collect tetrahedra that contains the considered edge and identify the triangles holding the edge
        # - get barycenter of each triangle
        # - compute the normal to the triangle at each barycenter, noted nt, which gives direction of J
        # - compute the segments connecting the centroids, noted r.  Centroids should be ordered properly, and the order will
        #   determine the direction of the "normal" on which the curl is projected
        # - compute the dot product between J and r and sum over segments
        # - compute midpoint of the considered segment
        # - compute area of the triangles formed by the midpoint and each segment, and sum over all

        # find edges on the mesh boundary
        be = self._edges[np.unique(self._face_edges[self.boundary_face_list])]
        be = be.tolist()
        # loop over edges
        for e in self._edges:
            # process edge not on boundary
            if e.tolist() not in be:

                # find tetrahedra holding the edge
                tet = []
                tet_no = []
                for n, t in enumerate(self.simplices):
                    if e[0] in t and e[1] in t:
                        tet.append(t)
                        tet_no.append(n)

                # find triangles holding the edge
                triangles = []
                tri_no = []
                for nt in np.unique(self._simplex_faces[tet_no]):
                    f = self._faces[nt]
                    if e[0] in f and e[1] in f:
                        triangles.append(f)
                        tri_no.append(nt)

                # get barycenter of each triangle
                bary = []
                for t in triangles:
                    bary.append((self.nodes[t[0]] + self.nodes[t[1]] + self.nodes[t[2]]) / 3.)

                # start with a tetrahedron and look for one next to it
                order = []
                tet_to_process = copy.copy(tet_no)
                tet_to_process.append(tet_no[0])   # to allow finding triangle opposite to the first that will be picked
                tet_processed = []
                t_no = tet_to_process[0]
                while len(tet_to_process) > 1:
                    if len(tet_processed) > 1 and tet_no[0] in tet_processed:
                        tet_processed.remove(tet_no[0])   # remove to be able to reprocess it with the last tet
                    t_faces = self._simplex_faces[t_no]
                    for nb in self.neighbors[t_no]:
                        if nb == -1 or nb in tet_processed:
                            continue
                        nb_faces = self._simplex_faces[nb]
                        for nf in t_faces:
                            if nf in nb_faces:
                                o = np.where(nf == tri_no)[0]
                                order.append(o[0])
                                break
                        tet_processed.append(t_no)
                        tet_to_process.remove(t_no)
                        t_no = nb
                        break

                mid_pt = 0.5 * (self.nodes[e[0]] + self.nodes[e[1]])

                r = []
                area = []
                for n in range(len(order)-1):
                    r.append(bary[order[n+1]] - bary[order[n]])
                    v1 = bary[order[n]] - mid_pt
                    v2 = bary[order[n+1]] - mid_pt
                    area.append(0.5 * np.linalg.norm(np.cross(v1, v2)))
                r.append(bary[order[0]] - bary[order[-1]])
                v1 = bary[order[0]] - mid_pt
                v2 = bary[order[-1]] - mid_pt
                area.append(0.5 * np.linalg.norm(np.cross(v1, v2)))

                self._save_edge_data(e, bary, order, triangles)

    def build_D(self):
        return self.face_divergence

    def build_G(self):
        # centers to faces
        return self.stencil_cell_gradient

    def build_M(self, v, harmon=True):
        M = self.average_cell_to_face
        if harmon:
            tmp = M @ (1.0 / v)
            return sp.diags_array(1.0 / tmp, format='csr')
        else:
            return M @ v

    def build_A(self, sigma):
        """Build LHS matrix for DC resistivity forward modeling.

        Parameters
        ----------
        sigma : array_like
            conductivity model

        Returns
        -------
        tuple of 2 matrices:
            - A : LHS matrix
            - M : matrix of harmonic average of sigma
        """
        # Build LHS matrix
        if np.isscalar(sigma) is True:
            M = sigma * sp.eye_array(self.nf, self.nf)
        else:
            M = self.build_M(sigma)
        A = -self.D @ M @ self.G
        # I, J, V = sp.find(A[0, :])
        # for jj in J:
        #     A[0, jj] = 0.0
        # A[0, 0] = 1.0/(self.fv.hx[0] * self.fv.hy[0] * self.fv.hz[0])
        A[0, 0] += 1.0 / self.volume_voxels()[0]

        return A, M

    def compute_u_homog(self, c1c2, sigma, cs):
        x, y, z = self.centre_voxels()
        u0 = np.empty((self.nc, c1c2.shape[0]))
        # turn off warning b/c we get a divide by zero that we will fix later
        np.seterr(divide='ignore')
        if c1c2.shape[1] == 6:
            # keep track of electrodes below the surface
            below_surf_c1 = self.below_surface(c1c2[:, :3])
            below_surf_c2 = self.below_surface(c1c2[:, 3:6])
            # make sure electrodes are at least at the depth of the first cell center
            c1c2[:, :3] = self.process_surface_elec(c1c2[:, :3], np.logical_not(below_surf_c1))
            c1c2[:, 3:6] = self.process_surface_elec(c1c2[:, 3:6], np.logical_not(below_surf_c2))

            for i in np.arange(c1c2.shape[0]):
                pve1 = np.sqrt((x - c1c2[i, 0]) ** 2 + (y - c1c2[i, 1]) ** 2 + (z - c1c2[i, 2]) ** 2)
                # norm of negative current electrode and 1st potential electrode
                nve1 = np.sqrt((x - c1c2[i, 3]) ** 2 + (y - c1c2[i, 4]) ** 2 + (z - c1c2[i, 5]) ** 2)
                if below_surf_c1[i] or below_surf_c2[i]:
                    # norm of imaginary positive current electrode and 1st potential electrode
                    pveimag1 = np.sqrt((x - c1c2[i, 0]) ** 2 + (y - c1c2[i, 1]) ** 2 + (z + c1c2[i, 2]) ** 2)
                    nveimag1 = np.sqrt((x - c1c2[i, 3]) ** 2 + (y - c1c2[i, 4]) ** 2 + (z + c1c2[i, 5]) ** 2)
                    gf = 4.0
                else:
                    pveimag1 = np.inf
                    nveimag1 = np.inf
                    gf = 2.0
                u0[:, i] = cs[i] / (sigma*gf*np.pi) * (1./pve1 - 1./nve1 + 1./pveimag1 - 1./nveimag1).flatten()
            # note: this works for electrodes at infinity, numpy recognizes that 1./np.inf is 0

        elif c1c2.shape[1] == 3:
            # keep track of electrodes below the surface
            below_surf_c1 = self.below_surface(c1c2)
            # make sure electrodes are at least at the depth of the first cell center
            c1c2 = self.process_surface_elec(c1c2, np.logical_not(below_surf_c1))

            for i in np.arange(c1c2.shape[0]):
                pve1 = np.sqrt((x - c1c2[i, 0]) ** 2 + (y - c1c2[i, 1]) ** 2 + (z - c1c2[i, 2]) ** 2)
                if below_surf_c1[i]:
                    # norm of imaginary positive current electrode and 1st potential electrode
                    pveimag1 = np.sqrt((x - c1c2[i, 0]) ** 2 + (y - c1c2[i, 1]) ** 2 + (z + c1c2[i, 2]) ** 2)
                    gf = 4.0
                else:
                    pveimag1 = np.inf
                    gf = 2.0
                u0[:, i] = cs[i] / (sigma * gf * np.pi) * (1. / pve1 + 1. / pveimag1).flatten()
        else:
            raise ValueError("c1c2 must be either n x 3 or n x 6")
        np.seterr(divide='warn')

        # check for singularities due to the source being on a tet center
        for i in range(c1c2.shape[0]):
            ind = np.nonzero(np.isinf(u0[:, i]))
            if ind[0].size > 0:
                raise NotImplementedError('now is the time to raise your sleeves')

        return u0


    def toVTK(self, fields, filename, component="s", on_face=True, metadata=None):

        if not isinstance(fields, dict):
            raise ValueError("'fields' must be a dict[str, ndarray]")

        mesh = vtk.vtkUnstructuredGrid()
        pts_vtk = vtk.vtkPoints()
        for n in range(self.nodes.shape[0]):
            pts_vtk.InsertNextPoint(self.nodes[n, :])
        mesh.SetPoints(pts_vtk)

        vtk_tet = vtk.vtkCellArray()
        for n in range(self.simplices.shape[0]):
            vtk_tet.InsertNextCell(4, self.simplices[n, :])

        mesh.SetCells(vtk.VTK_TETRA, vtk_tet)

        for fn in fields:
            # TODO: make this work for vectors
            field = fields[fn]
            data = vtk.vtkDoubleArray()
            for i in field:
                data.InsertNextValue(i)
            data.SetName(fn)
            if component == "s":
                mesh.GetCellData().AddArray(data)
            else:
                mesh.GetPointData().AddArray(data)

        if metadata is not None:
            string_array = vtk.vtkStringArray()
            string_array.SetNumberOfTuples(len(metadata))
            for nm, key in enumerate(metadata):
                string_array.SetValue(nm, metadata[key])
                string_array.SetName(key)
            mesh.GetFieldData().AddArray(string_array)

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(filename + ".vtu")
        writer.SetInputData(mesh)
        writer.Write()

    def _save_edge_data(self, e, bary, order, triangles):
        # save edge for visualization
        pts = vtk.vtkPoints()
        pts.InsertNextPoint(self.nodes[e[0]])
        pts.InsertNextPoint(self.nodes[e[1]])
        pdata = vtk.vtkPolyData()
        pdata.SetPoints(pts)
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(2)
        line.GetPointIds().SetId(0, 0)
        line.GetPointIds().SetId(1, 1)
        ca = vtk.vtkCellArray()
        ca.InsertNextCell(line)
        pdata.SetLines(ca)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName('/tmp/edge.vtp')
        writer.SetInputData(pdata)
        writer.Update()

        pts = vtk.vtkPoints()
        pts.InsertNextPoint(0.5 * (self.nodes[e[0]] + self.nodes[e[1]]))
        pdata = vtk.vtkPolyData()
        pdata.SetPoints(pts)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName('/tmp/mid_pt.vtp')
        writer.SetInputData(pdata)
        writer.Update()

        pts = vtk.vtkPoints()
        for b in bary:
            pts.InsertNextPoint(b)
        pdata = vtk.vtkPolyData()
        pdata.SetPoints(pts)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName('/tmp/bary.vtp')
        writer.SetInputData(pdata)
        writer.Update()

        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(order)+1)
        for n, o in enumerate(order):
            line.GetPointIds().SetId(n, o)
        line.GetPointIds().SetId(len(order), 0)
        ca = vtk.vtkCellArray()
        ca.InsertNextCell(line)
        pdata.SetLines(ca)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName('/tmp/seg.vtp')
        writer.SetInputData(pdata)
        writer.Update()

        pts = vtk.vtkPoints()
        pt_no = []
        for n, p in enumerate(np.unique(triangles).tolist()):
            pts.InsertNextPoint(self.nodes[p])
            pt_no.append(n)
        tri = vtk.vtkCellArray()
        for t in triangles:
            tri.InsertNextCell(3, t)
        mesh = vtk.vtkUnstructuredGrid()
        mesh.SetPoints(pts)
        mesh.SetCells(vtk.VTK_TRIANGLE, tri)

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName('/tmp/tri_edge.vtu')
        writer.SetInputData(mesh)
        writer.Write()

    def _inside_tet(self, tet_no, p):
        v1 = self.nodes[self.simplices[tet_no, 0]]
        v2 = self.nodes[self.simplices[tet_no, 1]]
        v3 = self.nodes[self.simplices[tet_no, 2]]
        v4 = self.nodes[self.simplices[tet_no, 3]]

        # if not inside_tetrahedron_box(v1, v2, v3, v4, p):
        #     return False

        M1 = tetra_coord(v1, v2, v3, v4)
        # apply the transform to P (v1 is the origin)
        newp = M1.dot(p-v1)
        # perform test
        return np.all(newp >= 0.) and np.all(newp <= 1.) and np.sum(newp) <= 1.

    def _inside_tri(self, tri_no, p):
        v1 = np.array(self.nodes[self.tri_surf[tri_no, 0]])
        v2 = np.array(self.nodes[self.tri_surf[tri_no, 1]])
        v3 = np.array(self.nodes[self.tri_surf[tri_no, 2]])

        if not inside_triangle_box(v1, v2, v3, p):
            return False

        b = barycentric(v1, v2, v3, p)
        return b[1] >= 0. and b[2] >= 0. and (b[1] + b[2]) <= 1.

    def print_info(self, file=None):
        print('    Mesh: {0:d} voxels, {1:d} nodes'.format(len(self.simplices), len(self.nodes)), file=file)
        # print('      X min: {0:e}\tX max: {1:e}'.format(self.x[0], self.x[-1]), file=file)
        # print('      Y min: {0:e}\tY max: {1:e}'.format(self.y[0], self.y[-1]), file=file)
        # print('      Z min: {0:e}\tZ max: {1:e}'.format(self.z[0], self.z[-1]), file=file)


# %% GridFV


class Grid25FV(BaseFV):
    """Class to manage rectilinear grids for 2.5D finite volume modelling.

    Parameters
    ----------
    param: tuple
        parameters to instantiate the finite volume grid
        x : array_like
            Node coordinates along x
        z : array_like
            Node coordinates along z
    comm : MPI Communicator, optional
        If None, use MPI_COMM_WORLD

    Notes
    -----
    Voxels are sorted column major, i.e. x is the fast axis
    (choice dictated by convention used by VTK).

    """

    def __init__(self, param, comm=None, n_threads=1):
        BaseFV.__init__(self, comm, n_threads)
        x, z = param
        self.x = x
        self.z = z
        # wavenumbers & weights of the inverse Fourier transform, u = 1/2 sum_i g_i U(k_i),
        # normalized so that sum_i g_i K0(k_i r) ~ 1/r.  Default values from Pidlisecky & Knight (2008),
        # accurate for r in 0.5 - 20 m; with auto_kg True, they are optimized for the survey (see optimize_k_g)
        self.k = np.array([0.0217102, 0.2161121, 1.0608400, 5.0765870])
        self.g = np.array([0.0463660, 0.2365931, 1.0382080, 5.3648010])
        self.auto_kg = True
        self.n_k = 11
        self.surface_images = True   # see _with_images
        self._kg_range = None
        self.dim = 2.5

    def ind(self, i, j, k, component=None):
        """
        Returns the index of a voxel.

        Parameters
        ----------
        i : int or array_like of int
            indice(s) along x
        j : int or array_like of int
            indice(s) along y (ignored)
        k : int or array_like of int
            indice(s) along z
        component : str, optional
            component considered, possible values are:
                - None: the method returns voxel index
                - 'ex': component x defined on an edge
                - 'ey': component y defined on an edge
                - 'ez': component z defined on an edge
                - 'fx': component x defined on a face
                - 'fy': component y defined on a face
                - 'fz': component z defined on a face
        Returns
        -------
        int or array of int
        """
        if component is None:
            nx = self.nx
            nz = self.nz
        elif component == "ex":
            nx = self.nx
            nz = self.nz - 1
        elif component == "ey":
            nx = self.nx - 1
            nz = self.nz - 1
        elif component == "ez":
            nx = self.nx - 1
            nz = self.nz
        elif component == "fx":
            nx = self.nx - 1
            nz = self.nz
        elif component == "fy":
            nx = self.nx
            nz = self.nz
        elif component == "fz":
            nx = self.nx
            nz = self.nz - 1
        else:
            raise ValueError("unknown component")

        if np.size(i) > 1:
            i = np.array(i)
            i = i.flatten()
        if np.size(k) > 1:
            k = np.array(k)
            k = k.flatten()
        if np.any(i < 0) or np.any(i >= nx) or np.any(k < 0) or np.any(k >= nz):
            raise IndexError("Index outside grid")
        if np.size(i) > 1 or np.size(k) > 1:
            ii = np.kron(np.ones((np.size(k),), dtype=np.int64), i)
            kk = np.kron(k, np.ones((np.size(i),), dtype=np.int64))
            return np.sort(kk * nx + ii)
        else:
            return k * nx + i

    def revind(self, ind, component=None):
        """
        Returns indices i, j, k of a voxel or of a component on a face
        or an edge.

        Parameters
        ----------
        ind : int
            voxel index
        component : str, optional
            component considered, possible values are:
                - None: voxel index
                - 'ex': component x defined on an edge
                - 'ey': component y defined on an edge
                - 'ez': component z defined on an edge
                - 'fx': component x defined on a face
                - 'fy': component y defined on a face
                - 'fz': component z defined on a face

        Returns
        -------
        tuple of int

        """
        if component is None:
            nx = self.nx
        elif component == "ex":
            nx = self.nx
        elif component == "ey":
            nx = self.nx - 1
        elif component == "ez":
            nx = self.nx - 1
        elif component == "fx":
            nx = self.nx - 1
        elif component == "fy":
            nx = self.nx
        elif component == "fz":
            nx = self.nx
        else:
            raise ValueError("unknown component")

        k = int(ind / nx)
        i = ind - k * nx
        return i, 0, k

    def __update_n(self):
        if "_x" in self.__dict__ and "_z" in self.__dict__:
            self.nx = len(self._x) - 1   # number of voxels along x
            self.nz = len(self._z) - 1   # number of voxels along z
            self.nc = self.nx * self.nz  # number of voxels
            self.nfx = (self.nx - 1) * self.nz  # number of faces with normal vector along x
            self.nfz = self.nx * (self.nz - 1)  # number of faces with normal vector along z
            self.nf = self.nfx + self.nfz       # total number of faces
            self.nex = self.nx * (self.nz - 1)  # number of edges with vector along x
            self.nez = (self.nx - 1) * self.nz  # number of edges with vector along z
            self.ne = self.nex + self.nez       # total number of edges
            # wavenumber-domain vector fields (2.5D MMR, see build_C_k): the y component of a vector
            # defined on faces lives at the cell centres, the y component defined on edges at the nodes
            self.ney = (self.nx - 1) * (self.nz - 1)  # number of (interior) nodes
            self.nf_k = self.nfx + self.nc + self.nfz  # vector on faces: x, y (centres), z
            self.ne_k = self.nex + self.ney + self.nez  # vector on edges: x, y (nodes), z

            # Divergence & gradient matrices are precomputed because essentially always used
            self.D = self.build_D()
            self.G = self.build_G()

    @property
    def x(self):
        """Node coordinates along x."""
        return self._x

    @x.setter
    def x(self, val):
        tmp = np.sort(np.array(val, dtype=np.float64))
        if tmp.ndim != 1:
            raise ValueError("1D array needed")
        if len(tmp) < 2:
            raise ValueError("2 nodes or more needed")

        self._x = tmp
        self.hx = np.diff(tmp)
        self.xc = (tmp[1:] + tmp[:-1]) / 2
        self.dx = np.diff(self.xc)
        self.__update_n()

    @property
    def z(self):
        """Node coordinates along z."""
        return self._z

    @z.setter
    def z(self, val):
        tmp = np.sort(np.array(val, dtype=np.float64))
        if tmp.ndim != 1:
            raise ValueError("1D array needed")
        if len(tmp) < 2:
            raise ValueError("2 nodes or more needed")

        self._z = tmp
        self.hz = np.diff(tmp)
        self.zc = (tmp[1:] + tmp[:-1]) / 2
        self.dz = np.diff(self.zc)
        self.__update_n()

    @property
    def A(self):
        """Base matrix A = -D M G (without the k^2 sigma term)."""
        if self.solver_A is None:
            return None
        return self.solver_A.A

    @A.setter
    def A(self, val):
        if self.solver_A is None:
            raise RuntimeError("Solver not defined, cannot set A")
        self.solver_A.update(val, self._sigma)

    def set_system(self, A, keep_solver=True, verbose=False):
        """Factorize the systems A + k_i^2 sigma, one per wavenumber.

        Parameters
        ----------
        A : sparse array
            Base matrix returned by build_A (conductivity model is the one given to build_A).
        keep_solver : bool, optional
            Reuse existing solvers (refactorization only) if wavenumbers have not changed.
        verbose : bool, optional
        """
        # note: set_solver instantiates an (empty) Solver, replaced here by a Solver25
        if (not isinstance(self.solver_A, Solver25) or keep_solver is False or
                not self.solver_A.same_kg(self.k, self.g)):
            self.solver_A = Solver25(self.get_solver_params(), A, self._sigma, self.k, self.g, verbose=verbose,
                                     comm=self.comm, n_threads=self.n_threads)
        else:
            self.solver_A.update(A, self._sigma)

    # The 2.5D grid is seen as a 3D grid one cell thick along y, centred on y = 0, for input/output
    # (VTK files) and for the code shared with 3D (region of interest, regularization).
    ny = 1
    nfy = 0
    y = np.array([-0.5, 0.5])
    yc = np.array([0.0])
    hy = np.array([1.0])

    def toVTK(self, fields, filename, component="s", on_face=True, metadata=None):
        """Save fields in a VTK file (rectilinear grid one cell thick along y, see GridFV.toVTK).

        Parameters
        ----------
        fields : dict[str, ndarray]
            dict holding the name to assign to the data (key) and the data array (value)
        filename : string
            name of file without extension
        component : string, optional
            - 's' : field is a scalar (defined at centre of voxels)
            - 'x' : X component of a field defined on faces
            - 'z' : Z component of a field defined on faces
        on_face : bool, optional
            must be True (fields on edges are not defined in 2.5D)
        metadata : dict[str, str], optional
            metadata to store in field data

        Notes
        -----
        Files written for 2.5D models can be used as model files in run_job.py (2.5D is selected when
        the grid has a single cell along y).
        """
        if component == "y":
            raise ValueError("no Y component on faces in 2.5D")
        if not on_face:
            raise NotImplementedError("fields defined on edges not available in 2.5D")
        # GridFV.toVTK only uses the coordinates (x, y, z and centres), defined for the 2.5D grid
        GridFV.toVTK(self, fields, filename, component=component, on_face=on_face, metadata=metadata)

    def fromVTK(self, fieldname, filename):
        """Extract scalar field from a VTK file (rectilinear grid one cell thick along y).

        Returns
        -------
        ndarray (or None if field not in file)
        """
        # GridFV.fromVTK only uses the grid dimensions (nx, ny, nz), defined for the 2.5D grid
        return GridFV.fromVTK(self, fieldname, filename)

    def volume_faces(self):
        """Returns the dual area associated with the (interior) faces of the grid.

        Returns
        -------
        ndarray
            face length times distance between the centres of the two adjacent cells, ordered as the
            faces (x, then z).

        Notes
        -----
        With Vc = volume_voxels() and Vf = volume_faces(), D = -Vc^-1 G^T Vf, so that diag(Vc) @ L_k,
        with L_k = -D M G + k^2 sigma, is symmetric.  Used to apply L_k^-T in the adjoint computations.
        """
        vx = np.kron(self.hz, self.dx)
        vz = np.kron(self.dz, self.hx)
        return np.r_[vx, vz]

    def extract_Gxyz(self):
        """Extract Gx, Gy & Gz from G (Gy is empty in 2.5D)."""
        Gx = self.G[:self.nfx, :]
        Gy = sp.csr_array((0, self.nc))
        Gz = self.G[self.nfx:, :]
        return Gx, Gy, Gz

    def extract_xyz_faces(self, v):
        """Split a field defined on faces in its X, Y (empty in 2.5D) and Z components."""
        if v.ndim == 1:
            v = v.reshape(-1, 1)
        return v[:self.nfx, :], v[:0, :], v[self.nfx:, :]

    def distance_weighting(self, xo, beta):
        """Calculate distance weighting.

        Parameters
        ----------
        xo : array_like
            coordinates of observation points (y is ignored).
        beta : float
            Determines weighting intensity.

        Returns
        -------
        ndarray (nc,)
            Weights.

        Notes
        -----
        Eq 19 of Li & Oldenburg 2000 Geophysics, vol 65, no 2, pp. 540-552, as in GridFV, with distances
        in the x-z plane and the cell areas.
        """
        xo = np.atleast_2d(xo)
        dA = self.volume_voxels()
        xc, _, zc = self.centre_voxels()
        R0 = 0.5 * np.sqrt(dA.min())
        Q = np.zeros((self.nc,))
        for i in np.arange(xo.shape[0]):
            R = np.sqrt((xc - xo[i, 0])**2 + (zc - xo[i, -1])**2)
            Q += (dA / (R + R0)**3)**2
        return Q ** (0.25 * beta)

    def is_inside(self, x, y, z):
        """Check if point is inside grid.

        Parameters
        ----------
        x : float
            coordinate along X.
        y : float
            coordinate along Y (ignored).
        z : float
            coordinate along Z.

        Returns
        -------
        True if point is inside grid

        """
        return self.x[0] <= x <= self.x[-1] and self.z[0] <= z <= self.z[-1]

    def volume_voxels(self):
        """Returns the volume (area actually for 2.5D) of the voxels of the grid.

        Returns
        -------
        ndarray
            Volumes of voxels.

        """
        return np.kron(self.hz, self.hx)

    def centre_voxels(self):
        """Returns coordinates at the centres of voxels.

        Returns
        -------
        tuple of 3 ndarray
            coordinates X, Y, Z

        Notes
        -----
        Y is returned for the method to be compatible with the 3D codebase.  It is set to 0
        """
        x = np.kron(np.ones((self.nz,)), self.xc)
        z = np.kron(self.zc, np.ones((self.nx,)))
        y = np.zeros(x.shape)
        return x, y, z

    def linear_interp(self, x, y, z, component=None):
        """Calculate interpolation matrix.

        Matrix that allows interpolating a variable defined at the voxel
        centre, on a face, or on an edge, at arbitrary coordinates.

        Parameters
        ----------
        x : float or array of float
            coordinate along X.
        y : float or array of float
            coordinate along Y (ignored in 2.5D).
        z : float or array of float
            coordinate along Z.
        component : str, optional
            component considered, possible values are:
                - None: voxel index
                - 'ex': component x defined on une edge
                - 'ey': component y defined on une edge
                - 'ez': component z defined on une edge
                - 'fx': component x defined on une face
                - 'fy': component y defined on une face
                - 'fz': component z defined on une face

        Returns
        -------
        csr_matrix, npts x nc

        Notes
        -----
        - If any coordinate is at infinity, the corresponding matrix element
        is set to zero.
        """
        if component is None:
            nx = self.nx
            nz = self.nz
            xc = self.xc
            zc = self.zc
        elif component == "ex":
            nx = self.nx
            nz = self.nz - 1
            xc = self.xc
            zc = self.z[1:-1]
        elif component == "ey":
            nx = self.nx - 1
            nz = self.nz - 1
            xc = self.x[1:-1]
            zc = self.z[1:-1]
        elif component == "ez":
            nx = self.nx - 1
            nz = self.nz
            xc = self.x[1:-1]
            zc = self.zc
        elif component == "fx":
            nx = self.nx - 1
            nz = self.nz
            xc = self.x[1:-1]
            zc = self.zc
        elif component == "fy":
            nx = self.nx
            nz = self.nz
            xc = self.xc
            zc = self.zc
        elif component == "fz":
            nx = self.nx
            nz = self.nz - 1
            xc = self.xc
            zc = self.z[1:-1]
        else:
            raise ValueError("unknown component")

        nc = nx * nz

        dx = [0.0, 0.0]
        dz = [0.0, 0.0]
        ix = [0, 0]
        iz = [0, 0]

        if np.isscalar(x) is True:
            x = np.array([x])
            z = np.array([z])

        Q = sp.lil_array((len(x), nc))
        for i in range(len(x)):

            if np.any(np.array([x[i], z[i]]) == np.inf):
                # check if point at np.inf -> set Q to 0
                Q[i, 0] = 0.0
                continue

            # points beyond the first or last node are extrapolated linearly from the two nearest nodes
            im = np.argmin(np.abs(x[i] - xc))
            if (x[i] <= xc[im] and im > 0) or im == len(xc) - 1:
                ix[0] = im - 1
                ix[1] = im
            else:
                ix[0] = im
                ix[1] = im + 1
            dx[0] = x[i] - xc[ix[0]]
            dx[1] = xc[ix[1]] - x[i]

            im = np.argmin(np.abs(z[i] - zc))
            if (z[i] <= zc[im] and im > 0) or im == len(zc) - 1:
                iz[0] = im - 1
                iz[1] = im
            else:
                iz[0] = im
                iz[1] = im + 1
            dz[0] = z[i] - zc[iz[0]]
            dz[1] = zc[iz[1]] - z[i]

            Dx = xc[ix[1]] - xc[ix[0]]
            Dz = zc[iz[1]] - zc[iz[0]]

            Q[i, self.ind(ix[0], 0, iz[0], component)] = (1 - dx[0] / Dx) * (1 - dz[0] / Dz)
            Q[i, self.ind(ix[1], 0, iz[0], component)] = (1 - dx[1] / Dx) * (1 - dz[0] / Dz)
            Q[i, self.ind(ix[0], 0, iz[1], component)] = (1 - dx[0] / Dx) * (1 - dz[1] / Dz)
            Q[i, self.ind(ix[1], 0, iz[1], component)] = (1 - dx[1] / Dx) * (1 - dz[1] / Dz)

        return Q.tocsr()

    def build_A(self, sigma):
        """Build LHS matrix for DC resistivity forward modeling.

        Parameters
        ----------
        sigma : array_like
            conductivity model

        Returns
        -------
        tuple of 2 matrices:
            - A : LHS matrix
            - M : matrix of harmonic average of sigma
        """
        # Build LHS matrix
        if np.isscalar(sigma) is True:
            M = sigma * sp.eye_array(self.nf, self.nf)
        else:
            M = self.build_M(sigma)
        self._sigma = sigma
        # no reference potential needed: A + k^2 sigma is nonsingular for k > 0
        A = -self.D @ M @ self.G

        return A, M

    def build_D(self):
        """Construction of divergence matrix.

        Returns
        -------
        csr_matrix
            Divergence matrix

        """

        # Dx

        M = self.nx
        N = self.nx - 1
        i = np.hstack((np.arange(N), 1 + np.arange(N)))
        j = np.hstack((np.arange(N), np.arange(N)))
        nval = i.size
        ii = np.zeros((self.nz * nval,), dtype=np.int64)
        jj = np.zeros((self.nz * nval,), dtype=np.int64)
        for n in np.arange(self.nz):
            ii[(n * nval) : ((n + 1) * nval)] = i + n * M
            jj[(n * nval) : ((n + 1) * nval)] = j + n * N
        s = np.tile(np.hstack((1.0 / self.hx[:-1], -1.0 / self.hx[1:])), (self.nz,))
        Dx = sp.coo_array((s, (ii, jj)))

        # Dz

        N = self.nx * (self.nz - 1)
        i = np.hstack((np.arange(N), self.nx + np.arange(N)))
        j = np.hstack((np.arange(N), np.arange(N)))
        s = np.hstack(
            (
                np.kron(1.0 / self.hz[:-1], np.ones((self.nx,))),
                np.kron(-1.0 / self.hz[1:], np.ones((self.nx,))),
            )
        )
        Dz = sp.coo_array((s, (i, j)))

        # assemblage

        return sp.hstack((Dx, Dz)).tocsr()

    def build_M(self, v, harmon=True):
        """Calculate harmonic or arithmetic average of a variable.

        Parameters
        ----------
        v : ndarray or tuple of 2 ndarray
            Variable defined at the centre of voxels
            if v is a tuple, medium is anisotropic, with 3 arrays
            corresponding to terms on the diagonal of the tensor
            ie
                |v_xx    0|
            v = |   0 v_zz|

        harmon : bool, optional
            if True, harmonic average is computed, otherwise the
            arithmetic mean is computed.

        Returns
        -------
        csr_matrix
            diagonal matrix containing the average on the faces of the voxels

        """
        if type(v) is tuple:
            vx, vz = v
        else:
            vx = v
            vz = v

        # Mx

        d = np.kron(np.ones((self.nz,)), 2 * self.dx)
        hi = np.kron(np.ones((self.nz,)), self.hx[1:])
        him1 = np.kron(np.ones((self.nz,)), self.hx[:-1])
        tmp = np.ones((self.nx,), dtype=bool)
        tmp[0] = 0
        ind1 = np.kron(np.ones((self.nz,), dtype=bool), tmp)
        tmp[0] = 1
        tmp[-1] = 0
        ind2 = np.kron(np.ones((self.nz,), dtype=bool), tmp)
        if harmon:
            Mx = d / (hi / vx[ind1] + him1 / vx[ind2])
        else:
            Mx = (hi * vx[ind1] + him1 * vx[ind2]) / d

        # Mz

        d = np.kron(2 * self.dz, np.ones((self.nx,)))
        hi = np.kron(self.hz[1:], np.ones((self.nx,)))
        him1 = np.kron(self.hz[:-1], np.ones((self.nx,)))
        ind1 = np.hstack(
            (np.zeros((self.nx,), dtype=bool), np.ones((self.nx * (self.nz - 1),), dtype=bool))
        )
        ind2 = np.hstack(
            (np.ones((self.nx * (self.nz - 1),), dtype=bool), np.zeros((self.nx,), dtype=bool))
        )
        if harmon:
            Mz = d / (hi / vz[ind1] + him1 / vz[ind2])
        else:
            Mz = (hi * vz[ind1] + him1 * vz[ind2]) / d

        # assemblage

        return sp.coo_array((np.hstack((Mx, Mz)), (np.arange(self.nf), np.arange(self.nf)))).tocsr()

    def build_G(self, v=None):
        """Construction of gradient matrix.

        Parameters
        ----------
        v : ndarray, optional
            if None, gradient for a scalar defined at the voxel centre,
            otherwise, gradient of vector v defined on the edges.
            (dVx/dx, dVz/dz).

        Returns
        -------
        csr_matrix
            Gradient matrix
        """
        if v is None:
            # centers to faces
            return self._gradient_centre()
        else:
            return self._gradient_faces(v)

    def _gradient_centre(self):

        # Gx

        M = self.nx - 1
        N = self.nx
        i = np.hstack((np.arange(M), np.arange(M)))
        j = np.hstack((np.arange(M), np.arange(1, N)))
        nval = i.size
        ii = np.zeros((self.nz * nval,), dtype=np.int64)
        jj = np.zeros((self.nz * nval,), dtype=np.int64)
        for n in np.arange(self.nz):
            ii[(n * nval) : ((n + 1) * nval)] = i + n * M
            jj[(n * nval) : ((n + 1) * nval)] = j + n * N
        s = np.tile(np.hstack((-1.0 / self.dx, 1.0 / self.dx)), (self.nz,))
        Gx = sp.coo_array((s, (ii, jj)))

        # Gz

        i = np.hstack((np.arange(self.nfz), np.arange(self.nfz)))
        j = np.hstack((np.arange(self.nfz), self.nx + np.arange(self.nfz)))
        s = np.hstack(
            (
                np.kron(-1.0 / self.dz, np.ones((self.nx,))),
                np.kron(1.0 / self.dz, np.ones((self.nx,))),
            )
        )
        Gz = sp.coo_array((s, (i, j)))

        # assemblage
        return sp.vstack((Gx, Gz), format="csr")

    def _gradient_faces(self, v):

        v_x = sp.diags_array(v[: self.nfx])
        v_z = sp.diags_array(v[(self.nfx) :])

        # Gx

        dVf = 0.5 * (self.hx[1:] + self.hx[:-1])

        M = self.nx - 1
        N = self.nx
        i = np.hstack((np.arange(M), np.arange(M)))
        j = np.hstack((np.arange(M), np.arange(1, N)))
        nval = i.size
        ii = np.zeros((self.nz * nval,), dtype=np.int64)
        jj = np.zeros((self.nz * nval,), dtype=np.int64)
        for n in np.arange(self.nz):
            ii[(n * nval) : ((n + 1) * nval)] = i + n * M
            jj[(n * nval) : ((n + 1) * nval)] = j + n * N
        s = np.tile(np.hstack((0.5 * self.hx[:-1] / dVf, 0.5 * self.hx[1:] / dVf)), (self.nz,))
        Gx = v_x @ sp.coo_array((s, (ii, jj)))

        # Gz

        dVf = 0.5 * (self.hz[1:] + self.hz[:-1])

        i = np.hstack((np.arange(self.nfz), np.arange(self.nfz)))
        j = np.hstack((np.arange(self.nfz), self.nx + np.arange(self.nfz)))
        s = np.hstack(
            (
                np.kron(0.5 * self.hz[:-1] / dVf, np.ones((self.nx,))),
                np.kron(0.5 * self.hz[1:] / dVf, np.ones((self.nx,))),
            )
        )
        Gz = v_z @ sp.coo_array((s, (i, j)))

        # assemblage
        return sp.vstack((Gx, Gz), format="csr")

    def build_G_faces(self):
        """Construction of gradient matrix to use with vector defined on edges.

        Returns
        -------
        csr_matrix
            Gradient matrix
        """
        # Gx

        dVf = 0.5 * (self.hx[1:] + self.hx[:-1])

        M = self.nx - 1
        N = self.nx
        i = np.hstack((np.arange(M), np.arange(M)))
        j = np.hstack((np.arange(M), np.arange(1, N)))
        nval = i.size
        ii = np.zeros((self.nz * nval,), dtype=np.int64)
        jj = np.zeros((self.nz * nval,), dtype=np.int64)
        for n in np.arange(self.nz):
            ii[(n * nval) : ((n + 1) * nval)] = i + n * M
            jj[(n * nval) : ((n + 1) * nval)] = j + n * N
        s = np.tile(np.hstack((0.5 * self.hx[:-1] / dVf, 0.5 * self.hx[1:] / dVf)), (self.nz,))
        Gx = sp.coo_array((s, (ii, jj)))

        # Gz

        dVf = 0.5 * (self.hz[1:] + self.hz[:-1])

        i = np.hstack((np.arange(self.nfz), np.arange(self.nfz)))
        j = np.hstack((np.arange(self.nfz), self.nx + np.arange(self.nfz)))
        s = np.hstack(
            (
                np.kron(0.5 * self.hz[:-1] / dVf, np.ones((self.nx,))),
                np.kron(0.5 * self.hz[1:] / dVf, np.ones((self.nx,))),
            )
        )
        Gz = sp.coo_array((s, (i, j)))

        return sp.vstack((Gx, Gz), format="csr")

    # ------------------------------------------------------------------------------------------------
    # Wavenumber-domain operators for vector fields (2.5D MMR), see docs/mmr25.md
    #
    # Fields are cosine (even in y) or sine (odd in y) transforms, all real:
    #   vector on faces  [Fx (x-faces, cos), Fy (cell centres, sin), Fz (z-faces, cos)],  nf_k values
    #   vector on edges  [Ex (x-edges, sin), Ey (nodes, cos),        Ez (z-edges, sin)],  ne_k values
    # x-edges coincide with z-faces (xc, z node) and z-edges with x-faces (x node, zc).  A derivative
    # along y maps a cosine coefficient F to the sine coefficient -k F, and a sine coefficient F to the
    # cosine coefficient +k F.  Only interior faces, edges and nodes are unknowns (zero on the boundary),
    # as in 3D (GridFV.build_C).
    # ------------------------------------------------------------------------------------------------

    @staticmethod
    def _diff_n(h):
        """Difference between n values at cell centres, divided by h (n-1 values): (n-1) x n."""
        n = h.size + 1
        i = np.r_[np.arange(n - 1), np.arange(n - 1)]
        j = np.r_[np.arange(n - 1), np.arange(1, n)]
        return sp.coo_array((np.r_[-1.0 / h, 1.0 / h], (i, j)), shape=(n - 1, n)).tocsr()

    @staticmethod
    def _diff_c(h):
        """Difference between the n-1 interior nodes (0 on the boundary nodes), divided by h: n x (n-1)."""
        n = h.size
        i = np.r_[np.arange(n - 1), np.arange(1, n)]
        j = np.r_[np.arange(n - 1), np.arange(n - 1)]
        return sp.coo_array((np.r_[1.0 / h[:-1], -1.0 / h[1:]], (i, j)), shape=(n, n - 1)).tocsr()

    def build_G_k(self, k):
        """Gradient of a cosine-transformed scalar at the centres, to faces (nf_k x nc).

        Notes
        -----
        G_k = [Gx; -k I; Gz].  With M_k = build_M_k(sigma), -D_k M_k G_k = -D M G + k^2 sigma is the
        operator of 2.5D DC modelling, and J_k = -M_k G_k U_k is the transformed current density
        (Jy, a sine coefficient, is sigma k U_k).
        """
        I = sp.eye_array(self.nc, format='csr')
        return sp.vstack((self.G[:self.nfx, :], -k * I, self.G[self.nfx:, :]), format='csr')

    def build_D_k(self, k):
        """Divergence of a vector on faces, to the cell centres (nc x nf_k): D_k = [Dx, k I, Dz].

        Notes
        -----
        D_k = -Vc^-1 G_k^T Vf_k, with Vc = volume_voxels() and Vf_k = volume_faces_k().
        """
        I = sp.eye_array(self.nc, format='csr')
        return sp.hstack((self.D[:, :self.nfx], k * I, self.D[:, self.nfx:]), format='csr')

    def build_M_k(self, v, harmon=True):
        """Average of a variable on faces, the y component (cell centres) being the variable itself.

        Parameters
        ----------
        v : ndarray or tuple of 3 ndarray
            Variable at the centre of voxels, or diagonal terms (v_xx, v_yy, v_zz) of a tensor.
        harmon : bool, optional
            Harmonic (True) or arithmetic average on the x and z faces.

        Returns
        -------
        sparse array, nf_k x nf_k (diagonal)
        """
        if type(v) is tuple:
            if len(v) != 3:
                raise ValueError('build_M_k: tensor needs 3 terms (v_xx, v_yy, v_zz)')
            M = self.build_M((v[0], v[2]), harmon).diagonal()
            vy = v[1]
        else:
            M = self.build_M(v, harmon).diagonal()
            vy = v
        vy = vy * np.ones((self.nc,))
        return sp.diags_array(np.r_[M[:self.nfx], vy, M[self.nfx:]], format='csr')

    def build_C_k(self, k, to_faces=True):
        """Curl of a transformed vector field.

        Parameters
        ----------
        k : float
            Wavenumber.
        to_faces : bool, optional
            If True, curl of a vector on edges, evaluated on faces (nf_k x ne_k); otherwise curl of a
            vector on faces, evaluated on edges (ne_k x nf_k).

        Notes
        -----
        Faces to edges, for A = [Ax (cos), Ay (sin), Az (cos)]:
            Bx = -k Az - dAy/dz,   By = dAx/dz - dAz/dx,   Bz = dAy/dx + k Ax
        Edges to faces, for B = [Bx (sin), By (cos), Bz (sin)]:
            Hx = k Bz - dBy/dz,    Hy = dBx/dz - dBz/dx,   Hz = dBy/dx - k Bx
        With Ve_k = volume_edges_k() and Vf_k = volume_faces_k(), Ve_k C_f = (Vf_k C_e)^T.
        """
        nx, nz = self.nx, self.nz
        if to_faces:
            Ix = sp.eye_array(nx - 1, format='csr')
            Iz = sp.eye_array(nz - 1, format='csr')
            dBy_dz = sp.kron(self._diff_c(self.hz), Ix)    # nodes -> x-faces
            dBy_dx = sp.kron(Iz, self._diff_c(self.hx))    # nodes -> z-faces
            dBx_dz = self.D[:, self.nfx:]                  # x-edges (= z-faces) -> centres
            dBz_dx = self.D[:, :self.nfx]                  # z-edges (= x-faces) -> centres
            kx = k * sp.eye_array(self.nfx, format='csr')
            kz = k * sp.eye_array(self.nfz, format='csr')
            return sp.block_array([[None, -dBy_dz, kx],
                                   [dBx_dz, None, -dBz_dx],
                                   [-kz, dBy_dx, None]], format='csr')
        else:
            Ix = sp.eye_array(nx - 1, format='csr')
            Iz = sp.eye_array(nz - 1, format='csr')
            dAx_dz = sp.kron(self._diff_n(self.dz), Ix)    # x-faces -> nodes
            dAz_dx = sp.kron(Iz, self._diff_n(self.dx))    # z-faces -> nodes
            dAy_dz = self.G[self.nfx:, :]                  # centres -> x-edges (= z-faces)
            dAy_dx = self.G[:self.nfx, :]                  # centres -> z-edges (= x-faces)
            kx = k * sp.eye_array(self.nfx, format='csr')
            kz = k * sp.eye_array(self.nfz, format='csr')
            return sp.block_array([[None, -dAy_dz, -kz],
                                   [dAx_dz, None, -dAz_dx],
                                   [kx, dAy_dx, None]], format='csr')

    def build_L_k(self, k):
        """Vector Helmholtz operator C_e C_f - G_k D_k, i.e. -lap + k^2 (nf_k x nf_k).

        Notes
        -----
        Discrete counterpart of curl curl - grad div, as in 3D MMR modelling (GridMMR._build_A, without
        the factor 1/mu0).  The components are uncoupled, and Vf_k L_k is symmetric positive definite
        for k > 0.
        """
        return (self.build_C_k(k) @ self.build_C_k(k, to_faces=False) -
                self.build_G_k(k) @ self.build_D_k(k)).tocsr()

    def volume_faces_k(self):
        """Dual volumes of the components of a vector on faces (x-faces, cell centres, z-faces)."""
        return np.r_[self.volume_faces()[:self.nfx], self.volume_voxels(), self.volume_faces()[self.nfx:]]

    def volume_edges_k(self):
        """Dual volumes of the components of a vector on edges (x-edges, nodes, z-edges)."""
        return np.r_[np.kron(self.dz, self.hx), np.kron(self.dz, self.dx), np.kron(self.hz, self.dx)]

    def below_surface(self, pts):
        """Return indices of electrodes that are not at the surface."""
        if pts.shape[1] == 3:
            iz = 2
        else:
            iz = 1
        return pts[:, iz] != self.z[-1]

    def _with_images(self, pts):
        """Return True for electrodes whose analytic field must include the image source.

        Notes
        -----
        Surface electrodes are moved to the depth of the first cell centre (process_surface_elec).  If
        surface_images is True, their analytic field includes the image above the surface, so that it
        satisfies the no-flux condition at z = 0 as the discrete solution does.  Without the image, the
        source term of the boundary correction contains a spurious flux at the surface, which yields an
        error of order h in heterogeneous models.
        """
        if self.surface_images:
            return np.ones((pts.shape[0],), dtype=bool)
        return self.below_surface(pts)

    def process_surface_elec(self, elec):
        """Process surface Tx coordinate to make sure electrodes are at least at the depth of the first cell center.

        Parameters
        ----------
        elec : ndarray
            coordinates of electrodes, n x 3

        Returns
        -------
        ndarray
            coordinates after correction
        Note
        ----
        Vertical axis is elevation, ie positive upwards.
        """
        if elec.shape[1] == 3:
            iz = 2
        else:
            iz = 1
        ind = elec[:, iz] > self.zc[-1]
        elec[ind, iz] = self.zc[-1]
        return elec

    def solve(self, b, return_k=False):
        """Solve for the potential in the spatial domain.

        Parameters
        ----------
        b : array_like or KSource
            Source term, the same for all wavenumbers (array) or one per wavenumber (KSource)
        return_k : bool, optional
            Also return the fields of each wavenumber (see Solver25.solve)
        """
        if not self.solver_A.same_kg(self.k, self.g):
            # wavenumbers changed since the solvers were built (e.g. new survey with no new model)
            self.set_system(self.solver_A.A, keep_solver=False, verbose=self.solver_A.verbose)
        return self.solver_A.solve(b, return_k=return_k)

    def compute_u_homog(self, c1c2, sigma, cs):
        """3D potential of a homogeneous half-space at the centre of the voxels (y = 0).

        Parameters
        ----------
        c1c2 : ndarray
            Injection electrodes, n x 6 (dipoles) or n x 3 (poles).
        sigma : float or ndarray
            Conductivity, scalar, or one value per current electrode (n x 2 for dipoles, n for poles).
        cs : ndarray
            Current intensity, n.
        """
        return self._u_homog(c1c2.copy(), sigma, cs, lambda r: 1. / r)

    def compute_U_homog_k(self, c1c2, sigma, cs, k):
        """Fourier-domain potential of a homogeneous half-space at the centre of the voxels, wavenumber k.

        Notes
        -----
        Same as compute_u_homog with 1/r replaced by 2 K0(k r), so that 1/2 sum_i g_i U(k_i) = u.
        """
        return self._u_homog(c1c2.copy(), sigma, cs, lambda r: 2. * k0(k * r))

    def boundary_source(self, c1c2, sigma0, cs, correct_quadrature=True, k=None):
        """Source terms of the boundary correction, one per wavenumber.

        Parameters
        ----------
        c1c2 : ndarray
            Coordinates of injection electrodes (n x 6 for dipoles, n x 3 for poles).
        sigma0 : float or ndarray
            Conductivity of the reference model, scalar or one value per current electrode
            (n x 2 for dipoles, n for poles).
        cs : ndarray
            Current intensity.
        correct_quadrature : bool, optional
            Keep the analytic fields so that the error of the inverse Fourier transform applies to the
            secondary field only (the solution is exact for the homogeneous half-space).
        k : array_like, optional
            Wavenumbers (default: self.k).

        Returns
        -------
        tuple
            KSource, and analytic 3D potential u0 at the voxel centres (nc x n)

        Notes
        -----
        Correction of Pidlisecky et al. (2007) applied in the wavenumber domain: q_k = L_k(sigma0) U0_k,
        with U0_k the analytic solution and L_k = sigma0 (-D G + k^2).  Because U0_k is proportional to
        1/sigma0, q_k does not depend on sigma0 (nor does the solution); sigma0 only defines the reference
        fields U0_k and u0 used to correct the quadrature and the interpolation at the measurement electrodes.
        These corrections work best when the reference fields have the same singular behaviour as the
        solution near the current electrodes, i.e. with sigma0 equal to the conductivity at each electrode.
        """
        L1 = -(self.D @ self.G)
        I = sp.eye_array(self.nc, self.nc, format='csr')
        k_all = self.k if k is None else np.atleast_1d(np.asarray(k, dtype=float))
        q = np.empty((k_all.size, self.nc, c1c2.shape[0]))
        U0 = np.empty(q.shape) if correct_quadrature else None
        for i, k in enumerate(k_all):
            q[i] = (L1 + (k * k) * I) @ self.compute_U_homog_k(c1c2, 1.0, cs, k)
            if correct_quadrature:
                U0[i] = self.compute_U_homog_k(c1c2, sigma0, cs, k)
        u0 = self.compute_u_homog(c1c2, sigma0, cs)
        return KSource(q, U0, u0 if correct_quadrature else None), u0

    # ------------------------------------------------------------------------------------------------
    # Electrodes on conductivity contrasts (GridDC.contact_interp)
    #
    # Measurement electrodes: interpolation of the cell-centred potential with continuity of the normal
    # current (flux_interp).  Current electrodes: reference model of the boundary correction made of two
    # quarter-spaces separated by the vertical face nearest to the electrode (contact_reference).

    def _stencil(self, c, p):
        """Indices of the 2 nodes of c bracketing each p, as in linear_interp (extrapolation at the ends)."""
        im = np.argmin(np.abs(p[:, None] - c[None, :]), axis=1)
        lower = ((p <= c[im]) & (im > 0)) | (im == c.size - 1)
        i0 = np.where(lower, im - 1, im)
        return i0, i0 + 1

    @staticmethod
    def _flux_weights(p, ca, cb, f, sa, sb):
        """Weights of cells a and b for the potential at p, with continuity of the current at face f.

        The potential varies linearly in each half-cell, the potential at the face being
        (sa ua / da + sb ub / db) / (sa / da + sb / db).  Linear extrapolation outside [ca, cb].  Also
        returns phi, 1 at the face and 0 at the cell centres.  sa and sb may be complex (complex-step
        derivatives).
        """
        da, db = f - ca, cb - f
        ka, kb = sa / da, sb / db
        wfa, wfb = ka / (ka + kb), kb / (ka + kb)
        left = p <= f
        t = np.where(left, (p - ca) / da, (p - f) / db)
        wa = np.where(left, 1 - t + t * wfa, (1 - t) * wfa)
        wb = np.where(left, t * wfb, (1 - t) * wfb + t)
        phi = np.where(left, t, 1 - t)
        out = (p < ca) | (p > cb)
        t = (p - ca) / (cb - ca)
        wa = np.where(out, 1 - t, wa)
        wb = np.where(out, t, wb)
        phi = np.where(out, 0.0, phi)
        return wa, wb, phi

    def _flux_local(self, x, z, s):
        """Weights of the 4 cells of the stencil and contrast factor (see flux_interp).

        s : n x 4 conductivity of cells (ix0, iz0), (ix1, iz0), (ix0, iz1), (ix1, iz1), may be complex
        """
        ix0, ix1 = self._stencil(self.xc, x)
        iz0, iz1 = self._stencil(self.zc, z)
        xa, xb, xf = self.xc[ix0], self.xc[ix1], self.x[ix1]
        za, zb, zf = self.zc[iz0], self.zc[iz1], self.z[iz1]
        w = []
        c = 1.0
        s_row = []
        for r in range(2):
            sa, sb = s[:, 2 * r], s[:, 2 * r + 1]
            wa, wb, phi = self._flux_weights(x, xa, xb, xf, sa, sb)
            w.append((wa, wb))
            s_row.append(wa * sa + wb * sb)
            c = c * (1 - phi * (sa - sb)**2 / (sa + sb)**2)
        wa, wb, phi = self._flux_weights(z, za, zb, zf, s_row[0], s_row[1])
        c = c * (1 - phi * (s_row[0] - s_row[1])**2 / (s_row[0] + s_row[1])**2)
        W = np.stack((wa * w[0][0], wa * w[0][1], wb * w[1][0], wb * w[1][1]), axis=1)
        return W, c

    def flux_interp(self, x, z, sigma, derivatives=False):
        """Interpolation of the cell-centred potential accounting for conductivity contrasts.

        Like linear_interp, but in each direction the potential varies linearly in each half-cell, with
        continuity of the normal current at the face (flux continuity).  Same as linear_interp in a
        homogeneous model.  At a contact, the potential at the face is close to the value of the most
        conductive cell, whereas linear_interp takes the average of the 2 cells.

        Parameters
        ----------
        x, z : ndarray
            Coordinates of the points (at most at the depth of the first cell centre, see
            process_surface_elec).  Points at infinity give a row of zeros.
        sigma : ndarray
            Conductivity model.
        derivatives : bool, optional
            Also return the derivatives with respect to ln sigma of the 4 cells of the stencil.

        Returns
        -------
        Q : csr_array, npts x nc
            Interpolation matrix.
        c : ndarray
            Contrast factor in ]0, 1]: product over the faces of the stencil of 1 - phi (sa - sb)^2 / (sa + sb)^2,
            sa and sb being the conductivities on either side of the face and phi = 1 at the face, 0 at the
            cell centres.  1 in a homogeneous model; (4 sa sb / (sa + sb)^2)^2 for a surface electrode on a
            vertical contact (one factor for each of the 2 rows of the stencil).
        cells : ndarray, npts x 4 (if derivatives)
            Cells of the stencil.
        dW : ndarray, npts x 4 x 4 (if derivatives)
            dW[:, i, j] derivative of the weight of cell i with respect to ln sigma of cell j.
        dc : ndarray, npts x 4 (if derivatives)
            Derivative of c with respect to ln sigma of the cells.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        z = np.atleast_1d(np.asarray(z, dtype=float))
        sigma = np.asarray(sigma, dtype=float) * np.ones((self.nc,))
        n = x.size
        ok = np.isfinite(x) & np.isfinite(z)
        xo, zo = x[ok], z[ok]
        ix0, ix1 = self._stencil(self.xc, xo)
        iz0, iz1 = self._stencil(self.zc, zo)
        cells_ok = np.stack((ix0 + iz0 * self.nx, ix1 + iz0 * self.nx,
                             ix0 + iz1 * self.nx, ix1 + iz1 * self.nx), axis=1)
        s = sigma[cells_ok]
        W, c_ok = self._flux_local(xo, zo, s)
        rows = np.repeat(np.where(ok)[0], 4)
        Q = sp.coo_array((W.ravel(), (rows, cells_ok.ravel())), shape=(n, self.nc)).tocsr()
        c = np.ones((n,))
        c[ok] = c_ok
        if not derivatives:
            return Q, c
        cells = np.zeros((n, 4), dtype=np.int64)
        cells[ok] = cells_ok
        dW = np.zeros((n, 4, 4))
        dc = np.zeros((n, 4))
        h = 1e-30
        for j in range(4):
            sj = s.astype(complex)
            sj[:, j] *= np.exp(1j * h)      # complex step on ln sigma_j
            Wj, cj = self._flux_local(xo, zo, sj)
            dW[ok, :, j] = Wj.imag / h
            dc[ok, j] = cj.imag / h
        return Q, c, cells, dW, dc

    def contact_reference(self, poles, sigma):
        """Reference model of the boundary correction for each current electrode.

        Two quarter-spaces separated by the vertical face (node of x) nearest to the electrode, with the
        conductivities sa and sb of the 2 cells on either side of the face, in the row of the electrode.
        For an electrode on the face, the reference field is that of a half-space of conductivity
        (sa + sb) / 2, which is the exact solution for an electrode on a vertical contact.  For an
        electrode beyond the outer nodes, a homogeneous half-space with the conductivity of its cell.

        Parameters
        ----------
        poles : ndarray
            Current electrodes, n x 3 (rows at infinity allowed).
        sigma : float or ndarray
            Conductivity model (float: homogeneous reference with that conductivity).

        Returns
        -------
        ContactRef
        """
        poles = np.atleast_2d(np.asarray(poles, dtype=float))
        n = poles.shape[0]
        ok = np.all(np.isfinite(poles), axis=1)
        e = self.process_surface_elec(poles.copy())
        ia = np.full((n,), -1, dtype=np.int64)
        ib = np.full((n,), -1, dtype=np.int64)
        xf = np.zeros((n,))
        if np.any(ok):
            j = np.argmin(np.abs(e[ok, 0][:, None] - self.x[None, :]), axis=1)
            iz = np.clip(np.searchsorted(self.z, e[ok, 2]) - 1, 0, self.nz - 1)
            ixa = np.clip(j - 1, 0, self.nx - 1)
            ixb = np.clip(j, 0, self.nx - 1)
            ia[ok] = ixa + iz * self.nx
            ib[ok] = ixb + iz * self.nx
            xf[ok] = self.x[j]
        if np.isscalar(sigma) or np.size(sigma) == 1:
            sa = sb = float(sigma) * np.ones((n,))
            homogeneous = True
        else:
            sigma = np.asarray(sigma, dtype=float).ravel()
            sa = np.where(ok, sigma[np.maximum(ia, 0)], 1.0)
            sb = np.where(ok, sigma[np.maximum(ib, 0)], 1.0)
            homogeneous = False
        return ContactRef(ia, ib, sa, sb, xf, homogeneous)

    def _ref_field(self, pole, ref, i, cs, kernel=None, pts=None, derivatives=False, offset_y=False):
        """Field of current electrode pole (3,) in its reference model (row i of ref), at the cell centres
        (kernel: 1/r or 2 K0(k r), or a list of kernels: the fields are then n_kernels x nc) or at the points
        pts (3D potential, see u_homog_at).

        With pts, pole can also be n x 3 (one position per point, same x and z, e.g. different y with
        offset_y).  With derivatives, returns (u, du/d ln sa, du/d ln sb).
        """
        multi = pts is None and not callable(kernel)
        if not np.all(np.isfinite(pole)):
            z = np.zeros((self.nc if pts is None else pts.shape[0],))
            if multi:
                z = np.zeros((len(kernel), self.nc))
            return (z, z, z) if derivatives else z
        sa, sb, xf = ref.sa[i], ref.sb[i], ref.xf[i]
        img = np.array(pole, dtype=float)
        img[..., 0] = 2 * xf - img[..., 0]
        if pts is None:
            kernels = kernel if multi else [kernel]
            with_img = sa != sb or derivatives
            F = self._u_homog_poles(np.vstack((pole, img)) if with_img else pole[None, :],
                                    cs * np.ones((2,)), kernels)
            h = F[..., 0] if multi else F[0, :, 0]
            hi = (F[..., 1] if multi else F[0, :, 1]) if with_img else 0.0
            x = self.centre_voxels()[0]
        else:
            m = pts.shape[0]
            h = self.u_homog_at(np.broadcast_to(pole, (m, 3)), pts, 1.0, cs, offset_y=offset_y, check=False)
            # the image is on the other side of the face: a point there may coincide with it (unused term)
            hi = self.u_homog_at(np.broadcast_to(img, (m, 3)), pts, 1.0, cs, offset_y=offset_y, check=False) \
                if sa != sb or derivatives else 0.0
            x = pts[:, 0]
        left = np.ravel(pole)[0] <= xf
        ss, so = (sa, sb) if left else (sb, sa)
        same = x <= xf if left else x > xf
        S = sa + sb
        kap = (ss - so) / S
        hi = np.where(same, hi, 0.0)
        u = np.where(same, (h + kap * hi) / ss, 2 * h / S)
        if pts is not None and np.any(np.isinf(u)):
            raise ValueError('Evaluation point at the position of a current electrode')
        if not derivatives:
            return u
        dk = 2 * ss * so / S**2               # d kappa / d ln ss = - d kappa / d ln so
        du_ss = np.where(same, -(h + kap * hi) / ss + hi * dk / ss, -2 * h * ss / S**2)
        du_so = np.where(same, -hi * dk / ss, -2 * h * so / S**2)
        return (u, du_ss, du_so) if left else (u, du_so, du_ss)

    def _ref_sigma(self, ref, i):
        """Conductivity of the reference model of row i of ref, and mask of the cells of conductivity sa."""
        mask_a = self.centre_voxels()[0] <= ref.xf[i]
        return np.where(mask_a, ref.sa[i], ref.sb[i]), mask_a

    def contact_source(self, c1c2, refs, cs, correct_quadrature=False, k=None):
        """Source terms of the boundary correction with the reference models of contact_reference.

        Same as boundary_source, with q_k = L_k(sigma_ref) U0_k for each current electrode, sigma_ref and
        U0_k being the conductivity and the field of its reference model.

        Parameters
        ----------
        c1c2 : ndarray
            Current electrodes, n x 6 (dipoles) or n x 3 (poles).
        refs : list of ContactRef
            Reference models of the electrodes c1 (and c2).
        cs : ndarray
            Current intensity.

        Returns
        -------
        tuple
            KSource, and the 3D field of the reference models u0 at the voxel centres (nc x n)
        """
        k_all = self.k if k is None else np.atleast_1d(np.asarray(k, dtype=float))
        n = c1c2.shape[0]
        cs = cs * np.ones((n,))
        q = np.zeros((k_all.size, self.nc, n))
        U0 = np.zeros(q.shape) if correct_quadrature else None
        u0 = np.zeros((self.nc, n))
        for e, sign in ((0, 1.0), (1, -1.0)):
            if e >= len(refs):
                break
            ref = refs[e]
            kernels = [lambda r, kk=kk: 2. * k0(kk * r) for kk in k_all] + [lambda r: 1. / r]
            for i in range(n):
                pole = c1c2[i, 3 * e:3 * e + 3]
                if not np.all(np.isfinite(pole)):
                    continue
                s_ref, _ = self._ref_sigma(ref, i)
                m_ref = self.build_M(s_ref).diagonal()
                F = self._ref_field(pole, ref, i, cs[i], kernel=kernels)
                for ik, kk in enumerate(k_all):
                    q[ik, :, i] += sign * (-(self.D @ (m_ref * (self.G @ F[ik]))) + kk * kk * s_ref * F[ik])
                    if correct_quadrature:
                        U0[ik, :, i] += sign * F[ik]
                u0[:, i] += sign * F[-1]
        return KSource(q, U0, u0 if correct_quadrature else None), u0

    def contact_source_derivative(self, pole, ref, i, cs, k, Gf=None):
        """Derivative of the source term q_k of current electrode pole with respect to ln sa (row i of ref).

        q_k depends on the ratio sa / sb only: d q_k / d ln sb = - d q_k / d ln sa.  Gf: build_G_faces().

        Returns
        -------
        ndarray, nc (k scalar) or n_k x nc
        """
        s_ref, mask_a = self._ref_sigma(ref, i)
        M = self.build_M(s_ref).diagonal()
        if Gf is None:
            Gf = self.build_G_faces()
        # d m_f / d ln sigma_i = m_f^2 w_fi / sigma_i (harmonic average)
        dM = M**2 * (Gf @ (mask_a / s_ref))
        ks = np.atleast_1d(np.asarray(k, dtype=float))
        U, dU, _ = self._ref_field(pole, ref, i, cs, kernel=[lambda r, kk=kk: 2. * k0(kk * r) for kk in ks],
                                   derivatives=True)
        dq = np.empty((ks.size, self.nc))
        for n, kk in enumerate(ks):
            dq[n] = (-(self.D @ (dM * (self.G @ U[n]))) + kk * kk * s_ref * mask_a * U[n]
                     - (self.D @ (M * (self.G @ dU[n]))) + kk * kk * s_ref * dU[n])
        return dq[0] if np.isscalar(k) else dq

    @staticmethod
    def _electrode_sigma(sigma, n, dipole):
        """Conductivity at each current electrode, returns 2 arrays of size n."""
        s = np.asarray(sigma, dtype=float)
        if s.ndim == 0:
            s1 = s2 = s * np.ones((n,))
        elif dipole:
            s = s.reshape(n, 2)
            s1, s2 = s[:, 0], s[:, 1]
        else:
            s1 = s2 = s.reshape(n)
        return s1, s2

    def u_homog_at(self, c1c2, pts, sigma, cs, offset_y=False, check=True):
        """3D potential of a homogeneous half-space, source of row i evaluated at point of row i.

        Parameters
        ----------
        c1c2 : ndarray
            Injection electrodes, n x 6 (dipoles) or n x 3 (poles).
        pts : ndarray
            Evaluation points, n x 3.
        sigma : float or ndarray
            Conductivity, scalar or one value per current electrode (n x 2 for dipoles, n for poles).
        cs : ndarray
            Current intensity, n.
        offset_y : bool
            Include the offset along y between the electrode and the point in the distance.  False
            (GridDC): the distance is in the x-z plane, consistent with the numerical 2.5D solution,
            computed for an offset of 0.  True (pod_e of GridMMR): true 3D distance.
        check : bool
            Raise an error if a point is at the position of a current electrode (inf otherwise).

        Notes
        -----
        Exact solution at the true positions of the electrodes and points (surface electrodes are not
        moved to the depth of the first cell centre and have no image), used as primary field at the
        measurement electrodes.
        """
        c1c2 = np.atleast_2d(np.array(c1c2, dtype=float))
        pts = np.atleast_2d(np.array(pts, dtype=float))
        cs = cs * np.ones((c1c2.shape[0],))
        s1, s2 = self._electrode_sigma(sigma, c1c2.shape[0], c1c2.shape[1] == 6)

        def pot(c, s):
            """Field of one electrode (image and gf = 4 if it is below the surface)."""
            below = self.below_surface(c)
            with np.errstate(divide='ignore', invalid='ignore'):
                h2 = (pts[:, 0] - c[:, 0])**2
                if offset_y:
                    h2 = h2 + (pts[:, 1] - c[:, 1])**2
                r = np.sqrt(h2 + (pts[:, 2] - c[:, 2])**2)
                ri = np.sqrt(h2 + (pts[:, 2] + c[:, 2] - 2 * self.z[-1])**2)
                u = np.where(below, (1. / r + 1. / ri) / 4.0, 1. / (2.0 * r)) / s
            # electrode or point at infinity (e.g. pole-pole): no contribution (avoids inf - inf)
            remote = np.logical_not(np.logical_and(np.all(np.isfinite(c), axis=1),
                                                   np.all(np.isfinite(pts), axis=1)))
            u[remote] = 0.0
            return u

        # sum of the fields of the electrodes, as in _u_homog
        u = pot(c1c2[:, :3], s1)
        if c1c2.shape[1] == 6:
            u = u - pot(c1c2[:, 3:6], s2)
        u = cs / np.pi * u
        if check and np.any(np.isinf(u)):
            raise ValueError('Evaluation point at the position of a current electrode')
        return u

    def _u_homog(self, c1c2, sigma, cs, kernel):
        """Potential of a homogeneous half-space, with kernel(r) = 1/r (3D) or 2 K0(k r) (Fourier domain)."""
        x, _, z = self.centre_voxels()
        u0 = np.empty((self.nc, c1c2.shape[0]))
        s1, s2 = self._electrode_sigma(sigma, c1c2.shape[0], c1c2.shape[1] == 6)
        # turn off warning b/c we get a divide by zero that we will fix later
        if c1c2.shape[1] == 6:
            # dipole = difference of the fields of the two electrodes, each with its own treatment of the
            # singularity (as when the fields are computed per electrode, for the sensitivity)
            return (self._u_homog(c1c2[:, :3].copy(), s1, cs, kernel) -
                    self._u_homog(c1c2[:, 3:6].copy(), s2, cs, kernel))

        if c1c2.shape[1] != 3:
            raise ValueError("c1c2 must be either n x 3 or n x 6")
        return self._u_homog_poles(c1c2, np.asarray(cs, dtype=float) / s1, [kernel])[0]

    def _u_homog_poles(self, poles, cs, kernels):
        """Fields of poles (n x 3) in a half-space of unit conductivity, for each kernel (see _u_homog).

        Returns
        -------
        ndarray, n_kernels x nc x n
        """
        x, _, z = self.centre_voxels()
        # keep track of electrodes below the surface
        below = self._with_images(poles)
        # make sure electrodes are at least at the depth of the first cell center
        poles = self.process_surface_elec(np.array(poles, dtype=float))
        u0 = np.empty((len(kernels), self.nc, poles.shape[0]))
        with np.errstate(divide='ignore'):
            for i in range(poles.shape[0]):
                r = np.sqrt((x - poles[i, 0]) ** 2 + (z - poles[i, 2]) ** 2)
                # image of the electrode above the surface
                ri = np.sqrt((x - poles[i, 0]) ** 2 + (z + poles[i, 2] - 2 * self.z[-1]) ** 2) if below[i] else None
                gf = 4.0 if below[i] else 2.0
                for n, kernel in enumerate(kernels):
                    v = kernel(r) if ri is None else kernel(r) + kernel(ri)
                    u0[n, :, i] = cs[i] / (gf * np.pi) * v

        # singularities due to a source at the centre of a cell: mean of the finite values of the
        # neighbouring cells (the two electrodes of a dipole can be in adjacent cells)
        for i in range(poles.shape[0]):
            for j in np.nonzero(np.isinf(u0[0, :, i]))[0]:
                ix, _, iz = self.revind(j)
                nb = [self.ind(ix + dx, 0, iz + dz) for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))
                      if 0 <= ix + dx < self.nx and 0 <= iz + dz < self.nz]
                for n in range(len(kernels)):
                    v = u0[n, nb, i]
                    u0[n, j, i] = np.mean(v[np.isfinite(v)])

        return u0

    def optimize_k_g(self, num, r_min=None, r_max=None):
        """Optimize wavenumbers and weights of the inverse Fourier transform.

        Parameters
        ----------
        num : int
            Number of wavenumbers.
        r_min : float, optional
            Smallest distance for which the transform should be accurate.  Default: smallest cell size / 4.
        r_max : float, optional
            Largest distance for which the transform should be accurate.  Default: 4 x grid length.

        Notes
        -----
        Weights are normalized such that sum_i g_i K0(k_i r) ~ 1/r, as the default values (approach of
        SimPEG, least-squares fit at log-spaced distances).

        Calling this method sets auto_kg to False, i.e. the wavenumbers are not optimized automatically
        for the survey at the next forward modelling.
        """
        self._optimize_k_g(num, r_min, r_max)
        self.auto_kg = False

    def _optimize_k_g(self, num, r_min=None, r_max=None):
        def get_phi(r):
            e = np.ones_like(r)

            def phi(k):
                # use log10 transform to enforce positivity
                k = 10 ** k
                A = r[:, None] * k0(r[:, None] * k)
                v_i = A @ np.linalg.solve(A.T @ A, A.T @ e)
                dv = (e - v_i) / len(r)
                return np.linalg.norm(dv)

            def g(k):
                A = r[:, None] * k0(r[:, None] * k)
                return np.linalg.solve(A.T @ A, A.T @ e)

            return phi, g

        if r_min is None:
            r_min = min(np.r_[self.hx, self.hz]) / 4
        if r_max is None:
            r_max = 4 * (self.x[-1] - self.x[0])
        # generate test points log spaced between these two end members
        rs = np.logspace(np.log10(r_min), np.log10(r_max), 100)

        min_rinv = -np.log10(rs).max()
        max_rinv = -np.log10(rs).min()
        # a decent initial guess of the k_i's for the optimization = 1/rs
        k_i = np.linspace(min_rinv, max_rinv, num)

        # these functions depend on r, so grab them
        func, g_func = get_phi(rs)

        # just use scipy's minimize for ease
        out = minimize(func, k_i)
        if self.verbose and self.comm.rank == 0:
            print(f"optimized ks converged? : {out['success']}")
            print(f"Estimated transform Error: {out['fun']}")
        # transform the solution back to normal points
        self.k = 10 ** out["x"]
        # sum g K0(k r) ~ 1/r; the factor 1/2 is applied in Solver25.solve
        self.g = g_func(self.k)
        self._kg_range = (num, r_min, r_max)

    def print_info(self, file=None):
        print('    Grid: {0:d} x {1:d} voxels'.format(self.nx, self.nz), file=file)
        print('      X min: {0:e}\tX max: {1:e}'.format(self.x[0], self.x[-1]), file=file)
        print('      Z min: {0:e}\tZ max: {1:e}'.format(self.z[0], self.z[-1]), file=file)

    def optimize_k_g0(self, src_term, num):
        if src_term.shape[1] == 6:
            # we have xyz pairs, remove y values
            src_term = src_term[:, (0, 2, 3, 5)]

        # set the maximum number of iterations for the optimization routine
        itsmax = 25

        # Max number of radii to search over
        max_num = 2000

        # Number of linesearch steps
        lsnum = 10
        # Line Search parameters
        #  lower bound
        ls_low_lim = 0.01
        #  upper bound
        ls_up_lim = 1

        # Define observation distances
        rpos = np.array([])
        rneg = np.array([])
        rpos_im = np.array([])
        rneg_im = np.array([])

        # hard-wired search radius for determining k and g.
        x_radius = np.array([0.1, 0.5, 1, 5, 10, 20, 30])
        x_radius = np.r_[np.zeros(x_radius.shape), x_radius]
        z_radius = x_radius[::-1]

        for n in range(src_term.shape[0]):
            xr = x_radius + src_term[n, 0]
            zr = z_radius + src_term[n, 1]

            # norm of positive current electrode and 1st potential electrode
            rpost = np.sqrt((xr - src_term[n, 0])**2 + (zr - src_term[n, 1])**2)

            # norm of negative current electrode and 1st potential electrode
            rnegt = np.sqrt((xr - src_term[n, 2])**2 + (zr - src_term[n, 3])**2)

            # norm of imaginary positive current electrode and 1st potential electrode
            rpos_imt = np.sqrt((xr - src_term[n, 0]) ** 2 + (zr + src_term[n, 1]) ** 2)

            # norm of imaginary negative current electrode and 1st potential electrode
            rneg_imt = np.sqrt((xr - src_term[n, 2]) ** 2 + (zr + src_term[n, 3]) ** 2)

            rpos = np.r_[rpos, rpost]
            rneg = np.r_[rneg, rnegt]
            rpos_im = np.r_[rpos_im, rpos_imt]
            rneg_im = np.r_[rneg_im, rneg_imt]

        # now we remove all non-unique radii
        rtot = np.c_[rpos.reshape(-1, 1), rneg.reshape(-1, 1), rpos_im.reshape(-1, 1), rneg_im.reshape(-1, 1)]
        rtot = np.unique(rtot, axis=0)
        ind = rtot == 0.0
        ind = np.sum(ind, axis=1) == 0
        rtot = rtot[ind, :]

        tnum = rpos.size
        if tnum > max_num:
            step = np.ceil(tnum / max_num)
            rpos = rpos[::step]
            rneg = rneg[::step]
            rpos_im = rpos_im[::step]
            rneg_im = rneg_im[::step]

        # initialize a starting guess for k0
        k0 = np.logspace(-2, 0.5, num)

        # calculate the A matrix
        # set up a matrix of radii
        rinv = 1. / (1. / rtot[:, 0] - 1. / rtot[:, 1] + 1. / rtot[:, 2] - 1. / rtot[:, 3])
        i = np.where(np.isfinite(np.sum(1. / rtot, axis=1) + rinv))[0]
        rtot = rtot[i, :]
        rinv = rinv[i].reshape(-1, 1)

        # Form matrices for computation
        rinv1 = rinv @ np.ones((1, num))
        rpos1 = rtot[:, 0].reshape(-1, 1) @ np.ones((1, num))
        rneg1 = rtot[:, 1].reshape(-1, 1) @ np.ones((1, num))
        rpos_im1 = rtot[:, 2].reshape(-1, 1) @ np.ones((1, num))
        rneg_im1 = rtot[:, 3].reshape(-1, 1) @ np.ones((1, num))

        # identity vector
        I = np.ones((rpos1.shape[0], 1))
        # K values matrix
        Km = I @ k0.reshape(1, -1)

        # Calculate the A matrix
        A = rinv1 * np.real(kv(0, rpos1*Km) - kv(0, rneg1*Km) + kv(0, rpos_im1*Km) - kv(0, rneg_im1*Km))

        # Estimate g for the given K values
        v = A @ np.linalg.solve(A.T @ A, A.T @ I)
        # Evaluate the objective function for the initial guess
        obj = [((1.-v).T.dot(1.-v)).item()]

        # Start counter and initialize the optimization
        its = 0  # iteration counter
        knew = k0.copy().reshape(-1, 1)  # updated k vector
        reduction = 1  # Variable for ensure sufficient decrease between iterations
        #                Optimization terminates if objective function is not
        #                reduced by at least 5% at each iteration

        while obj[-1] > 1.e-5 and its < itsmax and reduction > 0.05:
            # Create the derivative matrix
            dvdk = np.zeros((v.size, num))
            for i in range(num):
                Ktemp = Km.copy()
                Ktemp[:, i] *= 1.01  # 1.05
                A = rinv1 * np.real(kv(0, rpos1 * Ktemp) - kv(0, rneg1 * Ktemp) + kv(0, rpos_im1 * Ktemp) -
                                    kv(0, rneg_im1 * Ktemp))
                L = A.T @ A

                # Estimate g for the given K values
                vT = A @ np.linalg.solve(A.T @ A, A.T @ I)

                # Calculate the derivative for the appropriate column
                dvdk[:, i] = (vT - v).flatten() / (Ktemp[:, i] - Km[:, i])

            # Apply some smallness regularization
            beta = 1.0e-9
            h = dvdk.T @ (I-v) + beta * np.eye(knew.size) @ knew
            dk = np.linalg.solve((dvdk.T @ dvdk + beta * np.eye(knew.size)), h)

            # Perform a line-search to maximize the descent
            ls = np.linspace(ls_low_lim, ls_up_lim, lsnum)
            ls_res = np.empty((ls.size, 2))
            for j in range(lsnum):
                ktemp = knew + ls[j] * dk

                Km = np.ones((rpos1.shape[0], 1)) @ ktemp.T

                A = rinv1 * np.real(kv(0, rpos1 * Km) - kv(0, rneg1 * Km) + kv(0, rpos_im1 * Km) - kv(0, rneg_im1 * Km))
                v = A @ np.linalg.solve(A.T @ A, A.T @ I)
                objt = ((1. - v).T.dot(1. - v)).item()
                if np.isnan(objt):
                    objt = np.inf  # nan is picked as min below...
                ls_res[j, :] = np.c_[objt, ls[j]]

            # Find the smallest objective function from the line-search
            c = ls_res[:, 0].argmin()

            # Create a new guess for k
            knew += ls[c] * dk
            # eval obj funct
            Km = np.ones((rpos1.shape[0], 1)) @ knew.T

            # Calculate the A matrix
            A = rinv1 * np.real(kv(0, rpos1 * Km) - kv(0, rneg1 * Km) + kv(0, rpos_im1 * Km) - kv(0, rneg_im1 * Km))
            v = A @ np.linalg.solve(A.T @ A, A.T @ I)

            obj.append(((1. - v).T.dot(1. - v)).item())
            reduction = obj[its] / obj[its + 1] - 1
            its += 1

            if np.linalg.cond(A.T @ A, p=1) < 1.e-20:
                knew -= ls[c] * dk
                break

        # RMS fit
        err = [np.sqrt(x/rpos.size) for x in obj]
        # final k values
        self.k = np.abs(knew.flatten())
        Km = np.ones((rpos1.shape[0], 1)) @ knew.T

        # reform A to obtain the final g values
        A = rinv1 * np.real(kv(0, rpos1 * Km) - kv(0, rneg1 * Km) + kv(0, rpos_im1 * Km) - kv(0, rneg_im1 * Km))
        self.g = np.linalg.solve(A.T @ A, A.T @ I).flatten()
        self.auto_kg = False
        return obj, err


# %% Solver


class ContactRef:
    """Reference models of current electrodes, see Grid25FV.contact_reference.

    Attributes
    ----------
    ia, ib : ndarray of int
        Cells on either side of the face (-1 for electrodes at infinity).
    sa, sb : ndarray
        Conductivity of the reference model on either side of the face.
    xf : ndarray
        Position of the face.
    homogeneous : bool
        True if the reference conductivity does not come from the model cells (sa = sb = constant).
    """

    def __init__(self, ia, ib, sa, sb, xf, homogeneous=False):
        self.ia = ia
        self.ib = ib
        self.sa = sa
        self.sb = sb
        self.xf = xf
        self.homogeneous = homogeneous


class KSource:
    """Source terms defined for each wavenumber (2.5D modelling).

    Parameters
    ----------
    q : ndarray
        Source terms, n_k x nc x ns.
    U0 : ndarray, optional
        Analytic Fourier-domain fields of the reference model, n_k x nc x ns.
    u0 : ndarray, optional
        Analytic spatial-domain field of the reference model, nc x ns.

    Notes
    -----
    If U0 and u0 are given, the solution is corrected as u = u0 + 1/2 sum_i g_i (U_i - U0_i), so that the
    error of the inverse Fourier transform applies to the secondary field only.

    Supports slicing of the columns, i.e. q[:, cols], as done for sparse source terms.
    """

    def __init__(self, q, U0=None, u0=None):
        self.q = q
        self.U0 = U0
        self.u0 = u0

    @property
    def shape(self):
        return self.q.shape[1:]

    def __getitem__(self, key):
        rows, cols = key
        if rows != slice(None):
            raise IndexError('KSource: only slicing of columns is supported')
        if isinstance(cols, (int, np.integer)):
            cols = [cols]
        return KSource(self.q[:, :, cols],
                       None if self.U0 is None else self.U0[:, :, cols],
                       None if self.u0 is None else self.u0[:, cols])


class Solver25:
    """Solvers for 2.5D modelling, one per wavenumber.

    Parameters
    ----------
    solver_par : tuple
        Solver parameters (see Solver).
    A : sparse array
        Base matrix -D M G.
    sigma : ndarray or float
        Conductivity model.
    k : ndarray
        Wavenumbers.
    g : ndarray
        Weights, normalized such that sum_i g_i K0(k_i r) ~ 1/r.
    verbose : bool, optional
    """

    def __init__(self, solver_par, A, sigma, k, g, verbose=False, comm=None, n_threads=1):
        self.k = np.array(k, dtype=float)
        self.g = np.array(g, dtype=float)
        self.verbose = verbose
        self._A = A
        self.solvers = []
        for n, k_i in enumerate(self.k):
            if verbose:
                print('    Wavenumber {0:d} of {1:d}'.format(n + 1, self.k.size))
            self.solvers.append(Solver(solver_par, self._L(A, sigma, k_i), verbose=verbose,
                                       comm=comm, n_threads=n_threads))

    @staticmethod
    def _L(A, sigma, k):
        s = sigma * np.ones((A.shape[0],))
        return (A + (k * k) * sp.diags_array(s, format='csr')).tocsr()

    @property
    def A(self):
        """Base matrix -D M G."""
        return self._A

    def update(self, A, sigma):
        """Refactorize for a new conductivity model."""
        self._A = A
        for s, k_i in zip(self.solvers, self.k):
            s.A = self._L(A, sigma, k_i)

    def same_kg(self, k, g):
        return (np.size(k) == self.k.size and np.size(g) == self.g.size and
                np.allclose(k, self.k, rtol=1e-12, atol=0) and np.allclose(g, self.g, rtol=1e-12, atol=0))

    def solve(self, rhs, verbose=None, return_k=False):
        """Solve for the potential in the spatial domain, u = 1/2 sum_i g_i U(k_i).

        Parameters
        ----------
        rhs : array_like or KSource
            Source term, the same for all wavenumbers, or one per wavenumber.
        return_k : bool, optional
            Also return the fields in the wavenumber domain U(k_i), n_k x nc x ns (used for sensitivity).
        """
        if verbose is None:
            verbose = self.verbose
        u = None
        U_k = None
        for n, s in enumerate(self.solvers):
            b = rhs.q[n] if isinstance(rhs, KSource) else rhs
            Un = s.solve(b, verbose=verbose)
            if return_k:
                if U_k is None:
                    U_k = np.empty((self.k.size,) + Un.shape)
                U_k[n] = Un
            U = 0.5 * self.g[n] * Un
            if isinstance(rhs, KSource) and rhs.U0 is not None:
                U -= 0.5 * self.g[n] * rhs.U0[n]
            u = U if u is None else u + U
        if isinstance(rhs, KSource) and rhs.u0 is not None:
            u += rhs.u0
        if return_k:
            return u, U_k
        return u

    def set_mumps_print_level(self, level):
        for s in self.solvers:
            s.set_mumps_print_level(level)

    def print_info(self, file=None):
        self.solvers[0].print_info(file)
        print('    2.5D: {0:d} wavenumbers'.format(self.k.size), file=file)


class Solver:
    """
    Class to solve a system Ax = b with a choice of solvers

    Parameters
    ----------
    solver_par : tuple
        Parameters of the solver, 1st element can be str or callable
        If str: name of solver (pardiso, umfpack, or mumps)
        If callable (iterative solver of scipy.sparse.linalg, eg `bicgstab`), the remaining elements are:
        `atol`, `rtol`, `max_it`, `precon`, `do_perm`
    A : spmatrix, optional
        Matrix on left-hand side
    verbose : bool, optional
        Display progress messages
    comm : MPI Communicator or None
        for mumps solver
    """

    def __init__(self, solver_par, A=None, verbose=False, comm=None, n_threads=1):
        self._A = A
        self.precon = '0'
        self.do_perm = False
        self.Mpre = None
        self.perm = None
        self.inv_perm = None
        self.x0 = None
        self.verbose = verbose
        self.ctx = None
        self.want_pardiso = False
        self.want_umfpack = False
        self.want_superlu = False
        self.want_pastix = False
        self.pastix_solver = None
        self.umfpack = None
        self.pardiso = False
        if comm is not None:
            self.verbose = verbose and (comm.rank == 0)
        self.n_threads = n_threads
        if callable(solver_par[0]):
            # solveur itératif

            self.it_slv = solver_par[0]
            self.atol = solver_par[1]
            self.rtol = solver_par[2]
            self.max_it = solver_par[3]
            self.precon = solver_par[4]
            self.do_perm = solver_par[5]

            if A is None:
                return

            if self.do_perm:
                if self.verbose:
                    print("  Applying Inverse Cuthill-McKee Permutation ... ", end="", flush=True)
                self.perm = reverse_cuthill_mckee(A)
                self.inv_perm = np.argsort(self.perm)

                permut = sp.csr_array((np.ones((self.perm.size,)), (self.perm, np.arange(self.perm.size))))
                A = A @ permut
                self._A = (A.T @ permut).T
                if self.verbose:
                    print("done.")

            if self.precon != '0':
                if self.verbose:
                    print("  Computing preconditioning matrix ... ", end="", flush=True)
                if self.precon == 'diag':
                    Ainv = sp.diags_array(1.0 / self.A.diagonal())
                    self.Mpre = sp.linalg.aslinearoperator(Ainv)
                elif self.precon == 'ilu':
                    try:
                        self.Mpre = sp.linalg.spilu(self.A.tocsc())
                        self.Mpre = sp.linalg.LinearOperator(self.A.shape, self.Mpre.solve)
                    except RuntimeError as err:
                        if self.verbose:
                            print(err)
                            print("Switching to using diagonal of A")
                        Ainv = sp.diags_array(1.0 / self.A.diagonal())
                        self.Mpre = sp.linalg.aslinearoperator(Ainv)
                else:
                    raise ValueError("Unknown preconditioning solver")
                if self.verbose:
                    print("done.")

            self.solver = lambda A, b: self.it_slv(A, b, x0=self.x0, atol=self.atol, rtol=self.rtol,
                                                   maxiter=self.max_it, M=self.Mpre)
        elif solver_par[0] == "mumps":
            self.ctx = mumps.DMumpsContext(sym=0, par=1, comm=comm)
            self.ctx.set_icntl(4, 1)  # print only error messages by default
            self.solver = self._solve_mumps
            if A is None:
                return
            A = _canonical_csr(A)   # see A.setter
            self._A = A
            if self.ctx.myid == 0:
                self.ctx.set_centralized_sparse(A)
            if verbose and self.ctx.myid == 0:
                print("    Analyzing & factorizing matrix A ... ", end="", flush=True)
            self._mumps_factorize(job=4)  # Analysis & Factorization
            if verbose and self.ctx.myid == 0:
                print("done.")
            self.solver = self._solve_mumps
        elif solver_par[0] == "pardiso":
            self.want_pardiso = True
            self.solver = pypardiso.spsolve
            self.pardiso = True
        elif solver_par[0] == "pastix":
            self.want_pastix = True
            if A is None:
                return
            self.pastix_solver = pypastix.solver(A)
            self.solver = lambda A, b: self.pastix_solver.solve(b.flatten(), refine=False)
        elif solver_par[0] == "umfpack":
            self.want_umfpack = True
            if A is None:
                return
            umf_family, A = _get_umf_family(A.tocsc().sorted_indices())
            self.umfpack = um.UmfpackContext(umf_family)
            self.umfpack.control[um.UMFPACK_PRL]
            # use_solver(useUmfpack=True, assumeSortedIndices=True)
            if verbose:
                print("    Factorizing matrix A ... ", end="", flush=True)
            self.umfpack.numeric(A)
            # solve = factorized(A.tocsc().sorted_indices())
            if verbose:
                print("done.")
            # self.solver = lambda A, b : solve(np.array(b).flatten())
            umf = self.umfpack

            def slv(AA, b):
                with np.errstate(divide="ignore", invalid="ignore"):
                    # Ignoring warnings with numpy >= 1.23.0, see gh-16523
                    result = umf.solve(um.UMFPACK_A, A, b.flatten(), autoTranspose=True)
                    # umf.report_info()
                return result

            self.solver = slv
        elif solver_par[0] == "superlu":
            self.want_superlu = True
            use_solver(useUmfpack=False)
            if A is None:
                return
            if verbose:
                print("    Factorizing matrix A ... ", end="", flush=True)
            solve = factorized(A.tocsc())
            if verbose:
                print("done.")
            self.solver = lambda A, b: solve(b.flatten())
        else:
            raise RuntimeError("Solver not implemented")

    def __del__(self):

        if self.pastix_solver is not None:
            self.pastix_solver.finalize()
        elif self.umfpack is not None:
            try:
                del self.umfpack
            except AttributeError as e:
                print(e)
        elif self.ctx is not None:
            self.ctx.destroy()

    def set_mumps_print_level(self, level):
        if self.ctx is None:
            return
        self.ctx.set_icntl(4, level)

    def solve(self, rhs, x0=None, verbose=None):
        """
        Solve Ax = b

        Parameters
        ----------
        rhs : array_like
            right hand side term.
        x0 : array_like, optional
            Initial solution.
        verbose : bool, optional
            print info messages.

        Returns
        -------
        spmatrix
            solution

        """

        if verbose is None:
            verbose = self.verbose

        if verbose:
            if self.ctx is None:
                print("      Solving system ...", flush=True)
            elif self.ctx.myid == 0:
                print("      Solving system ...", flush=True)

        # direct solvers mumps & pardiso can take matrices as rhs

        if self.ctx is not None and rhs.ndim == 2:
            assert self.do_perm is False
            if self.ctx.myid == 0:
                if sp.issparse(rhs):
                    x = rhs.toarray(order="F")
                else:
                    # we must have a numpy array
                    x = rhs.copy(order="F")
                self.ctx.set_rhs(x)
            else:
                x = np.empty(rhs.shape)
            self.ctx.run(job=3)  # Solve
            self.ctx.comm.Bcast(x, root=0)
            if self.ctx.myid == 0 and verbose:
                res = np.linalg.norm(rhs - self._A @ x, axis=0)
                print("        Residuals: ", res)
                print("      done.")

            return x
        elif self.pardiso is True and rhs.ndim == 2:
            assert self.do_perm is False
            if sp.issparse(rhs):
                x = rhs.toarray(order="F")
            else:
                # we must have a numpy array
                x = rhs.copy(order="F")
            x = self.solver(self._A, x)
            if verbose:
                res = np.linalg.norm(rhs - self._A @ x, axis=0)
                print("        Residuals: ", res)
                print("      done.")
            return x

        if rhs.ndim == 1:
            rhs = np.atleast_2d(rhs).T

        if sp.issparse(rhs):
            rhs = rhs.toarray()

        sz = rhs.shape[1]
        v = np.empty((self._A.shape[0], sz))
        if is_free_threading():
            chunklen = (sz + self.n_threads - 1) // self.n_threads
            threads = [Thread(target=Solver._chunk_solve, args=(self, x, x + chunklen if x + chunklen < sz else sz, rhs, v, x0, verbose)) for x in range(0, sz, chunklen)]
            for i in threads:
                i.start()
            for i in threads:
                i.join()
        else:
            Solver._chunk_solve(self, 0, sz, rhs, v, x0, verbose)

        if verbose:
            print("      done.")

        return v

    def _chunk_solve(self, n0, n1, rhs, v, x0, verbose):
        for ns in range(n0, n1):
            if verbose:
                if ns == n0:
                    msg = ""
                    pre_msg = ""
                nback = len(msg) - len(pre_msg)
                pre_msg = ""
                for _ in range(nback):
                    pre_msg += "\b"
                msg = pre_msg + "        iteration " + str(ns + 1) + "/" + str(rhs.shape[1]) + " "
                print(msg, end="", flush=True)

            if x0 is not None:
                if self.do_perm:
                    self.x0 = x0[self.perm, ns]
                else:
                    self.x0 = x0[:, ns]
            else:
                self.x0 = None

            if self.do_perm:
                qi = rhs[self.perm, ns]
            else:
                qi = rhs[:, ns]

            u = self.solver(self._A, qi)
            if type(u) == tuple:
                u, info = u
            else:
                info = 0

            if info > 0:
                conv = np.linalg.norm(qi - self._A @ u)
                print(
                    "{0:}: convergence not achieved, stopped after {1:d} \
iterations for atol = {2:g}, rtol = {3:g}, with ||b|| = {4:3.2e} and residuals = {5:g}".format(
                        self.solver.__name__, info, self.atol, self.rtol, np.linalg.norm(qi), conv
                    )
                )
            elif info < 0:
                print("{0:s}: illegal input or breakdown, switching to spsolve".format(self.solver.__name__))
                u = spsolve(self._A, qi)

            if verbose > 0 >= info:
                conv = np.linalg.norm(qi - self._A @ u)
                print(", residuals = {0:g}".format(conv), flush=True)

            if self.do_perm:
                v[:, ns] = u[self.inv_perm]
            else:
                v[:, ns] = u

    @property
    def A(self):
        """Matrix to invert."""
        return self._A

    @A.setter
    def A(self, val):
        # check if we have same structure
        same_struct = False
        if self._A is not None:
            i1, j1, _ = sp.find(self._A)
            i2, j2, _ = sp.find(val)
            if i1.shape == i2.shape and j1.shape == j2.shape:
                if np.all(i1 == i2) and np.all(j1 == j2):
                    same_struct = True
        self._A = val
        if self.want_pastix:
            if self.pastix_solver is not None:
                self.pastix_solver.setup(val)
            else:
                self.pastix_solver = pypastix.solver(val)
                self.solver = lambda A, b: self.pastix_solver.solve(b.flatten(), refine=False)
            return
        elif self.want_umfpack:
            umf_family, A = _get_umf_family(val.tocsc().sorted_indices())
            self.umfpack = um.UmfpackContext(umf_family)
            self.umfpack.control[um.UMFPACK_PRL]
            self.umfpack.numeric(A)
            umf = self.umfpack

            def slv(AA, b):
                with np.errstate(divide="ignore", invalid="ignore"):
                    # Ignoring warnings with numpy >= 1.23.0, see gh-16523
                    result = umf.solve(um.UMFPACK_A, A, b.flatten(), autoTranspose=True)
                    # umf.report_info()
                return result

            self.solver = slv
            return
        elif self.want_superlu:
            solve = factorized(val.tocsc())
            self.solver = lambda A, b: solve(b.flatten())
            return
        elif self.ctx is not None:
            # we are using MUMPS
            if val.shape[0] != val.shape[1]:
                raise RuntimeError("MUMPS: matrix must be square")
            # values passed to MUMPS when only they change must be in the same order as the
            # entries given at analysis: use canonical CSR (sorted indices, no duplicates) everywhere
            val = _canonical_csr(val)
            self._A = val
            if self.ctx.myid == 0:
                if not same_struct:
                    self.ctx.set_shape(val.shape[0])
                    self.ctx.set_centralized_sparse(val)
                else:
                    self.ctx.set_centralized_assembled_values(val.data)
            if self.verbose:
                if same_struct:
                    print("    Factorizing matrix A ... ", end="", flush=True)
                else:
                    print("    Analyzing & factorizing matrix A ... ", end="", flush=True)
            self._mumps_factorize(job=2 if same_struct else 4)
            if self.verbose:
                print("done.")
            return

        # solveur itératif
        if self.do_perm:
            if self.verbose:
                print("  Applying Inverse Cuthill-McKee Permutation ... ", end="", flush=True)
            self.perm = reverse_cuthill_mckee(val)
            self.inv_perm = np.argsort(self.perm)

            I = sp.csr_array((np.ones((self.perm.size,)), (self.perm, np.arange(self.perm.size))))
            val = val @ I
            self._A = (val.T @ I).T
            if self.verbose:
                print("done.")

        if self.precon != '0':
            if self.verbose:
                print("  Computing preconditioning matrix ... ", end="", flush=True)
            if self.precon == 'diag':
                Ainv = sp.diags_array(1.0 / self.A.diagonal())
                self.Mpre = sp.linalg.aslinearoperator(Ainv)
            elif self.precon == 'ilu':
                try:
                    self.Mpre = sp.linalg.spilu(self.A.tocsc())
                    self.Mpre = sp.linalg.LinearOperator(self.A.shape, self.Mpre.solve)
                except RuntimeError as err:
                    if self.verbose:
                        print(err)
                        print("Switching to using diagonal of A")
                    Ainv = sp.diags_array(1.0 / self.A.diagonal())
                    self.Mpre = sp.linalg.aslinearoperator(Ainv)
            else:
                raise ValueError("Unknown preconditioning solver")
            if self.verbose:
                print("done.")
        self.solver = lambda A, b: self.it_slv(A, b, x0=self.x0, atol=self.atol, rtol=self.rtol, maxiter=self.max_it,
                                               M=self.Mpre)

    def _mumps_factorize(self, job, max_retry=5):
        """Run MUMPS factorization (job=2) or analysis & factorization (job=4).

        If the workspace is too small (error -9), it is increased by 20% and the
        factorization is attempted again.
        """
        if self._A is not None and not np.all(np.isfinite(self._A.data)):
            # MUMPS would fail with a misleading error code (-9 or -10)
            raise RuntimeError("MUMPS: the matrix has non-finite values (NaN or inf), check the conductivity model")
        for _ in range(max_retry + 1):
            try:
                self.ctx.run(job=job)
                return
            except RuntimeError:
                if self.ctx.id.infog[0] != -9:
                    raise RuntimeError(f"MUMPS: factorization failed with error code {self.ctx.id.infog[0]}")
                if self.verbose:
                    print(" workspace too small, increasing ... ", end="", flush=True)
                self.ctx.id.icntl[13] += 20
        raise RuntimeError("MUMPS: factorization failed, workspace too small (error -9)")

    def _solve_mumps(self, A, b):
        if self.ctx.myid == 0:
            x = b.flatten().copy()
            self.ctx.set_rhs(x)
        else:
            x = np.empty(b.flatten().shape)
        self.ctx.run(job=3)  # Solve
        self.ctx.comm.Bcast(x, root=0)
        return x

    def print_info(self, file=None):
        if self.want_pardiso:
            print("    Solver: pardiso")
        elif self.want_pastix:
            print("    Solver: PaStiX")
        elif self.want_umfpack:
            print("    Solver: UMFPACK")
        elif self.want_superlu:
            print("    Solver: SuperLU")
        elif self.ctx is not None:
            print("    Solver: MUMPS")
        else:
            print("    Solver: " + self.it_slv.__name__)
            print("      max_it: " + str(self.max_it))
            print("      abs tolerance: " + str(self.atol))
            print("      rel tolerance: " + str(self.rtol))
            if self.do_perm:
                print("    Inverse Cuthill-McKee Permutation: used")
            else:
                print("    Inverse Cuthill-McKee Permutation: not used")
            if self.precon != '0':
                print("    Preconditioning: " + self.precon)
            else:
                print("    Preconditioning: not used")

            if is_free_threading() and self.n_threads > 1:
                print("    Number of threads: " + str(self.n_threads))


if __name__ == "__main__":

    pts = np.array(
        [
            [0.05, 0.0, -0.8],
            [0.0, 0.05, 0.75],
            [-0.5, 0.0, 0.1],
            [0.0, 0.7, 0.05],
            [0.5, 0.5, -0.2],
            [0.45, -0.65, 0.0],
            [-0.35, -0.55, 0.03],
        ]
    )

    tet = np.array([[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 5], [0, 1, 5, 6], [0, 1, 6, 2]], dtype=np.int64)
    sigma = np.array([1., 2., 3., 4., 5.])

    mesh = MeshFV((pts, tet, np.array([])))

    G = mesh.build_G()
    # C = mesh.build_C()

    mesh.toVTK({'sigma': sigma}, "/tmp/test_mesh")
