#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for finite volume modeling

@author: giroux

References:
    
@book{haber2014computational,
  title={Computational methods in geophysical electromagnetics},
  author={Haber, Eldad},
  year={2014},
  publisher={SIAM}
}

@MastersThesis{lelievre03,
  author       = {Peter George Leli\`evre},
  school       = {University of British Columbia},
  title        = {Forward modeling and inversion of geophysical magnetic data},
  year         = {2003}
}

"""
import copy
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import bicgstab, spsolve, use_solver, factorized
from scipy.sparse.csgraph import reverse_cuthill_mckee

import vtk
from vtk.util.numpy_support import vtk_to_numpy

try:
    import pypardiso
    has_pardiso = True
except:
    has_pardiso = False
    
try:
    import scikits.umfpack as um
    has_umfpack = True
except:
    has_umfpack = False

try:
    import mumps
    has_mumps = True
except:
# except ImportError as err:
#     print(err)
    has_mumps = False
    
try:
    import pypastix
    has_pastix = True
except:
    has_pastix = False


# %% Some functions


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
        Paddaing array holding cell sizes.

    Notes
    -----
    Cell size increases following

    factor**n

    where n represent n^th cell.

    """
    return dx * factor**np.arange(1, n_cells+1)


def build_from_vtk(grid_class, filename):
    """Create grid from VTK file.

    The file must contain a rectilinear grid

    Parameters
    ----------
    grid_class : class
        Class for building grid (must be derived from GridFV)
    filename : string
        Name of VTK file

    Returns
    -------
    instance of the grid
    """
    reader = vtk.vtkXMLRectilinearGridReader()
    reader.SetFileName(filename)
    reader.Update()
    x = vtk_to_numpy(reader.GetOutput().GetXCoordinates())
    y = vtk_to_numpy(reader.GetOutput().GetYCoordinates())
    z = vtk_to_numpy(reader.GetOutput().GetZCoordinates())
    return grid_class(x, y, z)


def _get_umf_family(A):
    """Get umfpack family string given the sparse matrix dtype.
    
    This function was taken in scipy.sparse.linalg file _dsolve/linsolve.py"""
    _families = {
        (np.float64, np.int32): 'di',
        (np.complex128, np.int32): 'zi',
        (np.float64, np.int64): 'dl',
        (np.complex128, np.int64): 'zl'
    }

    f_type = np.sctypeDict[A.dtype.name]
    i_type = np.sctypeDict[A.indices.dtype.name]

    try:
        family = _families[(f_type, i_type)]

    except KeyError as e:
        msg = 'only float64 or complex128 matrices with int32 or int64' \
            ' indices are supported! (got: matrix: %s, indices: %s)' \
            % (f_type, i_type)
        raise ValueError(msg) from e

    # See gh-8278. Considered converting only if
    # A.shape[0]*A.shape[1] > np.iinfo(np.int32).max,
    # but that didn't always fix the issue.
    family = family[0] + "l"
    A_new = copy.copy(A)
    A_new.indptr = np.array(A.indptr, copy=False, dtype=np.int64)
    A_new.indices = np.array(A.indices, copy=False, dtype=np.int64)

    return family, A_new


# %% GridFV

class GridFV:
    """Class to manage rectilinear grids for finite volume modelling.
    
    Parameters
    ----------
    x : array_like
        Node coordinates along x
    y : array_like
        Node coordinates along y
    z : array_like
        Node coordinates along z
    comm : MPI Communicator or None
        If None, use MPI_COMM_WORLD

    Notes
    -----
    Voxels are sorted column major, i.e. x is the fast axis
    (choice dictated by convention used by VTK).
    
    """

    def __init__(self, x, y, z, comm=None):
        if comm is None:
            from mpi4py import MPI
        
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.myid = comm.rank

        self.x = x
        self.y = y
        self.z = z
        self.solver = bicgstab
        self.tol = 1e-9
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
        self.precon = False
        self.do_perm = False

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
        elif component == 'ex':
            nx = self.nx
            ny = self.ny - 1
            nz = self.nz - 1
        elif component == 'ey':
            nx = self.nx - 1
            ny = self.ny
            nz = self.nz - 1
        elif component == 'ez':
            nx = self.nx - 1
            ny = self.ny - 1
            nz = self.nz
        elif component == 'fx':
            nx = self.nx - 1
            ny = self.ny
            nz = self.nz
        elif component == 'fy':
            nx = self.nx
            ny = self.ny - 1
            nz = self.nz
        elif component == 'fz':
            nx = self.nx
            ny = self.ny
            nz = self.nz - 1
        else:
            raise ValueError('unknown component')

        if np.size(i) > 1:
            i = np.array(i)
            i = i.flatten()
        if np.size(j) > 1:
            j = np.array(j)
            j = j.flatten()
        if np.size(k) > 1:
            k = np.array(k)
            k = k.flatten()
        if np.any(i < 0) or np.any(i >= nx) or np.any(j < 0) or \
           np.any(j >= ny) or np.any(k < 0) or np.any(k >= nz):
            raise IndexError('Index outside grid')
        if np.size(i) > 1 or np.size(j) > 1 or np.size(k) > 1:
            ii = np.kron(np.ones((np.size(j)*np.size(k),), dtype=np.int64), i)
            jj = np.kron(np.kron(np.ones((np.size(k),), dtype=np.int64), j),
                         np.ones((np.size(i),), dtype=np.int64))
            kk = np.kron(k, np.ones((np.size(i)*np.size(j),), dtype=np.int64))
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
        elif component == 'ex':
            nx = self.nx
            ny = self.ny - 1
        elif component == 'ey':
            nx = self.nx - 1
            ny = self.ny
        elif component == 'ez':
            nx = self.nx - 1
            ny = self.ny - 1
        elif component == 'fx':
            nx = self.nx - 1
            ny = self.ny
        elif component == 'fy':
            nx = self.nx
            ny = self.ny - 1
        elif component == 'fz':
            nx = self.nx
            ny = self.ny
        else:
            raise ValueError('unknown component')

        k = int(ind / (nx * ny))
        j = int((ind - k*nx*ny)/nx)
        i = ind - (k * ny + j) * nx
        return i, j, k

    def __update_n(self):
        if '_x' in self.__dict__ and '_y' in self.__dict__ and \
           '_z' in self.__dict__:
            self.nx = len(self._x)-1                # number of voxels along x
            self.ny = len(self._y)-1                # number of voxels along y
            self.nz = len(self._z)-1                # number of voxels along z
            self.nc = self.nx * self.ny * self.nz   # number of voxels
            self.nfx = (self.nx-1)*self.ny*self.nz  # number of faces with normal vector along x
            self.nfy = self.nx*(self.ny-1)*self.nz  # number of faces with normal vector along y
            self.nfz = self.nx*self.ny*(self.nz-1)  # number of faces with normal vector along z
            self.nf = self.nfx + self.nfy + self.nfz    # total number of faces
            self.nex = self.nx*(self.ny-1)*(self.nz-1)  # number of edges with vector along x
            self.ney = (self.nx-1)*self.ny*(self.nz-1)  # number of edges with vector along y
            self.nez = (self.nx-1)*(self.ny-1)*self.nz  # number of edges with vector along z
            self.ne = self.nex + self.ney + self.nez    # total number of edges

            # Divergence & gradient matrices are precomputed because essentially always used
            self.D = self.build_D()
            self.G = self.build_G()

    @property
    def x(self):
        """Node coordinates along x."""
        return self._x

    @x.setter
    def x(self, val):
        tmp = np.array(val, dtype=np.float64)
        if tmp.ndim != 1:
            raise ValueError('1D array needed')
        if len(tmp) < 2:
            raise ValueError('2 nodes or more needed')

        self._x = tmp
        self.hx = np.diff(tmp)
        self.xc = (tmp[1:]+tmp[:-1])/2
        self.dx = np.diff(self.xc)
        self.__update_n()

    @property
    def y(self):
        """Node coordinates along y."""
        return self._y

    @y.setter
    def y(self, val):
        tmp = np.array(val, dtype=np.float64)
        if tmp.ndim != 1:
            raise ValueError('1D array needed')
        if len(tmp) < 2:
            raise ValueError('2 nodes or more needed')

        self._y = tmp
        self.hy = np.diff(tmp)
        self.yc = (tmp[1:]+tmp[:-1])/2
        self.dy = np.diff(self.yc)
        self.__update_n()

    @property
    def z(self):
        """Node coordinates along z."""
        return self._z

    @z.setter
    def z(self, val):
        tmp = np.array(val, dtype=np.float64)
        if tmp.ndim != 1:
            raise ValueError('1D array needed')
        if len(tmp) < 2:
            raise ValueError('2 nodes or more needed')

        self._z = tmp
        self.hz = np.diff(tmp)
        self.zc = (tmp[1:]+tmp[:-1])/2
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
        True if point si inside grid

        """
        return self.x[0] <= x <= self.x[-1] and self.y[0] <= y <= self.y[-1] and \
            self.z[0] <= z <= self.z[-1]

    def volume_voxels(self):
        """Returns the volume of the voxels of the grid.

        Returns
        -------
        ndarray
            Volumes of voxels.

        """
        return np.kron(self.hz, np.kron(self.hy, self.hx))
    
    def centre_voxels(self):
        """Returns coordinates at the centres of voxels.

        Returns
        -------
        ndarray nc x 3
            coordinates X, Y, Z

        """
        x = np.kron(np.ones((self.ny * self.nz,)), self.xc)
        y = np.kron(np.kron(np.ones((self.nz,)), self.yc),
                     np.ones((self.nx,)))
        z = np.kron(self.zc, np.ones((self.nx * self.ny,)))
        return np.c_[x, y, z]
    
    def distance_weighting(self, xo, beta):
        """Calculate distance weighting matrix.

        Parameters
        ----------
        xo : arraylike
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
        xyzc = self.centre_voxels()
        R0 = 0.5 * dV.min()**0.33333333333333
        Q = np.zeros((self.nc,))
        for i in np.arange(xo.shape[0]):
            R = np.sqrt((xyzc[:,0]-xo[i, 0])*(xyzc[:,0]-xo[i, 0]) +
                        (xyzc[:,1]-xo[i, 1])*(xyzc[:,1]-xo[i, 1]) +
                        (xyzc[:,2]-xo[i, 2])*(xyzc[:,2]-xo[i, 2]))
            R = (R+R0)**3
            R = (dV/R)**2
            Q += R
        return Q**(0.25*beta)

    def linear_interp(self, x, y, z, component=None):
        """Calculate interpolation matrix.

        Matrice that allows interpolating a variable defined at the voxel
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

        """
        if component is None:
            nx = self.nx
            ny = self.ny
            nz = self.nz
            xc = self.xc
            yc = self.yc
            zc = self.zc
        elif component == 'ex':
            nx = self.nx
            ny = self.ny - 1
            nz = self.nz - 1
            xc = self.xc
            yc = self.y[1:-1]
            zc = self.z[1:-1]
        elif component == 'ey':
            nx = self.nx - 1
            ny = self.ny
            nz = self.nz - 1
            xc = self.x[1:-1]
            yc = self.yc
            zc = self.z[1:-1]
        elif component == 'ez':
            nx = self.nx - 1
            ny = self.ny - 1
            nz = self.nz
            xc = self.x[1:-1]
            yc = self.y[1:-1]
            zc = self.zc
        elif component == 'fx':
            nx = self.nx - 1
            ny = self.ny
            nz = self.nz
            xc = self.x[1:-1]
            yc = self.yc
            zc = self.zc
        elif component == 'fy':
            nx = self.nx
            ny = self.ny - 1
            nz = self.nz
            xc = self.xc
            yc = self.y[1:-1]
            zc = self.zc
        elif component == 'fz':
            nx = self.nx
            ny = self.ny
            nz = self.nz - 1
            xc = self.xc
            yc = self.yc
            zc = self.z[1:-1]
        else:
            raise ValueError('unknown component')

        nc = nx * ny * nz

        dx = [0.0, 0.0]
        dy = [0.0, 0.0]
        dz = [0.0, 0.0]
        ix = [0, 0]
        iy = [0, 0]
        iz = [0, 0]

        if np.isscalar(x) is True:
            x = np.array([x])
            y = np.array([y])
            z = np.array([z])

        Q = sp.lil_matrix((len(x), nc))
        for i in range(len(x)):
            
            if np.any(np.array([x[i], y[i], z[i]]) == np.inf):
                # check if point at np.inf -> set Q to 0
                Q[i, 0] = 0.0
                continue

            im = np.argmin(np.abs(x[i] - xc))
            if x[i] < xc[im]:
                ix[0] = im-1
                ix[1] = im
            else:
                ix[0] = im
                ix[1] = im+1
            dx[0] = x[i] - xc[ix[0]]
            dx[1] = xc[ix[1]] - x[i]

            im = np.argmin(np.abs(y[i] - yc))
            if y[i] < yc[im]:
                iy[0] = im-1
                iy[1] = im
            else:
                iy[0] = im
                iy[1] = im+1
            dy[0] = y[i] - yc[iy[0]]
            dy[1] = yc[iy[1]] - y[i]

            im = np.argmin(np.abs(z[i] - zc))
            if z[i] < zc[im]:
                iz[0] = im-1
                iz[1] = im
            else:
                iz[0] = im
                iz[1] = im+1
            dz[0] = z[i] - zc[iz[0]]
            dz[1] = zc[iz[1]] - z[i]

            Dx = xc[ix[1]] - xc[ix[0]]
            Dy = yc[iy[1]] - yc[iy[0]]
            Dz = zc[iz[1]] - zc[iz[0]]

            Q[i, self.ind(ix[0], iy[0], iz[0], component)] = (1-dx[0]/Dx)*(1-dy[0]/Dy)*(1-dz[0]/Dz)
            Q[i, self.ind(ix[1], iy[0], iz[0], component)] = (1-dx[1]/Dx)*(1-dy[0]/Dy)*(1-dz[0]/Dz)
            Q[i, self.ind(ix[0], iy[1], iz[0], component)] = (1-dx[0]/Dx)*(1-dy[1]/Dy)*(1-dz[0]/Dz)
            Q[i, self.ind(ix[1], iy[1], iz[0], component)] = (1-dx[1]/Dx)*(1-dy[1]/Dy)*(1-dz[0]/Dz)
            Q[i, self.ind(ix[0], iy[0], iz[1], component)] = (1-dx[0]/Dx)*(1-dy[0]/Dy)*(1-dz[1]/Dz)
            Q[i, self.ind(ix[1], iy[0], iz[1], component)] = (1-dx[1]/Dx)*(1-dy[0]/Dy)*(1-dz[1]/Dz)
            Q[i, self.ind(ix[0], iy[1], iz[1], component)] = (1-dx[0]/Dx)*(1-dy[1]/Dy)*(1-dz[1]/Dz)
            Q[i, self.ind(ix[1], iy[1], iz[1], component)] = (1-dx[1]/Dx)*(1-dy[1]/Dy)*(1-dz[1]/Dz)

        return Q.tocsr()

    def build_D(self):
        """Construction of divergence matrix.

        Returns
        -------
        csr_matrix
            Divergence matrix

        """

        # Dx

        M = self.nx
        N = self.nx-1
        i = np.hstack((np.arange(N), 1+np.arange(N)))
        j = np.hstack((np.arange(N), np.arange(N)))
        nval = i.size
        ii = np.zeros((self.ny*self.nz*nval,), dtype=np.int64)
        jj = np.zeros((self.ny*self.nz*nval,), dtype=np.int64)
        for n in np.arange(self.ny*self.nz):
            ii[(n*nval):((n+1)*nval)] = i + n*M
            jj[(n*nval):((n+1)*nval)] = j + n*N
        s = np.tile(np.hstack((1.0/self.hx[:-1], -1.0/self.hx[1:])),
                    (self.ny*self.nz,))
        Dx = sp.coo_matrix((s, (ii, jj)))

        # Dy

        M = self.nx*self.ny
        N = self.nx*(self.ny-1)
        i = np.hstack((np.arange(N), self.nx+np.arange(N)))
        j = np.hstack((np.arange(N), np.arange(N)))
        nval = i.size
        ii = np.zeros((self.nz*nval,), dtype=np.int64)
        jj = np.zeros((self.nz*nval,), dtype=np.int64)
        for n in np.arange(self.nz):
            ii[(n*nval):((n+1)*nval)] = i + n*M
            jj[(n*nval):((n+1)*nval)] = j + n*N
        s = np.hstack((np.kron(1.0/self.hy[:-1], np.ones((self.nx,))),
                       np.kron(-1.0/self.hy[1:], np.ones((self.nx,)))))
        s = np.tile(s, (self.nz,))
        Dy = sp.coo_matrix((s, (ii, jj)))

        # Dz

        N = (self.nx*self.ny*(self.nz-1))
        i = np.hstack((np.arange(N), self.nx*self.ny+np.arange(N)))
        j = np.hstack((np.arange(N), np.arange(N)))
        s = np.hstack((np.kron(1./self.hz[:-1], np.ones((self.nx*self.ny,))),
                       np.kron(-1./self.hz[1:], np.ones((self.nx*self.ny,)))))
        Dz = sp.coo_matrix((s, (i, j)))

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

        d = np.kron(np.ones((self.ny*self.nz,)), 2*self.dx)
        hi = np.kron(np.ones((self.ny*self.nz,)), self.hx[1:])
        him1 = np.kron(np.ones((self.ny*self.nz,)), self.hx[:-1])
        tmp = np.ones((self.nx,), dtype=bool)
        tmp[0] = 0
        ind1 = np.kron(np.ones((self.ny*self.nz,), dtype=bool), tmp)
        tmp[0] = 1
        tmp[-1] = 0
        ind2 = np.kron(np.ones((self.ny*self.nz,), dtype=bool), tmp)
        if harmon:
            Mx = d / (hi/vx[ind1] + him1/vx[ind2])
        else:
            Mx = (hi*vx[ind1] + him1*vx[ind2]) / d

        # My

        d = np.kron(np.ones((self.nz,)), np.kron(2*self.dy,
                                                 np.ones((self.nx,))))
        hi = np.kron(np.ones((self.nz,)), np.kron(self.hy[1:],
                                                  np.ones((self.nx,))))
        him1 = np.kron(np.ones((self.nz,)), np.kron(self.hy[:-1],
                                                    np.ones((self.nx,))))

        tmp = np.ones((self.ny,), dtype=bool)
        tmp[0] = 0
        ind1 = np.kron(np.ones((self.nz,), dtype=bool),
                       np.kron(tmp, np.ones((self.nx,), dtype=bool)))
        tmp[0] = 1
        tmp[-1] = 0
        ind2 = np.kron(np.ones((self.nz,), dtype=bool),
                       np.kron(tmp, np.ones((self.nx,), dtype=bool)))
        if harmon:
            My = d / (hi/vy[ind1] + him1/vy[ind2])
        else:
            My = (hi*vy[ind1] + him1*vy[ind2]) / d

        # Mz

        d = np.kron(2*self.dz, np.ones((self.nx*self.ny,)))
        hi = np.kron(self.hz[1:], np.ones((self.nx*self.ny,)))
        him1 = np.kron(self.hz[:-1], np.ones((self.nx*self.ny,)))
        ind1 = np.hstack((np.zeros((self.nx*self.ny,), dtype=bool),
                          np.ones((self.nx*self.ny*(self.nz-1),), dtype=bool)))
        ind2 = np.hstack((np.ones((self.nx*self.ny*(self.nz-1),), dtype=bool),
                          np.zeros((self.nx*self.ny,), dtype=bool)))
        if harmon:
            Mz = d / (hi/vz[ind1] + him1/vz[ind2])
        else:
            Mz = (hi*vz[ind1] + him1*vz[ind2]) / d

        # assemblage

        return sp.coo_matrix((np.hstack((Mx, My, Mz)),
                             (np.arange(self.nf), np.arange(self.nf)))).tocsr()

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
        ii = np.zeros((self.ny*self.nz*nval,), dtype=np.int64)
        jj = np.zeros((self.ny*self.nz*nval,), dtype=np.int64)
        for n in np.arange(self.ny*self.nz):
            ii[(n*nval):((n+1)*nval)] = i + n*M
            jj[(n*nval):((n+1)*nval)] = j + n*N
        s = np.tile(np.hstack((-1.0/self.dx, 1.0/self.dx)), (self.ny*self.nz,))
        Gx = sp.coo_matrix((s, (ii, jj)))

        # Gy

        M = self.nx*(self.ny-1)
        N = self.nx*self.ny
        i = np.hstack((np.arange(M), np.arange(M)))
        j = np.hstack((np.arange(M), self.nx+np.arange(M)))
        nval = i.size
        ii = np.zeros((self.nz*nval,), dtype=np.int64)
        jj = np.zeros((self.nz*nval,), dtype=np.int64)
        for n in np.arange(self.nz):
            ii[(n*nval):((n+1)*nval)] = i + n*M
            jj[(n*nval):((n+1)*nval)] = j + n*N
        s = np.hstack((np.kron(-1.0/self.dy, np.ones((self.nx,))),
                       np.kron(1.0/self.dy, np.ones((self.nx,)))))
        s = np.tile(s, (self.nz,))
        Gy = sp.coo_matrix((s, (ii, jj)))

        # Gz

        i = np.hstack((np.arange(self.nfz), np.arange(self.nfz)))
        j = np.hstack((np.arange(self.nfz),
                       self.nx*self.ny+np.arange(self.nfz)))
        s = np.hstack((np.kron(-1.0/self.dz, np.ones((self.nx*self.ny,))),
                       np.kron(1.0/self.dz, np.ones((self.nx*self.ny,)))))
        Gz = sp.coo_matrix((s, (i, j)))

        # assemblage
        return sp.vstack((Gx, Gy, Gz), format='csr')

    def _gradient_faces(self, v):

        v_x = sp.diags(v[:self.nfx])
        v_y = sp.diags(v[self.nfx:(self.nfx+self.nfy)])
        v_z = sp.diags(v[(self.nfx+self.nfy):])

        # Gx

        dVf = 0.5 * (self.hx[1:] + self.hx[:-1])

        M = self.nx - 1
        N = self.nx
        i = np.hstack((np.arange(M), np.arange(M)))
        j = np.hstack((np.arange(M), np.arange(1, N)))
        nval = i.size
        ii = np.zeros((self.ny*self.nz*nval,), dtype=np.int64)
        jj = np.zeros((self.ny*self.nz*nval,), dtype=np.int64)
        for n in np.arange(self.ny*self.nz):
            ii[(n*nval):((n+1)*nval)] = i + n*M
            jj[(n*nval):((n+1)*nval)] = j + n*N
        s = np.tile(np.hstack((0.5*self.hx[:-1]/dVf, 0.5*self.hx[1:]/dVf)), (self.ny*self.nz,))
        Gx = v_x @ sp.coo_matrix((s, (ii, jj)))

        # Gy

        dVf = 0.5 * (self.hy[1:] + self.hy[:-1])

        M = self.nx*(self.ny-1)
        N = self.nx*self.ny
        i = np.hstack((np.arange(M), np.arange(M)))
        j = np.hstack((np.arange(M), self.nx+np.arange(M)))
        nval = i.size
        ii = np.zeros((self.nz*nval,), dtype=np.int64)
        jj = np.zeros((self.nz*nval,), dtype=np.int64)
        for n in np.arange(self.nz):
            ii[(n*nval):((n+1)*nval)] = i + n*M
            jj[(n*nval):((n+1)*nval)] = j + n*N
        s = np.hstack((np.kron(0.5*self.hy[:-1]/dVf, np.ones((self.nx,))),
                       np.kron(0.5*self.hy[1:]/dVf, np.ones((self.nx,)))))
        s = np.tile(s, (self.nz,))
        Gy = v_y @ sp.coo_matrix((s, (ii, jj)))

        # Gz

        dVf = 0.5 * (self.hz[1:] + self.hz[:-1])

        i = np.hstack((np.arange(self.nfz), np.arange(self.nfz)))
        j = np.hstack((np.arange(self.nfz),
                       self.nx*self.ny+np.arange(self.nfz)))
        s = np.hstack((np.kron(0.5*self.hz[:-1]/dVf, np.ones((self.nx*self.ny,))),
                       np.kron(0.5*self.hz[1:]/dVf, np.ones((self.nx*self.ny,)))))
        Gz = v_z @ sp.coo_matrix((s, (i, j)))

        # assemblage
        return sp.vstack((Gx, Gy, Gz), format='csr')


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
        M = (self.nx-1)*self.ny
        N = (self.nx-1)*(self.ny-1)
        i = np.hstack((np.arange(N), (self.nx-1)+np.arange(N)))
        j = np.hstack((np.arange(N), np.arange(N)))
        nval = i.size
        ii = np.zeros((self.nz*nval,), dtype=np.int64)
        jj = np.zeros((self.nz*nval,), dtype=np.int64)
        for n in np.arange(self.nz):
            ii[(n*nval):((n+1)*nval)] = i + n*M
            jj[(n*nval):((n+1)*nval)] = j + n*N
        s = np.hstack((np.kron(1.0/self.hy[:-1], np.ones((self.nx-1,))),
                       np.kron(-1.0/self.hy[1:], np.ones((self.nx-1,)))))
        s = np.tile(s, (self.nz,))
        Dzy = sp.coo_matrix((s, (ii, jj)))

        # dHy / dz
        N = ((self.nx-1)*self.ny*(self.nz-1))
        i = np.hstack((np.arange(N), (self.nx-1)*self.ny+np.arange(N)))
        j = np.hstack((np.arange(N), np.arange(N)))
        s = np.hstack((np.kron(1./self.hz[:-1], np.ones(((self.nx-1)*self.ny,))),
                       np.kron(-1./self.hz[1:], np.ones(((self.nx-1)*self.ny,)))))
        Dyz = sp.coo_matrix((s, (i, j)))

        # dHx / dz
        N = (self.nx*(self.ny-1)*(self.nz-1))
        i = np.hstack((np.arange(N), self.nx*(self.ny-1)+np.arange(N)))
        j = np.hstack((np.arange(N), np.arange(N)))
        s = np.hstack((np.kron(1./self.hz[:-1], np.ones((self.nx*(self.ny-1),))),
                       np.kron(-1./self.hz[1:], np.ones((self.nx*(self.ny-1),)))))
        Dxz = sp.coo_matrix((s, (i, j)))

        # dHz / dx
        M = self.nx
        N = self.nx-1
        i = np.hstack((np.arange(N), 1+np.arange(N)))
        j = np.hstack((np.arange(N), np.arange(N)))
        nval = i.size
        ii = np.zeros(((self.ny-1)*self.nz*nval,), dtype=np.int64)
        jj = np.zeros(((self.ny-1)*self.nz*nval,), dtype=np.int64)
        for n in np.arange((self.ny-1)*self.nz):
            ii[(n*nval):((n+1)*nval)] = i + n*M
            jj[(n*nval):((n+1)*nval)] = j + n*N
        s = np.tile(np.hstack((1.0/self.hx[:-1], -1.0/self.hx[1:])),
                    ((self.ny-1)*self.nz,))
        Dzx = sp.coo_matrix((s, (ii, jj)))

        # dHy / dx
        M = self.nx
        N = self.nx-1
        i = np.hstack((np.arange(N), 1+np.arange(N)))
        j = np.hstack((np.arange(N), np.arange(N)))
        nval = i.size
        ii = np.zeros((self.ny*(self.nz-1)*nval,), dtype=np.int64)
        jj = np.zeros((self.ny*(self.nz-1)*nval,), dtype=np.int64)
        for n in np.arange(self.ny*(self.nz-1)):
            ii[(n*nval):((n+1)*nval)] = i + n*M
            jj[(n*nval):((n+1)*nval)] = j + n*N
        s = np.tile(np.hstack((1.0/self.hx[:-1], -1.0/self.hx[1:])),
                    (self.ny*(self.nz-1),))
        Dyx = sp.coo_matrix((s, (ii, jj)))

        # dHx / dy
        M = self.nx*self.ny
        N = self.nx*(self.ny-1)
        i = np.hstack((np.arange(N), self.nx+np.arange(N)))
        j = np.hstack((np.arange(N), np.arange(N)))
        nval = i.size
        ii = np.zeros(((self.nz-1)*nval,), dtype=np.int64)
        jj = np.zeros(((self.nz-1)*nval,), dtype=np.int64)
        for n in np.arange(self.nz-1):
            ii[(n*nval):((n+1)*nval)] = i + n*M
            jj[(n*nval):((n+1)*nval)] = j + n*N
        s = np.hstack((np.kron(1.0/self.hy[:-1], np.ones((self.nx,))),
                       np.kron(-1.0/self.hy[1:], np.ones((self.nx,)))))
        s = np.tile(s, (self.nz-1,))
        Dxy = sp.coo_matrix((s, (ii, jj)))

        return sp.vstack((sp.hstack((sp.csr_matrix((self.nfx, self.nex)), -Dyz, Dzy), format='csr'),
                          sp.hstack((Dxz, sp.csr_matrix((self.nfy, self.ney)), -Dzx), format='csr'),
                          sp.hstack((-Dxy, Dyx, sp.csr_matrix((self.nfz, self.nez))), format='csr')))

    def _curl_edges(self):

        # dAz / dy
        M = self.nx*(self.ny-1)
        N = self.nx*self.ny
        i = np.hstack((np.arange(M), np.arange(M)))
        j = np.hstack((np.arange(M), self.nx+np.arange(M)))
        nval = i.size
        ii = np.zeros(((self.nz-1)*nval,), dtype=np.int64)
        jj = np.zeros(((self.nz-1)*nval,), dtype=np.int64)
        for n in np.arange(self.nz-1):
            ii[(n*nval):((n+1)*nval)] = i + n*M
            jj[(n*nval):((n+1)*nval)] = j + n*N
        s = np.hstack((np.kron(-1.0/self.dy, np.ones((self.nx,))),
                       np.kron(1.0/self.dy, np.ones((self.nx,)))))
        s = np.tile(s, (self.nz-1,))
        Dzy = sp.coo_matrix((s, (ii, jj)))

        # dAy / dz
        N = self.nx*(self.ny-1)*(self.nz-1)
        i = np.hstack((np.arange(N), np.arange(N)))
        j = np.hstack((np.arange(N), self.nx*(self.ny-1)+np.arange(N)))
        s = np.hstack((np.kron(-1.0/self.dz, np.ones((self.nx*(self.ny-1),))),
                       np.kron(1.0/self.dz, np.ones((self.nx*(self.ny-1),)))))
        Dyz = sp.coo_matrix((s, (i, j)))

        # dAx / dz
        N = (self.nx-1)*self.ny*(self.nz-1)
        i = np.hstack((np.arange(N), np.arange(N)))
        j = np.hstack((np.arange(N), (self.nx-1)*self.ny+np.arange(N)))
        s = np.hstack((np.kron(-1.0/self.dz, np.ones(((self.nx-1)*self.ny,))),
                       np.kron(1.0/self.dz, np.ones(((self.nx-1)*self.ny,)))))
        Dxz = sp.coo_matrix((s, (i, j)))

        # dAz / dx
        M = self.nx - 1
        N = self.nx
        i = np.hstack((np.arange(M), np.arange(M)))
        j = np.hstack((np.arange(M), np.arange(1, N)))
        nval = i.size
        ii = np.zeros((self.ny*(self.nz-1)*nval,), dtype=np.int64)
        jj = np.zeros((self.ny*(self.nz-1)*nval,), dtype=np.int64)
        for n in np.arange(self.ny*(self.nz-1)):
            ii[(n*nval):((n+1)*nval)] = i + n*M
            jj[(n*nval):((n+1)*nval)] = j + n*N
        s = np.tile(np.hstack((-1.0/self.dx, 1.0/self.dx)), (self.ny*(self.nz-1),))
        Dzx = sp.coo_matrix((s, (ii, jj)))

        # dAy / dx
        M = self.nx - 1
        N = self.nx
        i = np.hstack((np.arange(M), np.arange(M)))
        j = np.hstack((np.arange(M), np.arange(1, N)))
        nval = i.size
        ii = np.zeros(((self.ny-1)*self.nz*nval,), dtype=np.int64)
        jj = np.zeros(((self.ny-1)*self.nz*nval,), dtype=np.int64)
        for n in np.arange((self.ny-1)*self.nz):
            ii[(n*nval):((n+1)*nval)] = i + n*M
            jj[(n*nval):((n+1)*nval)] = j + n*N
        s = np.tile(np.hstack((-1.0/self.dx, 1.0/self.dx)), ((self.ny-1)*self.nz,))
        Dyx = sp.coo_matrix((s, (ii, jj)))

        # dAx / dy
        M = (self.nx-1)*(self.ny-1)
        N = (self.nx-1)*self.ny
        i = np.hstack((np.arange(M), np.arange(M)))
        j = np.hstack((np.arange(M), (self.nx-1)+np.arange(M)))
        nval = i.size
        ii = np.zeros((self.nz*nval,), dtype=np.int64)
        jj = np.zeros((self.nz*nval,), dtype=np.int64)
        for n in np.arange(self.nz):
            ii[(n*nval):((n+1)*nval)] = i + n*M
            jj[(n*nval):((n+1)*nval)] = j + n*N
        s = np.hstack((np.kron(-1.0/self.dy, np.ones((self.nx-1,))),
                       np.kron(1.0/self.dy, np.ones((self.nx-1,)))))
        s = np.tile(s, (self.nz,))
        Dxy = sp.coo_matrix((s, (ii, jj)))

        return sp.vstack((sp.hstack((sp.csr_matrix((self.nex, self.nfx)), -Dyz, Dzy), format='csr'),
                          sp.hstack((Dxz, sp.csr_matrix((self.ney, self.nfy)), -Dzx), format='csr'),
                          sp.hstack((-Dxy, Dyx, sp.csr_matrix((self.nez, self.nfz))), format='csr')))

    def toVTK(self, fields, filename, component='s', on_face=True):
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
        """

        if type(fields) != dict:
            raise ValueError('\'fields\' must be a dict[str, ndarray]')

        xCoords = vtk.vtkFloatArray()
        yCoords = vtk.vtkFloatArray()
        zCoords = vtk.vtkFloatArray()

        if component == 's':
            x = self.x
            y = self.y
            z = self.z
        elif component =='x':
            if on_face:
                x = self.x[1:-1]
                y = self.yc
                z = self.zc
            else:
                x = self.xc
                y = self.y[1:-1]
                z = self.z[1:-1]
        elif component == 'y':
            if on_face:
                x = self.xc
                y = self.y[1:-1]
                z = self.zc
            else:
                x = self.x[1:-1]
                y = self.yc
                z = self.z[1:-1]
        elif component == 'z':
            if on_face:
                x = self.xc
                y = self.yc
                z = self.z[1:-1]
            else:
                x = self.x[1:-1]
                y = self.y[1:-1]
                z = self.zc
        else:
            raise ValueError('unknown component')

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
            field = fields[fn]
            data = vtk.vtkDoubleArray()
            for i in field:
                data.InsertNextValue(i)
            data.SetName(fn)
            if component == 's':
                rgrid.GetCellData().AddArray(data)
            else:
                rgrid.GetPointData().AddArray(data)

        writer = vtk.vtkXMLRectilinearGridWriter()
        writer.SetInputData(rgrid)
        writer.SetFileName(filename+'.vtr')
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
        if dims[0] != self.nx+1 or dims[1] != self.ny+1 or dims[2] != self.nz+1:
            raise ValueError('Grid size incompatible')
        data = reader.GetOutput().GetCellData().GetArray(fieldname)
        if data is None:
            return data
        else:
            return vtk_to_numpy(data)

    def set_solver(self, name, tol=1e-9, max_it=1000, precon=False, do_perm=False, comm=None):
        """Define parameters of solver to be used during forward modelling.

        Parameters
        ----------
        name : `string` or `callable`
            If `string`: name of solver (mumps, pardiso, umfpack, or superlu)
            If `callable`: (iterative solver from scipy.sparse.linalg, eg bicgstab)
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
        if callable(name):
            self.solver = name
            self.tol = tol
            self.max_it = max_it
            self.want_pardiso = False
            self.want_pastix = False
            self.want_superlu = False
            self.want_umfpack = False
            self.want_mumps = False
        elif 'superlu' in name:
            self.want_pardiso = False
            self.want_pastix = False
            self.want_superlu = True
            self.want_umfpack = False
            self.want_mumps = False
        elif 'pardiso' in name:
            self.want_pardiso = True
            self.want_pastix = False
            self.want_superlu = False
            self.want_umfpack = False
            self.want_mumps = False
        elif 'pastix' in name:
            self.want_pastix = True
            self.want_pardiso = False
            self.want_superlu = False
            self.want_umfpack = False
            self.want_mumps = False
        elif 'umfpack' in name:
            self.want_umfpack = True
            self.want_superlu = False
            self.want_pardiso = False
            self.want_pastix = False
            self.want_mumps = False
        elif 'mumps' in name:
            self.want_pardiso = False
            self.want_pastix = False
            self.want_superlu = False
            self.want_umfpack = False
            self.want_mumps = True
        else:
            raise RuntimeError('Solver '+name+' not implemented')

        self.precon = precon
        self.do_perm = do_perm
        self.comm = comm
            
    def get_solver(self):
        """Return parameters needed to instantiate Solver."""
        if self.want_mumps:
            return 'mumps', self.comm
        elif self.want_pardiso:
            return ('pardiso',)
        elif self.want_pastix:
            return ('pastix',)
        elif self.want_superlu:
            return ('superlu',)
        elif self.want_umfpack:
            return ('umfpack',)
        else:
            return self.solver, self.tol, self.max_it, self.precon, self.do_perm
            
    @property
    def want_pardiso(self):
        """Using solver pardiso (if available)."""
        return self._want_pardiso

    @want_pardiso.setter
    def want_pardiso(self, val):
        if val == True and has_pardiso == False:
            warnings.warn('Pardiso not available, default solver used.', RuntimeWarning, stacklevel=2)
            self._want_pardiso = False
        else:
            self._want_pardiso = val

    @property
    def want_pastix(self):
        """Using solver pastix (if available)."""
        return self._want_pastix

    @want_pastix.setter
    def want_pastix(self, val):
        if val == True and has_pastix == False:
            warnings.warn('Pastix not available, default solver used.', RuntimeWarning, stacklevel=2)
            self._want_pastix = False
        else:
            self._want_pastix = val

    @property
    def want_umfpack(self):
        """Using solver umfpack (if available)."""
        return self._want_umfpack

    @want_umfpack.setter
    def want_umfpack(self, val):
        if val == True and has_umfpack == False:
            warnings.warn('UMFPACK not available, default solver used.', RuntimeWarning, stacklevel=2)
            self._want_umfpack = False
        else:
            self._want_umfpack = val

    @property
    def want_mumps(self):
        """Using solver mumps (if available)."""
        return self._want_mumps

    @want_mumps.setter
    def want_mumps(self, val):
        if val == True and has_mumps == False:
            warnings.warn('MUMPS not available, default solver used.', RuntimeWarning, stacklevel=2)
            self._want_mumps = False
        else:
            self._want_mumps = val

    def print_solver_info(self, file=None):
        if self.want_pardiso:
            print('    Solver: pardiso')
        elif self.want_pastix:
            print('    Solver: PaStiX')
        elif self.want_umfpack:
            print('    Solver: UMFPACK')
        elif self.want_superlu:
            print('    Solver: SuperLU')
        elif self.want_mumps:
            print('    Solver: MUMPS')
        else:
            print('    Solver: '+self.solver.__name__)
            print('      max_it: '+str(self.max_it))
            print('      tolerance: '+str(self.tol))
        if self.do_perm:
            print('    Inverse Cuthill-McKee Permutation: used')
        else:
            print('    Inverse Cuthill-McKee Permutation: not used')
        if self.precon:
            print('    Preconditionning: used')
        else:
            print('    Preconditionning: not used')


# %% Solveur

class Solver:
    """
    Classe pour solutionner un système Ax = b avec un choix of solveurs
    
    Parameters
    ----------
    A : spmatrix
        terme du gauche du système
    solver : string or callable
        Si string: nom du solveur (pardiso, umfpack, or mumps)
        Si callable (solver itératif of scipy.sparse.linalg, eg bicgstab)
    verbose : bool, optionel
        Affiche des messages informatifs
    """
    def __init__(self, A, solver, verbose=False):
        self._A = A
        self.precon = False
        self.do_perm = False
        self.Mpre = None
        self.perm = None
        self.inv_perm = None
        self.x0 = None
        self.verbose = verbose
        self.ctx = None
        self.want_umfpack = False
        self.want_superlu = False
        self.pastix_solver = None
        self.umfpack = None
        self.pardiso = False
        if len(solver) == 5:
            # solveur itératif

            slv = solver[0]
            self.tol = solver[1]
            max_it = solver[2]
            self.precon = solver[3]
            self.do_perm = solver[4]
            
            if self.do_perm:
                if self.verbose:
                    print('  Applying Inverse Cuthill-McKee Permutation ... ', end='', flush=True)
                self.perm = reverse_cuthill_mckee(A)
                self.inv_perm = np.argsort(self.perm)

                I = sp.csr_matrix((np.ones((self.perm.size,)), (self.perm, np.arange(self.perm.size))))
                A = A @ I
                self._A = (A.T @ I).T
                if self.verbose:
                    print('done.')

            if self.precon:
                if self.verbose:
                    print('  Computing preconditionning matrix ... ', end='', flush=True)
                try:
                    self.Mpre = sp.linalg.spilu(self.A.tocsc())
                    self.Mpre = sp.linalg.LinearOperator(self.A.shape, self.Mpre.solve)
                except RuntimeError as err:
                    if self.verbose:
                        print(err)
                        print('Switching to using diagonal of A')
                    Ainv = sp.spdiags(1./self.A.diagonal(), 0, self.A.shape[0], self.A.shape[0])
                    self.Mpre = sp.linalg.aslinearoperator(Ainv)
                if self.verbose:
                    print('done.')

            self.solver = lambda A, b : slv(A, b, x0=self.x0, tol=self.tol, max_iter=max_it, M=self.Mpre)
        elif solver[0] == 'mumps':
            self.ctx = mumps.DMumpsContext(sym=0, par=1, comm=solver[1])
            self.ctx.set_icntl(4, 1)  # print only error messages
            if self.ctx.myid == 0:
                self.ctx.set_centralized_sparse(A)
            if verbose:
                print('    Factorizing matrix A ... ', end='', flush=True)
            self.ctx.run(job=4)  # Analysis & Factorization
            if verbose:
                print('done.')
            self.solver = self._solve_mumps
        elif solver[0] == 'pardiso':
            self.solver = pypardiso.spsolve
            self.pardiso = True
        elif solver[0] == 'pastix':
            self.pastix_solver = pypastix.solver(A)
            self.solver = lambda A, b : self.pastix_solver.solve(b.flatten(), refine=False)
        elif solver[0] == 'umfpack':
            self.want_umfpack = True
            umf_family, A = _get_umf_family(A.tocsc().sorted_indices())
            self.umfpack = um.UmfpackContext(umf_family)
            self.umfpack.control[um.UMFPACK_PRL]
            # use_solver(useUmfpack=True, assumeSortedIndices=True)
            if verbose:
                print('    Factorizing matrix A ... ', end='', flush=True)
            self.umfpack.numeric( A )
            # solve = factorized(A.tocsc().sorted_indices())
            if verbose:
                print('done.')
            # self.solver = lambda A, b : solve(np.array(b).flatten())
            umf = self.umfpack
            def slv(AA, b):
                with np.errstate(divide="ignore", invalid="ignore"):
                    # Ignoring warnings with numpy >= 1.23.0, see gh-16523
                    result = umf.solve(um.UMFPACK_A, A, b.flatten(), autoTranspose=True)
                    # umf.report_info()
                return result
            self.solver = slv
        elif solver[0] == 'superlu':
            self.want_superlu = True
            use_solver(useUmfpack=False)
            if verbose:
                print('    Factorizing matrix A ... ', end='', flush=True)
            solve = factorized(A.tocsc())
            if verbose:
                print('done.')
            self.solver = lambda A, b : solve(b.flatten())
        else:
            raise RuntimeError('Solver not implemented')

    def __del__(self):

        if self.pastix_solver is not None:
            self.pastix_solver.finalize()
        elif self.umfpack is not None:
            try:
                del(self.umfpack)
            except AttributeError as e:
                print(e)
        elif self.ctx is not None:
            self.ctx.destroy()

    def solve(self, rhs, x0=None, verbose=None):
        """
        Solve Ax = b

        Parameters
        ----------
        rhs : arraylike
            right hand side term.
        x0 : arraylike, optional
            Initial solution initiale.
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
            print('      Solving system ...', end='', flush=True)
            
        # direct solvers mumps & pardiso can take matrices as rhs

        if self.ctx is not None and rhs.ndim == 2:
            assert self.do_perm is False
            if self.ctx.myid == 0:
                if sp.isspmatrix(rhs):
                    x = rhs.toarray(order='F')
                else:
                    # we must have a numpy array
                    x = rhs.copy(order='F')
                self.ctx.set_rhs(x)
            self.ctx.run(job=3)  # Solve
            if self.ctx.myid == 0:
                if verbose:
                    print(' done.')
                return x
        elif self.pardiso is True and rhs.ndim == 2:
            assert self.do_perm is False
            if sp.isspmatrix(rhs):
                x = rhs.toarray(order='F')
            else:
                # we must have a numpy array
                x = rhs.copy(order='F')
            x = self.solver(self._A, x)
            if verbose:
                print(' done.')
            return x
            
        if verbose > 1:
            res = []
            
        if rhs.ndim == 1:
            rhs = np.atleast_2d(rhs).T
            
        if sp.isspmatrix(rhs):
            rhs = rhs.toarray()

        v = np.empty((self._A.shape[0], rhs.shape[1]))
        for ns in range(rhs.shape[1]):
            if verbose:
                if ns == 0:
                    msg = ''
                    pre_msg = ''
                nback = len(msg) - len(pre_msg)
                pre_msg = ''
                for _ in range(nback):
                    pre_msg += '\b'
                msg = pre_msg+'  iteration '+str(ns+1)+'/'+str(rhs.shape[1])+' '
                print(msg, end='', flush=True)

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
                print('{0:}: convergence not achieved, stopped after {1:d} \
    iterations for tol = {2:g}'.format(self.solver.__name__, info, self.tol))
            elif info < 0:
                print('{0:s}: illegal input or breakdown, switching to spsolve'.format(self.solver.__name__))
                u = spsolve(self._A, qi)
                
            if verbose > 1:
                res.append(np.linalg.norm(self._A @ u - np.array(qi).flatten()))

            if self.do_perm:
                v[:, ns] = u[self.inv_perm]
            else:
                v[:, ns] = u

        if verbose:
            print('done.')
            
        if verbose > 1:
            print('        norm of residuals =', end='')
            [ print('   {:3.2e}'.format(x), end='') for x in res ]
            print('')

        return v
    
    @property
    def A(self):
        """Matrix to invert."""
        return self._A
    
    @A.setter
    def A(self, val):
        self._A = val
        if self.pastix_solver is not None:
            self.pastix_solver.setup(val)
        elif self.want_umfpack:
            umf_family, A = _get_umf_family(val.tocsc().sorted_indices())
            self.umfpack = um.UmfpackContext(umf_family)
            self.umfpack.control[um.UMFPACK_PRL] 
            self.umfpack.numeric( A )
            umf = self.umfpack
            def slv(AA, b):
                with np.errstate(divide="ignore", invalid="ignore"):
                    # Ignoring warnings with numpy >= 1.23.0, see gh-16523
                    result = umf.solve(um.UMFPACK_A, A, b.flatten(), autoTranspose=True)
                    # umf.report_info()
                return result
            self.solver = slv
        elif self.want_superlu:
            solve = factorized(val.tocsc())
            self.solver = lambda A, b : solve(b.flatten())
        elif self.ctx is not None:
            # we are using MUMPS
            if val.shape[0] != val.shape[1]:
                raise RuntimeError('MUMPS: matrix must be square')
            if self.ctx.myid == 0:
                self.ctx.set_shape(val.shape[0])
                self.ctx.set_centralized_sparse(val)
            if self.verbose:
                print('    Factorizing matrix A ... ', end='', flush=True)
            self.ctx.run(job=4)  # Analysis & Factorization
            if self.verbose:
                print('done.')

        # solveur itératif
        if self.do_perm:
            if self.verbose:
                print('  Applying Inverse Cuthill-McKee Permutation ... ', end='', flush=True)
            self.perm = reverse_cuthill_mckee(val)
            self.inv_perm = np.argsort(self.perm)

            I = sp.csr_matrix((np.ones((self.perm.size,)), (self.perm, np.arange(self.perm.size))))
            val = val @ I
            self._A = (val.T @ I).T
            if self.verbose:
                print('done.')

        if self.precon:
            if self.verbose:
                print('  Computing preconditionning matrix ... ', end='', flush=True)
            try:
                self.Mpre = sp.linalg.spilu(self.A.tocsc())
                self.Mpre = sp.linalg.LinearOperator(self.A.shape, self.Mpre.solve)
            except RuntimeError as err:
                if self.verbose:
                    print(err)
                    print('Switching to using diagonal of A')
                Ainv = sp.spdiags(1./self.A.diagonal(), 0, self.A.shape[0], self.A.shape[0])
                self.Mpre = sp.linalg.aslinearoperator(Ainv)
            if self.verbose:
                print('done.')
            
    def _solve_mumps(self, A, b):
        if self.ctx.myid == 0:
            x = b.flatten().copy()
            self.ctx.set_rhs(x)
        self.ctx.run(job=3)  # Solve
        if self.ctx.myid == 0:
            return x
    

# %% main
if __name__ == '__main__':

    mkFig = False
    test_solver = False

    x = [1, 2, 3, 3.5]
    y = [1, 2.5, 3, 4, 5.5]
    z = np.arange(6)
    gvf = GridFV(x, y, z)

    print(gvf.nc, gvf.dx)

    ind = gvf.ind([1, 2], 2, [3, 0])
    print(ind)

    B0 = np.array([1., 2., 3.])
    chi = np.zeros((gvf.nc,))
    chi[gvf.ind(2, 2, 3)] = 1.0
    mu0 = 4 * np.pi * 1.e-7
    mu = mu0 * (1.+chi)
    D = gvf.build_D()
    M = gvf.build_M(mu)
    G = gvf.build_G()

    gvf.toVTK({'chi': chi}, 'chi')
    C1 = gvf._curl_edges()
    C2 = gvf._curl_faces()
    
    xyzc = gvf.centre_voxels()
