import os
from pathlib import Path

import ufl

os.environ["PETSC_DIR"] = r"C:\Users\obour\GitHub\petsc"
os.environ["PETSC_ARCH"] = "arch-msvc-intelmpi-mkl"
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

_dll_directories = [
    Path(r"C:\Users\obour\GitHub\fenicsx-install\bin"),
    Path(r"C:\Users\obour\GitHub\petsc\arch-msvc-intelmpi-mkl\lib"),
    Path(r"C:\Program Files (x86)\Intel\oneAPI\mkl\2026.1\bin"),
    Path(r"C:\Program Files (x86)\Intel\oneAPI\compiler\2026.1\bin"),
    Path(r"C:\Program Files (x86)\Intel\oneAPI\mpi\2021.18\bin"),
    Path(
        r"C:\Program Files (x86)\Intel\oneAPI"
        r"\mpi\2021.18\opt\mpi\libfabric\bin"
    ),
]

_dll_handles = [
    os.add_dll_directory(str(directory))
    for directory in _dll_directories
    if directory.is_dir()
]

# Toujours charger PETSc avant DOLFINx
from petsc4py import PETSc

if not PETSc.Sys.hasExternalPackage("mkl_pardiso"):
    raise RuntimeError("Le PETSc chargé ne contient pas MKL PARDISO.")

from dolfinx import fem, mesh, io
from mpi4py import MPI
import numpy as np
from itertools import combinations
from dolfinx.fem.petsc import (
    LinearProblem,
    assemble_matrix,
    discrete_curl,
    discrete_gradient,
)
from dolfinx.io import gmsh as gmshio
from dolfinx.plot import vtk_mesh
import dolfinx.mesh as dmesh
import dolfinx.cpp.mesh
import gmsh
from petsc4py import PETSc
from scipy.sparse import csr_matrix

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

from pymmr.finite_volume import BaseFV, GridFV, Solver

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
    A_new.indptr = np.array(A.indptr, copy=False, dtype=np.int64)
    A_new.indices = np.array(A.indices, copy=False, dtype=np.int64)

    return family, A_new


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


class BaseFE(BaseFV):
    def func():
        pass


class GridFE(BaseFE):
    """Class to manage unstructured grids for finite element modelling.

    Parameters
    ----------
    param: tuple
        parameters to instantiate the finite element grid
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
        BaseFE.__init__(self, comm, n_threads)
        x, y, z = param
        self.x = x
        self.y = y
        self.z = z
        self.dim = 3


        print("Dans Finite element")

        # Initialize the mesh
        gmsh.initialize()
        self.model = gmsh.model # the model
        self.model.add("GridFE")

        # Create a box in Gmsh
        self.volume_tag = self.model.occ.addBox(0.0, 0.0, -(self.z[-1]-self.z[0]), 
                                            self.x[-1]-self.x[0], self.y[-1]-self.y[0], self.z[-1]-self.z[0])

        # Set the mesh size for the box with a single size value
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.x[1]-self.x[0])
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.x[1]-self.x[0])


        self.model.occ.synchronize() # Synchronize changes

        volume_group = self.model.addPhysicalGroup(
            3, [self.volume_tag], tag=1
        )
        self.model.setPhysicalName(3, volume_group, "domain")

        self.model.mesh.generate(self.dim) # Generate the mesh

        mesh_data = gmshio.model_to_mesh(self.model, MPI.COMM_WORLD, rank=0, gdim=3)

        self.domain = mesh_data.mesh
        self.cell_tags = mesh_data.cell_tags
        self.facet_tags = mesh_data.facet_tags

        nodeTags, self.nodes, _ = self.model.mesh.getNodes()
        elementType = self.model.mesh.getElementType("tetrahedron", 1)
        faceNodes = self.model.mesh.getElementFaceNodes(elementType, 3)
        edgeNodes = self.model.mesh.getElementEdgeNodes(elementType)


        self.nodes_idx = np.array(nodeTags, dtype=int)
        self.nodes = np.reshape(self.nodes, (int(len(self.nodes)/3), 3))
        self.faces_idx = np.reshape(faceNodes, (int(len(faceNodes)/3), 3))

        edges = np.reshape(edgeNodes, (int(len(edgeNodes)/2), 2))

        def get_unique_edges(lst):
            unique_entries = set()

            for entry in lst:
                sorted_entry = tuple(sorted(entry))
                unique_entries.add(sorted_entry)

            return np.array([list(entry) for entry in unique_entries])

        self.edges_idx = get_unique_edges(edges)

        self.nc = self.volume_voxels().size # Total number of voxels
        self.nf = self.faces_idx.shape[0] # Total number of faces
        self.ne = self.edges_idx.shape[0] # Total number of edges

        self.xc = self.centre_voxels()[0] # X coordinates of the voxel centers
        self.yc = self.centre_voxels()[1] # Y coordinates of the voxel centers
        self.zc = self.centre_voxels()[2] # Z coordinates of the voxel centers


        # This line halts script execution and opens the interactive viewer
        gmsh.fltk.run()


    def _build_entity_indices(self):
        gm = self.model.mesh
        tet_type = gm.getElementType("tetrahedron", 1)
        types, tags, _ = gm.getElements(dim=3)
        if list(types) != [tet_type]:
            raise ValueError("Expected a mesh of first-order tetrahedra")

        self.cell_element_tags = tags[0]
        edges = gm.getElementEdgeNodes(tet_type).reshape(-1, 2)
        faces = gm.getElementFaceNodes(tet_type, 3).reshape(-1, 3)

        self.edges_idx = np.unique(np.sort(edges, axis=1), axis=0)
        self.faces_idx = np.unique(np.sort(faces, axis=1), axis=0)

        self._entity_indices = {
            None: {tag: i for i, tag in enumerate(self.cell_element_tags)},
            "edge": {tuple(nodes): i for i, nodes in enumerate(self.edges_idx)},
            "face": {tuple(nodes): i for i, nodes in enumerate(self.faces_idx)},
        }


    def ind(self, entity, component=None):
        """Return the zero-based index of a cell, edge, or face."""
        self._build_entity_indices()

        if component not in self._entity_indices:
            raise ValueError("component must be None, 'edge', or 'face'")

        if component is None and isinstance(entity, tuple) and len(entity) == 3:
            axes = tuple(np.asarray(ids) for ids in entity)

        if all(ids.ndim == 1 for ids in axes):
            for ids in axes:
                if ids.size and ids.dtype.kind not in "iu":
                    raise TypeError("Cell indices must be integers")
                if np.any((ids < 0) | (ids >= self.nc)):
                    raise IndexError("Cell index outside grid")

            ix, iy, iz = (ids.astype(np.intp, copy=False) for ids in axes)
            return np.intersect1d(np.intersect1d(ix, iy), iz)

        entities = np.asarray(entity)
        if entities.size and entities.dtype.kind not in "iu":
            raise TypeError("Gmsh tags must be integers")

        index = self._entity_indices[component]
        try:
            if component is None:
                result = [index[tag] for tag in entities.ravel()]
                shape = entities.shape
            else:
                width = 2 if component == "edge" else 3
                if entities.ndim == 0 or entities.shape[-1] != width:
                    raise ValueError(f"Each {component} requires {width} node tags")

                keys = np.sort(entities.reshape(-1, width), axis=1)
                result = [index[tuple(key)] for key in keys]
                shape = entities.shape[:-1]
        except KeyError as exc:
            raise IndexError(f"Unknown mesh entity: {exc.args[0]}") from None

        if shape == ():
            return int(result[0])
        return np.asarray(result, dtype=np.intp).reshape(shape)

    # def ind(self, i, component=None):
    #     """
    #     Return the coordinate of a node from its gmsh index.
    #     """

    #     idx = np.where(self.nodes_idx == i)

    #     return self.nodes[idx[0]][0] if len(idx) > 0 else None
    

    def revind(self, coord, component=None):
        """
        Return the gmsh index of a node from its coordinate.
        """

        idx = np.where(np.all(self.nodes == coord, axis=1))

        return self.nodes_idx[idx[0]][0] if len(idx) > 0 else None


    def __update_n(self):
        # if "_x" in self.__dict__ and "_y" in self.__dict__ and "_z" in self.__dict__:
        # Divergence & gradient matrices are precomputed because essentially always used
        self.D = self.build_D()
        self.G = self.build_G()


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

        _, element_tags, _ = self.model.mesh.getElements(dim=3)

        for e_tags in element_tags:
            # Get the volume for each element tag in this type
            volumes = self.model.mesh.getElementQualities(e_tags, "volume")

        return volumes


    def centre_voxels(self):
        """Returns coordinates at the centres of voxels.

        Returns
        -------
        tuple of 3 ndarray
            coordinates X, Y, Z
        """

        cell_centers = []

        element_types, _, _ = gmsh.model.mesh.getElements(dim=3, tag=-1)

        for element_type in element_types:

            centers = gmsh.model.mesh.getBarycenters(elementType=element_type, tag=-1, fast=False, primary=True)
            centers = np.asarray(centers).reshape(-1, 3)

            cell_centers.append(centers)

        cell_centers = np.vstack(cell_centers)

        X, Y, Z = cell_centers[:, 0], cell_centers[:, 1], cell_centers[:, 2]

        return X, Y, Z


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
        self.nc = self.volume_voxels().size
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

        pass


    def build_A(self, sigma, bcs=()):
        """Build LHS matrix for DC resistivity forward modeling.

        Parameters
        ----------
        sigma : array_like
            conductivity model

        bcs : list of DirichletBC, optional
            Boundary conditions

        Returns
        -------
        tuple of 2 matrices:
            - A : LHS matrix
            - M : matrix of harmonic average of sigma
        """
        V_space = fem.functionspace(self.domain, ("Nedelec 1st kind H(curl)", 1)) # Lagrange 1 (DG, 0)

        u = ufl.TrialFunction(V_space)
        v = ufl.TestFunction(V_space)

        a = fem.form(
            sigma * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        )

        A = fem.petsc.assemble_matrix(a, bcs=bcs)
        A.assemble()

        return A, None


    def build_D(self):
        """Construction of divergence matrix.

        Returns
        -------
        csr_matrix
            Divergence matrix

        https://docu.ngsolve.org/v6.2.1907/i-tutorials/unit-2.3-hcurlhdiv/hcurlhdiv.html

        """
        mesh = self.domain

        V_face = fem.functionspace(mesh, ("RT", 1))  # Raviart-Thomas elements for H(div)
        Q_cells = fem.functionspace(mesh, ("DG", 0)) # Discontinuous Lagrange elements for scalar fields

        u = ufl.TrialFunction(V_face) # Function defined on faces (Raviart-Thomas)
        v = ufl.TestFunction(Q_cells) # Function defined on cells (Discontinuous Lagrange)

        div_form = fem.form(ufl.div(u) * v * ufl.dx) # 

        D = assemble_matrix(div_form) # Assemble the divergence matrix
        D.assemble() # assemble the matrix


        return D


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
        pass


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
        node_tags, node_coordinates, _ = self.model.mesh.getNodes()
        
        node_tags = np.asarray(node_tags, dtype=np.int64)
        coordinates = np.asarray(node_coordinates).reshape(-1, 3)

        element_tags, element_node_tags = (self.model.mesh.getElementsByType(4))

        connectivity_tags = np.asarray(element_node_tags, dtype=np.int64).reshape(-1, 4)

        # Convert Gmsh node tags to array indices
        order = np.argsort(node_tags)
        positions = np.searchsorted(node_tags[order], connectivity_tags)

        connectivity = order[positions]

        # Coordinates of tetrahedron nodes
        X = coordinates[connectivity]  # (n_cells, 4, 3)

        # Geometrical transformation matrix
        A = X[:, 1:, :] - X[:, 0:1, :]

        # Reference operator:
        # [u1-u0, u2-u0, u3-u0]
        C = np.array([
            [-1.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 1.0],
        ])

        # Local gradient matrices, shape (n_cells, 3, 4)
        B = np.linalg.solve(A, np.broadcast_to(C, (A.shape[0], 3, 4)))

        n_cells = connectivity.shape[0]
        n_nodes = node_tags.size

        # Assemble global sparse gradient matrix
        rows = np.repeat(np.arange(3 * n_cells), 4)

        cols = np.repeat(connectivity[:, None, :], 3, axis=1).ravel()

        G = sp.coo_matrix(
            (B.ravel(), (rows, cols)),
            shape=(3 * n_cells, n_nodes)
        ).tocsr()

        return G



    def _gradient_centre(self):
        pass


    def _gradient_faces(self, v):
        pass


    def build_G_faces(self):
        """Construction of gradient matrix to use with vector defined on edges.

        Returns
        -------
        csr_matrix
            Gradient matrix
        """
        pass


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

        https://docu.ngsolve.org/v6.2.1907/i-tutorials/unit-2.3-hcurlhdiv/hcurlhdiv.html

        """

        if to_faces:
            return self._curl_faces()
        else:
            return self._curl_edges()


    def _curl_faces(self):

        V_edges = fem.functionspace(self.domain, ("N1curl", 1))
        
        V_faces = fem.functionspace(self.domain, ("RT", 1))

        C = discrete_curl(V_edges, V_faces)

        C.assemble()
        
        return C


    def _curl_edges(self):

        V_edges = fem.functionspace(self.domain, ("N1curl", 1))
                
        V_faces = fem.functionspace(self.domain, ("RT", 1))

        C = discrete_curl(V_edges, V_faces)

        C.assemble()

        return C.transpose()


    def extract_Gxyz(self):
        """Extract Gx, Gy & Gz from G.
        """
        G = self.build_G()

        return G[0::3, :].tocsr().T, G[1::3, :].tocsr().T, G[2::3, :].tocsr().T


    def extract_xyz_faces(self, v):
        V_faces = fem.functionspace(self.domain, ("RT", 1))
        if isinstance(v, fem.Function):
            field = v
        else:
            values = v.getArray(readonly=True) if isinstance(v, PETSc.Vec) else v
            values = np.asarray(values).reshape(-1)
            field = fem.Function(V_faces)
            owned = V_faces.dofmap.index_map.size_local * V_faces.dofmap.index_map_bs
            if values.size not in (owned, field.x.array.size):
                raise ValueError("v must contain the owned RT DoFs, optionally with ghosts")
            field.x.array[:values.size] = values
        field.x.scatter_forward()

        topology = self.domain.topology
        cell_dim = topology.dim
        face_dim = cell_dim - 1
        topology.create_entities(face_dim)
        topology.create_connectivity(face_dim, cell_dim)
        topology.create_connectivity(face_dim, 0)
        face_cells = topology.connectivity(face_dim, cell_dim)
        faces = np.arange(topology.index_map(face_dim).size_local, dtype=np.int32)
        midpoints = dmesh.compute_midpoints(self.domain, face_dim, faces)

        counts = np.array([len(face_cells.links(int(f))) for f in faces], dtype=np.int32)
        if np.any(counts == 0):
            raise RuntimeError("A face has no adjacent cell")
        face_ids = np.repeat(faces, counts)
        cell_ids = np.concatenate([face_cells.links(int(f)) for f in faces]).astype(np.int32)
        side_values = field.eval(midpoints[face_ids], cell_ids)

        xyz = np.zeros((faces.size, 3), dtype=side_values.dtype)
        np.add.at(xyz, face_ids, side_values)
        xyz /= counts[:, None]

        # For a tetrahedral face, determine which Cartesian direction its
        # normal is closest to. Work in DOLFINx geometry ordering, not the
        # (potentially repeated) Gmsh self.faces_idx ordering.
        geometry_ids = dmesh.entities_to_geometry(self.domain, face_dim, faces)
        corners = self.domain.geometry.x[geometry_ids[:, :3]]
        normals = np.cross(corners[:, 1] - corners[:, 0],
                           corners[:, 2] - corners[:, 0])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        direction = np.argmax(np.abs(normals), axis=1)
        # Choose the sign so the largest normal component points along +x,
        # +y, or +z, independent of the triangle's vertex ordering.
        orientation = np.where(normals[np.arange(faces.size), direction] < 0,
                               -1.0, 1.0)
        normals *= orientation[:, None]

        normal_value = np.einsum("ij,ij->i", xyz, normals)
        self.face_xyz_indices = tuple(np.flatnonzero(direction == k)
                                      for k in range(3))
        return tuple(normal_value[ids, None] for ids in self.face_xyz_indices)


    def below_surface(self, pts):
        """Return indices of electrodes that are not at the surface."""
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("pts must have shape (n, 3)")
        return pts[:, 2] != 0.0


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

        elec = np.array(elec, dtype=float, copy=True)
        zc = np.asarray(self.centre_voxels()[2], dtype=float)
        
        top_center = np.max(zc)
        finite = np.all(np.isfinite(elec), axis=1)
        elec[finite & (elec[:, 2] > top_center), 2] = top_center
        return elec


    def solve(self, b):
        return self.solver_A.solve(b)


    def compute_u_homog(self, c1c2, sigma, cs, points=None):
        """Compute the potential at the center of the voxels for a given set of electrodes.

        Note
        ----
        Current intensity is multiplied by -1 because we are working with elevation (z positive upward) and first
        electrode is the source (current "going down in the ground") and second is the sink (current "coming back")
        """
        from scipy.spatial import cKDTree

        electrodes = np.array(c1c2, dtype=float, copy=True)
        if electrodes.ndim == 1:
            electrodes = electrodes.reshape(1, -1)
        
        currents = np.asarray(cs, dtype=float).reshape(-1)
        nsrc = electrodes.shape[0]
        if currents.size == 1:
            currents = np.full(nsrc, currents[0])

        centres = np.column_stack(self.centre_voxels())

        at_centres = points is None
        xyz = centres if at_centres else np.asarray(points, dtype=float)

        electrodes = electrodes.reshape(nsrc, electrodes.shape[1] // 3, 3)
        finite = np.all(np.isfinite(electrodes), axis=2)
        # Record the same surface/depth flags as GridFV BEFORE correction.
        # Retain its pair-level image convention, including remote electrodes.
        buried = np.any(electrodes[:, :, 2] != 0.0, axis=1)
        for k in range(electrodes.shape[1]):
            electrodes[:, k, :] = self.process_surface_elec(electrodes[:, k, :])

        u0 = np.zeros((xyz.shape[0], nsrc))
        for i in range(nsrc):
            if currents[i] == 0.0:
                continue
            if electrodes.shape[1] == 2 and np.array_equal(electrodes[i, 0], electrodes[i, 1]):
                continue  # coincident source and sink cancel, even at the source
            gf = 4.0 if buried[i] else 2.0
            for k in range(electrodes.shape[1]):
                if not finite[i, k]:
                    continue  # 1/r = 0 for an electrode at infinity
                electrode = electrodes[i, k]
                r = np.linalg.norm(xyz - electrode, axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    term = 1.0 / r
                    if buried[i]:
                        image = electrode * np.array([1.0, 1.0, -1.0])
                        term += 1.0 / np.linalg.norm(xyz - image, axis=1)
                    sign = 1.0 if k == 0 else -1.0
                    u0[:, i] += -currents[i] * sign / (sigma * gf * np.pi) * term

        # Tetrahedra sharing three vertices share a face: the unstructured
        # equivalent of the adjacent voxels used by the FV implementation.
        if not np.all(np.isfinite(u0)):
            element_type = self.model.mesh.getElementType("tetrahedron", 1)
            _, node_tags = self.model.mesh.getElementsByType(element_type)
            tetrahedra = np.asarray(node_tags).reshape(-1, 4)
            
            original = u0.copy()
            for cell, i in np.argwhere(~np.isfinite(original)):
                neighbours = np.flatnonzero(np.sum(np.isin(tetrahedra, tetrahedra[cell]), axis=1) == 3)
                values = original[neighbours, i]
                values = values[np.isfinite(values)]
                
                u0[cell, i] = np.mean(values)

        if at_centres:
            # DOLFINx reorders Gmsh cells and DOFs. Match DG0 DOF coordinates
            # to Gmsh centres instead of assigning u0 in the wrong order.
            Q = fem.functionspace(self.domain, ("DG", 0))
            dof_xyz = Q.tabulate_dof_coordinates()
            distance, gmsh_cell = cKDTree(centres).query(dof_xyz)
            scale = max(1.0, float(np.max(np.abs(centres))))
            tol = 100 * np.finfo(self.domain.geometry.x.dtype).eps * scale
            
            functions = []
            for i in range(nsrc):
                field = fem.Function(Q)
                field.name = f"u_homog_{i}"
                field.x.array[:] = u0[gmsh_cell, i]
                field.x.scatter_forward()
                functions.append(field)
            self.u_homog_functions = functions

        return u0

    
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

        xCoords = vtk.vtkFloatArray()
        yCoords = vtk.vtkFloatArray()
        zCoords = vtk.vtkFloatArray()

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
            field = fields[fn]
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
        data = reader.GetOutput().GetCellData().GetArray(fieldname)
        if data is None:
            return data
        else:
            return vtk_to_numpy(data)


    def print_info(self, file=None):
        print('      X min: {0:e}\tX max: {1:e}'.format(self.x[0], self.x[-1]), file=file)
        print('      Y min: {0:e}\tY max: {1:e}'.format(self.y[0], self.y[-1]), file=file)
        print('      Z min: {0:e}\tZ max: {1:e}'.format(self.z[0], self.z[-1]), file=file)



class MeshFE(BaseFE, SimplexMesh):
    """Class to manage tetrahedral meshes for finite element modelling.

    Parameters
    ----------
    param: tuple
        parameters to instantiate the finite element mesh
        pts : array_like
            Coordinates of the nodes making the mesh
        tet : array_like of int
            Indices of the nodes forming the tetrahedra
        tri_surf : array_like of int
            Indices of the nodes forming the triangulated surface of the ground
    """

    def __init__(self, param, comm=None, n_threads=1):
        BaseFE.__init__(self, comm, n_threads)
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
            return sp.diags(1.0 / tmp, format='csr')
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
        V_space = fem.functionspace(self.domain, ("Nedelec 1st kind H(curl)", 1)) # Lagrange 1 (DG, 0)
        
        u = ufl.TrialFunction(V_space)
        v = ufl.TestFunction(V_space)

        a = fem.form(
            sigma * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        )

        A = fem.petsc.assemble_matrix(a)
        A.assemble()

        return A


    def compute_u_homog(self, c1c2, sigma, cs, points=None):
        from scipy.spatial import cKDTree
        
        electrodes = np.array(c1c2, dtype=float, copy=True)
        if electrodes.ndim == 1:
            electrodes = electrodes.reshape(1, -1)
        
        currents = np.asarray(cs, dtype=float).reshape(-1)
        nsrc = electrodes.shape[0]
        if currents.size == 1:
            currents = np.full(nsrc, currents[0])

        centres = np.column_stack(self.centre_voxels())

        at_centres = points is None
        xyz = centres if at_centres else np.asarray(points, dtype=float)

        electrodes = electrodes.reshape(nsrc, electrodes.shape[1] // 3, 3)
        finite = np.all(np.isfinite(electrodes), axis=2)
        # Record the same surface/depth flags as GridFV BEFORE correction.
        # Retain its pair-level image convention, including remote electrodes.
        buried = np.any(electrodes[:, :, 2] != 0.0, axis=1)
        for k in range(electrodes.shape[1]):
            electrodes[:, k, :] = self.process_surface_elec(electrodes[:, k, :])

        u0 = np.zeros((xyz.shape[0], nsrc))
        for i in range(nsrc):
            if currents[i] == 0.0:
                continue
            if electrodes.shape[1] == 2 and np.array_equal(electrodes[i, 0], electrodes[i, 1]):
                continue  # coincident source and sink cancel, even at the source
            gf = 4.0 if buried[i] else 2.0
            for k in range(electrodes.shape[1]):
                if not finite[i, k]:
                    continue  # 1/r = 0 for an electrode at infinity
                electrode = electrodes[i, k]
                r = np.linalg.norm(xyz - electrode, axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    term = 1.0 / r
                    if buried[i]:
                        image = electrode * np.array([1.0, 1.0, -1.0])
                        term += 1.0 / np.linalg.norm(xyz - image, axis=1)
                    sign = 1.0 if k == 0 else -1.0
                    u0[:, i] += -currents[i] * sign / (sigma * gf * np.pi) * term

        # Tetrahedra sharing three vertices share a face: the unstructured
        # equivalent of the adjacent voxels used by the FV implementation.
        if not np.all(np.isfinite(u0)):
            element_type = self.model.mesh.getElementType("tetrahedron", 1)
            _, node_tags = self.model.mesh.getElementsByType(element_type)
            tetrahedra = np.asarray(node_tags).reshape(-1, 4)
            
            original = u0.copy()
            for cell, i in np.argwhere(~np.isfinite(original)):
                neighbours = np.flatnonzero(np.sum(np.isin(tetrahedra, tetrahedra[cell]), axis=1) == 3)
                values = original[neighbours, i]
                values = values[np.isfinite(values)]
                
                u0[cell, i] = np.mean(values)

        if at_centres:
            # DOLFINx reorders Gmsh cells and DOFs. Match DG0 DOF coordinates
            # to Gmsh centres instead of assigning u0 in the wrong order.
            Q = fem.functionspace(self.domain, ("DG", 0))
            dof_xyz = Q.tabulate_dof_coordinates()
            distance, gmsh_cell = cKDTree(centres).query(dof_xyz)
            scale = max(1.0, float(np.max(np.abs(centres))))
            tol = 100 * np.finfo(self.domain.geometry.x.dtype).eps * scale
            
            functions = []
            for i in range(nsrc):
                field = fem.Function(Q)
                field.name = f"u_homog_{i}"
                field.x.array[:] = u0[gmsh_cell, i]
                field.x.scatter_forward()
                functions.append(field)
            self.u_homog_functions = functions

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


class Solver(Solver):
    def func():
        pass



# if __name__ == "__main__":

#     x = np.arange(-60.0 * 16, 60.01 * 16, 960)
#     y = np.arange(-60.0 * 16, 60.01 * 16, 960)
#     z = np.arange(-60.0 * 16, 0.01, 960)

#     g = GridFE((x, y, z))

#     c1c2 = np.array([0., 0., 0., 0., 0., -10.]).reshape(1, 6)

#     print(g.compute_u_homog(c1c2, 1/1000, [1.0]))


#     gmsh.finalize()