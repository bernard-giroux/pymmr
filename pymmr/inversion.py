#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for inverting ERT & MMR data.

@author: giroux

Important references
--------------------

For data weighting using the Jacobian

@article{abubakar2009inversion,
  title={Inversion algorithms for large-scale geophysical electromagnetic measurements},
  author={Abubakar, A and Habashy, TM and Li, M and Liu, J},
  journal={Inverse Problems},
  volume={25},
  number={12},
  pages={123012},
  year={2009},
  publisher={IOP Publishing}
}

@article{hu2009joint,
  title={Joint electromagnetic and seismic inversion using structural constraints},
  author={Hu, Wenyi and Abubakar, Aria and Habashy, Tarek M},
  journal={Geophysics},
  volume={74},
  number={6},
  pages={R99--R109},
  year={2009},
  publisher={Society of Exploration Geophysicists}
}




"""
from collections import namedtuple
import os
import pickle
import warnings

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from pymmr.finite_volume import Solver, has_mumps, has_pardiso, has_pastix, has_umfpack


# %%  Define namedtuple for input data

DataMMR = namedtuple("DataMMR", "xs xo data wt cs date")
DataERT = namedtuple("DataERT", "c1c2 p1p2 data wt cs date")


# %% Some functions


class RegPreconditioner:
    """Preconditioner of `cglscd`: factorization of the regularization matrix C.

    The Gauss-Newton matrix is J^T D^2 J + beta C, and J^T D^2 J has a rank of at most n_data.  With
    (beta C)^-1 as preconditioner, the preconditioned matrix is the identity plus a matrix of rank n_data
    or less, and CG converges in about n_data + 1 iterations whatever the conditioning.  C does not
    depend on beta: it is factorized once and the preconditioner is applied as C^-1 r / beta.

    Parameters
    ----------
    CTC : sparse matrix
        Regularization matrix (CTC = C.T @ C), symmetric positive (semi-)definite.
    solver_par : tuple, optional
        Solver parameters (`BaseFV.get_solver_params`).  An iterative solver is replaced by the first
        direct solver available among MUMPS, Pardiso, UMFPACK, PaStiX and SuperLU.
    comm : MPI Communicator, optional
        For MUMPS.
    n_threads : int, optional
        Number of threads.

    Notes
    -----
    C is singular without smallness term (al_s = 0): a shift of `SHIFT` times its largest diagonal
    term is added (without effect on the solution, and on the convergence if C is not singular).
    """

    SHIFT = 1e-6

    def __init__(self, CTC, solver_par=None, comm=None, n_threads=1):
        if solver_par is None or callable(solver_par[0]):
            solver_par = (self.direct_solver(),)
        C = sp.csr_array(CTC)
        C = C + sp.diags_array(np.full(C.shape[0], self.SHIFT * np.abs(C.diagonal()).max()), format='csr')
        self.solver = Solver(solver_par, A=C, comm=comm, n_threads=n_threads)
        # Some solvers (Pardiso) factorize at the first solve: solve once here so that a failure is raised
        # now, as RuntimeError, and not within cglscd
        try:
            x = self.solve(np.ones((C.shape[0], 1)), 1.0)
        except Exception as err:
            raise RuntimeError("solve with the factorized regularization matrix failed: {0}".format(err)) from err
        if not np.all(np.isfinite(x)):
            raise RuntimeError("factorization of the regularization matrix gives non-finite values")

    @staticmethod
    def direct_solver():
        """Name of the preferred direct solver available (SuperLU as last resort)."""
        for name, ok in (("mumps", has_mumps), ("pardiso", has_pardiso), ("umfpack", has_umfpack),
                         ("pastix", has_pastix)):
            if ok:
                return name
        return "superlu"

    def solve(self, r, beta):
        """Return (beta C)^-1 r."""
        return np.asarray(self.solver.solve(r.ravel())).reshape(-1, 1) / beta


def _diag_DtDJtJ(J, D):
    """Diagonal of J^T D^T D J, without copy of J if D is diagonal."""
    d = D.diagonal() if sp.issparse(D) else np.diag(D)
    if sp.issparse(D) and (D - sp.diags_array(d)).count_nonzero() == 0:
        return np.einsum('ij,ij,i->j', J, J, d * d)
    DJ = D @ J
    return np.sum(DJ * DJ, axis=0)


def cglscd(J, x, b, beta, CTC, dxc, D, max_it, tol, reg_var, P=None, alpha=0, WTWt=0, precond=None):
    """Compute perturbation with a preconditioned conjugate-gradient least-squares algorithm.

    Minimizes  1/2 ||D (b - J x)||^2 + 1/2 beta R(x), i.e. solves (J^T D^2 J + beta C) x = J^T D^2 b - ...

    Parameters
    ----------
    J : ndarray
        Jacobian matrix.
    x : array_like
        initial solution.
    b : array_like
        dobs - d.
    beta : float
        Regularization factor.
    CTC : sparse matrix
        Regularization matrix (CTC = C.T @ C).
    dxc : array_like
        m - m_ref.
    D : sparse matrix
        Data weighting matrix.
    max_it : int or None
        Max number of iterations.  If None: 2 n_data + 100 with a `RegPreconditioner`, 1000 otherwise.
    tol : float
        Tolerance on the norm of the gradient, relative to its value for x = 0.
    reg_var : str
        Regularization variable:
        - 'model': R(x) = (x + dxc)^T C (x + dxc)
        - 'model perturbation': R(x) = x^T C x
        - 'time-lapse': R(x) = x^T C x, plus the term alpha WTWt dxc in the gradient
    P : array_like
        Constraint array to fix solution (0: fixed, 1: free).
    alpha : float
        Time-lapse regularization factor.
    WTWt : sparse matrix
        Time-lapse regularization matrix.
    precond : RegPreconditioner, optional
        Factorization of `CTC`.  If None, the diagonal of J^T D^2 J + beta C is used (Jacobi).

    Returns
    -------
    tuple: x, error, it
        x : ndarray
            Solution
        error : float
            Norm of error
        it : int
            number of iterations performed

    Notes
    -----

    The algorithm (with Jacobi preconditioning) is described in

    @article{article,
    author = {Günther, Thomas and Rücker, Carsten and Spitzer, Klaus},
    year = {2006},
    month = {08},
    pages = {506 - 517},
    title = {Three-dimensional modelling and inversion of DC resistivity data incorporating topography – II. Inversion},
    volume = {166},
    journal = {Geophysical Journal International},
    doi = {10.1111/j.1365-246X.2006.03011.x}
    }

    Without preconditioning by C, the tolerance of 1e-9 can take more than 10 000 iterations, and
    stopping early degrades the model (the regularization part of the step converges last).
    """
    if reg_var == "model":
        def reg(x_):
            return beta * (CTC @ (x_ + dxc))
    elif reg_var == "model perturbation":
        def reg(x_):
            return beta * (CTC @ x_)
    elif reg_var == "time-lapse":
        tl = alpha * (WTWt @ dxc.reshape(-1, 1))

        def reg(x_):
            return beta * (CTC @ x_) + tl
    else:
        raise ValueError("Wrong value for reg_var: " + str(reg_var))

    if max_it is None:
        max_it = 2 * J.shape[0] + 100 if precond is not None else 1000

    x = x.reshape(-1, 1)
    dxc = dxc.reshape(-1, 1)
    P = np.ones(x.shape) if P is None else P.reshape(-1, 1)
    x *= P

    if precond is None:
        M = _diag_DtDJtJ(J, D) + beta * CTC.diagonal()
        M[M == 0] = np.finfo(float).eps
        M = 1 / M.reshape(-1, 1)

        def apply_prec(r_):
            return M * r_
    else:
        def apply_prec(r_):
            return P * precond.solve(r_, beta)

    def gradient(x_, zz_):
        return P * ((zz_.T @ D) @ J).T - P * reg(x_)

    b = b.reshape(-1, 1)
    zz = D @ (b - J @ x)                                 # weighted data residual
    bnrm2 = np.linalg.norm((((D @ (D @ b)).T @ J).T - reg(np.zeros(x.shape))))   # gradient for x = 0
    if bnrm2 == 0.0:
        bnrm2 = 1.0
    r = gradient(x, zz)
    error = np.linalg.norm(r) / bnrm2

    it = -1
    rho_1 = 1.0
    p = None
    if error > tol:
        for it in range(max_it):
            z = apply_prec(r)
            rho = (r.T @ z).item()
            p = z if it == 0 else z + (rho / rho_1) * p
            q = D @ (J @ p)
            alpha_k = rho / (q.T @ q + beta * (p.T @ (CTC @ p))).item()
            x += alpha_k * p
            zz -= alpha_k * q
            r = gradient(x, zz)
            error = np.linalg.norm(r) / bnrm2
            if error <= tol:
                break
            rho_1 = rho

        if error > tol:
            warnings.warn(
                "cglscd: convergence not achieved after {0:d} it (tol = {1:3.2e}, error = {2:5.4e}).".format(
                    it + 1, tol, error
                ),
                stacklevel=2,
            )

    return x, error, it + 1


def calc_WdW(wt, dobs, par):
    """Compute data weighting matrix.

    Parameters
    ----------
    wt : array_like
        Measurement error in %.
    dobs : array_like
        Observed data.
    par :
        Weighting parameters.
    Returns
    -------
    output : `csr_matrix`
        Data weighting matrix.

    Notes
    -----
    The noise model is described in the book Eldad Haber (eq 6.6)
    Computational Methods in Geophysical Electromagnetics
    """
    dtw = 0.01 * wt.flatten() * np.abs(dobs.flatten()) + par.e
    dtw = 1. / dtw

    # normalisation
    dtw = dtw / dtw.max()

    # data with % error larger than cutoff have no weight
    dtw[wt.flatten() > par.max_err] = 0.0

    return sp.csr_array((dtw, (np.arange(dobs.size), np.arange(dobs.size))))


def df_to_data(df):
    """Create namedtuple from DataFrame."""

    c1c2 = np.c_[
        df["c1_x"].to_numpy().reshape(-1, 1),
        df["c1_y"].to_numpy().reshape(-1, 1),
        df["c1_z"].to_numpy().reshape(-1, 1),
        df["c2_x"].to_numpy().reshape(-1, 1),
        df["c2_y"].to_numpy().reshape(-1, 1),
        df["c2_z"].to_numpy().reshape(-1, 1),
    ]

    vintage = None
    if "date" in df:
        vintage = df["date"].to_numpy()

    if "Bx" in df:
        # we have MMR data
        try:
            data = np.c_[
                df["Bx"].to_numpy().reshape(-1, 1),
                df["By"].to_numpy().reshape(-1, 1),
                df["Bz"].to_numpy().reshape(-1, 1),
            ]
            xo = np.c_[
                df["obs_x"].to_numpy().reshape(-1, 1),
                df["obs_y"].to_numpy().reshape(-1, 1),
                df["obs_z"].to_numpy().reshape(-1, 1),
            ]
        except KeyError:
            raise ValueError("Invalid MMR data file")
        if "cs" in df:
            cs = df["cs"].to_numpy()
        else:
            warnings.warn("Source current not defined, using 1 A", stacklevel=2)
            cs = np.ones((data.shape[0],))
        if "wt_x" in df:
            wt = np.hstack(
                (
                    df["wt_x"].to_numpy(),
                    df["wt_y"].to_numpy(),
                    df["wt_z"].to_numpy(),
                )
            )
        else:
            warnings.warn("MMR measurement error not defined, using 1%", stacklevel=2)
            wt = np.ones((3 * data.shape[0],))

        if vintage is None:
            return DataMMR(xs=c1c2, xo=xo, data=data, wt=wt, cs=cs, date=None)
        else:
            # wt: all x, then all y, then all z
            wt = wt.reshape(3, -1)
            out = []
            for vin in np.unique(vintage):
                ind = vintage == vin
                out.append(DataMMR(xs=c1c2[ind, :], xo=xo[ind, :], data=data[ind, :], wt=wt[:, ind].ravel(),
                                   cs=cs[ind], date=vin))
            return out
    elif "V" in df:
        # we have ERT data
        try:
            data = df["V"].to_numpy()
            p1p2 = np.c_[
                df["p1_x"].to_numpy().reshape(-1, 1),
                df["p1_y"].to_numpy().reshape(-1, 1),
                df["p1_z"].to_numpy().reshape(-1, 1),
                df["p2_x"].to_numpy().reshape(-1, 1),
                df["p2_y"].to_numpy().reshape(-1, 1),
                df["p2_z"].to_numpy().reshape(-1, 1),
            ]
        except KeyError:
            raise ValueError("Invalid ERT data file")
        if "cs" in df:
            cs = df["cs"].to_numpy()
        else:
            warnings.warn("Source current not defined, using 1 A", stacklevel=2)
            cs = np.ones((data.shape[0],))
        if "wt" in df:
            wt = df["wt"].to_numpy()
        else:
            warnings.warn("ERT measurement error not defined, using 1%", stacklevel=2)
            wt = np.ones((data.shape[0],))

        if vintage is None:
            return DataERT(c1c2=c1c2, p1p2=p1p2, data=data, wt=wt, cs=cs, date=None)
        else:
            out = []
            for vin in np.unique(vintage):
                ind = vintage == vin
                out.append(DataERT(c1c2=c1c2[ind, :], p1p2=p1p2[ind, :], data=data[ind], wt=wt[ind], cs=cs[ind],
                                   date=vin))
            return out
    else:
        raise ValueError("Invalid DataFrame file")


class Inversion:
    def __init__(self):
        self.method = "Gauss-Newton"
        """Algorithm: 'Gauss-Newton', 'Quasi-Newton'."""

        self.param_transf = "log_conductivity"
        """working variable: 'conductivity', 'log_conductivity', 'resistivity', 'log_resistivity'."""

        self.beta = 2500.0
        """Regularization parameter."""

        self.beta_min = 100.0
        """Minimum value of beta."""

        self.beta_cooling = 2
        """Decrease factor of beta."""

        self.max_it = 5
        """Number of iterations."""

        # smoothing
        self.alx = 1.0
        """Weighting factor for smoothing along X."""

        self.aly = 1.0
        """Weighting factor for smoothing along Y."""

        self.alz = 1.0
        """Weighting factor for smoothing along Z."""

        self.als = 1e-2
        """Smallness factor."""

        # Data weighting
        self.e = 0.1
        """instrument error or error floor (in measurement units)."""

        self.max_err = 10.0
        """Acceptable measurement error in %."""

        self.beta_dw = 0.5
        """Beta for distance weighting."""

        self.smooth_type = "smooth"
        """Type of smoothing: 'smooth', 'blocky', 'ekblom', 'min. support'."""

        self.reg_var = "model"
        """Regularization variable: 'model', 'model perturbation', 'time-lapse'."""

        self.data_weighting = "variance"
        """Data weighting method: 'variance', 'jacobian', 'variance*jacobian', 'variance+jacobian'."""

        self.model_weighting = "distance"
        """Model weighting method: 'distance', 'jacobian'."""

        self.max_it_cglscd = None
        """Maximum number of iterations in cglscd (None: 2 n_data + 100, or 1000 with 'jacobi')."""

        self.tol_cglscd = 1.0e-9
        """Tolerance for cglscd."""

        self.precond_cglscd = "regularization"
        """Preconditioner of cglscd: 'regularization' (factorization of the regularization matrix, see
        `RegPreconditioner`) or 'jacobi' (diagonal of the Gauss-Newton matrix)."""

        self.max_step_size = 10
        """Maximum perturbation step."""

        self.sigma_max = 1.0e4
        """Upper cut-off value for conductivity."""

        self.sigma_min = 2.0e-5
        """Lower cut-off value for conductivity."""

        self.minsupport_e = 0.000001
        """Small value (epsilon) used in 'ekblom' and 'min. support' weights."""

        self.ekblom_p = 1.0
        """Exponent of the Ekblom measure, (x^2 + e^2)^(p/2), used with smooth_type 'ekblom'."""

        self.ekblom_smallness_max = 2.0
        """Upper limit of the weights of the smallness term with smooth_type 'ekblom' (None: no limit).
        Without limit, cells equal to the reference model get weights of p / minsupport_e."""

        self.minsupport_form = "portniaguine"
        """Weights for smooth_type 'min. support': 'portniaguine' (Portniaguine & Zhdanov 1999 reweighting,
        sqrt(2) e / (c^2 + e^2)) or 'general' (IRLS weight rho'(c)/c, 2 e^2 / (c^2 + e^2)^2)."""

        self.data_transf = None
        """Function to transform data: 'asinh' ou None."""

        self.verbose = True

        self.show_plots = False

        self.save_plots = False

        self.basename = "inv"

        self.checkpointing = False

        self.start_from_chkpt = False

    def run(
            self,
            g,
            m0,
            m_ref,
            data_mmr=None,
            data_ert=None,
            m_weight=None,
            m_active=None,
    ):
        """Run inversion.

        Parameters
        ----------
        g : Grid instance
            GridMMR (or GridDC if `data_mmr` is None)
        m0 : array_like
            Initial model (equal to m_ref if None).
        m_ref : array_like
            Reference model (S/m).
        data_mmr : DataMMR, optional
            MMR data.  B-field units are pT. Data weight (std dev) in %.
        data_ert : DataERT, optional
            DC resistivity data.  Voltage should be in mV. Data weight (std dev) in %.
        m_weight : array_like, optional
            Model weights.
        m_active : array_like, optional
            Part of the model allowed to change during inversion.

        Returns
        -------
        sigma_inv : list of ndarray
            Conductivity model at each iteration
        data_inv : list of ndarray
            Computed data at each iteration
        rms : list of float
            Weighted RMS error at each iteration
        misfit : list of float
            Misfit at each iteration
        smy : list of float
            Parameter variation function at each iteration

        Notes
        -----
        The diagonal elements of the weighting matrix built from data weights are

        1 / (0.01 * `wt` * abs(data) + 𝜖)

        where `wt` is the weight in %, data is the B-field or the voltage, and 𝜖 is attribute `e` of the class.

        The matplotlib backend is not changed.  Diagnostic figures are created only if `show_plots` or
        `save_plots` is True; when they are only saved, they are closed once written.
        """

        if data_mmr is None and data_ert is None:
            raise ValueError("No data to invert")

        dobs = None
        wt = None
        nobs_mmr = 0
        if data_mmr is not None:
            try:
                g.set_survey_mmr(data_mmr.xs, data_mmr.xo, data_mmr.cs)
                g.check_acquisition()
                dobs = g.data_to_obs(data_mmr.data)
                # weights are given per datum and component (all x, then y, then z, see df_to_data): same
                # ordering as dobs
                wt = g.data_to_obs(np.asarray(data_mmr.wt).reshape(3, -1).T).flatten()
                nobs_mmr = dobs.size
            except AttributeError:
                raise RuntimeError("Format of MMR data incorrect")

        if data_ert is not None:
            try:
                g.set_survey_ert(data_ert.c1c2, data_ert.p1p2, data_ert.cs)
                if dobs is None:
                    dobs = data_ert.data.reshape(-1, 1)
                else:
                    dobs = np.r_[dobs, data_ert.data.reshape(-1, 1)]
                if wt is None:
                    wt = data_ert.wt
                else:
                    wt = np.r_[wt, data_ert.wt]
            except AttributeError:
                raise RuntimeError("Format of ERT data incorrect")

        if self.verbose:
            print("\nInversion")
            g.print_info()
            if m_active is not None:
                if m_active.dtype == bool:
                    npar = np.sum(m_active)
                else:
                    npar = m_active.size
                print("    Number of parameters to estimate: {0:d}".format(npar))
            else:
                print("    Number of parameters to estimate: {0:d}".format(g.fv.nc))
            if nobs_mmr > 0:
                print("    Number of MMR data: {0:d}".format(nobs_mmr))
            if data_ert is not None:
                print("    Number of ERT data: {0:d}".format(g.c1c2.shape[0]))
            print("    Algorithm: " + self.method)
            print("    Maximum number of iterations: {0:d}".format(self.max_it))
            print("    Working variable: " + self.param_transf)
            if self.data_transf is not None:
                print("    Function applied to data: " + self.data_transf)
            print("    Type of smoothing: " + self.smooth_type)
            print("      Smoothing factor along x: {0:g}".format(self.alx))
            print("      Smoothing factor along y: {0:g}".format(self.aly))
            print("      Smoothing factor along z: {0:g}".format(self.alz))
            print("      Smallness factor: {0:g}".format(self.als))
            print("    Regularization variable: " + self.reg_var)
            print("    Regularization β: {0:g}".format(self.beta))
            print("      Cooling factor: {0:g}".format(self.beta_cooling))
            print("      Min value: {0:g}".format(self.beta_min))
            print("    Data weighting: " + self.data_weighting)
            print("      error floor 𝜖: {0:g}".format(self.e))
            print("    Model weighting: " + self.model_weighting)
            if self.model_weighting == "distance":
                print("      Distance weighting β: {0:g}".format(self.beta_dw))
            print("    cglscd: preconditioner " + self.precond_cglscd + ", tol {0:g}, max_it {1}".format(
                self.tol_cglscd, self.max_it_cglscd if self.max_it_cglscd is not None else "auto"))
            print("    Conductivity bounds: {0:g} < 𝜎 < {1:g}".format(self.sigma_min, self.sigma_max))
            g.fv.solver_A.print_info()
            if self.checkpointing:
                print("    Checkpointing: ON")
            else:
                print("    Checkpointing: OFF")
            if self.start_from_chkpt:
                print("    Inversion will start from checkpoint")

        if m_weight is None:
            if self.model_weighting == "distance":
                xo = None
                if data_mmr is not None:
                    xo = g.xo_all
                if data_ert is not None:
                    if xo is None:
                        xo = np.unique(np.r_[g.p1p2[:, :3], g.p1p2[:, 3:]], axis=0)
                    else:
                        xo = np.unique(np.r_[xo, g.p1p2[:, :3], g.p1p2[:, 3:]], axis=0)
                m_weight = g.fv.distance_weighting(xo, self.beta_dw)
            else:
                m_weight = np.ones(m_ref.shape)
        if m_active is None:
            m_active = np.ones(m_ref.shape, dtype=bool)
            # TODO: if m_active is not set, this should be transferred to g to use ROI

        if m0 is None:
            m0 = m_ref

        if self.param_transf == "conductivity":
            xt = m0[m_active].copy()
        elif self.param_transf == "log_conductivity":
            xt = np.log(m0[m_active])
        elif self.param_transf == "log_resistivity":
            xt = np.log(1 / m0[m_active])
        elif self.param_transf == "resistivity":
            xt = 1 / m0[m_active]
        else:
            raise ValueError("Wrong value for param_transf")

        np.seterr(divide='ignore')
        if self.param_transf == "log_conductivity":
            ind = m_ref == 0.0
            m_ref = np.log(m_ref)
            m_ref[ind] = 0.0
        elif self.param_transf == "log_resistivity":
            ind = m_ref == 0.0
            m_ref = np.log(1 / m_ref)
            m_ref[ind] = 0.0
        elif self.param_transf == "resistivity":
            ind = m_ref == 0.0
            m_ref = 1 / m_ref
            m_ref[ind] = 0.0
        np.seterr(divide='warn')

        g.in_inv = True
        xc0 = xt.copy()
        xc = xt.copy()
        sigma = m0.copy()

        WTW = g.calc_WtW(m_weight, self, m_active)
        if "variance" in self.data_weighting:
            D = calc_WdW(wt, dobs, self)
        elif "*" in self.data_weighting:
            D = sp.eye_array(dobs.size)
        else:
            D = 0.0

        data_inv = []
        rms = []
        misfit = []
        smy = []
        s = np.array([1.0])
        sigma_inv = []

        WGx = WGy = WGz = None

        beta = self.beta

        checkpt_loaded = False

        if self.precond_cglscd not in ("regularization", "jacobi"):
            raise ValueError("precond_cglscd must be 'regularization' or 'jacobi'")
        precond = None
        precond_of = None    # WTW factorized in precond

        for i in range(self.max_it):
            if self.start_from_chkpt and checkpt_loaded is False:
                # checkpointing
                if os.path.isfile(self.basename + "_chkpt.pkl"):
                    with open(self.basename + "_chkpt.pkl", "rb") as f:
                        if self.verbose:
                            print("    Loading checkpoint data")
                        (
                            i,
                            J,
                            d,
                            D,
                            WTW,
                            WGx,
                            WGy,
                            WGz,
                            xc,
                            xt,
                            sigma,
                            beta,
                            s,
                            sigma_inv,
                            data_inv,
                            rms,
                            misfit,
                            smy,
                        ) = pickle.load(f)
                        i += 1
                        checkpt_loaded = True

            if i == 0 or self.method == "Gauss-Newton":
                if self.verbose:
                    print("  ------------------------------------------------------------")
                    print("  *** Iteration no {0:d} ***".format(i + 1))
                    print(
                        "    Forward modelling & computation of jacobian ... ",
                        end="",
                        flush=True,
                    )
                d, J = g.fwd_mod(sigma, calc_sens=True, keep_solver=True)
                J = J.T
                if self.data_transf == "asinh":
                    d = np.arcsinh(d)
                    fact = 1 / np.sqrt(1 + np.sinh(d) ** 2).flatten()
                    for ii in np.arange(J.shape[0]):
                        J[ii, :] *= fact[ii]
                if self.verbose:
                    print("done.")
            else:
                if self.verbose:
                    print("  *** Iteration no {0:d} ***".format(i + 1))
                    print("    Forward modelling ... ", end="", flush=True)
                d = g.fwd_mod(sigma, keep_solver=True)
                if self.data_transf == "asinh":
                    d = np.arcsinh(d)
                if self.verbose:
                    print("done.")

            d = d.reshape(-1, 1)
            d_1 = d.copy()
            data_inv.append(d.flatten().copy())

            if i > 0 and self.method == "Quasi-Newton":
                # Broyden update
                p = s.T @ s
                dJ = (d - d_1 - J @ s) / p
                J += dJ @ s.T

            if i == 0:
                if "jacobian" in self.data_weighting:
                    D2 = 1 / np.sqrt(np.sum(J * J, axis=1))
                    if data_ert is not None and data_mmr is not None:
                        # scaling MMR & ERT parts in joint inversion
                        # TODO compare this with Athanasiou et al. 2007
                        tmp1 = np.median(D2[:nobs_mmr])
                        tmp2 = np.median(D2[nobs_mmr:])
                        D2[nobs_mmr:] *= tmp1 / tmp2
                    if "*" in self.data_weighting:
                        D = D @ sp.diags_array(D2)
                    else:
                        D = D + sp.diags_array(D2)

                if self.model_weighting == "jacobian":
                    m_weight = 1 / np.sqrt(np.sum(J * J, axis=0))
                    WTW = g.calc_WtW(m_weight, self, m_active)

            rms.append(np.sqrt(((d - dobs).T @ D @ (d - dobs)) / dobs.size).item())
            misfit.append(np.sqrt(np.sum((d - dobs) ** 2) / np.sum(dobs * dobs)))
            smy.append(np.sum((xc - xc0) ** 2))
            xc0 = xc.copy()
            if self.verbose:
                print("    Weighted RMS: {0:g}, misfit: {1:g}, smy: {2:g}".format(rms[-1], misfit[-1], smy[-1]))

            if self.verbose:
                print("    Computing perturbation with cglscd ... ", end="", flush=True)

            if self.precond_cglscd == "regularization" and precond_of is not WTW:
                # WTW changes only with IRLS weights (or model weighting at the 1st iteration)
                precond = self._reg_preconditioner(g, WTW)
                precond_of = WTW

            s, err, iter1 = cglscd(
                J,
                np.zeros(xt.shape),
                dobs - d,
                beta,
                WTW,
                xt - m_ref[m_active],
                D,
                P=None,
                max_it=self.max_it_cglscd,
                tol=self.tol_cglscd,
                reg_var=self.reg_var,
                precond=precond,
            )
            if self.show_plots or self.save_plots:
                fig, ax = plt.subplots(3, 3, figsize=(9, 9))
                ax = ax.flatten()
                ax[0].plot(s), ax[0].set_title("s")
                ax[1].plot(WTW.diagonal()), ax[1].set_title("WTW")
                ax[2].plot(D.diagonal()), ax[2].set_title("D")
                ax[3].plot(dobs), ax[3].set_title("dobs")
                ax[4].plot(d), ax[4].set_title("d")
                ax[5].plot(dobs - d), ax[5].set_title("dobs-d")
                ax[6].plot(xt - m_ref[m_active]), ax[6].set_title("xt-mref")

                ax[7].text(0.2, 0.75, "RMS = {0:g}".format(rms[-1]))
                ax[7].text(0.2, 0.60, "$\\beta$ = {0:g}".format(beta))
                ax[7].text(0.2, 0.45, "$\\|J\\|$ = {0:g}".format(np.linalg.norm(J)))
                ax[7].text(0.2, 0.30, "$\\|WTW\\|$ = {0:g}".format(sp.linalg.norm(WTW)))
                ax[7].text(0.2, 0.15, "$\\|D\\|$ = {0:g}".format(sp.linalg.norm(D)))
                ax[7].axis("off")
                ax[8].axis("off")

                fig.suptitle(f"Iteration {i + 1}")
                fig.tight_layout()
                if self.save_plots:
                    filename = self.basename + "_it{0:02d}".format(i + 1) + ".pdf"
                    fig.savefig(filename)
                if self.show_plots:
                    plt.show(block=False)
                    plt.draw()
                    plt.pause(0.1)
                else:
                    # only saved: close it (would otherwise be displayed later, e.g. in notebooks)
                    plt.close(fig)
            if self.verbose:
                print("done.\n      err = {0:e}, iter = {1:d}".format(err, iter1))

            if np.max(np.abs(s)) > self.max_step_size:
                s = s / np.max(np.abs(s)) * self.max_step_size

            # Parabolic line search
            s, xt = self._line_search(s, d, dobs, g, D, xc, sigma, m_active)

            self._update_sigma(sigma, xt, m_active)

            if self.param_transf == "conductivity":
                xt = sigma[m_active].copy()
            elif self.param_transf == "log_conductivity":
                xt = np.log(sigma[m_active])
            elif self.param_transf == "log_resistivity":
                xt = np.log(1 / sigma[m_active])
            elif self.param_transf == "resistivity":
                xt = 1 / sigma[m_active]

            sigma_inv.append(sigma[m_active].copy())
            xc = xt.copy()

            beta /= self.beta_cooling
            if beta < self.beta_min:
                beta = self.beta_min

            if self.smooth_type == "blocky" or self.smooth_type == "ekblom" or self.smooth_type == "min. support":
                if self.verbose:
                    print("  Computing regularization term")

                # with reg_var 'model', the regularization applies to m - m_ref (see cglscd)
                xref = m_ref if self.reg_var == "model" else 0.0
                if self.param_transf == "conductivity":
                    tmp = sigma
                elif self.param_transf == "log_conductivity":
                    tmp = np.log(sigma)
                elif self.param_transf == "log_resistivity":
                    tmp = np.log(1 / sigma)
                elif self.param_transf == "resistivity":
                    tmp = 1 / sigma
                # IRLS weights are recomputed from the current model (base weights of 1); the weights
                # returned are kept for checkpointing only, they must not be fed back (they would compound)
                WTW, WGx, WGy, WGz = g.calc_reg(m_weight, self, tmp, xref, None, None, None, m_active)

            # checkpoint
            if self.checkpointing:
                with open(self.basename + "_chkpt.pkl", "wb") as f:
                    pickle.dump(
                        [
                            i,
                            J,
                            d,
                            D,
                            WTW,
                            WGx,
                            WGy,
                            WGz,
                            xc,
                            xt,
                            sigma,
                            beta,
                            s,
                            sigma_inv,
                            data_inv,
                            rms,
                            misfit,
                            smy,
                        ],
                        f,
                    )

        if self.verbose:
            print("  ------------------------------------------------------------")
            print("End of inversion.")

        # inversion completed -> delete checkpoint file
        if self.checkpointing:
            os.remove(self.basename + "_chkpt.pkl")

        return sigma_inv, data_inv, rms, misfit, smy

    def _reg_preconditioner(self, g, WTW):
        """Factorize the regularization matrix for cglscd (falls back to Jacobi if it fails)."""
        if self.verbose:
            print("    Factorizing regularization matrix ... ", end="", flush=True)
        try:
            precond = RegPreconditioner(WTW, g.fv.get_solver_params(), comm=g.fv.comm, n_threads=g.fv.n_threads)
        except RuntimeError as err:
            warnings.warn("cglscd: factorization of the regularization matrix failed ({0}), "
                          "using Jacobi preconditioning".format(err), stacklevel=2)
            precond = None
        if self.verbose:
            print("done.")
        return precond

    def _line_search(self, s, d, dobs, g, D, xc, sigma, m_active):
        if self.verbose:
            print("    Computing step length (parabolic line search)")
        fd0 = (0.5 * (d - dobs).T @ (D * D) @ (d - dobs)).item()
        mu1 = 1
        ils = 1
        # Evaluate the new objective function for mu1=1
        xt = xc + mu1 * s.flatten()

        self._update_sigma(sigma, xt, m_active)

        if self.verbose:
            print("      Forward modelling ... ", end="", flush=True)
        d = g.fwd_mod(sigma, keep_solver=True)
        if self.verbose:
            print("done.")

        d = d.reshape(-1, 1)

        if self.data_transf == "asinh":
            d = np.arcsinh(d)

        fd1 = (0.5 * (d - dobs).T @ (D * D) @ (d - dobs)).item()
        fd11 = fd1

        while True:
            # Evaluate the new objective function for mu=mu/3 with mu_initial=1;
            xt = xc + (mu1 / 3) * s.flatten()

            self._update_sigma(sigma, xt, m_active)

            if self.verbose:
                print("      Forward modelling ... ", end="", flush=True)
            d = g.fwd_mod(sigma, keep_solver=True)
            if self.verbose:
                print("done.")

            d = d.reshape(-1, 1)

            if self.data_transf == "asinh":
                d = np.arcsinh(d)

            fd2 = (0.5 * (d - dobs).T @ (D * D) @ (d - dobs)).item()

            # if self.verbose:
            #     print('      fd: {0:g}   {1:g}   {2:g}'.format(fd0, fd1, fd2))

            if fd2 < fd0 and fd1 > fd2:
                # stationary point is egal to minimum of fitted parabola
                p = np.polyfit(np.array([0, mu1 / 3, mu1]), np.array([fd0, fd2, fd1]), 2)
                mu1 = -p[1] / (2 * p[0])
                xt = xc + mu1 * s.flatten()
                s *= mu1
                break
            elif fd0 > fd2 > fd1 and ils == 1:
                xt = xc + s.flatten()
                break
            elif fd0 > fd2 > fd1:
                # stationary point can have any value greater than mu of fd1 and
                # less than mu of fd1 of the last iteration
                p = np.polyfit(
                    np.array([0, mu1 / 3, mu1, 3 * mu1]),
                    np.array([fd0, fd2, fd1, fd11]),
                    2,
                )
                mu1 = -p[1] / (2 * p[0])
                xt = xc + mu1 * s.flatten()
                s = mu1 * s
                break
            elif mu1 < 1e-6 or ils == 3:
                xt = xc + mu1 * s.flatten()
                s = mu1 * s
                break
            else:
                ils += 1
                mu1 = mu1 / 3
                fd11 = fd1
                fd1 = fd2

        return s, xt

    def _update_sigma(self, sigma, xt, m_active):
        if self.param_transf == "conductivity":
            sigma[m_active] = xt
        elif self.param_transf == "log_conductivity":
            sigma[m_active] = np.exp(xt)
        elif self.param_transf == "log_resistivity":
            sigma[m_active] = np.exp(-xt)
        elif self.param_transf == "resistivity":
            sigma[m_active] = 1 / xt

        sigma[sigma > self.sigma_max] = self.sigma_max
        sigma[sigma < self.sigma_min] = self.sigma_min
