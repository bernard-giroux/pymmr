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


# %%  Define namedtuple for input data

DataMMR = namedtuple("DataMMR", "xs xo data wt cs date")
DataERT = namedtuple("DataERT", "c1c2 p1p2 data wt cs date")


# %% Some functions


def cglscd(J, x, b, beta, CTC, dxc, D, max_it, tol, reg_var, P=None, alpha=0, WTWt=0):
    """Compute perturbation with a conjugate-gradient least-squares algorithm.

    Parameters
    ----------
    J : sparse matrix
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
    max_it : int
        Max number of iterations.
    tol : float
        tolerance.
    reg_var : str
        Regularization variable: 'model', 'model perturbation', 'time-lapse'.
    P : array_like
        Constraint array to fix solution (0: fixed, 1: free).
    alpha : float
        Time-lapse regularization factor.
    WTWt : sparse matrix
        Time-lapse regularization matrix.

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

    This algorithm is described in

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
    """
    it = 0

    x = x.reshape(-1, 1)
    dxc = dxc.reshape(-1, 1)

    if P is None:
        P = np.ones(x.shape)
    else:
        P = P.reshape(-1, 1)

    x *= P

    M = np.sum(J * J, axis=0) + beta * CTC.diagonal()
    M[M == 0] = np.finfo(float).eps
    M = 1 / M.reshape(-1, 1)

    zz = D @ (b - J @ x)  # initialisation du gradient

    if reg_var == "model":
        b1 = (((D * D) @ b).T @ J).T - beta * CTC @ dxc
        bnrm2 = np.linalg.norm(b1)

        if bnrm2 == 0.0:
            bnrm2 = 1.0
        r = P * (((zz.T @ D) @ J).T - beta * CTC @ (x + dxc))

        error = np.linalg.norm(r) / bnrm2  # initialisation de l'erreur

        if error < tol:
            return x, error, it

        rho_1 = 1.0
        for it in np.arange(max_it):
            z = M * r  # calcul du gradient modifié
            rho = (r.T @ z).item()
            if it == 0:
                p = z.copy()
            else:
                beta_k = rho / rho_1  # calcul du coefficient beta_k
                p = z + beta_k * p

            q = D @ (J @ p)
            alpha_k = rho / (q.T @ q + beta * (p.T @ CTC @ p)).item()  # calcul du pas
            x += alpha_k * p  # calcul de la descente
            zz -= alpha_k * q
            r = P * (((zz.T @ D) @ J).T - beta * CTC @ (x + dxc))  # gradient
            error = np.linalg.norm(r) / bnrm2
            if error <= tol:
                break
            rho_1 = rho

    elif reg_var == "model perturbation":
        r = P * ((zz.T @ D) @ J).T - beta * CTC @ x
        b1 = (((D * D) @ b).T @ J).T
        bnrm2 = np.linalg.norm(b1)
        error = np.linalg.norm(r) / bnrm2  # initialisation de l'erreur

        if error < tol:
            return x, error, it

        rho_1 = 1.0
        for it in np.arange(max_it):
            z = M * r  # calcul du gradient modifié
            rho = (r.T @ z).item()
            if it == 0:
                p = z.copy()
            else:
                beta_k = rho / rho_1  # calcul du coefficient beta_k
                p = z + beta_k * p

            q = D @ (J @ p)
            alpha_k = rho / (q.T @ q + beta * (p.T @ CTC @ p)).item()  # calcul du pas
            x += alpha_k * p  # calcul de la descente
            zz -= alpha_k * q
            r = P * (((zz.T @ D) @ J).T - beta * CTC @ x)  # gradient
            error = np.linalg.norm(r) / bnrm2
            if error <= tol:
                break
            rho_1 = rho

    elif reg_var == "time-lapse":
        b1 = (((D * D) @ b).T @ J).T - alpha * WTWt @ dxc
        bnrm2 = np.linalg.norm(b1)

        if bnrm2 == 0.0:
            bnrm2 = 1.0

        r = P * (((zz.T @ D) @ J).T - alpha * WTWt @ dxc - CTC @ x)

        error = np.linalg.norm(r) / bnrm2  # initialisation de l'erreur

        if error < tol:
            return x, error, it

        rho_1 = 1.0
        for it in np.arange(max_it):
            z = M * r  # calcul du gradient modifié
            rho = (r.T @ z).item()
            if it == 0:
                p = z.copy()
            else:
                beta_k = rho / rho_1  # calcul du coefficient beta_k
                p = z + beta_k * p

            q = D @ J @ p
            alpha_k = rho / (q.T @ q + beta * (p.T @ CTC @ p)).item()  # calcul du pas
            x += alpha_k * p  # calcul de la descente
            zz -= alpha_k * q
            r = P * (((zz.T @ D) @ J).T - alpha * WTWt @ dxc - CTC @ x)  # calcul du gradient
            error = np.linalg.norm(r) / bnrm2
            if error <= tol:
                break
            rho_1 = rho

    if error > tol:
        warnings.warn(
            "cglscd: convergence not achieved after {0:d} it (tol = {1:3.2e}, error = {2:5.4e}).".format(
                max_it, tol, error
            )
        )

    return x, error, it + 1


def calc_WdW(wt, dobs, par):
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
        vintage = df["date"].to_numpy().reshape(-1, 1)

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
            data = []
            for vin in np.unique(vintage):
                ind = vintage == vin
                data.append(
                    DataMMR(xs=c1c2[ind, :], xo=xo[ind, :], data=data[ind, :], wt=wt[ind, :], cs=cs[ind], date=vin)
                )
            return data
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
            data = []
            for vin in np.unique(vintage):
                ind = vintage == vin
                data.append(
                    DataERT(c1c2=c1c2[ind, :], p1p2=p1p2[ind, :], data=data[ind], wt=wt[ind], cs=cs[ind], date=vin)
                )
            return data
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
        """epsilon min, to reduce weight of low values."""

        self.max_err = 10.0
        """Acceptable std-dev."""

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

        self.max_it_cglscd = 1000
        """Maximum number of iterations in cglscd."""

        self.tol_cglscd = 1.0e-9
        """Tolerance for cglscd."""

        self.max_step_size = 10
        """Maximum perturbation step."""

        self.sigma_max = 1.0e4
        """Upper cut-off value for conductivity."""

        self.sigma_min = 2.0e-5
        """Lower cut-off value for conductivity."""

        self.minsupport_e = 0.000001

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
            m_ref,
            data_mmr=None,
            data_ert=None,
            m0=None,
            m_weight=None,
            m_active=None,
    ):
        """Run inversion.

        Parameters
        ----------
        g : Grid instance
            GridMMR (or GridDC if `data_mmr` is None)
        m_ref : array_like
            Reference model (S/m).
        data_mmr : DataMMR, optional
            MMR data.  B-field units are pT. Data weight (std dev) in %.
        data_ert : DataERT, optional
            DC resistivity data.  Voltage should be in mV. Data weight (std dev) in %.
        m0 : array_like, optional
            Initial model (equal to m_ref if None).
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
                wt = data_mmr.wt
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
                print("    Number of parameters to estimate: {0:d}".format(g.nc))
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
            print("      𝜖: {0:g}".format(self.e))
            print("    Model weighting: " + self.model_weighting)
            if self.model_weighting == "distance":
                print("      Distance weighting β: {0:g}".format(self.beta_dw))
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
            m0 = m_ref.copy()

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

        if self.param_transf == "log_conductivity":
            m_ref = np.log(m_ref)
        elif self.param_transf == "log_resistivity":
            m_ref = np.log(1 / m_ref)
        elif self.param_transf == "resistivity":
            m_ref = 1 / m_ref

        g.in_inv = True
        xc0 = xt.copy()
        xc = xt.copy()
        sigma = m0.copy()

        WTW = g.calc_WtW(m_weight, self, m_active)
        if "variance" in self.data_weighting:
            D = calc_WdW(wt, dobs, self)
        elif "*" in self.data_weighting:
            D = sp.eye(dobs.size)
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
                        D = D @ sp.diags(D2, 0)
                    else:
                        D = D + sp.diags(D2, 0)

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
                ax[7].text(0.2, 0.45, "$\|J\|$ = {0:g}".format(np.linalg.norm(J)))
                ax[7].text(0.2, 0.30, "$\|WTW\|$ = {0:g}".format(sp.linalg.norm(WTW)))
                ax[7].text(0.2, 0.15, "$\|D\|$ = {0:g}".format(sp.linalg.norm(D)))
                ax[7].axis("off")
                ax[8].axis("off")

                fig.suptitle(f"Iteration {i+1}")
                fig.tight_layout()
                if self.save_plots:
                    filename = self.basename + "_it{0:02d}".format(i + 1) + ".pdf"
                    fig.savefig(filename)
                if self.show_plots:
                    plt.show(block=False)
                    plt.draw()
                    plt.pause(0.1)
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

                xref = 0.0
                if self.param_transf == "conductivity":
                    tmp = sigma
                elif self.param_transf == "log_conductivity":
                    tmp = np.log(sigma)
                elif self.param_transf == "log_resistivity":
                    tmp = np.log(1 / sigma)
                elif self.param_transf == "resistivity":
                    tmp = 1 / sigma
                WTW, WGx, WGy, WGz = g.calc_reg(m_weight, self, tmp, xref, WGx, WGy, WGz, m_active)

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
        os.remove(self.basename + "_chkpt.pkl")

        return sigma_inv, data_inv, rms, misfit, smy

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
