# Off-line measurements in 2.5D DC (`GridDC`)

## Problem

The 2.5D potential is computed from 2D fields U(k_i) at a few wavenumbers. `GridDC` uses 11
wavenumbers whose weights g_i are fitted so that Σ g_i K0(k_i ρ) ≈ 1/ρ: the inverse transform is
accurate for an offset of 0 along y between the current and potential electrodes only. Electrodes
of a datum at different y (off-line measurements, e.g. 3D surveys with parallel lines, or cross-line
dipoles) were computed as if all electrodes were at y = 0. With a pole at x = 0 and a potential
electrode at x = 10 m, the computed potential was 1.41 times the true one at y = 10 m and 3.16
times at y = 30 m.

## Method

The model does not vary along y, so one 2D field per current electrode gives the potential at any y.
For a current electrode and a potential electrode at a distance Δy along y:

    u = Σ_i W_i(Δy) Q (U_i − U0_i) + u0

- k_i: wavenumbers of `select_k(r_min, r_max, n_k_offline)` (30 by default), with r_min the smallest
  x–z distance between current and potential electrodes (at least half the smallest cell) and r_max
  4 times the largest 3D distance between them (at least 100 r_min);
- W_i(Δy): even weights of `KQuadrature` (spline interpolation of the spectrum in ln k, see
  `docs/mmr25.md`, section 6), valid for any Δy;
- Q: interpolation in the x–z plane (electrodes at least at the depth of the first cell centre);
- U0_i: analytic fields of the boundary correction (`Grid25FV.boundary_source`, conductivity at
  each current electrode with `local_sigma0`), and u0 the analytic potential at the true positions,
  Δy included (`u_homog_at(..., offset_y=True)`). Without `apply_bc` or `exact_primary`, the U_i alone.

A datum is the sum of the terms of its pairs of current and potential electrodes (up to 4), each
weighted by ±I. The fields are computed once per unique (x, z) position of the current electrodes,
for a unit current; only their values at the potential electrodes are kept, one wavenumber at a
time. `correct_quadrature` does not apply: U0_i is always removed in the wavenumber domain.

## Usage

`GridDC.offline25`:

- `'auto'` (default): the off-line method is used when the electrodes of at least one datum are at
  different y; surveys on a line (any y, electrodes at infinity allowed) keep the fitted
  wavenumbers, which are more accurate there and cost about 3 times less (11 factorizations instead
  of 30);
- `True`: always (e.g. to compare);
- `False`: never; off-line data are computed as if all electrodes were at y = 0, with a warning.

The number of wavenumbers is `GridDC.n_k_offline`. Sensitivity (`calc_sens`) and current density
(`calc_J`) are not implemented yet with the off-line method (`NotImplementedError`), so off-line data
cannot be inverted yet.

## Accuracy (`tests/test_dc25.py`, `TestOffLineModelling`)

Dipole–dipole and pole–dipole data, potential dipoles along x and y at y = 0, 10 and 30 m, h = 2.5 m:

- homogeneous half-space: exact (1e-8);
- two-layer earth (100 Ω·m over 10 Ω·m at 10 m), against the image series: within 2%, or 0.2% of the
  sum of the terms of a datum when they nearly cancel (dipole–dipole far off the line: 5.7% of the
  datum with 30 wavenumbers, 3.6% with 50). Pole–dipole at y = 0 and 10 m: within 1.2%;
- on the line, same data as the fitted wavenumbers within 0.5% for a heterogeneous model.

2D bodies (a conductive block 0.2 S/m and a resistive one 0.002 S/m in 0.01 S/m, infinite along y),
against 3D `GridDC` with the same cells: dipole–dipole data within 1.3%, most within 0.1%. Pole data
with the pole in the resistive block differ by up to 20%, but the 2.5D results do not depend on the
method (same values with the fitted wavenumbers on the line, 30 or 60 wavenumbers off the line), and
the 3D results converge towards them as the padding along y increases (0.73, 0.80 and 0.91 of the
2.5D value with 78 m, 416 m and 2 km of padding): the 3D grid cannot hold a conductor infinite along y.

## Not yet done

- Sensitivity: in `_sens25`, g_i / 2 becomes W_i(Δy) for each pair of electrodes (up to 4 per datum),
  plus the derivative of the exact-primary term.
- Inversion, `run_job` and joint MMR–ERT inversion with off-line ERT data.
