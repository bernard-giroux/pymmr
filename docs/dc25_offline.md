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

The number of wavenumbers is `GridDC.n_k_offline`. Current density (`calc_J`) is not available with the
off-line method (`NotImplementedError`): the field depends on y.

## Sensitivity

The source terms of the boundary correction do not depend on σ, so with V_i the field of a unit
source at the potential electrode (adjoint field; L_i⁻ᵀ = Vc L_i⁻¹ Vc⁻¹, Dᵀ = −Vf G Vc⁻¹):

    d(Q U_i)/d(ln σ) = −[ Dm Gfᵀ (G U_i ∘ m² Vf ∘ G V_i) + k_i² σ Vc U_i ∘ V_i ]

as in `_sens25` for the fitted wavenumbers, with g_i / 2 replaced by W_i(Δy) for each pair of
electrodes. U0_i and u0 are proportional to 1/σ0, σ0 being the conductivity at the current electrode
(`local_sigma0`) or the geometric mean of the model, which adds −R d(ln σ0)/d(ln σ), R being their
contribution to the datum (for each pair, since σ0 depends on its current electrode).

The fields are solved one wavenumber at a time, the current and potential electrodes together (one
factorization); the products are computed once per unique pair of (x, z) positions (parallel lines
share them), in chunks.

Tests (`tests/test_dc25.py`, `TestOffLineSensitivity`): against central finite differences, cells at
the current electrodes, between and below them, dipole–dipole, pole–dipole and pole–pole data
off the line: 1e-6 (local σ0 or geometric mean); without `apply_bc`, 1e-4, limited by the round-off of
the finite differences (the error grows as the step decreases). On the line, the sensitivity agrees with
the fitted wavenumbers within 0.2%. The region of interest (`set_roi`) gives the rows of the full matrix.

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

## Inversion, `run_job` and joint inversion

No change was needed: `Inversion.run` sets the survey with `set_survey_ert` and calls
`fwd_mod(sigma, calc_sens=True)`, `run_job` passes the coordinates of the electrodes as they are, and the
joint 2.5D MMR–ERT inversion computes the ERT data and sensitivity with `GridDC.fwd_mod`. Tests
(`tests/test_dc25.py`, `TestOffLineInversion`): forward modelling through `run_job`, a few Gauss–Newton
iterations, and the ERT columns of the joint sensitivity equal to those of `GridDC`.

Example: two parallel lines 10 m apart (h = 2.5 m), in-line dipole–dipole on each line and cross-line
pole–dipole (213 data, 113 off-line), conductive 2D block (0.1 S/m in 0.01 S/m), 5 Gauss–Newton
iterations from a homogeneous model. Model error: distance between the final and true models (log σ),
relative to the starting model.

| method                          | RMS (iterations 1 to 5)        | model error |
|---------------------------------|--------------------------------|-------------|
| off-line (`offline25 = 'auto'`) | 20.1, 4.1, 3.0, 2.2, 1.6       | 0.60        |
| y ignored (`offline25 = False`) | 116, 53, 47, 48, 48            | 3.70        |

Joint inversion with MMR receivers off the line (y = 5 and 15 m) and the same ERT data: RMS 13.4, 5.2,
4.2 in 3 iterations.
