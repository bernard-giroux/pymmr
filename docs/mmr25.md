# 2.5D MMR modelling: formulation

This note fixes the conventions used by the 2.5D MMR code (`Grid25FV.build_*_k`, and later
`GridMMR` with a 2D grid). It extends the 2.5D DC formulation already in `Grid25FV`/`Solver25`.

## 1. Setting and Fourier convention

The conductivity σ(x, z) is invariant along the strike direction y. A current pole is at
(x_s, y_s, z_s); all fields depend on y only through Δy = y − y_s, so each pole is modelled with
its source at y = 0 and shifted afterwards.

A field even in Δy is represented by its cosine transform, a field odd in Δy by its sine
transform (both real):

    F^c(k) = ∫_{-∞}^{∞} f(y) cos(ky) dy        f(y) = (1/π) ∫_0^∞ F^c(k) cos(ky) dk
    F^s(k) = ∫_{-∞}^{∞} f(y) sin(ky) dy        f(y) = (1/π) ∫_0^∞ F^s(k) sin(ky) dk

This is the convention of the existing DC code: for a pole, u = I/(4πσr) transforms to
I/(4πσ)·2K0(kρ) (`compute_U_homog_k`), and the quadrature u ≈ ½ Σ g_i U(k_i) of `Solver25.solve`
approximates (1/π)∫_0^∞ … dk at Δy = 0 (weights fitted so that Σ g_i K0(k_i ρ) ≈ 1/ρ).

Derivatives along y (integration by parts):

    ∂f/∂y, f even  →  sine coefficient    −k F^c
    ∂f/∂y, f odd   →  cosine coefficient  +k F^s

## 2. Parity of the fields of a pole

| even (cos)                       | odd (sin)              |
|----------------------------------|------------------------|
| U, Jx, Jz, Ax, Az, **By**        | Jy, Ay, **Bx**, **Bz** |

Consequence: on the plane of the source (Δy = 0), **Bx = Bz = 0** by symmetry; only By is non
zero. Receivers off the line of the source are required to observe Bx and Bz (e.g. the Bz anomaly
of `mmr.VerticalDyke`).

## 3. Staggering

The 2D grid is the 3D staggered grid with the y direction removed. A component along y of a vector
on faces lives at the cell centres; along y of a vector on edges, at the nodes.

| quantity        | x component               | y component           | z component               |
|-----------------|---------------------------|-----------------------|---------------------------|
| on faces (A, J) | x-faces (x node, zc), cos | cell centres, sin     | z-faces (xc, z node), cos |
| on edges (B)    | x-edges (xc, z node), sin | nodes, cos            | z-edges (x node, zc), sin |

x-edges coincide with z-faces and z-edges with x-faces. Ordering of vectors is x, y, z (as in 3D):
`nf_k = nfx + nc + nfz` values on faces, `ne_k = nex + ney + nez` on edges (`ney` = interior nodes).
Only interior faces, edges and nodes are unknowns, as in 3D. In `linear_interp`, components
`'ex'`, `'ey'`, `'ez'` interpolate Bx, By, Bz from these locations.

## 4. Discrete operators (all real, linear in k)

    G_k = [Gx; −k I; Gz]                         (nf_k × nc)   gradient of a cos scalar
    D_k = [Dx,  k I, Dz]                         (nc × nf_k)   divergence
    M_k = diag(Mx, σ, Mz)                        (nf_k × nf_k) σ on faces (harmonic mean on x, z)
    C_f(k): faces → edges  (ne_k × nf_k)
        Bx = −k Az − ∂z Ay,   By = ∂z Ax − ∂x Az,   Bz = ∂x Ay + k Ax
    C_e(k): edges → faces  (nf_k × ne_k)
        Hx =  k Bz − ∂z By,   Hy = ∂z Bx − ∂x Bz,   Hz = ∂x By − k Bx

Properties (exact, tested in `tests/test_mmr25_operators.py`):

- D_k C_e = 0 and C_f G_k = 0 (div curl = curl grad = 0)
- D_k = −Vc⁻¹ G_kᵀ Vf_k and Ve_k C_f = (Vf_k C_e)ᵀ, with Vc = `volume_voxels`,
  Vf_k = `volume_faces_k`, Ve_k = `volume_edges_k`
- −D_k M_k G_k = −D M G + k²σ, i.e. the DC operator of `Solver25` (so the transformed current
  density is J_k = −M_k G_k U_k, with Jy = σ k U_k)
- L_k = C_e C_f − G_k D_k = −∇²_xz + k² with **uncoupled** components; Vf_k L_k is symmetric
  positive definite for k > 0
- C_f and L_k are second-order accurate for smooth fields on uniform grids

## 5. MMR problem for one wavenumber

As in 3D (`GridMMR._build_A`), with constant μ0 and the Coulomb gauge:

    (1/μ0) L_k A_k = P J_k,     B_k = C_f(k) A_k

where P maps the earth faces of the DC grid to the faces of the grid extended with air layers
(J = 0 in air). L_k depends only on the grid and k: it is factorized once per wavenumber and kept
across inversion iterations. Because the components are uncoupled, the three blocks could also be
factorized separately; the y block is the DC operator with σ = 1.

Boundary conditions are those of 3D: normal component of A and tangential B zero on the outer
boundary. They are not exact, and the error decreases as exp(−k·distance): the smallest wavenumber
sets the padding needed (air above, earth below and on the sides).

## 6. Back to the spatial domain

For a pole at y_s and a receiver at y_o, Δy = y_o − y_s:

    By(Δy)     = (1/π) ∫_0^∞ B̃y(k) cos(kΔy) dk
    Bx,z(Δy)   = (1/π) ∫_0^∞ B̃x,z(k) sin(kΔy) dk

A dipole is the difference of its two poles, each with its own Δy. The DC wavenumbers (fitted for
1/ρ at Δy = 0) do not apply; the MMR wavenumbers are a fixed set with weights W(Δy) such that

    B(Δy) ≈ Σ_i W_i(Δy) B̃(k_i)

Since W is explicit, the discrete forward map is linear in the B̃(k_i), and its adjoint
for the sensitivity is exact. The DC fields U_k must be computed at the same wavenumbers.

Implementation (`pymmr/kquad.py`):

- `select_k(r_min, r_max, n_k=30)`: log-spaced wavenumbers from 0.05 / r_max to 15 / r_min, where
  r_min is the smallest distance in the x-z plane that must be resolved and r_max the largest
  distance (including Δy).
- `KQuadrature(k).weights(dy, parity)`: the samples F(k_i) are interpolated by a cubic spline in
  ln k (cardinal splines, so linear in the samples). The integral of the spline times cos(kΔy) or
  sin(kΔy) is computed by composite Gauss–Legendre, with at most a quarter of a period per
  sub-interval, so any Δy is handled exactly for the interpolant. Tails:
  - k < k_1: even fields take F = a + b ln k through the first two samples (the log singularity
    of fields decaying as 1/r), integrated analytically with Si. Odd fields hold F = F_1, a small
    contribution since sin(kΔy) ≈ kΔy there.
  - k > k_n: F = F_n exp(−(k − k_n)·decay) with decay = 15 / k_n, integrated analytically. This
    avoids the oscillating error ~F_n/Δy that a hard truncation causes.

Accuracy on analytic pairs (1/r, 1/r³, ρ/(ρ²+y²) even; y/r³, y/(ρ²+y²) odd), with ρ from 0.5 to
100 m, |Δy| ≤ 200 m and k from `select_k(0.5, 300)`. The largest error is relative to the field's
size at each ρ:

| n_k | 20     | 30     | 40     |
|-----|--------|--------|--------|
| max | 1.2e-2 | 1.2e-3 | 2.2e-4 |

With 30 wavenumbers, the pointwise relative error is below 1% wherever the field is at least 1% of
its size at that ρ (`tests/test_kquad.py`). Weights are cached per |Δy|; 500 distinct offsets take
about 1 s.

## 7. Adjoint relations (for the sensitivity)

    L_k⁻ᵀ    = Vf_k L_k⁻¹ Vf_k⁻¹
    L_dc,k⁻ᵀ = Vc L_dc,k⁻¹ Vc⁻¹      (L_dc,k = −D_k M_k G_k)
    D_kᵀ     = −Vf_k G_k Vc⁻¹

These are the same relations as used in 3D (`GridMMR._adj`) and in `GridDC._sens25`.

### Sensitivity (`GridMMR._sens25`, `fwd_mod(sigma, calc_sens=True)`)

For one wavenumber and one current electrode of unit current, the datum is R A with R = Q C_f
(interpolation of one component of B at one measurement point), (1/μ0) L A = P J, J = −M G U and
L_dc U = q. The source q does not depend on σ (`Grid25FV.boundary_source`), so

    dJ = −[I + M G L_dc⁻¹ D] dM (G U)

and, with the relations above,

    ∂(R A)/∂ln σ = −Jmᵀ [ G U ⊙ Vf ⊙ (X + G L_dc⁻¹ D M X) ],    X = P ᵀ (L/μ0)⁻¹ Vf⁻¹ Rᵀ

where Jm = ∂M/∂ln σ: m² w/σ on the x and z faces (harmonic mean with weights w,
`build_G_faces`) and σ for the y component. X and L_dc⁻¹ D M X depend only on the measurement
points. For each wavenumber, two solves with 3 n_rec right-hand sides (one with the MMR
factorization, one with the DC factorization kept from the forward problem) serve all the
electrodes. The sum over wavenumbers and over the electrodes of the dipoles uses the weights
W(Δy) of the forward problem, so the result is the exact derivative of the discrete data.

The rows are the cells of `dc.ind_roi`. The columns follow `data_to_obs`: for each dipole, Bx, By
and Bz of its measurement points, as in 3D. In a joint inversion the ERT columns of
`GridDC._sens25` follow.

Checks (`tests/test_mmr25.py`, `TestSensitivity`): against central differences in ln σ
(ε = 1e-4), on a heterogeneous model with a dipole across strike and a pole, receivers on and off
the line. The relative difference is 1e-8 to 1e-10 for all three components, also in inversion
mode with paired dipoles and receivers and ERT data.

Cost, h = 5 m grid (4000 cells), 36 receivers, n_k = 30, MUMPS: the forward problem takes 1 s
when the MMR factorizations are kept, and 5.7 s (1 dipole) or 6.6 s (4 dipoles) with the
sensitivity.

## 8. Forward modelling (`GridMMR` with a 2D grid)

`GridMMR((x, z))` builds the 2.5D model. Current electrodes and measurement points are given in
3D (any y), as in 3D modelling. `fwd_mod(sigma)`:

1. Air layers are added above the DC grid (`_add_air`), at least as thick as the earth part of the
   grid (see section 9).
2. Once per survey (`_setup25`), the current electrodes are reduced to the unique (x, z) positions
   (an electrode's y only enters through Δy). The wavenumbers come from `select_k(r_min, r_max, n_k)`:
   - r_min is the smallest x–z distance between an electrode and a measurement point, and at
     least half the smallest cell.
   - r_max is 4 times the largest 3D electrode–receiver distance, and at least 100 r_min.
   - `n_k = 30` by default.

   The weights W(Δy) of every electrode–receiver pair are computed at this step.
3. DC fields U_k of each electrode, for unit current. With `dc.apply_bc`, the source term is the
   per-wavenumber boundary correction (`Grid25FV.boundary_source` with the MMR wavenumbers), using
   the conductivity at each electrode (`dc.local_sigma0`). As in 3D, surface electrodes have no
   image in the reference fields (`GridMMR.surface_images = False`): images make B less accurate
   (13% vs 5% error in the half-space test below).
4. J_k = −M_k G_k U_k is mapped to the faces of the grid with air, and (1/μ0) L_k A_k = J_k is
   solved. The factorizations depend only on the grid and the wavenumbers; they are kept between
   calls (`keep_solver=True`). Then B_k = C_f A_k, interpolated at the receivers (`'ex'`, `'ey'`,
   `'ez'`).
5. Inverse transform: Bx, Bz with the sine weights and By with the cosine weights. The dipole data
   are the sum over its electrodes, each weighted by ±I.

The DC data of a joint inversion are computed by `GridDC` with its own wavenumbers. Not yet
available in 2.5D: electric dipoles at the receivers (`pod_e`).

Usage in the rest of the package (phase 6):

- `run_job.py`: a VTK model one cell thick along y gives a 2.5D model for MMR as for DC (`fwd mmr`,
  `compute sensitivity`, `inv` with MMR and/or ERT data). `save_sensitivity` writes 2.5D grids.
- `Inversion.run` needs no change: the joint balancing of MMR and ERT data (Jacobian weighting)
  does not depend on the dimension. Example: `examples/example_mmr25.py`, forward modelling of the
  dyke and inversion of synthetic Bx, By, Bz data.

### Half-space test (`tests/test_mmr25.py`)

The test uses a dipole at x = ±60 m on the surface, σ = 0.01 S/m, and surface receivers at
x ∈ [−40, 40] m, y ∈ {0, 10, 30, −50} m. Errors are the largest absolute error divided by max |By|,
against `normal_field`:

| h (m) | Bx     | By     | time  |
|-------|--------|--------|-------|
| 5     | 1.2e-2 | 4.8e-2 | 1.1 s |
| 2.5   | 6.5e-3 | 2.6e-2 | 2.8 s |
| 1.25  | 4.1e-3 | 1.7e-2 | 9.5 s |

Bz is zero to machine precision. The error does not change with the padding (20 vs 30 cells) or
with the wavenumbers (30 to 60, k_max up to 12 1/m), so it is spatial discretization. It is
largest at the receivers closest to an electrode.

The same model with h = 10 m, 2.5D vs `GridMMR` 3D (same x grid along y), errors as above:

|                 | Bx     | By     | time   |
|-----------------|--------|--------|--------|
| 2.5D            | 2.5e-2 | 1.2e-1 | 0.4 s  |
| 3D              | 3.0e-2 | 1.7e-1 | 66.5 s |
| 2.5D − 3D       | 5.3e-3 | 5.3e-2 |        |

Both share the same discretization bias near the electrodes; away from them they agree to about 1%.
Section 9 shows that this bias comes from the horizontal cell size at the electrodes.

## 9. Validation (phase 4)

Scripts for the numbers below are summarized in `tests/test_mmr25.py` (classes `TestHalfSpace` and
`TestVerticalDyke`).

### Vertical dyke, Bz off the line

Analytic solution: `mmr.VerticalDyke` (Edwards et al., 1978). The model is a dyke 20 m thick
(x from 0 to 20 m), infinite in depth and along y, with ρ = (100, 10, 300) Ω·m from left to right. A
pole is at the surface with its return electrode at infinity (`xs` with `np.inf`). Receivers are at
x ∈ [−80, 80] m and y = 10, 20, 40, 100 m. The grid has h = 5 m, top cells of 0.25 m, 15 padding
cells and n_k = 30. Errors are the largest absolute error divided by max |Bz|, for each y:

| source                  | y = 10 | y = 20 | y = 40 | y = 100 |
|-------------------------|--------|--------|--------|---------|
| x = −50 (left)          | 2.0e-3 | 1.8e-3 | 2.2e-3 | 2.4e-3  |
| x = 10 (in the dyke)    | 9.0e-3 | 2.7e-3 | 1.6e-3 | 1.6e-3  |
| x = 25 (right, at 5 m)  | 1.2e-2 | 3.5e-3 | 1.8e-3 | 1.8e-3  |
| x = 60 (right)          | 1.5e-3 | 1.2e-3 | 1.6e-3 | 1.2e-3  |

This comparison exposed two errors in `VerticalDyke` (electrode outside the dyke), both now fixed:

- Bz on the far side of the dyke (`Bz3`, flagged "TODO double check") was wrong; it was off by
  two orders of magnitude next to the dyke. The corrected expression is

      Bz3 = μI/(4πy) [ k13 + f(x − b) − (1 + k12)(1 − k32) Σ_n (k12 k32)^n f(x − b + 2nd) ],
      f(ξ) = ξ / √(y² + ξ²)

  with x and b measured from the left flank. It is continuous with Bz2 at x = d, vanishes for
  x → ∞, and reduces to the single contact for k12 = 0 and to zero for a uniform medium. The
  coefficients were identified from the continuity with Bz2 (fit residual 1e-14 for several
  resistivity contrasts) and are confirmed by the finite-volume solution above.
- The expressions assume an electrode on the left of the dyke or within it. An electrode on the
  right is now handled by reflection about the centre of the dyke: ρ1 and ρ3 are swapped, and Bz,
  a component of a pseudovector, changes sign.

### Sources of error

- **Electrode depth.** Surface electrodes are placed at the first cell centre, z = −hz/2. With
  uniform h = 5 m, Bz of the dyke is 2% too large for a source outside the dyke and 10% too large
  for a source in it. The error is first order in hz. Refining only the top cells fixes it:

  | top cells (m) | 5     | 1     | 0.25  |
  |---------------|-------|-------|-------|
  | source left   | 1.023 | 1.005 | 1.000 |
  | source in dyke| 1.10  | 1.03  | 1.01  |

  (least-squares ratio of computed to analytic Bz at y = 10 m, ρ = (100, 10, 100) Ω·m.)
- **Air thickness.** The air layers used to be sized from the top cell (4 cells, then 15 cells
  growing by 1.3), only 55 m with 0.25 m top cells. A_z = 0 at the top of the air was then too
  close: 2.3% error instead of 0.2% (dyke, y = 10 m). The 2.5D code now adds air at least as thick as the earth part
  of the grid, padding included, and so does the 3D code. In 3D (half-space of section 8, h = 10 m,
  top cells of 0.25 m), the air goes from 55 m to 2.8 km and the By error from 20% to 15% (least-squares
  ratio 0.82 to 0.92); the rest is the horizontal cell size at the electrodes (below). Time: 34 to 46 s.
- **Horizontal cell size at the electrodes.** In the half-space test By is 2% too small everywhere
  and 4.8% at the receivers closest to an electrode (h = 5 m). This error depends on hx, not hz
  or the top cells, and is first order. Refining x within 10 m of each electrode (h = 5 m
  elsewhere) is enough:

  | hx at the electrodes (m) | 5      | 1      | 0.5    | 0.25   |
  |--------------------------|--------|--------|--------|--------|
  | By error                 | 4.8e-2 | 1.0e-2 | 6.1e-3 | 4.3e-3 |
  | Bx error                 | 1.2e-2 | 4.4e-3 | 6.2e-3 | 7.1e-3 |
  | time (s)                 | 1.1    | 1.6    | 2.1    | 3.1    |

  The same bias is present in 3D (section 8).

Practical rule: refine the grid around the current electrodes, horizontally and at the top. A
primary/secondary split of B, considered in the plan, is not needed.

### Convergence in wavenumbers and padding

Dyke with ρ = (100, 10, 100) Ω·m, source in the dyke, h = 5 m, top cells 0.25 m, 20 padding cells.
The wavenumbers span 9e-5 to 120 1/m:

| n_k        | 10     | 15     | 20     | 30     | 40     | 60     |
|------------|--------|--------|--------|--------|--------|--------|
| y = 10 m   | 1.4e-1 | 2.9e-2 | 1.2e-2 | 9.0e-3 | 8.7e-3 | 8.6e-3 |
| y = 40 m   | 6.2e-2 | 1.9e-3 | 4.5e-3 | 1.6e-3 | 1.5e-3 | 1.4e-3 |
| time (s)   | 0.7    | 1.1    | 1.4    | 2.1    | 2.9    | 4.3    |

The default n_k = 30 is converged: the remaining error is spatial. With n_k = 30, 10, 15, 20 or
30 padding cells give the same errors (within 4e-4). The air follows the earth padding.

The wavenumber range was also computed with r_min from the 3D electrode–receiver distances instead
of the x–z distances. That gives a narrower range (k_max 1.5 instead of 6 to 120 1/m) and the same
errors in these tests. The x–z distance is kept, which is the safe choice for receivers at small
x–z distances but large Δy.
