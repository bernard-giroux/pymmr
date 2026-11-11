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
