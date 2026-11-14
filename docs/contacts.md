# Electrodes on conductivity contrasts in DC modelling (`GridDC.contact_interp`)

## Problem

Potentials are defined at cell centres. Before this change, the potential at an electrode was a
linear interpolation of the 2 neighbouring cells (`linear_interp`), and the reference model of the
boundary correction was a homogeneous half-space with the conductivity interpolated at the current
electrode. Both are wrong when the electrode is on a face between cells of different
conductivity, which is common: electrodes are often placed on grid nodes, and in an inversion
neighbouring cells always differ. The error is first order in the cell size.

Example: dipole-dipole (a = 1 m) across a vertical dyke (100, 1000 and 50 Ω·m) with electrodes on
the contacts. Errors relative to the analytic solution (`VerticalDyke`), for 0.5 m cells:

| | previous | `contact_interp` |
|---|---|---|
| potential electrode on a contact | 467 % | 4.4 % |
| current electrode on a contact (reciprocal survey) | 410 % | 8.3 % |
| other stations (median) | 0.02 % | 0.02 % |

On a random model (σ × 10^±0.2 per cell, electrodes on nodes), with the cells divided by 2 and 4,
the error relative to a grid 8 times finer is 52, 27 and 13 % with the previous method and 2.7,
0.55 and 0.11 % with `contact_interp`: the convergence becomes second order.

## Potential electrodes: interpolation with flux continuity

`Grid25FV.flux_interp`: in each direction, the potential varies linearly in each half-cell, and the
normal current is continuous at the face between cells a and b:

    u_f = (σa ua / da + σb ub / db) / (σa / da + σb / db)

with da and db the distances from the cell centres to the face. The potential at the face is close
to that of the most conductive cell (1/21 of the way to it for a contrast of 20, instead of 1/2).
In a homogeneous model the weights are those of `linear_interp`. Along z, the conductivities of the
2 rows are interpolated along x at the electrode. Surface electrodes are extrapolated linearly along
z from the 2 top cells, as before.

With the boundary correction and `exact_primary`, the data are

    d = Qf u + c (P − Q u0)       (for each potential electrode)

with Qf the flux-continuity interpolation of the total field u, Q the linear interpolation, P the
field of the reference model (below) at the true position of the electrode and u0 at the cell
centres. The term P − Q u0 corrects the interpolation of the singular field near the current
electrodes. It assumes that the field near the potential electrode has the shape of the reference
field, which is not true at a contact. The contrast factor c removes it there:

    c = Π (1 − φ (σa − σb)² / (σa + σb)²)

over the faces of the stencil, with φ = 1 at the face and 0 at the cell centres (c = 1 in a
homogeneous model). This factor is empirical. It works well for the dyke, where applying Qf to
u − u0, or keeping the full correction, gave errors of 90 to 170 % at a potential electrode on a
contact near the current electrodes.

## Current electrodes: reference model with a vertical contact

`Grid25FV.contact_reference`: the reference model of each current electrode is made of 2
quarter-spaces separated by the vertical face (node of x) nearest to the electrode, with the
conductivities σa and σb of the 2 cells on either side of the face, in the row of the electrode.
Its analytic field has one image across the face (κ = (σs − σo)/(σa + σb), s on the side of the
source, o on the other side):

    side of the source:  u = I / (2π σs) (1/r + κ/r')
    other side:          u = I / (π (σa + σb) r)

For an electrode on the contact, this is a half-space of conductivity (σa + σb)/2, the exact
solution. The source terms of the boundary correction are q_k = L_k(σref) U0_k
(`Grid25FV.contact_source`), so the numerical solution is exact for the reference model (with
`correct_quadrature`, tested). With `local_sigma0 = False`, σa = σb = geometric mean of the model
(same as before).

Without the boundary correction (`apply_bc = False`), the point sources are distributed with the
weights of `flux_interp`, so that reciprocity holds exactly (swapping current and potential
electrodes gives the same data).

## 3D

The same method is used on 3D rectilinear grids (`GridFV`).

- `GridFV.flux_interp`: flux continuity along x, then along y (conductivities of the rows interpolated
  along x), then along z. The factor of the faces along x is averaged over the 2 rows along y, so that
  c is the same as in 2.5D for a model invariant along y.
- `GridFV.contact_reference`: an electrode on a vertical edge of the grid (node of x and y) touches 4
  columns of cells. The reference model is made of the 4 quarter-spaces around the edge, with the
  conductivities of the 4 cells in the row of the electrode. The boundaries are radial planes through
  the electrode, so the potential I / (2π σ̄ r), σ̄ being the mean of the 4 conductivities, is the exact
  solution. Otherwise, as in 2.5D, 2 quarter-spaces separated by the nearest vertical face (normal to
  x or to y), with one image. The source term is q = A(σref) u0 (`GridFV.contact_source`). In 3D, it
  includes the reference potential term of A (A[0, 0]), which is proportional to u0, so q depends on
  the conductivity of the reference model even when it is homogeneous.

Dyke with electrodes on the contacts (on nodes, y = 0, and on faces, y = 0.5), 1 m cells: maximum
error 460–815 % with the previous method, 19–66 % with `contact_interp`, and 10–16 % with 0.5 m cells
(second order). On a random 3D model (electrodes on nodes and faces), the error relative to a grid
4 times finer, normalized by the largest datum, is 2.2 and 0.83 % (6.3 and 4.6 % with the previous
method) with the cells divided by 1 and 2. Convergence is closer to first order there: electrodes near
the corners where 4 blocks of different conductivity meet are not exactly described by either reference
model.

## MMR (`GridMMR`)

- Electric dipoles at the measurement points (`pod_e`, 2.5D and 3D): potentials, treated as in
  `GridDC` when `g.dc.contact_interp` is True (reference models of the current electrodes,
  flux-continuity interpolation at the ends of the dipoles, contrast factor), on and off the line.
  Dyke (100, 1000, 50 Ω·m), dipoles with an end on a contact and the current electrode on the other
  contact, 1 m cells: 387 % error before, 0.06 % now (2.5D).
- B: the fields keep the previous treatment of the current electrodes. B depends on the current
  density, not on the potential at a point. With the current electrode on a contact of a dyke, the
  new reference model changes Bz by less than 0.1 % (the error, up to 27 % with 5 m cells at the
  receiver across the contact, comes from the resolution of the field near the contact, and decreases
  with the cell size). The sensitivity of B is therefore unchanged.

## Sensitivity

In addition to the usual adjoint term (adjoint sources Qfᵀ at the potential electrodes):

- the weights Qf and the factor c depend on the 4 cells (8 in 3D) around each potential electrode. Their
  derivatives are computed by complex step (`flux_interp(..., derivatives=True)`);
- P, u0 and q_k depend on the conductivities of the reference model of each current electrode (2 or 4
  cells). In 2.5D, q_k depends on the ratio σa/σb only: its term is the product of the adjoint fields
  with dq_k/d ln σa (`Grid25FV.contact_source_derivative`). In 3D, see
  `GridFV.contact_source_derivatives`;
- without the boundary correction, the point sources depend on the cells around the current
  electrodes (reciprocal of the term of Qf).

These terms are added in `GridDC._contact_sens_term` (on-line 2.5D and 3D) and in `_fwd_offline25`
(off-line, with the weights W_i(Δy) instead of g_i/2). They are checked against finite differences in
`tests/test_dc25.py` (`TestContact`, `TestOffLineSensitivity`) and `tests/test_dc3d.py`
(`TestContact`).

## Scope and limitations

- `GridDC` with a 2.5D grid (on-line and off-line measurements) or a 3D rectilinear grid. Set
  `g.contact_interp = False` for the previous method.
- `GridMMR`: the electric dipoles (`pod_e`) follow `g.dc.contact_interp`; the B fields use the previous
  method (see above). In a joint inversion, the DC data use `contact_interp`.
- Not on tetrahedral meshes.
- The reference model handles vertical contacts only. A buried current electrode on a horizontal
  contact keeps a homogeneous reference along z (the interpolation at potential electrodes handles
  both directions: a potential electrode on the boundary of a two-layer earth is within 1 % of the
  image series solution, 10 % before).
- About 30 % more time for the forward modelling of a small grid (fields of the image sources).
