# Gauss-Newton step: `cglscd`

At each iteration, `Inversion.run` computes the model perturbation s with `cglscd`, a preconditioned
conjugate-gradient solver for

    (J^T D^2 J + β C) s = J^T D^2 (d_obs − d) − β C (m − m_ref)

(`reg_var = 'model'`; with `'model perturbation'` the last term is absent). J is the Jacobian
(n_data × n_cells), D the data weighting (diagonal) and C the regularization matrix (`WTW`, sparse).
The solver stops when the norm of the gradient has decreased by a factor `tol_cglscd` (1e-9).

## Preconditioner

J^T D^2 J has a rank of at most n_data. Preconditioned by (β C)^-1, the matrix becomes the identity
plus a matrix of rank n_data or less, and CG converges in n_data + 1 iterations in exact arithmetic,
whatever the conditioning (2e7 to 2e8 in the example below). `RegPreconditioner` factorizes C with
the direct solver of the grid (`g.fv.get_solver_params()`; if the grid uses an iterative solver, the
first direct solver available among MUMPS, Pardiso, UMFPACK, PaStiX and SuperLU) and
applies C^-1 r / β, so that C is factorized again only when it changes:

- `smooth_type = 'smooth'`: once per inversion (twice with `model_weighting = 'jacobian'`);
- `'blocky'`, `'ekblom'`, `'min. support'`: at each iteration (IRLS weights).

C is singular without smallness term (`als = 0`); a shift of 1e-6 times its largest diagonal term
is then needed. It changes the preconditioner only, not the solution.

`precond_cglscd = 'jacobi'` gives the former preconditioner, the diagonal of J^T D^2 J + β C. It
used to be computed without D (diag of J^T J): with the variance weighting, D spans 3 decades and the
diagonal was wrong by a median factor of 100 (up to 2000). This is corrected.

Parameters (`Inversion` attributes, `run_job` keywords):

| attribute        | keyword                 | default                                      |
|------------------|-------------------------|----------------------------------------------|
| `precond_cglscd` | `cglscd preconditioner` | `'regularization'`                           |
| `tol_cglscd`     | `cglscd tol`            | 1e-9                                         |
| `max_it_cglscd`  | `cglscd max_it`         | None: 2 n_data + 100 (1000 with `'jacobi'`)  |

If the factorization fails, `Inversion` warns and uses `'jacobi'`.

## Example (`examples/example_mmr25.py`, 297 data, 1000 cells, 6 iterations)

Iterations of `cglscd` at each Gauss-Newton iteration, misfit after 6 iterations, and distance
between the final and true models (log σ), relative to the starting model:

| preconditioner                  | iterations      | misfit | model error |
|---------------------------------|-----------------|--------|-------------|
| Jacobi without D, max_it 1000   | 1000 (warning)  | 0.0153 | 0.735       |
| Jacobi without D, converged     | 6 773 – 14 056  | 0.0152 | 0.710       |
| Jacobi with D, converged        | 1 854 – 7 686   | 0.0152 | 0.710       |
| regularization                  | 137 – 285       | 0.0152 | 0.710       |
| regularization, `blocky`        | 134 – 239       | 0.0151 | 0.653       |
| regularization, `als = 0`       | 236 – 581       | 0.0080 | 2.84        |

A looser tolerance is not a substitute: with Jacobi and tol = 1e-3 or 1e-2 (8 to 316 iterations) the
misfit is lower but the model error is 1.8 to 2.7. The part of the step that enforces the
regularization converges last, and stopping early drops it. Adding the diagonal of J^T D^2 J to β C
in the preconditioner destroys the low-rank structure (1 655 to 6 886 iterations).

## Cost in 3D

Factorization of C (MUMPS) and cost per iteration, compared with the products by J
(n_data = 1000):

| cells   | factorization | C^-1 r     | J p and J^T q |
|---------|---------------|------------|---------------|
| 108 000 | 1.0 s         | 34 ms      | 67 ms         |
| 256 000 | 5.1 s         | 78 ms      | 183 ms        |
| 500 000 | 11.7 s        | 163 ms     | 319 ms        |

An iteration costs about 1.5 times more than with Jacobi, for 10 to 50 times fewer iterations.
With many data (ERT, n_data ≥ 10 000), the number of iterations can approach n_data; it is not
capped at 1000 by default, unlike before.

## Data and model weighting with the Jacobian

`data_weighting = 'jacobian'` (and `'variance*jacobian'`, `'variance+jacobian'`) weights each datum
by 1/‖its row of J‖, and `model_weighting = 'jacobian'` each cell by 1/‖its column of J‖
(`jacobian_weights`). Some data have no sensitivity at all: Bx and Bz at receivers on the line of
the source are 0 by symmetry, whatever the model (66 of the 297 data of the example). Their weight
was infinite, the Gauss-Newton system was NaN, and the NaN step gave a NaN model on which MUMPS failed
with error -9. Rows or columns with a norm of at most 1e-10 times the largest one now get a weight of
0 (data, with a warning giving their number) or the largest weight of the other cells (cells). In
joint inversion, the medians that balance the MMR and ERT weights exclude these data.

Example, 6 iterations, misfit and model error as above:

| data weighting      | misfit | model error |
|---------------------|--------|-------------|
| `variance`          | 0.0152 | 0.710       |
| `variance+jacobian` | 0.0156 | 0.721       |
| `variance*jacobian` | 0.0474 | 0.971       |
| `jacobian`          | 0.0047 | 0.827       |

Non-finite values are no longer silent: `cglscd` raises `ValueError` if the gradient is not
finite (it used to return a zero step, and the inversion went on without changing the model), and
MUMPS factorization raises `RuntimeError` if the matrix has NaN or inf values (instead of error -9
or -10).

## Other changes

- The three `reg_var` cases share one loop.
- `'time-lapse'`: the gradient used C x but the step length used β C; both use β C now. This case
  is not used by `Inversion.run` (no `alpha`, `WTWt`).
- `'model perturbation'`: the constraint array P is applied to the whole initial gradient, as in
  the iterations (P is not used by `Inversion.run`).
- The convergence warning gives the number of iterations performed.

Tests: `tests/test_cglscd.py` (exact solution for the three cases and both preconditioners,
iteration counts, singular C, fixed cells, same inversion result with both preconditioners).
