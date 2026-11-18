# Iterative solvers for the MMR system

    g.set_solver(bicgstab, precon='amg')    # scipy.sparse.linalg.bicgstab, needs pyamg

`precon` can be `'amg'`, `'ilu'`, `'diag'` or `'0'`. With `'amg'`, BiCGStab is the fastest iterative
option tested for the MMR system; it needs `pyamg` (`pip install pyamg`), which is optional.

## Structure of the matrix

On a rectilinear grid, the MMR matrix A = M (C_e C_f − G D) has these properties:

- it is three decoupled scalar Laplacians, one per face orientation (the blocks coupling them are at
  the level of round-off, 1e-17 of the norm of A);
- it does not depend on the conductivity, which only enters the right-hand side;
- it is not symmetric on a stretched grid (‖A − Aᵀ‖ / ‖A‖ = 0.04 with padding), but a diagonal scaling
  makes it symmetric to round-off.

Multigrid is well suited to such a matrix. The preconditioner ignores entries below 1e-12 √(a_ii a_jj),
takes the connected components of what remains (3 for the MMR matrix, 1 for a DC matrix), and applies
one classical (Ruge-Stüben) V-cycle per component. The Krylov solver still uses the complete matrix.

## Comparison

3D grid of `examples/example_mmr.ipynb` with a conductive block (σ = 0.1 S/m in 0.001 S/m), 2 current
dipoles, `rtol = 1e-5`, `atol = 1e-9`, one thread. "B error" is the relative error of B at the
receivers against MUMPS. Setup is the preconditioner (factorization for MUMPS); solve is per right-hand
side. Iterations are the larger count of the two right-hand sides.

1 273 944 unknowns (h = 30 m):

| solver                               | setup (s) | solve (s) | iterations | B error | available with `set_solver` |
|--------------------------------------|----------:|----------:|-----------:|--------:|-----------------------------|
| BiCGStab + AMG (Ruge-Stüben)         |       2.3 |       1.4 |          4 | 1.5e-5  | yes                         |
| CG + AMG (Ruge-Stüben), symmetrized  |       3.3 |       3.1 |         17 | 7.6e-7  | no                          |
| CG + smoothed aggregation, symmetrized |     3.1 |      10.5 |         59 | 1.0e-6  | no                          |
| CG + diagonal, symmetrized           |       1.1 |       6.4 |        615 | 1.1e-6  | no                          |
| BiCGStab + diagonal                  |       0.0 |       7.5 |        385 | 3.3e-5  | yes                         |
| BiCGStab, no preconditioner          |       0.0 |       9.9 |        524 | 1.1e-4  | yes                         |
| BiCGStab + ILU (`spilu` defaults)    |       112 |      44.6 |        295 | 1.5e-5  | yes                         |
| GMRES(50) + diagonal                 |       0.0 |      21.1 |        748 | 7.2e-6  | no (needs `restart`)        |
| MUMPS (38 GB)                        |       227 |      1.25 |          – | –       | yes                         |

357 656 unknowns (h = 60 m):

| solver                       | setup (s) | solve (s) | iterations | B error |
|------------------------------|----------:|----------:|-----------:|--------:|
| BiCGStab + AMG (Ruge-Stüben) |       0.7 |      0.45 |          4 | 6.2e-7  |
| BiCGStab + diagonal          |       0.0 |      1.56 |        244 | 2.1e-5  |
| BiCGStab + ILU               |      13.8 |      6.52 |        147 | 1.2e-5  |
| MUMPS (7.4 GB)               |      27.0 |      0.21 |          – | –       |

- AMG is faster than the diagonal from the first right-hand side, and its number of iterations does
  not grow with the grid.
- ILU is the slowest option here: it costs more per iteration than it saves in iterations.
- For the same residual, CG gives a B about 20 times more accurate than BiCGStab, but it needs the
  symmetrizing scaling, which `Solver` does not apply.
- MUMPS overtakes AMG after about 110 right-hand sides at 358 000 unknowns, and 1 750 at 1.27 million,
  with about 15 to 20 times the memory. A does not change with the conductivity, so its setup, like that of
  AMG, can be kept for the whole inversion (`keep_solver`).

These runs are for one model, on rectilinear grids. In 2.5D, AMG agreed with MUMPS to 8e-11 on a small
MMR example (`rtol = 1e-9`); its speed was not measured. Tetrahedral meshes were not tested, and DC
systems were only tested on the small grid of `tests/test_solver.py`.

## Tolerances

An iteration stops at max(`rtol` ‖b‖, `atol`). With the default `atol = 1e-9` and ‖b‖ of order 1e-4 to
1e-3 (1 A sources), `atol` is the limit as soon as `rtol` is below 1e-6 to 5e-6: asking for `rtol = 1e-8`
stopped at relative residuals of 1e-6 to 5e-6. Lower `atol` as well to obtain a more accurate solution.

## Running in parallel

Right-hand sides of iterative solvers are distributed over `n_threads` threads, given when the grid is
created (`GridMMR(param, n_threads=8)`, `number of threads` in the parameter file). Threads are used
only with a free-threaded Python **running without the GIL**. mpi4py and pyamg do not declare support for
free threading, so importing them enables the GIL again, with only a warning, unless `PYTHON_GIL=0`:

    PYTHON_GIL=0 python3.15t script.py

`free-threading.env` holds this setting. The state is displayed with `verbose` set:

    Threads: 4 threads, GIL disabled, right-hand sides solved in parallel
      Parallel solve: 4 right-hand sides on 4 threads, wall time 0.07 s, CPU time 0.25 s, CPU/wall = 3.8

The first line comes with the solver description. If threads cannot be used, it says why:

    Threads: 4 requested, but the GIL is enabled (set PYTHON_GIL=0, an imported module enabled it): right-hand sides solved sequentially
    Threads: 4 requested, but this Python is not a free-threaded build: right-hand sides solved sequentially

The second line is measured after the solve. **CPU/wall close to the number of threads shows that they
really run at the same time**; a warning is added if it is below half of that number. It does not give
the speedup, which is lower when threads compete for memory bandwidth: only a comparison with
`n_threads = 1` does. The values are in `solver.last_solve`. In 2.5D, the line sums all wavenumbers.

16 right-hand sides, BiCGStab, 1 273 944 unknowns, Python 3.15t, `OMP_NUM_THREADS=1`, 24 logical cores:

| threads | AMG wall (s) | AMG CPU/wall | AMG speedup | diag wall (s) | diag CPU/wall | diag speedup |
|--------:|-------------:|-------------:|------------:|--------------:|--------------:|-------------:|
|       1 |         21.1 |          1.0 |        1.00 |         101.0 |           1.0 |         1.00 |
|       2 |         10.8 |          1.9 |        1.95 |          52.1 |           2.0 |         1.94 |
|       4 |          5.5 |          3.9 |        3.85 |          28.1 |           3.8 |         3.59 |
|       8 |          3.1 |          7.5 |        6.85 |          20.9 |           7.5 |         4.84 |
|      16 |          2.0 |         14.8 |       10.35 |          19.5 |          14.8 |         5.18 |

With the diagonal, CPU/wall is 14.8 on 16 threads for a speedup of 5.2: the threads are all running, but
are limited by memory access. Note that:

- the number of threads actually used cannot exceed the number of right-hand sides;
- the AMG setup (2.5 s here) is sequential, as is the construction of the matrices;
- direct solvers do not use `n_threads`;
- BLAS threads add to CPU time and can hide a problem: use `OMP_NUM_THREADS=1` when checking.
