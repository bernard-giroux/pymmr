# pymmr
Code for magnetometric resistivity data inversion in python

## Main features of the current version

- MMR and ERT
- Discretisation is limited to rectilinear grids
- Finite-Volume forward solver
- joint MMR-ERT inversion
- 2.5D modelling and inversion (conductivity invariant along y) for ERT and MMR: a grid defined by
  2 coordinate arrays `(x, z)`, or a VTK model one cell thick along y with `run_job.py`.  For MMR,
  current electrodes and B-field sensors can be anywhere along y; see `docs/mmr25.md` and
  `examples/example_mmr25.py`
- time-lapse constraints not yet implemented


### Warning
This code is under continuous development and should not be considered as mature.