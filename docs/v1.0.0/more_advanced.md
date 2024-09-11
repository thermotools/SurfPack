---
layout: default
version: 1.0.0
title: Advanced Usage
permalink: /v1.0.0/more_advanced.html
---

# Tuning the solvers

Some density profiles may be more difficult to converge than others. The numerical solver routines used by `SurfPack`
are designed to be as easily tunable as possible.

See [pyExamples/solver_setup.py](https://github.com/thermotools/SurfPack/blob/main/pyExamples/solver_setup.py) for examples
on how to set up a tuned `SequentialSolver` for a specific problem. 