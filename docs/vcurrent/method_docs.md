---
layout: default
version: 
title: Documentation
permalink: /vcurrent/method_docs.html
---

Models in SurfPack are Helmholtz energy Functionals, implemented as classes. This means that almost all user-end 
functionality is concentrated in the `Functional` class, while inheriting classes simply override the methods neccessary
to compute the Helmholtz energy density and derivatives.

Numerical solvers are found in `solvers.py`, and are in practice interfaced through either the `SequentialSolver` or 
the `GridRefiner` class. All methods that require a numerical solver use a default solver if no solver is provided,
but for problems that are difficult to converge, you can likely achieve quite significant speedups by tweaking the 
solver yourself.

For Example usage, see the [pyExamples]() directory.

The inheritance structure in the program is

* Functional
  * SAFT
    * PC_SAFT
    * SAFT_VR_Mie

The `Profile` class is the core data structure used in SurfPack to represend discretized density profiles. It should be
documented well somewhere.