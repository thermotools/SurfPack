---
layout: default
version: 
title: Getting started
permalink: /vcurrent/getting_started.html
---

Currently, the primary documentation to help you get started using `SurfPack` will be the examples
found in the [pyExamples directory]().

SurfPack is designed much the same way as [`ThermoPack`](/thermopack/index.html) and 
[`KineticGas`](/KineticGas/index.html): Models are classes, and the primary user-interface to
computations is the parent class `Functional`. For a full overview of available method, see the
[Documentation for the `Functional` class](/SurfPack/vcurrent/Functional_methods.html).

# Initialising a model

To initialise a model, we supply a comma-separated string of [component identifiers](/thermopack/vcurrent/Component-name-mapping.html), 
as
```python
from surfpack.pcsaft import PC_SAFT
dft = PC_SAFT('C3,NC6') # Initialise PC-SAFT DFT model for propane-hexane mixture
```
`SurfPack` supports all components for which `ThermoPack` has parameters for the corresponding
equation of state.

Each DFT model holds an internal equation of state object, called `eos`, so we can access the
full `ThermoPack` functionality as
```python
# Continiued
T = 300 # Kelvin
vle, l1ve, l2ve = dft.eos.get_binary_pxy(T) # Computed using ThermoPack pc-saft eos with the same parameters as the DFT is using
```

# Setting up the Grid

When doing DFT calculations, it is seldom enough to specify the components in our system. In addition,
we typically need to specify a spatial discretisation and a geometry. This information is stored
in an instance of the `Grid` class, or its childred, `PlanarGrid` and `SphericalGrid`, which are
initialised as
```python
from surfpack import Grid, PlanarGrid, SphericalGrid, Geometry

n_points = 500
domain_size = 30 # Å
grid1 = Grid(n_points, Geometry.PLANAR, domain_size)
grid2 = PlanarGrid(n_points, domain_size) # Exactly equal to grid1

grid3 = Grid(n_points, Geometry.SPHERICAL, domain_size)
grid4 = SphericalGrid(n_points, domain_size) # Exactly equal to grid3
```
A grid will by default start at the origin, and extend in the positive direction. We may be
interested in modelling a region further away from the origin, in which case we can use the
`domain_start` keyword argument, as
```python
from surfpack import Grid, PlanarGrid, SphericalGrid, Geometry

n_points = 500
domain_size = 40 # Å
domain_start = 10 # Å 
grid = SphericalGrid(n_points, domain_size, domain_start=domain_start)
```
Will generate a grid in a spherical geometry, extending from 10-50 Å from the origin. This can
be useful for example if we are modelliing a bubble with a radius of approximately 30 Å. Similarly
```python
from surfpack import Grid, PlanarGrid, SphericalGrid, Geometry

n_points = 500
domain_size = 30 # Å
domain_start = 10 # Å 
grid = PlanarGrid(n_points, domain_size, domain_start=domain_start)
```
will give a grid for a planar geometry, extending from z = 10 to z = 40.

# The Profile data structure

The profiles we compute in DFT, be they particle density profiles, Helmholtz energy densities,
etc. are inextricably tied to a spacial discretisation and a control volume. Therefore, the profiles
computed using `SurfPack` are returned in a `Profile` object, which holds a grid, and contains
some convenience methods.

The most important attributes and methods of the `Profile` class are
* `grid` - The connected `Grid` object
* `save`/`load` - Convenience methods to save and load profiles to a (kind of) human readable format
* `integrate` - Method that integrates the `Profile` over the control volume.
  * This method implicitly handles the geometry of the `Grid`.

The `Profile` class inherits `numpy ndarray`, such that it can be treated as a numpy array in 
calculations.

For methods that compute several profiles, for example the density profiles of a mixture, a 
list of profiles is returned.

Thus, if we wish to plot a density profile, we typically do this as
```python
# pf is a previously computed list of density profiles
plt.plot(pf[0].grid.z, pf[0], label='Component 1')
plt.plot(pf[1].grid.z, pf[1], label='Component 2')
plt.xlabel(r'$z$ [Å]')
plt.ylabel(r'$\rho$ [Å$^{-3}$')
```

# Units

All IO in `SurfPack` is currently in SI *with the exception* of particle numbers and lengths, 
which are in ''particles'' and Å respectively. This gives us the internally consistent unit system

| Quantity  | SI Unit | SurfPack Unit  |
|-----------|---------|----------------|
| Particles | mol     | -              |
| Lenght    | m       | Å              |
| Energy    | J       | J              |
| Mass      | kg      | kg             |
| Time      | s       | $$10^{-20} s$$ |

# For Developers

If you are implementing a new Functional, or otherwise developing the package, it may be fruitfull to take a look at the files
`arrayshapes.md` and `readme.md` in the `surfpack` directory.