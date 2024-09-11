---
layout: default
version: 1.0.0
title: Getting started
permalink: /v1.0.0/getting_started.html
---

Currently, the primary documentation to help you get started using `SurfPack` will be the examples
found in the [pyExamples directory]().

SurfPack is designed much the same way as [`ThermoPack`](/thermopack/index.html) and 
[`KineticGas`](/KineticGas/index.html): Models are classes, and the primary user-interface to
computations is the parent class `Functional`. For a full overview of available method, see the
[Documentation for the `Functional` class](/SurfPack/v1.0.0/Functional_methods.html).

# Initialising a model

To initialise a model, we supply a comma-separated string of [component identifiers](/thermopack/v1.0.0/Component-name-mapping.html), 
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

# Computing Surface- and Interfacial properties

Using `SurfPack` we can compute a variety of properties, such as surface tensions, adsorbtions, curvature expansion coefficients such as 
the Tolman length, nucleation energies, and excess properties of surfaces. Methods for computing Properties are divided into two major categories

* [Profile Property interfaces](Functional_methods.html#profile-property-interfaces)
  * Compute properties for a given density profile, *without* ensuring that the profile is a stationary point of the Grand Potential of the system (an equilibrium profile).
* Direct Property interfaces
  * Compute properties for given bulk conditions or boundary conditions, after (implicitly) converging the density profile for these conditions.
  * Currently consists of
    * [$\rho$-T Property interfaces](Functional_methods.html#$\rho---t$-property-interfaces): Compute properties for given bulk densities and temperature.
  * Likely to include in the future
    * $\mu$-T Property interfaces: Compute properties for given chemical potential and temperature
    * $V_{ext}$-T Property interfaces: Compute properties for given external potential and temperature
    * (possibly) p-T Property interfaces: Compute properties for given bulk pressure and temperature

In practice, the latter are often more convenient for an end-user, which may want to compute an adsorbtion isotherm, or a surface tension 
isotherm for some mixture. However, because the latter methods implicitly need to converge a density profile, they are significantly more
costly. The former can be used explicitly if one has computed and saved a set of density profiles, and wishes to compute properties for them.

## Direct property calculations

*Note:* Take a look at the section on [Caching profiles](#saving-results-for-later) before you start doing computations in order to
save yourself some runtime while learing how to use the different interfaces.

*Note:* To do computations for different geometries (e.g. spherical bubbles/droplets etc.) take a look at the section
on [Setting up the Grid](#setting-up-the-grid). Methods in the Direct Property interfaces accept a `grid` argument that can be used to specify
the geometry of the system.

Once we have initialized a model, we can compute properties directly as

```python
import matplotlib.pyplot as plt
from surfpack.pcsaft import PC_SAFT

dft_hex = PC_SAFT('NC6')  # Initialise PC-SAFT DFT model for pure hexane
dft_mix = PC_SAFT('C3,NC6')  # Initialise PC-SAFT DFT model for propane/hexane mixture

# Compute the surface tension of hexane as a function of temperature, at 10 points from
# 0.7 * T_c to 0.9 * T_c, where T_c is the critical temperature obtained from the eos.
surf_tens, temps = dft_hex.surface_tension_singlecomp(n_points=10, t_min=0.7, t_max=0.9)
plt.plot(temps, surf_tens, label='NC6')
plt.xlabel('T [K]')
plt.ylabel(r'$\gamma$ [J Å$^{-2}$]') # NOTE: Units of Å for length
plt.show()

# Compute the surface tension of the mixture as a function of composition, at a given temperature (300 K)
surf_tens, x_C3 = dft_mix.surface_tension_isotherm(300, n_points=10)
plt.plot(x_C3, surf_tens)
plt.xlabel(r'Mole fraction Propane in Liquid')
plt.ylabel(r'$\gamma$ [J Å$^{-2}]$')
plt.show()
```

for mixtures, we can get the vapour composition and saturation pressure in the same call, as a 
ThermoPack [XYDiagram object](/thermopack/v2.2.0/thermo_methods.html#get_binary_pxyself-temp-maximum_pressure150000000-minimum_pressure1000000-maximum_dz0003-maximum_dlns001)
as

```python
surf_tens, lve = dft_mix.surface_tension_isotherm(300, n_points=10, calc_lve=True)

_, axs = plt.subplots(1, 3, sharey='row')
axs[0].plot(lve[0], surf_tens)
axs[0].set_xlabel('Liquid mole fraction C3')
axs[1].plot(lve[1], surf_tens)
axs[1].set_xlabel('Vapour mole fraction C3')
axs[2].plot(lve[2], surf_tens)
axs[2].set_xlabel('Pressure [Pa]')
axs[0].set_ylabel(r'$\gamma$ [J Å$^{-2}$]')
plt.show()
```

Keep in mind that properties computed through calls to thermopack (in this case the saturation pressure)
always use SI units.

We can compute adsorbtion isotherms the same way:

```python
# Continiued
ads, lve = dft_mix.adsorbtion_isotherm(300, n_points=10, calc_lve=True)
plt.plot(lve[0], ads[0], label='C3')
plt.plot(lve[0], ads[1], label='NC6')
plt.xlabel('Liquid mole fraction Propane')
plt.ylabel('Adsorbtion [particles / Å$^2$]')
plt.legend()
plt.show()
```

## Profile Property calculations

The profile property interfaces are in a sense simpler than the direct property interfaces: They simply accept a density profile,
temperature, and optionally an external potential, as arguments and compute properties assuming that the profile is converged.

Keep reading the next sections for information on how to converge a density profile.

We compute surface tensions and adsorbtions as
```python
from surfpack.pcsaft import PC_SAFT

dft = PC_SAFT('C3,NC6')  # Initialise PC-SAFT DFT model for propane/hexane mixture
T = 300 # K
# ... Compute equilibrium density profiles for some set of conditions ...
rho = # The density profiles

surf_tens = dft.surface_tension(rho, T, dividing_surface='t') # Compute the surface tension at the surface of tension
ads = dft.adsorbtion(rho, T, dividing_surface='e') # Compute adsorbtion at the equimolar surface
```

# Converging a density profile

Pretty much all calculations in DFT involve converging the equilibrium density profile of a system at some conditions.
This section will cover how you can set up and run these calculations, including how to use different system geometries
and tweaking the numerical solvers.

## The Profile data structure

The profiles we compute in DFT, be they particle density profiles, Helmholtz energy densities,
etc. are inextricably tied to a spacial discretisation and a control volume. Therefore, the profiles
computed using `SurfPack` are returned in a `Profile` object, which holds a grid, and contains
some convenience methods.

The most important attributes and methods of the `Profile` class are
* `grid` - The connected `Grid` object
* `save`/`load` - Convenience methods to save and load profiles to a (kind of) human readable format
* `integrate` - Method that integrates the `Profile` over the control volume.
  * This method implicitly handles the geometry of the `Grid`.

*Note:* It can sometimes be more convenient to use the methods in the `Functional` class to save the results of computations.
See the section on [saving profiles for later](#saving-results-for-later) for more on that.

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

You will rarely, if ever, need to initialize a `Profile` yourself, they are simply the data structure `SurfPack` uses internally,
and will return your results in.

## Setting up the Grid

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
Will generate a grid in a spherical geometry, extending 10-50 Å from the origin. This can
be useful for example if we are modelliing a bubble with a radius of approximately 30 Å. Similarly
```python
from surfpack import Grid, PlanarGrid, SphericalGrid, Geometry

n_points = 500
domain_size = 30 # Å
domain_start = 10 # Å 
grid = PlanarGrid(n_points, domain_size, domain_start=domain_start)
```
will give a grid for a planar geometry, extending from z = 10 to z = 40.

## Computing a density profile

Now that we have set up a grid, we can compute density profiles. This is done py calling the methods in the 
[Density Profile interfaces](Functional_methods.html/#density-profile-interfaces). Note that most of these methods are
simply convenience methods that will end up forwarding a call to [density_profile_twophase](#density_profile_twophaseself-rho_g-rho_l-t-grid-beta_v05-rho_0none-solvernone-verbose0)

Example: Computing the density profile of a planar interface

```python
import matplotlib.pyplot as plt
from surfpack.pcsaft import PC_SAFT
from surfpack import PlanarGrid

dft = PC_SAFT('C3,NC6')  # Initialise PC-SAFT DFT model for propane/hexane mixture
T = 300  # K
x = [0.2, 0.8]  # Liquid composition

n_points = 500
domain_size = 30  # Å
grid = PlanarGrid(n_points, domain_size)  # Setting up grid for computation

rho = dft.density_profile_tz(T, x, grid, z_phase=dft.LIQPH) # Indicating that the specified mole fractions apply to the liquid
# NOTE: The grid may be adapted during computation if it is initially too narrow to fit the profile extending into the bulk
# phase on each side of the interface. We should therefore use the grid held by the returned profiles to plot.
plt.plot(rho[0].grid.z, rho[0], label='C3') 
plt.plot(rho[1].grid.z, rho[1], label='NC6')
plt.xlabel(r'Position [Å]')
plt.ylabel(r'Density [particles / Å$^3$]')
plt.show()

# Specifying the vapour composition
y = [0.4, 0.6]
rho = dft.density_profile_tz(T, y, grid, z_phase=dft.VAPPH) # Indicating that specified mole fractions apply to vapour
plt.plot(rho[0].grid.z, rho[0], label='C3') 
plt.plot(rho[1].grid.z, rho[1], label='NC6')
plt.xlabel(r'Position [Å]')
plt.ylabel(r'Density [particles / Å$^3$]')
plt.show()
```

We can also choose to specify temperature, pressure and ''total'' composition, where the ''total'' composition refers
to the bulk compositions, i.e. the total composition one would see when using an EoS and a flash algorithm, neglecting
adsorbtion.

```python
# Continiued
z = [0.7, 0.3]
p = 1e6
T = 300
rho = dft.density_profile_tp(T, p, z, grid)
```

if the state supplied to `density_profile_tp` is a pure vapour or liquid phase (determined from a TP-flash) a warning will
be issued.

### Profiles at walls and in pores

If we wish to do computations for wall adsorbtion, or look at density profiles in pores, we need to specify an external
potential. In principle, the external potentials are just callables, but for convenience some classes are set up in `external_potential.py`.

```python
import matplotlib.pyplot as plt
from surfpack.external_potential import HardWall
from surfpack import PlanarGrid
from surfpack.pcsaft import PC_SAFT

dft = PC_SAFT('C3,NC6')  # Initialise PC-SAFT DFT model for propane/hexane mixture
T = 600  # K
p = 1e5
z = [0.2, 0.8]  # Bulk composition

# The external potential is a list, with one potential for each component
Vext = [HardWall(1, is_pore=False), HardWall(1, is_pore=False)]  # A hard wall, positioned a 1 Å
n_points = 500
domain_size = 30  # Å
grid = PlanarGrid(n_points, domain_size)  # Setting up grid for computation
rho = dft.density_profile_wall_tp(T, p, z, grid, Vext)
plt.plot(rho[0].grid.z, rho[0], label='C3')
plt.plot(rho[1].grid.z, rho[1], label='NC6')
plt.xlabel('Position [Å]')
plt.ylabel(r'Density [Å$^{-3}$]')
plt.legend()
plt.show()
```
## Saving results for later

Because it can be expensive to compute density profiles, we want to be able to save them.

The `Profile` class has a couple methods that can be used to store profiles in a human readable format, as
```python
from surfpack import Profile
rho = # ... some previously computed profile
Profile.save_list(rho, 'path/to/my_profile.json') # Saving the profile
rho_saved = Profile.load_file('path/to/my_profile.json') # Retriving the profile
```

However, for direct property computations, we are often less interested in spending time writing filenames. We can set
a model to automatically save and search for previously computed profiles as

```python
from surfpack.pcsaft import PC_SAFT

dft = PC_SAFT('AR,KR') # pc-saft for argon/krypton mixture
dft.set_cache_dir('directory/to/save/profiles/in') # This directory will be created if it does not exist
```
now, when we call e.g. `dft.adsorbtion_isotherm`, all computed density profiles will be stored in the indicated directory.

If we later do some other computation, for example `dft.surface_tension_isotherm`, the model will search the directory 
for previously computed density profiles, and only compute those that have not been previously evaluated.

The file names are generated in a way that ensures a unique file name for every model and state, but are less friendly 
to the human eye. If you need to inspect the saved profiles manually, the directory will also contain a file `000_REGISTER.txt`,
which contains the file name of each profile in the directory and information about the model and state for which it was computed.

# For Developers

If you are implementing a new Functional, or otherwise developing the package, it may be fruitfull to take a look at the files
`arrayshapes.md` and `readme.md` in the `surfpack` directory.