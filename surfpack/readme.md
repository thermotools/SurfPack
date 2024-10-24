# Table of Contents
* [Implementation structure](#implementation-structure)
* [Weighted densities](#weighted-densities)
* [Convolutions](#convolutions)
* [External potentials](#external-potentials)
* [Numerical routines](#solvers)
* [Units](#units)


# Implementation structure
This is how the model in the surfpack directory is organised

All DFT models inherit the `Functional` class, which serves a purpose analogous to the `thermo` class in ThermoPack. The inheritance tree at the moment of writing is

* `Functional`
  * `FMT_Functional` - ''Dumb'' FMT model (plain hard sphere, no temperature dependency) (in `hardsphere.py`)
    * `Rosenfeld`
    * `WhiteBear`
      * `SAFT_WhiteBear` - Implements temperature dependent hard sphere diameter (in `saft_hardsphere.py`)
    * `WhiteBearMarkII`
    * `SAFT_HardSphere` - Implements temperature dependent hard sphere diameter (in `saft_hardsphere.py`)
  * `SAFT` - Parent class for SAFT based functionals. Implements dispersion contribution to Helmholtz energy. Holds an internal `SAFT_HardSphere` model that is used to calculate hard sphere contribution.
            Using multiple inheritance may be a better solution here. It may even be viable to re-implement the FMT with temperature dependent hard-sphere diameters, or remove the `SAFT_HardSphere` class
            completely, and have it accessible through the saft-class by disabling other contributions.
    * `SAFT_VR_Mie` - The SAFT-VR Mie functional (in `saftvrmie.py`)
    * `PC_SAFT` - The PC-SAFT functional, implements association and chain contributions to Helmholtz energy

The most important utility classes are the `Grid` and `Profile` classes. The `Grid` class holds information about the geometry of the system, as well as a couple utility functions to compute 
the volume of the system, as well as the cross-sectional area. By using these methods, other functions that need either the volume of the system (or a sub-system) or the cross sectional area
(for planar systems) or the area of a "shell" at a certain radius (for spherical systems) can be agnostic as to the geometry of the system.

The `Profile` class inherits from `np.ndarray`. It is intended to hold e.g. a density profile. In addition to the profile values, it holds a `Grid` object. This way whenever a `Profile` is passed
to a method, all information about the system geometry, grid resolution, etc. is passed along with it. In addition, the `Profile` hold the attribute `is_vector_valued`, because some weighted 
densities are vector valued, and must be treated differently in convolutions. The `Profile` class also contains a couple utility methods

* `integrate` - Integrate the profile (implicitly handles geometry of integration domain)
* `save` - Save the profile to a file
* `load` - Load profile from file
* Several methods to initialise a profile of a certain shape (e.g. `tanh_profile` and `zeros` and `zeros_like`)
* `on_grid` - Transfer profile to a new `Grid`, using interpolation if the new grid has higher resolution
* `__call__` - Get profile value at a position by interpolating the grid.
* `get_position_idx` - Get nearest grid index for a given position.

In practice, a `Profile` rarely comes alone. Most commonly, we have a `list[Profile]` where each element in the list contains the density profile for one species.

* Optimally, a class should be implemented such that we can drop the `list[Profile]`, and rather have a `Profiles`, where every profile shares the same underlying `Grid`. This is probably best do in a language such as "NOT PYTHON" where we have access to pointers. 

# Weighted densities

Weighted densities come up often in DFT - so often that the file `arrayshapes.md` is dedicated to documenting how weights and weighted densities are
organised in arrays throughout the package. Here, we make some generic comments on how weighted densities are treated in such a way that we have
as few as possible "special cases".

## Note on segment densities

Segment densities are just special cases of a weighted density. We do not treat segment densities as anything different from other weighted densities.
Essentially, the local density can also be treated as a weighted density. The only issue in this regard is essentially a formal, mathematical one, not 
an implementation issue. 

The formal, mathematical issue arises from the fact that there is no function that is both the identity element of the 3D convolution,
and has an integral (over $R^3$) equal to unity. This is the same issue as with the definition of the delta function, and someone with
more mathematical background than me might have a formally acceptable solution. Implementation wise, we can define a 
`WeightFunction`, called `LocalDensity`, which is treated specially in the `Convolver` module

##

# Convolutions

Convolutions are handled in the functional module `Convolver`, which is based around the concept of convolving a discrete function (i.e. a `Profile`) with an analytical function (i.e. weight function).

The weight functions are defined in the `WeightFunction.py` module as callable classes that inherit from `Analytical`. These implement some functionality that make them behave like analytical functions.
please note that the `WeightFunction`s are implemented as their Fourier transforms, so that `myweight(k)` returns the Fourier transform of the weightfunction, not the real weight function.

The `WeightFunction` objects also hold an attribute indicating wheter or not they are vector valued.

Finally, the `Analytical` class implements the method `real_integral`, which returns the integral over R3 of the weight function. This is convenient to have access to 
when implementing bulk weighted densities, which are just 
$$
n = \rho * w.real_integral()
$$

# External potentials

External potentials can in principle be implemented as any callable. In the current implementation, I've used callable classes that also implement `__hash__`, which was done so that computations could 
be stored in a dict, where the dict key contained both information about the state point and the external potential. I don't know if that's neccessary to maintain in the long run, but so far it hasn't 
been much hassle.

# Solvers

Numerical solvers are found in `solvers.py`. The current state is a little messy, mostly due to some more-or-less obsolete
code not being cleaned out yet. The most important point is the following:

Some pre-tuned solvers can be retrieved using the `get_solver` function. This function accepts several keys, used to 
specify the type of mixture you are working with, and will return a solver tuned to that mixture. If no tuned solver exists,
a default solver is returned. If that solver does not work, you will need to tune your own solver.

## Tuning a solver

The numerical solver you want to use to converge a density profile is an instance of the `SequentialSolver` class. This
class is callable, and will iterate through a series of progressively more agressive solvers until the problem is converged,
with some fallback-mechanisms etc. to handle cases where the solvers are too agressive.

The `SequentialSolver` class implements the methods `add_anderson` and `add_picard` which you can use to build a solver.
For examples, see: `../pyExamples/solver_setup.py`

# Units

The current implementation uses SI units with the exception of the length unit (Å) and the particle number unit (dimensionless).

Thus, the conversion factor `1e30 / Avogadro` turns up some places throughout the code, specifically when converting densities
to/from SI units.

I've been thinking about using $k_B$ to reduce the energy (i.e. $E' = E / k_B$) because using particles / Å / Joule
often results in energies in the range of `1e-23`. I've also been thinking about implementing a `units.py`
module to handle generation of conversion factors. The latter would definitely reduce the probability of introducing silly bugs
when implementing new stuff.

The `HardSphere` model implements a method `weighted_densities_to_SI`, which can be useful.