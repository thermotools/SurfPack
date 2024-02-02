# Table of Contents
* [Implementation structure](#implementation-structure)
* [Weighted densities](#weighted-densities)
* [Convolutions](#convolutions)
* [External potentials](#external-potentials)
* [Numerical routines](#solvers)
* [Units](#units)


# Implementation structure
This is how the model in the src2 directory is organised

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

Numerical solvers are found in `solvers.py`. The current state is a little messy, but trust me, there is a plan behind it:

In order to obtain a converged density profile, you should be calling one of

* `solve_sequential_NT`
* `solve_sequential_muT`
* `solve_sequential_rhoT`

which are differentiated only in what constraints they take as arguments. All the above will just re-package the supplied arguments 
and forward a call to `solve_sequential`.

When calling one of the `solve_sequential_*` functions, you must supply a list of solvers, a list of tolerances and optionally a list
of `kwarg`s for each solver. Each constraint set has an associated pair of solvers

* `solve_sequential_NT` <= `picard_NT` and `anderson_NT`
* `solve_sequential_muT` <= `picard_muT` and `anderson_muT`
* `solve_sequential_rhoT` <= `picard_rhoT` and `anderson_rhoT`

The sequential solver will run the first solver in the supplied list of solvers until it reaches the first tolerance in the supplied list of tolerances.
Then it advances to the second solver, etc. In addition, the sequential solver implements functionality such as falling back to the previous 
solver if a solver exhibits divergent behaviour, or prompting the user for instructions if a solver reaches the maximum number of iterations
without converging, but is otherwise well-behaved.

The functions `picard_*` and `anderson_*` are fairly light-weight functions that define an appropriately formatted residual function, and call the
general `picard` and `anderson` functions. These solvers are at the bottom of the stack, and are more or less completely general numerical solvers
that simply accept a callable and an initial guess as arguments.

The purpose of this structure is essentially that the user should only have to deal with data structures that make sense (i.e. `Profile`s for the 
density profiles, `float`s and `ndarray[float]`s for constraints), while we can have general solvers internally that deal with big, messy vectors
that package all density profiles, fugacities, etc. into one long vector. This way, the user-interface makes sense, and the bottom-level solvers 
are readable, while intermediate functions handle the un- and re-packaging of the data structures.

# Units

The current implementation uses SI units with the exception of the length unit (Å) and the particle number unit (dimensionless).

Thus, the conversion factor `1e30 / Avogadro` turns up some places throughout the code, specifically when converting densities
to/from SI units.

I've been thinking about using $k_B$ to reduce the energy (i.e. $E' = E / k_B$) because it turns out that using particles / Å / Joule
often results in energies in the range of `1e-23` (which isn't surprising). I've also been thinking about implementing a `units.py`
module to handle generation of conversion factors. The latter would definitely reduce the probability of introducing silly bugs
when implementing new stuff.

The `HardSphere` model implements a method `weighted_densities_to_SI`, which can be useful.