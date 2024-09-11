---
layout: default
version: 1.0.0
title: Methods in the Functional class
permalink: /v1.0.0/Functional_methods.html
---

<!--- 
Generated at: 2024-02-13T11:55:48.496894
This is an auto-generated file, generated using the script at surfpack/docs/tools/markdown_from_docstrings.py
The file is created by parsing the docstrings of the methods in the 
Functional class. For instructions on how to use the parser routines, see the
file surfpack/docs/tools/markdown_from_docstrings.py--->

The `Functional` class, found in `surfpack/Functional.py`, is the core of SurfPack Density Functional
Theory. This is the interface to almost all practical computations, such as computation of surface
tensions, adsorbtion, etc. and also contains the interfaces to methods used for computing density
profiles in systems with various boundary conditions. All DFT models in SurfPack inherit the `Functional`
class.

## Table of contents
  * [Profile property interfaces](#profile-property-interfaces)
    * [N_adsorbed](#n_adsorbedself-rho-tnone-dividing_surfacee)
    * [adsorbtion](#adsorbtionself-rho-tnone-dividing_surfacee)
    * [correlation](#correlationself-rho-t)
    * [grand_potential](#grand_potentialself-rho-t-vextnone-bulkfalse-property_flagir)
    * [grand_potential_density](#grand_potential_densityself-rho-t-vextnone-property_flagir)
    * [reduced_helmholtz_energy_density](#reduced_helmholtz_energy_densityself-*args-**kwargs)
    * [residual_enthalpy_density](#residual_enthalpy_densityself-rho-t)
    * [residual_entropy_density](#residual_entropy_densityself-rho-t)
    * [residual_helmholtz_energy_density](#residual_helmholtz_energy_densityself-rho-t-bulkfalse)
    * [residual_internal_energy_density](#residual_internal_energy_densityself-rho-t)
    * [surface_tension](#surface_tensionself-rho-t-vextnone-dividing_surfaceequimolar)
    * [tolmann_length](#tolmann_lengthself-rho-t)
  * [$\rho - T$ property interfaces](#$\rho---t$-property-interfaces)
    * [adsorbtion_isotherm](#adsorbtion_isothermself-t-n_points30-dividing_surfacet-x_min0001-x_max0999-solvernone-rho0none-calc_lvefalse-verbosefalse)
    * [radial_distribution_functions](#radial_distribution_functionsself-rho_b-t-comp_idx0-gridnone)
    * [surface_tension_isotherm](#surface_tension_isothermself-t-n_points30-dividing_surfacet-solvernone-rho0none-calc_lvefalse-verbosefalse-cache_dir)
  * [Density profile interfaces](#density-profile-interfaces)
    * [density_profile_singlecomp](#density_profile_singlecompself-t-grid-rho_0none-solvernone-verbosefalse)
    * [density_profile_tp](#density_profile_tpself-t-p-z-grid-rho_0none-solvernone-verbosefalse)
    * [density_profile_twophase](#density_profile_twophaseself-rho_g-rho_l-t-grid-beta_v05-rho_0none-solvernone-verbose0)
    * [density_profile_tz](#density_profile_tzself-t-z-grid-z_phase1-rho_0none-solvernone-verbose0)
    * [density_profile_wall](#density_profile_wallself-rho_b-t-grid-vextnone-rho_0none-verbosefalse)
    * [density_profile_wall_tp](#density_profile_wall_tpself-t-p-z-grid-vext-rho0none-verbose0)
    * [drubble_profile_rT](#drubble_profile_rtself-rho_i-rho_o-t-r-grid-rho_0none)
  * [Bulk property interfaces](#bulk-property-interfaces)
    * [chemical_potential](#chemical_potentialself-rho-t-bulktrue-property_flagir)
    * [fugacity](#fugacityself-rho-t-vextnone)
    * [residual_chemical_potential](#residual_chemical_potentialself-rho-t-bulktrue)
  * [Pure fluid properties](#pure-fluid-properties)
    * [surface_tension_singlecomp](#surface_tension_singlecompself-n_points30-t_min05-t_max099-gridnone-solvernone-rho0none-verbose0)
  * [Weight function interfaces](#weight-function-interfaces)
    * [get_weights](#get_weightsself-t)
  * [Weighted density computations](#weighted-density-computations)
    * [get_weighted_densities](#get_weighted_densitiesself-*args-**kwargs)
  * [Utility methods](#utility-methods)
    * [clear_cache_dir](#clear_cache_dirself-clear_dir)
    * [dividing_surface_position](#dividing_surface_positionself-rho-tnone-dividing_surfacee)
    * [equimolar_surface_position](#equimolar_surface_positionrho)
    * [get_caching_id](#get_caching_idself)
    * [get_characteristic_lengths](#get_characteristic_lengthsself)
    * [get_load_dir](#get_load_dirself)
    * [get_save_dir](#get_save_dirself)
    * [reduce_temperature](#reduce_temperatureself-t-c0)
    * [set_cache_dir](#set_cache_dirself-cache_dir)
    * [set_load_dir](#set_load_dirself-load_dir)
    * [set_save_dir](#set_save_dirself-save_dir)
    * [surface_of_tension_position](#surface_of_tension_positionself-rho-t)
  * [Internal methods](#internal-methods)
    * [\_\_init\_\_](#__init__self-ncomps)
    * [\_\_repr\_\_](#__repr__self)
    * [sanitize_Vext](#sanitize_vextself-vext)
    * [split_component_weighted_densities](#split_component_weighted_densitiesself-n)
    * [validate_composition](#validate_compositionself-z)
  * [Deprecated methods](#deprecated-methods)
    * [_Functional\_\_density_profile_twophase](#_functional__density_profile_twophaseself-rho_left-rho_right-t-grid-rho_0none-constraintsnone)
    * [adsorbed_mean_density](#adsorbed_mean_densityself-profile)
    * [adsorbed_thickness](#adsorbed_thicknessself-profile)
    * [density_profile_NT](#density_profile_ntself-rho1-rho2-t-grid-rho-rho_is_frac-rho_0none)
    * [density_profile_muT](#density_profile_mutself-rho1-rho2-t-grid-rho-rho_is_frac-rho_0none)
    * [find_non_bulk_idx](#find_non_bulk_idxself-profile-rho_b-tol001-start_idx-1)
    * [interface_thickness](#interface_thicknessself-profile-positionsfalse)

## Profile property interfaces

Compute properties using a given density profile. Note that properties computed
using these methods do not generally check whether the Profile is a valid
equilibrium Profile. Properties are computed from the Profile as is.
For methods to compute equilibrium density profiles, see the Density Profile
section. For methods that implicitly compute the density profile for given
boundary conditions before computing a property see rho-T properties.

### Table of contents
  * [Profile property interfaces](#profile-property-interfaces)
    * [N_adsorbed](#n_adsorbedself-rho-tnone-dividing_surfacee)
    * [adsorbtion](#adsorbtionself-rho-tnone-dividing_surfacee)
    * [correlation](#correlationself-rho-t)
    * [grand_potential](#grand_potentialself-rho-t-vextnone-bulkfalse-property_flagir)
    * [grand_potential_density](#grand_potential_densityself-rho-t-vextnone-property_flagir)
    * [reduced_helmholtz_energy_density](#reduced_helmholtz_energy_densityself-*args-**kwargs)
    * [residual_enthalpy_density](#residual_enthalpy_densityself-rho-t)
    * [residual_entropy_density](#residual_entropy_densityself-rho-t)
    * [residual_helmholtz_energy_density](#residual_helmholtz_energy_densityself-rho-t-bulkfalse)
    * [residual_internal_energy_density](#residual_internal_energy_densityself-rho-t)
    * [surface_tension](#surface_tensionself-rho-t-vextnone-dividing_surfaceequimolar)
    * [tolmann_length](#tolmann_lengthself-rho-t)


### `N_adsorbed(self, rho, T=None, dividing_surface='e')`
Compute the adsorbtion of each on the interface in a given density profile

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[Profile]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The density profile for each component [1 / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K], only required if `dividing_surface` is 't' or 'tension'

&nbsp;&nbsp;&nbsp;&nbsp; **dividing_surface (str) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The dividing surface to use:

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 'e' or 'equimolar' for equimolar surface

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 't' or 'tension' for surface of tension

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **1D array :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The adsobtion of each component [1 / Å^2]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `adsorbtion(self, rho, T=None, dividing_surface='e')`
Compute the adsorbtion of each on the interface in a given density profile.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[Profile]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The density profile for each component [1 / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K], only required if `dividing_surface` is 't' or 'tension'

&nbsp;&nbsp;&nbsp;&nbsp; **dividing_surface (str) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The dividing surface to use:

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 'e' or 'equimolar' for equimolar surface

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 't' or 'tension' for surface of tension

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **1D array :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The adsobtion of each component [1 / Å^2]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `correlation(self, rho, T)`
Compute the one-body correlation function (sec. 4.2.3 in "introduction to DFT")

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[Profile]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Density profiles [particles / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **list[Profile] :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  One body correlation function of each species as a function of position,

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; indexed as c[<component idx>][<grid idx>]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `grand_potential(self, rho, T, Vext=None, bulk=False, property_flag='IR')`
Compute the Grand Potential, as defined in sec. 2.7 of R. Roth - Introduction to Density Functional Theory of Classical Systems: Theory and Applications.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[Profile] or Iterable[float]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The density profile for each component,

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; or the bulk density of each component [particles / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **Vext (list[callable] or callable) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  External potential for each component, it is recomended to use the

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; callables inherriting ExtPotential in external_potential.py. Defaults to None.

&nbsp;&nbsp;&nbsp;&nbsp; **bulk (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Is this a bulk computation? Defaults to False. Note: For bulk computations, the grand

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; potential density [J / Å^3] is returned.

&nbsp;&nbsp;&nbsp;&nbsp; **property_flag (str) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Return Residual ('R'), Ideal ('I') or total ('IR') grand potential

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **float :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The Grand Potential [J]. NOTE: For bulk computations, the grand potential

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; density (which is equal to the bulk pressure) [J / Å^3] is returned.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `grand_potential_density(self, rho, T, Vext=None, property_flag='IR')`
Compute the Grand Potential density.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[Profile] or Iterable[float]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The density profile for each component,

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; or the bulk density of each component [particles / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **Vext (list[callable] or callable) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  External potential for each component, it is recomended to use the

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; callables inherriting ExtPotential in external_potential.py. Defaults to None.

&nbsp;&nbsp;&nbsp;&nbsp; **property_flag (str) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Return Residual ('R'), Ideal ('I') or total ('IR') grand potential density

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; Returns

&nbsp;&nbsp;&nbsp;&nbsp; **Profile :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The Grand Potential density [J / Å^3].

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `reduced_helmholtz_energy_density(self, *args, **kwargs)`
Returns the reduced, residual helmholtz energy density, $\phi$ (i.e. the integrand of eq. 3.4 in "introduction to DFT")

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; $$\phi = \\frac{a^{res}}{k_B T} $$

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; where $a^{res}$ is the residual helmholtz energy density (per particle), in [J Å$^{-3}$]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `residual_enthalpy_density(self, rho, T)`
Compute the residual enthalpy density [J / Å^3]

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[Profile]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The density profile for each component [particles / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **Profile :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The residual enthalpy density [J / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `residual_entropy_density(self, rho, T)`
Compute the residual entropy density [J / Å^3 K]

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[Profile]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The density profile for each component [particles / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **Profile :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The residual entropy density [J / Å^3 K]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `residual_helmholtz_energy_density(self, rho, T, bulk=False)`
Compute the residual Helmholtz energy density [J / Å^3]

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[Profile] or 1d array [float]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The density profile or bulk density for each component [particles / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **bulk (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Whether to compute for bulk phase

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **Profile or float:** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The residual Helmholtz energy density [J / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `residual_internal_energy_density(self, rho, T)`
Compute the residual internal energy density [J / Å^3]

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[Profile]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The density profile for each component [particles / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **Profile :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The residual internal energy density [J / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `surface_tension(self, rho, T, Vext=None, dividing_surface='equimolar')`
Compute the surface tension for a given density profile Args: rho (list[Profile]) : Density profile for each component [particles / Å^3] T (float) : Temperature [K] Vext (list[callable], optional) : External potential dividing_surface (str, optional) : Which deviding surface to use 'e' or 'equimolar' for Equimolar surface 't' or 'tension' for Surface of tension Returns: float : Surface tension [J / Å^2] 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `tolmann_length(self, rho, T)`
Compute the Tolmann length, given a density profile.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[Profile]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The density profile of each species.

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **float :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The Tolmann lenth

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## $\rho - T$ property interfaces

Compute properties at a given density and temperature, ususally by first computing a density profile for the given state.

### Table of contents
  * [$\rho - T$ property interfaces](#$\rho---t$-property-interfaces)
    * [adsorbtion_isotherm](#adsorbtion_isothermself-t-n_points30-dividing_surfacet-x_min0001-x_max0999-solvernone-rho0none-calc_lvefalse-verbosefalse)
    * [radial_distribution_functions](#radial_distribution_functionsself-rho_b-t-comp_idx0-gridnone)
    * [surface_tension_isotherm](#surface_tension_isothermself-t-n_points30-dividing_surfacet-solvernone-rho0none-calc_lvefalse-verbosefalse-cache_dir)


### `adsorbtion_isotherm(self, T, n_points=30, dividing_surface='t', x_min=0.001, x_max=0.999, solver=None, rho0=None, calc_lve=False, verbose=False)`
Compute the adsorbtion as a function of molar composition along an isotherm

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **n_points (int) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Number of (evenly distriubted) points to compute. If an array is supplied, those points are used instead.

&nbsp;&nbsp;&nbsp;&nbsp; **dividing_surface (str, optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  't' or 'tension' for surface of tension, 'e' or 'equimolar' for equimolar surface

&nbsp;&nbsp;&nbsp;&nbsp; **x_min (float, optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Minimum liquid mole fraction of the first component.

&nbsp;&nbsp;&nbsp;&nbsp; **x_max (float, optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Maximum liquid mole fraction of the first component.

&nbsp;&nbsp;&nbsp;&nbsp; **solver (SequentialSolver, optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Custom solver object to use

&nbsp;&nbsp;&nbsp;&nbsp; **rho0 (list[Profile], optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Initial guess for denisty profile at x = [0, 1]

&nbsp;&nbsp;&nbsp;&nbsp; **calc_lve (bool, optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  If true, return a tuple (x, y, p) with pressure (p), liquid (x) and vapour (y) composition. If false, return only liquid composition.

&nbsp;&nbsp;&nbsp;&nbsp; **verbose (int, optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Print progress information, higher number gives more output, default 0

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; Returns

&nbsp;&nbsp;&nbsp;&nbsp; **tuple(gamma, x) or tuple(gamma, lve) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Adsorbtion and composition (of first component)

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `radial_distribution_functions(self, rho_b, T, comp_idx=0, grid=None)`
Compute the radial distribution functions $g_{i,j}$ for $i =$ `comp_idx` using the "Percus trick". To help convergence: First converge the profile for a planar geometry, exposed to an ExtendedSoft potential with a core radius $5R$, where $R$ is the maximum `characteristic_length` of the mixture. Then, shift that profile to the left, and use it as an initial guess for the spherical case. If that doesn't work, the profile can be shifted in several steps (by gradually reducing the core radius of the ExtendedSoft potential). The latter possibility is not implemented, but is just a matter of putting the "shift and recompute" part of this method in a for-loop, and adding some appropriate kwargs.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho_b (list[float]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The bulk densities [particles / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **comp_idx (int) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The first component in the pair, defaults to the first component

&nbsp;&nbsp;&nbsp;&nbsp; **grid (Grid) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The spatial discretisation (should have Spherical geometry for results to make sense)

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **list[Profile] :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The radial distribution functions around a particle of type `comp_idx`

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `surface_tension_isotherm(self, T, n_points=30, dividing_surface='t', solver=None, rho0=None, calc_lve=False, verbose=False, cache_dir='')`
Compute the surface tension as a function of molar composition along an isotherm

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **n_points (int) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Number of (evenly distriubted) points to compute. If an array is supplied, those points are used instead.

&nbsp;&nbsp;&nbsp;&nbsp; **dividing_surface (str, optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  't' or 'tension' for surface of tension, 'e' or 'equimolar' for equimolar surface

&nbsp;&nbsp;&nbsp;&nbsp; **solver (SequentialSolver, optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Custom solver object to use

&nbsp;&nbsp;&nbsp;&nbsp; **rho0 (list[Profile], optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Initial guess for denisty profile at x = [0, 1]

&nbsp;&nbsp;&nbsp;&nbsp; **calc_lve (bool, optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  If true, return a tuple (x, y, p) with pressure (p), liquid (x) and vapour (y) composition. If false, return only liquid composition.

&nbsp;&nbsp;&nbsp;&nbsp; **verbose (int, optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Print progress information, higher number gives more output, default 0

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; Returns

&nbsp;&nbsp;&nbsp;&nbsp; **tuple(gamma, x) or tuple(gamma, lve) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Surface tension and composition (of first component)

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## Density profile interfaces

Methods for converging a density profile given various boundary conditions.

### Table of contents
  * [Density profile interfaces](#density-profile-interfaces)
    * [density_profile_singlecomp](#density_profile_singlecompself-t-grid-rho_0none-solvernone-verbosefalse)
    * [density_profile_tp](#density_profile_tpself-t-p-z-grid-rho_0none-solvernone-verbosefalse)
    * [density_profile_twophase](#density_profile_twophaseself-rho_g-rho_l-t-grid-beta_v05-rho_0none-solvernone-verbose0)
    * [density_profile_tz](#density_profile_tzself-t-z-grid-z_phase1-rho_0none-solvernone-verbose0)
    * [density_profile_wall](#density_profile_wallself-rho_b-t-grid-vextnone-rho_0none-verbosefalse)
    * [density_profile_wall_tp](#density_profile_wall_tpself-t-p-z-grid-vext-rho0none-verbose0)
    * [drubble_profile_rT](#drubble_profile_rtself-rho_i-rho_o-t-r-grid-rho_0none)


### `density_profile_singlecomp(self, T, grid, rho_0=None, solver=None, verbose=False)`
Compute the equilibrium density profile across a gas-liquid interface. For multicomponent systems, twophase_tpflash is used to compute the composition of the two phases. For single component systems, p is ignored, and pressure is computed from dew_pressure.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **grid (Grid) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Spatial discretisation

&nbsp;&nbsp;&nbsp;&nbsp; **rho_0 (list[Profile], optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Initial guess for density profile

&nbsp;&nbsp;&nbsp;&nbsp; **solver (SequentialSolver or GridRefiner) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Solver to use, uses a default SequentialSolver if none is supplied

&nbsp;&nbsp;&nbsp;&nbsp; **verbose (bool, optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Whether to print progress info

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **list[Profile] :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The density profile for each component across the interface.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `density_profile_tp(self, T, p, z, grid, rho_0=None, solver=None, verbose=False)`
Compute the equilibrium density profile across a gas-liquid interface. For multicomponent systems, twophase_tpflash is used to compute the composition of the two phases. For single component systems, p is ignored, and pressure is computed from dew_pressure.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **p (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Pressure [Pa]

&nbsp;&nbsp;&nbsp;&nbsp; **z (Iterable) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Total Molar composition

&nbsp;&nbsp;&nbsp;&nbsp; **grid (Grid) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Spatial discretisation

&nbsp;&nbsp;&nbsp;&nbsp; **rho_0 (list[Profile], optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Initial guess for density profile

&nbsp;&nbsp;&nbsp;&nbsp; **solver (SequentialSolver or GridRefiner) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Solver to use, uses a default SequentialSolver if none is supplied

&nbsp;&nbsp;&nbsp;&nbsp; **verbose (bool, optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Whether to print progress info

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **list[Profile] :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The density profile for each component across the interface.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `density_profile_twophase(self, rho_g, rho_l, T, grid, beta_V=0.5, rho_0=None, solver=None, verbose=0)`
Compute the density profile separating two phases with denisties rho_g and rho_l

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho_g (list[float]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Density of each component in phase 1 [particles / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; **rho_l (list[float]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Density of each component in phase 2 [particles / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; **beta_V (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Liquid fraction

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **grid (Grid or GridSpec) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The spacial discretisation. Using a GridSpec is preferred, as the method then generates a

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; grid that is likely to be a suitable width.

&nbsp;&nbsp;&nbsp;&nbsp; **rho_0 (list[Profile], optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Initial guess, optional

&nbsp;&nbsp;&nbsp;&nbsp; **solver (SequentialSolver or GridRefiner) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Solver to use, uses a default SequentialSolver if none is supplied

&nbsp;&nbsp;&nbsp;&nbsp; **verbose (bool, options) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Print progress info while running solver

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **list[Profile] :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The density profile for of each component

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `density_profile_tz(self, T, z, grid, z_phase=1, rho_0=None, solver=None, verbose=0)`
Compute the density profile separating two phases at temperature T, with liquid (or optionally vapour) composition x

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **z (ndarray[float]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Molar composition of liquid phase (unless x_phase=2)

&nbsp;&nbsp;&nbsp;&nbsp; **grid (Grid or GridSpec) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The spatial discretization

&nbsp;&nbsp;&nbsp;&nbsp; **z_phase (int) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  ThermoPack Phase flag, indicating which phase has composition `x`. `x_phase=1` for liquid (default)

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; `x_phase=2` for vapour.

&nbsp;&nbsp;&nbsp;&nbsp; **rho_0 (list[Profile], optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Initial guess

&nbsp;&nbsp;&nbsp;&nbsp; **solver (SequentialSolver or GridRefiner, optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Solver to use

&nbsp;&nbsp;&nbsp;&nbsp; **verbose (int) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Print debugging information (higher number gives more output), default 0

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **list[Profile] :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The converged density profiles.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `density_profile_wall(self, rho_b, T, grid, Vext=None, rho_0=None, verbose=False)`
Calculate equilibrium density profile for a given external potential Note: Uses lazy evaluation for (rho_b, T, x, grid, Vext) to return a copy of previous result if the same calculation is done several times.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho_b (list[float]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The bulk densities [particles / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **grid (Grid) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Spatial discretization

&nbsp;&nbsp;&nbsp;&nbsp; **Vext (ExtPotential, optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  External potential as a function of position (default : Vext(r) = 0)

&nbsp;&nbsp;&nbsp;&nbsp; **Note:** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Must be hashable, to use with lazy evaluation

&nbsp;&nbsp;&nbsp;&nbsp; **Recomended:** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Use the callable classes inherriting ExtPotential

&nbsp;&nbsp;&nbsp;&nbsp; **rho_0 (list[Profile], optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Initial guess for density profiles.

&nbsp;&nbsp;&nbsp;&nbsp; **verbose (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Print progression information during run

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **list[Profile] :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The equilibrium density profiles

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `density_profile_wall_tp(self, T, p, z, grid, Vext, rho0=None, verbose=0)`
Calculate equilibrium density profile for a given external potential

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **p (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Pressure [Pa]

&nbsp;&nbsp;&nbsp;&nbsp; **z (list[float]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Bulk composition

&nbsp;&nbsp;&nbsp;&nbsp; **grid (Grid) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Spatial discretization

&nbsp;&nbsp;&nbsp;&nbsp; **Vext (ExtPotential, optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  External potential as a function of position (default : Vext(r) = 0)

&nbsp;&nbsp;&nbsp;&nbsp; **Note:** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Must be hashable, to use with lazy evaluation

&nbsp;&nbsp;&nbsp;&nbsp; **Recomended:** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Use the callable classes inherriting ExtPotential

&nbsp;&nbsp;&nbsp;&nbsp; **rho0 (list[Profile], optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Initial guess for density profiles.

&nbsp;&nbsp;&nbsp;&nbsp; **verbose (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Print progression information during run

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **list[Profile] :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The equilibrium density profiles

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `drubble_profile_rT(self, rho_i, rho_o, T, r, grid, rho_0=None)`
Compute the density profile across the interface of a droplet or bubble (drubble).

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho_i (list[float]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The particle densities inside the drubble.

&nbsp;&nbsp;&nbsp;&nbsp; **rho_o (list[float]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The particle densities outside the drubble.

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **r (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Drubble radius [Å]

&nbsp;&nbsp;&nbsp;&nbsp; **grid (Grid) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The grid to use (must have Geometry.SPHERICAL)

&nbsp;&nbsp;&nbsp;&nbsp; **rho_0 (list[Profile], optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Initial guess

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **list[Profile] :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The density profiles across the interface.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## Bulk property interfaces

Evaluating bulk properties.

### Table of contents
  * [Bulk property interfaces](#bulk-property-interfaces)
    * [chemical_potential](#chemical_potentialself-rho-t-bulktrue-property_flagir)
    * [fugacity](#fugacityself-rho-t-vextnone)
    * [residual_chemical_potential](#residual_chemical_potentialself-rho-t-bulktrue)


### `chemical_potential(self, rho, T, bulk=True, property_flag='IR')`
Compute the chemical potential [J]

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[float]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Density [particles / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **bulk (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Only True is implemented

&nbsp;&nbsp;&nbsp;&nbsp; **property_flag (str, optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  'I' for ideal, 'R' for residual, 'IR' for total.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **1d array (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The chemical potentials [J / particle]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `fugacity(self, rho, T, Vext=None)`
Compute the fugacity at given density and temperature

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[float]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Particle density of each species

&nbsp;&nbsp;&nbsp;&nbsp; **T (flaot) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **Vext (ExternalPotential, optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  External potential for each particle

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **1d array (flaot) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The fugacity of each species.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `residual_chemical_potential(self, rho, T, bulk=True)`
Compute the residual chemical potential [J]

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[float]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Density [particles / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **bulk (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Only True is implemented

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **1d array (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The chemical potentials [J / particle]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## Pure fluid properties

Methods to efficiently and conveninetly compute properties for pure fluids. Contain
some optimisations, tuning and convenience factors that are only possible for
pure fluids.

### Table of contents
  * [Pure fluid properties](#pure-fluid-properties)
    * [surface_tension_singlecomp](#surface_tension_singlecompself-n_points30-t_min05-t_max099-gridnone-solvernone-rho0none-verbose0)


### `surface_tension_singlecomp(self, n_points=30, t_min=0.5, t_max=0.99, grid=None, solver=None, rho0=None, verbose=0)`
Compute the surface tension of a pure component for a series of temperatures.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **n_points (int) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Number of points to compute

&nbsp;&nbsp;&nbsp;&nbsp; **t_min (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Start temperature, if 0 < t_min < 1, start temperature will be t_min * Tc, where Tc is the critical temperature.

&nbsp;&nbsp;&nbsp;&nbsp; **t_max (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Stop temperature, if 0 < t_max < 1, stop temperature will be t_max * Tc, where Tc is the critical temperature.

&nbsp;&nbsp;&nbsp;&nbsp; **grid (Grid) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Grid to use for initial calculation.

&nbsp;&nbsp;&nbsp;&nbsp; **solver (Solver) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Solver to use for all calculations

&nbsp;&nbsp;&nbsp;&nbsp; **rho0 (list[Profile]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Initial guess for first density profile.

&nbsp;&nbsp;&nbsp;&nbsp; **verbose (int) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Larger number gives more output during progress.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **tuple(gamma, T) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Where gamma and T are matching 1d arrays of the surface tension and temperature.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## Weight function interfaces

Get-methods for weight functions.

### Table of contents
  * [Weight function interfaces](#weight-function-interfaces)
    * [get_weights](#get_weightsself-t)


### `get_weights(self, T)`
Returns the weights for weighted densities in a 2D array, ordered as weight[<weight idx>][<component idx>]. See arrayshapes.md for a description of the ordering of different arrays. 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## Weighted density computations

Compute weighted densities

### Table of contents
  * [Weighted density computations](#weighted-density-computations)
    * [get_weighted_densities](#get_weighted_densitiesself-*args-**kwargs)


### `get_weighted_densities(self, *args, **kwargs)`
Compute the weighted densities, and optionally differentials

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (ndarray[Profile]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  2D array of component densities indexed as rho[<component index>][<position index>]

&nbsp;&nbsp;&nbsp;&nbsp; **bulk (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  If True, use simplified expressions for bulk - not requiring FFT

&nbsp;&nbsp;&nbsp;&nbsp; **dndrho (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Flag to activate calculation of differential

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **ndarray :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  array of weighted densities indexed as n[<weight index>][<position index>]

&nbsp;&nbsp;&nbsp;&nbsp; **ndarray :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  array of differentials

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## Utility methods

Methods for setting ... and getting ...

### Table of contents
  * [Utility methods](#utility-methods)
    * [clear_cache_dir](#clear_cache_dirself-clear_dir)
    * [dividing_surface_position](#dividing_surface_positionself-rho-tnone-dividing_surfacee)
    * [equimolar_surface_position](#equimolar_surface_positionrho)
    * [get_caching_id](#get_caching_idself)
    * [get_characteristic_lengths](#get_characteristic_lengthsself)
    * [get_load_dir](#get_load_dirself)
    * [get_save_dir](#get_save_dirself)
    * [reduce_temperature](#reduce_temperatureself-t-c0)
    * [set_cache_dir](#set_cache_dirself-cache_dir)
    * [set_load_dir](#set_load_dirself-load_dir)
    * [set_save_dir](#set_save_dirself-save_dir)
    * [surface_of_tension_position](#surface_of_tension_positionself-rho-t)


### `clear_cache_dir(self, clear_dir)`
Clear the directory `clear_dir`, after prompting for confirmation.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **clear_dir (str) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The name of the directory.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Raises:

&nbsp;&nbsp;&nbsp;&nbsp; **NotADirectoryError :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  If clear_dir does not exist or is not a directory.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `dividing_surface_position(self, rho, T=None, dividing_surface='e')`
Compute the position of a dividing surface on a given density profile.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[Profile]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The density profile for each species

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The temperature

&nbsp;&nbsp;&nbsp;&nbsp; **dividing_surface (str, optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Can be '(e)quimolar' (default) or '(t)ension'.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **float :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Position of the dividing surface.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `equimolar_surface_position(rho)`
Calculate the position of the equimolar surface for a given density profile

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[Profile]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The density profile for each component [1 / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **float :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The position of the equimolar surface [Å]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `get_caching_id(self)`
Returns a unique ID for an initialized model. Should include information about the model type, components, parameters and mixing parameters (if applicable). Used for caching profiles.

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **str :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  A model identifier

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `get_characteristic_lengths(self)`
Used to generate initial guesses for density profiles. Should return lengths that give an indication of molecular sizes. For example diameters of hard-spheres, or the Barker-Henderson diameter.

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **ndarray(float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The characteristic length of the molecules.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `get_load_dir(self)`
Get the current directory used to search and load Profiles

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **str :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Path to the current load directory

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `get_save_dir(self)`
Get the current directory used to save Profiles

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **str :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Path to the current save directory

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `reduce_temperature(self, T, c=0)`
Reduce the temperature in some meaningful manner, using LJ units when possible, doing nothing for hard spheres.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The temperature

&nbsp;&nbsp;&nbsp;&nbsp; **c (float, optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  ???

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **float :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The reduced temperature

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `set_cache_dir(self, cache_dir)`
Forwards call to `self.set_load_dir` and `self.set_save_dir`.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **cache_dir (str) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Name of directory save and load files in.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Raises:

&nbsp;&nbsp;&nbsp;&nbsp; **FileExistsError :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  If cache_dir is the name of an existing file that is not a directory.

&nbsp;&nbsp;&nbsp;&nbsp; **NotADirectoryError :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  If cache_dir does not exist after successfull call to `self.set_save_dir`

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `set_load_dir(self, load_dir)`
Sets this model to automatically search for computed profiles in `load_dir`.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **load_dir (str) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Name of directory to load files from.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Raises:

&nbsp;&nbsp;&nbsp;&nbsp; **NotADirectoryError :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  If load_dir does not exist.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `set_save_dir(self, save_dir)`
Sets this model to automatically save computed density profiles in the directory 'save_dir'.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **save_dir (str) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Name of directory to save to

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Raises:

&nbsp;&nbsp;&nbsp;&nbsp; **FileExistsError :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  If save_dir is the name of an existing file that is not a directory.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `surface_of_tension_position(self, rho, T)`
Calculate the position of the surface of tension on a given density profile

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[Profile]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The density profile for each component [1 / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **float :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The position of the surface of tension [Å]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## Internal methods

Methods for handling communication with the Fortran library.

### Table of contents
  * [Internal methods](#internal-methods)
    * [\_\_init\_\_](#__init__self-ncomps)
    * [\_\_repr\_\_](#__repr__self)
    * [sanitize_Vext](#sanitize_vextself-vext)
    * [split_component_weighted_densities](#split_component_weighted_densitiesself-n)
    * [validate_composition](#validate_compositionself-z)


### `__init__(self, ncomps)`
Handles initialisation that is common for all functionals

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **ncomps (int) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Number of components

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `__repr__(self)`
All Functionals must implement a unique `__repr__`, as these are used to generate the hashes that are used when saving profiles. The `__repr__` should contain a human-readable text with (at least) the name of the model and all model parameters. 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `sanitize_Vext(self, Vext)`
Ensure that Vext is a tuple with the proper ammount of elements. 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `split_component_weighted_densities(self, n)`
Unsure if this is still in use, but I believe it takes in fmt-weighted densities as a 1d array and returns the same densities as a 2d array.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **n (list[Profile]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  FMT-weighted densities

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **list[list[Profile]] :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  FMT-weighted densities, indexed as n[<component_idx>][<weight_idx>]

&nbsp;&nbsp;&nbsp;&nbsp; **:** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; return:

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `validate_composition(self, z)`
Check that the composition `z` has length equal to number of components, and sums to one.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **z (Iterable(float)) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The composition

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Raises:

&nbsp;&nbsp;&nbsp;&nbsp; **IndexError :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  If number of fractions does not match number of components.

&nbsp;&nbsp;&nbsp;&nbsp; **ValueError :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  If fractions do not sum to unity.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## Deprecated methods

Deprecated methods are not maintained, and may be removed in the future.

### Table of contents
  * [Deprecated methods](#deprecated-methods)
    * [_Functional\_\_density_profile_twophase](#_functional__density_profile_twophaseself-rho_left-rho_right-t-grid-rho_0none-constraintsnone)
    * [adsorbed_mean_density](#adsorbed_mean_densityself-profile)
    * [adsorbed_thickness](#adsorbed_thicknessself-profile)
    * [density_profile_NT](#density_profile_ntself-rho1-rho2-t-grid-rho-rho_is_frac-rho_0none)
    * [density_profile_muT](#density_profile_mutself-rho1-rho2-t-grid-rho-rho_is_frac-rho_0none)
    * [find_non_bulk_idx](#find_non_bulk_idxself-profile-rho_b-tol001-start_idx-1)
    * [interface_thickness](#interface_thicknessself-profile-positionsfalse)


### `_Functional__density_profile_twophase(self, rho_left, rho_right, T, grid, rho_0=None, constraints=None)`


&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `adsorbed_mean_density(self, profile)`
Compute the mean density of an adsorbed film.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **profile (Profile) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The density profile

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **float :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The mean density.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `adsorbed_thickness(self, profile)`
Find the thickness of an adsorbed film.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **profile (Profile) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The density profile

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **float :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The thickness of the adsorbed layer.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `density_profile_NT(self, rho1, rho2, T, grid, rho, rho_is_frac, rho_0=None)`


&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `density_profile_muT(self, rho1, rho2, T, grid, rho, rho_is_frac, rho_0=None)`


&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `find_non_bulk_idx(self, profile, rho_b, tol=0.01, start_idx=-1)`
Possibly no longer working, used to find the the first index in a Profile which is no longer considered a point in the bulk phase, by checking the change in density and comparing to `tol`.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **profile (Profile) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The density profile

&nbsp;&nbsp;&nbsp;&nbsp; **rho_b (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The bulk density

&nbsp;&nbsp;&nbsp;&nbsp; **tol (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Tolerance for relative change in density within bulk.

&nbsp;&nbsp;&nbsp;&nbsp; **start_idx (int) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Where to start searching. If start_idx < 0, search goes "right-to-left", otherwise "left-to-right".

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **int :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The index at which the bulk ends.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `interface_thickness(self, profile, positions=False)`
Find the thickness of an interface.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **profile (Profile) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The density profile

&nbsp;&nbsp;&nbsp;&nbsp; **positions (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Also return the position where the interface starts and stops.

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **float :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The thickness of the interface.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

