---
layout: default
version: 1.0.0
title: Methods in the saft class
permalink: /v1.0.0/saft_methods.html
---

<!--- 
Generated at: 2024-02-13T10:57:21.953235
This is an auto-generated file, generated using the script at surfpack/docs/tools/markdown_from_docstrings.py
The file is created by parsing the docstrings of the methods in the 
saft class. For instructions on how to use the parser routines, see the
file surfpack/docs/tools/markdown_from_docstrings.py--->

The `SAFT` class, found in `surfpack/saft.py`, is an abstract class, that is inherited
by the `SAFT_VR_Mie` and `PC_SAFT` classes. It contains some generic utility methods to
compute quantities of interest when investigating SAFT-type functionals.

## Table of contents
  * [Utility methods](#utility-methods)
    * [get_caching_id](#get_caching_idself)
    * [get_eps_div_k](#get_eps_div_kself-ic)
    * [get_sigma](#get_sigmaself-ic)
    * [pair_potential](#pair_potentialself-i-j-r)
    * [reduce_temperature](#reduce_temperatureself-t-c0)
    * [set_association_active](#set_association_activeself-active)
    * [set_chain_active](#set_chain_activeself-active)
    * [set_dispersion_active](#set_dispersion_activeself-active)
    * [set_eps_div_k](#set_eps_div_kself-ic-eps_div_k)
    * [set_multipole_active](#set_multipole_activeself-active)
    * [set_pure_assoc_param](#set_pure_assoc_paramself-ic-eps-beta)
    * [set_pure_fluid_param](#set_pure_fluid_paramself-ic-m-sigma-eps_div_k-*assoc_param)
    * [set_segment_number](#set_segment_numberself-ic-m)
    * [set_sigma](#set_sigmaself-ic-sigma)
  * [Internal methods](#internal-methods)
    * [\_\_init\_\_](#__init__self-comps-eos-hs_model<class-surfpacksaft_hardspheresaft_whitebear>-parameter_refdefault)
    * [\_\_repr\_\_](#__repr__self)
    * [refresh_hs_model](#refresh_hs_modelself)
  * [Profile property interfaces](#profile-property-interfaces)
    * [reduced_helmholtz_energy_density](#reduced_helmholtz_energy_densityself-rho-t-dphidnfalse-bulkfalse-asarrayfalse-dphidtfalse)
  * [Helmholtz energy contributions](#helmholtz-energy-contributions)
    * [dispersion_helmholtz_energy_density](#dispersion_helmholtz_energy_densityself-rho-t-bulkfalse-dphidnfalse-dphidtfalse-dphidrhofalse-n_dispnone)
  * [Weighted densities](#weighted-densities)
    * [get_dispersion_weighted_density](#get_dispersion_weighted_densityself-rho_arr-t-bulkfalse-dndrhofalse-dndtfalse)
    * [get_fmt_weighted_densities](#get_fmt_weighted_densitiesself-rho-t-bulkfalse-dndrhofalse-dndtfalse)
    * [get_weighted_densities](#get_weighted_densitiesself-rho-t-bulkfalse-dndrhofalse-dndtfalse)
  * [Weight functions](#weight-functions)
    * [get_dispersion_weights](#get_dispersion_weightsself-t-dwdtfalse-dnone-dd_dtnone)
    * [get_fmt_weights](#get_fmt_weightsself-t-dwdtfalse)
    * [get_weights](#get_weightsself-t-dwdtfalse)
  * [Deprecated methods](#deprecated-methods)
    * [pressure_tv](#pressure_tvself-rho-t)

## Utility methods

Methods for computing specific parameters and contributions to the residual
Helmholtz energy for SAFT-type equations of state

### Table of contents
  * [Utility methods](#utility-methods)
    * [get_caching_id](#get_caching_idself)
    * [get_eps_div_k](#get_eps_div_kself-ic)
    * [get_sigma](#get_sigmaself-ic)
    * [pair_potential](#pair_potentialself-i-j-r)
    * [reduce_temperature](#reduce_temperatureself-t-c0)
    * [set_association_active](#set_association_activeself-active)
    * [set_chain_active](#set_chain_activeself-active)
    * [set_dispersion_active](#set_dispersion_activeself-active)
    * [set_eps_div_k](#set_eps_div_kself-ic-eps_div_k)
    * [set_multipole_active](#set_multipole_activeself-active)
    * [set_pure_assoc_param](#set_pure_assoc_paramself-ic-eps-beta)
    * [set_pure_fluid_param](#set_pure_fluid_paramself-ic-m-sigma-eps_div_k-*assoc_param)
    * [set_segment_number](#set_segment_numberself-ic-m)
    * [set_sigma](#set_sigmaself-ic-sigma)


### `get_caching_id(self)`
See Functional for docs. 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `get_eps_div_k(self, ic)`
Get the epsilon parameter

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **ic (int) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Component index (zero-indexed)

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **float :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Model epsilon-parameter, divided by Boltzmanns constant [K]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `get_sigma(self, ic)`
Get the sigma parameter

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **ic (int) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Component index (zero-indexed)

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **float :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Model sigma-parameter [m]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `pair_potential(self, i, j, r)`
Evaluate the pair potential between component `i` and `j` at distance `r`

&nbsp;&nbsp;&nbsp;&nbsp; **i (int) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Component index

&nbsp;&nbsp;&nbsp;&nbsp; **j (int) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Component index

&nbsp;&nbsp;&nbsp;&nbsp; **r (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Distance [m]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **float :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Interaction potential energy [J]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `reduce_temperature(self, T, c=0)`
Compute the reduced temperature (LJ units)

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **c (int) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Component idx to use

&nbsp;&nbsp;&nbsp;&nbsp; **Return:** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

&nbsp;&nbsp;&nbsp;&nbsp; **float :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature in LJ units

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `set_association_active(self, active)`
Toggle association contribution on/off

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **active (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Whether association is active

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `set_chain_active(self, active)`
Toggle chain contribution on/off

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **active (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Whether chain is active

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `set_dispersion_active(self, active)`
Toggle dispersion contribution on/off

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **active (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Whether dispersion is active

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `set_eps_div_k(self, ic, eps_div_k)`
Set the model epsilon-parameter

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **ic (int) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Component index (zero-indexed)

&nbsp;&nbsp;&nbsp;&nbsp; **eps_div_k (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  epsilon-parameter divided by Boltzmanns constant [K]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `set_multipole_active(self, active)`
Toggle multipole contribution on/off

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **active (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Whether multipole is active

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `set_pure_assoc_param(self, ic, eps, beta)`
Set pure-conponent association parameters

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **ic (int) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Component index (zero indexed)

&nbsp;&nbsp;&nbsp;&nbsp; **eps (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Association energy [J / mol]

&nbsp;&nbsp;&nbsp;&nbsp; **beta (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Associaiton volume [-]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `set_pure_fluid_param(self, ic, m, sigma, eps_div_k, *assoc_param)`
Set all pure component parameters

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **ic (int) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Component index (zero indexed)

&nbsp;&nbsp;&nbsp;&nbsp; **m (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Segment number [-]

&nbsp;&nbsp;&nbsp;&nbsp; **sigma (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  sigma-parameter [m]

&nbsp;&nbsp;&nbsp;&nbsp; **eps_div_k (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  epsilon-parameter [K]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `set_segment_number(self, ic, m)`
Set the segment number

&nbsp;&nbsp;&nbsp;&nbsp; **ic (int) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Component index (zero indexed)

&nbsp;&nbsp;&nbsp;&nbsp; **m (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Segment number

&nbsp;&nbsp;&nbsp;&nbsp; **:** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; return:

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `set_sigma(self, ic, sigma)`
Set the model sigma-parameter

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **ic (int) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Component index (zero-indexed)

&nbsp;&nbsp;&nbsp;&nbsp; **sigma (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  sigma-parameter [m]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## Internal methods

Internal use, of little interest to outside users.

### Table of contents
  * [Internal methods](#internal-methods)
    * [\_\_init\_\_](#__init__self-comps-eos-hs_model<class-surfpacksaft_hardspheresaft_whitebear>-parameter_refdefault)
    * [\_\_repr\_\_](#__repr__self)
    * [refresh_hs_model](#refresh_hs_modelself)


### `__init__(self, comps, eos, hs_model=<class 'surfpack.saft_hardsphere.SAFT_WhiteBear'>, parameter_ref='default')`
This class is inherited by SAFT_VR_Mie and PC_SAFT. The only thing they do is pass the correct eos to the eos argument of this initialiser.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **comps (str) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Thermopack component string

&nbsp;&nbsp;&nbsp;&nbsp; **eos (thermo) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Thermopack EoS class (not initialised object)

&nbsp;&nbsp;&nbsp;&nbsp; **hs_model (SAFT_HardSphere) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  A class of the same type as SAFT_WhiteBear (inheriting from SAFT_HardSphere)

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `__repr__(self)`
Called from inheriting classes.

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **str :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  id. string with parameters, active contributions and hard sphere model identifier.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `refresh_hs_model(self)`
Update hard-sphere model such that parameters are in sync. 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## Profile property interfaces

Compute properties using a given density profile, without iterating the density
profile to ensure equilibrium.

### Table of contents
  * [Profile property interfaces](#profile-property-interfaces)
    * [reduced_helmholtz_energy_density](#reduced_helmholtz_energy_densityself-rho-t-dphidnfalse-bulkfalse-asarrayfalse-dphidtfalse)


### `reduced_helmholtz_energy_density(self, rho, T, dphidn=False, bulk=False, asarray=False, dphidT=False)`
Compute the reduced helmholtz energy density [1 / Å^3] (see the Functional class for explanation, this is simply an overlaod that puts toghether all the contributions.)

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[Profile] or list[float]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Density profiles [particles / Å^3], indexed as rho[<comp idx>][<grid idx>].

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; Takes list[Profile] if `bulk is False`, and list[float] if `bulk is True`.

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **dphidn (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Whether to compute differentials wrt. weighted densities

&nbsp;&nbsp;&nbsp;&nbsp; **bulk (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  passed on to get_weighted_density, because convolutions are not neccesary for bulk.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; If True, rho should be list[float].

&nbsp;&nbsp;&nbsp;&nbsp; **asarray (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Do not set to True (I dont think that works yet)

&nbsp;&nbsp;&nbsp;&nbsp; **Intent:** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  return helmholtz density as a 1d numpy array rather than a Profile

&nbsp;&nbsp;&nbsp;&nbsp; **dphidT (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Compute temperature derivative

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **Profile :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Reduced Helmholtz energy density [1 / Å^3]

&nbsp;&nbsp;&nbsp;&nbsp; **Optionally list[Profile] :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  derivatives wrt. weighted densities.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## Helmholtz energy contributions

Methods for computing various contributions to the Helmholtz energy
that are present in all SAFT-based functionals.

### Table of contents
  * [Helmholtz energy contributions](#helmholtz-energy-contributions)
    * [dispersion_helmholtz_energy_density](#dispersion_helmholtz_energy_densityself-rho-t-bulkfalse-dphidnfalse-dphidtfalse-dphidrhofalse-n_dispnone)


### `dispersion_helmholtz_energy_density(self, rho, T, bulk=False, dphidn=False, dphidT=False, dphidrho=False, n_disp=None)`


#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[Profile] or list[float]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Particle density for each species

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **bulk (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Default False. Set to True if `rho` is `list[float]`

&nbsp;&nbsp;&nbsp;&nbsp; **dphidn (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Compute derivative

&nbsp;&nbsp;&nbsp;&nbsp; **dphidT (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Compute derivative

&nbsp;&nbsp;&nbsp;&nbsp; **dphidrho (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Compute derivative

&nbsp;&nbsp;&nbsp;&nbsp; **n_disp (list[Profile], optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Pre-computed weighted densities.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **Profile or float :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The (reduced) dispersion helmholtz energy density [-]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## Weighted densities

Methods to compute various weighted densities required by the different
Helmholtz energy contributions.

### Table of contents
  * [Weighted densities](#weighted-densities)
    * [get_dispersion_weighted_density](#get_dispersion_weighted_densityself-rho_arr-t-bulkfalse-dndrhofalse-dndtfalse)
    * [get_fmt_weighted_densities](#get_fmt_weighted_densitiesself-rho-t-bulkfalse-dndrhofalse-dndtfalse)
    * [get_weighted_densities](#get_weighted_densitiesself-rho-t-bulkfalse-dndrhofalse-dndtfalse)


### `get_dispersion_weighted_density(self, rho_arr, T, bulk=False, dndrho=False, dndT=False)`
Get the weighted density for the dispersion term

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho_arr (list[Profile]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Density profiles [particles / Å^3], indexed as rho_arr[<comp idx>][<grid idx>]

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **bulk (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  For bulk, no convolution is required (this weighted density equals the bulk density)

&nbsp;&nbsp;&nbsp;&nbsp; **dndrho (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Whether to compute the derivatives (all are equal to unity)

&nbsp;&nbsp;&nbsp;&nbsp; **dndT (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Compute the derivatives

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **list[Profile] :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Weighted densities, indexed as n[<comp idx>][<grid idx>]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; optionally 2d array of ones (dndrho)

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `get_fmt_weighted_densities(self, rho, T, bulk=False, dndrho=False, dndT=False)`
Compute weighted densities from FMT model.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[Profile] or list[float]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Particle densities of each species

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **bulk (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Default False, set to True if `rho` is `list[float]`

&nbsp;&nbsp;&nbsp;&nbsp; **dndrho (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Also compute derivatives (only for bulk)

&nbsp;&nbsp;&nbsp;&nbsp; **dndT (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Also compute derivatives

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **list[Profile] or list[float] :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The weighted densities and optionally *one* differential.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `get_weighted_densities(self, rho, T, bulk=False, dndrho=False, dndT=False)`
Compute the weighted densities.

&nbsp;&nbsp;&nbsp;&nbsp; **Remember:** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  dphidn is a list[Profile], indexed as dphidn[<weight idx>][<grid idx>]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; Where <weight idx> gives the weight that was used to compute the weighted density.

&nbsp;&nbsp;&nbsp;&nbsp; **Remember also :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Each weight gives rise to ONE weighted density, becuase you sum over the components.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; The exception is the dispersion weight, which gives rise to ncomps weighted densities,

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; one for each component.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## Weight functions

Get-methods for various weight functions.

### Table of contents
  * [Weight functions](#weight-functions)
    * [get_dispersion_weights](#get_dispersion_weightsself-t-dwdtfalse-dnone-dd_dtnone)
    * [get_fmt_weights](#get_fmt_weightsself-t-dwdtfalse)
    * [get_weights](#get_weightsself-t-dwdtfalse)


### `get_dispersion_weights(self, T, dwdT=False, d=None, dd_dT=None)`
Get the weights for the dispersion term

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **dwdT (bool, optimal) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Compute derivative wrt. T? Defaults to False.

&nbsp;&nbsp;&nbsp;&nbsp; **d (1d array) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Pre-computed Barker-Henderson diameters

&nbsp;&nbsp;&nbsp;&nbsp; **dd_dT (1d array) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Pre-computed temperature derivatives of Barker-Henderson diameters.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **2d array of WeightFunction :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The weights for the dispersion weighted densities, indexed as wts[<wt_idx>][<comp_idx>]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `get_fmt_weights(self, T, dwdT=False)`
Get the FMT weight functions

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **dwdT (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Compute derivative instead

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **list[list[WeightFunction]] :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  FMT weight functions, indexed as wts[<wt_idx>][<comp_idx>]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `get_weights(self, T, dwdT=False)`
Get all the weights used for weighted densities in a 2D array, indexed as weight[<wt idx>][<comp idx>], where weight[:6] are the FMT weights, (w0, w1, w2, w3, wv1, wv2), and weight[6] is the list of dispersion weights (one for each component).

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K], used to get hard sphere diameters

&nbsp;&nbsp;&nbsp;&nbsp; **dwdT (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Compute temperature differentials

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

&nbsp;&nbsp;&nbsp;&nbsp; **Return:** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

&nbsp;&nbsp;&nbsp;&nbsp; **2D array [Analytical] :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Weight functions, indexed as weight[<wt idx>][<comp idx>]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## Deprecated methods

Deprecated methods are not maintained, and may be removed in the future.

### Table of contents
  * [Deprecated methods](#deprecated-methods)
    * [pressure_tv](#pressure_tvself-rho-t)


### `pressure_tv(self, rho, T)`


&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

