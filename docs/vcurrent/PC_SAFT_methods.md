---
layout: default
version: 
title: Methods in the saft class
permalink: /vcurrent/saft_methods.html
---

<!--- 
Generated at: 2024-02-02T20:19:14.177809
This is an auto-generated file, generated using the script at surfpack/docs/tools/markdown_from_docstrings.py
The file is created by parsing the docstrings of the methods in the 
saft class. For instructions on how to use the parser routines, see the
file surfpack/docs/tools/markdown_from_docstrings.py--->

The `PC_SAFT` class, found in `surfpack/pcsaft.py`, inherits the `SAFT` class and implements several
contributions to the Helmholtz energy density, such as association and chain contributions.

## Table of contents
  * [Constructor](#constructor)
    * [\_\_init\_\_](#__init__self-comps-hs_model<class-surfpacksaft_hardspheresaft_whitebear>-parameter_refdefault)
  * [Utility methods](#utility-methods)
    * [get_characteristic_lengths](#get_characteristic_lengthsself)
  * [Internal methods](#internal-methods)
    * [\_\_repr\_\_](#__repr__self)
  * [Profile property interfaces](#profile-property-interfaces)
    * [reduced_helmholtz_energy_density](#reduced_helmholtz_energy_densityself-rho-t-dphidnfalse-bulkfalse-asarrayfalse-dphidtfalse-dphidrhofalse)
  * [Helmholtz energy contributions](#helmholtz-energy-contributions)
    * [association_helmholtz_energy_density](#association_helmholtz_energy_densityself-rho-t-bulkfalse-dphidnfalse-dphidtfalse-dphidrhofalse-n_assocnone)
    * [chain_helmholtz_energy_density](#chain_helmholtz_energy_densityself-rho-t-dphidrhofalse-dphidnfalse-dphidtfalse-bulkfalse-n_chainnone)
  * [Weighted densities](#weighted-densities)
    * [get_assoc_weighted_densities](#get_assoc_weighted_densitiesself-rho-t-bulkfalse-dndrhofalse-dndtfalse)
    * [get_chain_weighted_densities](#get_chain_weighted_densitiesself-rho-t-bulkfalse-dndrhofalse-dndtfalse)
    * [get_weighted_densities](#get_weighted_densitiesself-rho-t-bulkfalse-dndrhofalse-dndtfalse)
  * [Weight functions](#weight-functions)
    * [get_association_weights](#get_association_weightsself-t-dwdtfalse)
    * [get_chain_weights](#get_chain_weightsself-t-dwdtfalse)
    * [get_weights](#get_weightsself-t-dwdtfalse)

## Constructor

Construction method(s).

### Table of contents
  * [Constructor](#constructor)
    * [\_\_init\_\_](#__init__self-comps-hs_model<class-surfpacksaft_hardspheresaft_whitebear>-parameter_refdefault)


### `__init__(self, comps, hs_model=<class 'surfpack.saft_hardsphere.SAFT_WhiteBear'>, parameter_ref='default')`
Initialises a PC-SAFT model.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **comps (str) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Comma separated component identifiers, following thermopack convention.

&nbsp;&nbsp;&nbsp;&nbsp; **hs_model (SAFT_HardSphere) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Model to use for the hard-sphere contribution. Default is SAFT_WhiteBear.

&nbsp;&nbsp;&nbsp;&nbsp; **parameter_ref (str) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Reference for parameter set to use (see ThermoPack).

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## Utility methods

Methods for computing specific parameters and contributions to the residual
Helmholtz energy for PC-SAFT-type equations of state

### Table of contents
  * [Utility methods](#utility-methods)
    * [get_characteristic_lengths](#get_characteristic_lengthsself)


### `get_characteristic_lengths(self)`
Compute a characteristic length for the system.

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **float :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The sigma-parameter of the first component.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## Internal methods

Internal use, mainly documented for maintainers and developers.

### Table of contents
  * [Internal methods](#internal-methods)
    * [\_\_repr\_\_](#__repr__self)


### `__repr__(self)`
Generates a unique string for this model, containing information about the parameters and hard-sphere model. 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## Profile property interfaces

Compute properties using a given density profile, without iterating the density
profile to ensure equilibrium.

### Table of contents
  * [Profile property interfaces](#profile-property-interfaces)
    * [reduced_helmholtz_energy_density](#reduced_helmholtz_energy_densityself-rho-t-dphidnfalse-bulkfalse-asarrayfalse-dphidtfalse-dphidrhofalse)


### `reduced_helmholtz_energy_density(self, rho, T, dphidn=False, bulk=False, asarray=False, dphidT=False, dphidrho=False)`
Compute the the reduced Helmholtz energy density

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

&nbsp;&nbsp;&nbsp;&nbsp; **n_chain (list[Profile], optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Pre-computed weighted densities.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **Profile or float :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The (reduced) chain helmholtz energy density [-]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## Helmholtz energy contributions

Methods for computing various contributions to the Helmholtz energy
that are present in all SAFT-based functionals.

### Table of contents
  * [Helmholtz energy contributions](#helmholtz-energy-contributions)
    * [association_helmholtz_energy_density](#association_helmholtz_energy_densityself-rho-t-bulkfalse-dphidnfalse-dphidtfalse-dphidrhofalse-n_assocnone)
    * [chain_helmholtz_energy_density](#chain_helmholtz_energy_densityself-rho-t-dphidrhofalse-dphidnfalse-dphidtfalse-bulkfalse-n_chainnone)


### `association_helmholtz_energy_density(self, rho, T, bulk=False, dphidn=False, dphidT=False, dphidrho=False, n_assoc=None)`
Compute the association contribution to the reduced Helmholtz energy density.

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

&nbsp;&nbsp;&nbsp;&nbsp; **n_assoc (list[Profile], optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Pre-computed weighted densities.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **Profile or float :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The (reduced) association helmholtz energy density [-]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `chain_helmholtz_energy_density(self, rho, T, dphidrho=False, dphidn=False, dphidT=False, bulk=False, n_chain=None)`
Compute the chain contribution to the reduced Helmholtz energy density

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

&nbsp;&nbsp;&nbsp;&nbsp; **n_chain (list[Profile], optional) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Pre-computed weighted densities.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **Profile or float :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The (reduced) chain helmholtz energy density [-]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## Weighted densities

Methods to compute various weighted densities required by the different
Helmholtz energy contributions.

### Table of contents
  * [Weighted densities](#weighted-densities)
    * [get_assoc_weighted_densities](#get_assoc_weighted_densitiesself-rho-t-bulkfalse-dndrhofalse-dndtfalse)
    * [get_chain_weighted_densities](#get_chain_weighted_densitiesself-rho-t-bulkfalse-dndrhofalse-dndtfalse)
    * [get_weighted_densities](#get_weighted_densitiesself-rho-t-bulkfalse-dndrhofalse-dndtfalse)


### `get_assoc_weighted_densities(self, rho, T, bulk=False, dndrho=False, dndT=False)`
Compute the component weighted densities

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[Profile]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Density profiles [particles / Å^3] indexed as rho_arr[<comp idx>][<grid idx>]

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K], used to get radii and weights (in case of temp. dependent radii)

&nbsp;&nbsp;&nbsp;&nbsp; **bulk (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  If true, take a list[float] for rho_arr, and return a list[float]

&nbsp;&nbsp;&nbsp;&nbsp; **dndrho (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  If true, also return derivative (only for bulk)

&nbsp;&nbsp;&nbsp;&nbsp; **dndT (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  If true, also return derivative

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **list[Profile] :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  weighted densities [particles / Å^3], indexed as wts[<comp idx>][<grid idx>]

&nbsp;&nbsp;&nbsp;&nbsp; **Optional 2d array (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  dndrho, derivatives of weighted densities, indexed as dndrho[<wt idx>][<comp idx>]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `get_chain_weighted_densities(self, rho, T, bulk=False, dndrho=False, dndT=False)`
Compute the weighted densities for the chain contribution

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[Profile]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Density profiles [particles / Å^3] indexed as rho_arr[<comp idx>][<grid idx>]

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K], used to get radii and weights (in case of temp. dependent radii)

&nbsp;&nbsp;&nbsp;&nbsp; **bulk (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  If true, take a list[float] for rho_arr, and return a list[float]

&nbsp;&nbsp;&nbsp;&nbsp; **dndrho (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  If true, also return derivative (only for bulk)

&nbsp;&nbsp;&nbsp;&nbsp; **dndT (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  If true, also return derivative

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **list[Profile] :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  weighted densities [particles / Å^3], indexed as wts[<comp idx>][<grid idx>]

&nbsp;&nbsp;&nbsp;&nbsp; **Optional 2d array (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  dndrho, derivatives of weighted densities, indexed as dndrho[<wt idx>][<comp idx>]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `get_weighted_densities(self, rho, T, bulk=False, dndrho=False, dndT=False)`
Compute all neccessary weighted densities for this model.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **rho (list[Profile] or list[float]) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The particle density of each species

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **bulk (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Default False. Set to True if `rho` is `list[float]`

&nbsp;&nbsp;&nbsp;&nbsp; **dndrho (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Compute derivative (only for `bulk=True`)

&nbsp;&nbsp;&nbsp;&nbsp; **dndT (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Compute derivative

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **list[Profile] or list[float] :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Weighted densities

&nbsp;&nbsp;&nbsp;&nbsp; **Optional 2d array (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  dndrho, derivatives of weighted densities, indexed as dndrho[<wt idx>][<comp idx>]

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

## Weight functions

Get-methods for various weight functions.

### Table of contents
  * [Weight functions](#weight-functions)
    * [get_association_weights](#get_association_weightsself-t-dwdtfalse)
    * [get_chain_weights](#get_chain_weightsself-t-dwdtfalse)
    * [get_weights](#get_weightsself-t-dwdtfalse)


### `get_association_weights(self, T, dwdT=False)`
Get the weight functions for the association weighted densities, indexed as wts[<wt_idx>][<comp_idx>], equivalent to the component weights from FMT.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **dwdT (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Compute derivative

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **list[list[Analytical]] :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The weight functions.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `get_chain_weights(self, T, dwdT=False)`
Get the weight functions for the chain contribution, indexed as wts[<wt_idx>][<comp_idx>]. The weights wts[0] correspond to the local density, wts[1] are eq. Eq. 53 in Sauer & Gross, 10.1021/acs.iecr.6b04551 (lambda) and wts[1] are Eq. 54 in the same paper.

#### Args:

&nbsp;&nbsp;&nbsp;&nbsp; **T (float) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Temperature [K]

&nbsp;&nbsp;&nbsp;&nbsp; **dwdT (bool) :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  Compute derivative instead.

#### Returns:

&nbsp;&nbsp;&nbsp;&nbsp; **list[list[Analytical]] :** 

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;  The weight functions, shape (3 * ncomps, ncomps)

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; 

### `get_weights(self, T, dwdT=False)`
Get all the weights used for weighted densities in a 2D array, indexed as weight[<wt idx>][<comp idx>].

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

