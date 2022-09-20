# Thermopack DFT
Classical density functinal theory for thermopack


# Issues
  * Binary weights and accuracy when not using Fourier weights
  * plot binary results vs. MC

# List of todos (Remove when accomplished)
  * Implement Lanczos sigma
  * Test the chain term implementation
  * Test with a wall potential
  * Implement postprocessing routine to extract useful variables
  * The thermopack association term implementation must be revised
  * Implement association into the DFT code [[2]](#2)
  * Learn "tricks of the trade" from Rolf
  * Implement one-dimensional spherical and cylindrical geometries
  * Extract radial distribution function and compare to simulations
  * Implement a consistent WCA-reference, test for the LJs fluid
  * Mixture implementation for saft-vrq Mie
  * Entropy for spherical
  * Set prefactors using sympy getting a more generic framework.

# References
<a id="1">[1]</a>
J. Mairhofer, J. Gross, Fluid Phase Equilibria, 444, 1-12, 2017.

<a id="2">[2]</a>
E. Sauer, J. Gross, Ind. Eng. Chem. Res., 56, 4119, 2017.