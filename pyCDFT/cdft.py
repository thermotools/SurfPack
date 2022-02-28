#!/usr/bin/env python3
import numpy as np
import fmt_functionals
from utility import weighted_densities_1D, differentials_1D,\
    packing_fraction_from_density
from weight_functions import planar_weights
import sys


class cdft1D:
    """
    Base classical DFT class for 1D problems
    """

    def __init__(self, bulk_density, wall, domain_length=40.0, functional="Rosenfeld", grid_dr=0.001, temperature=1.0):
        """
        Object holding specifications for classical DFT problem.
        Reduced particle size assumed to be d=1.0, and all other sizes are relative to this scale.

        Args:
            bulk_density (float): Bulk fluid density ()
            wall (str): Wall type
            domain_length (float): Length of domain
            functional (str): Name of hard sphere functional: Rosenfeld, WhiteBear, WhiteBear Mark II, Default Rosenfeld
            grid_dr (float) : Grid spacing
            temperature (float): Reduced temperature

        Returns:
            None
        """
        # Particle radius
        self.R = 0.5
        # Temperature
        self.T = temperature
        self.beta = 1.0/temperature
        # Bulk density
        self.bulk_density = bulk_density
        self.eta = packing_fraction_from_density(bulk_density)
        # Length
        self.domain_length = domain_length
        # Grid spacing
        self.dr = grid_dr

        # Get functional
        self.functional = fmt_functionals.get_functional(functional)

        # Get grid info
        self.N = round(domain_length / grid_dr) + 1
        self.NinP = 2*round(self.R / grid_dr)  # Number of grid points within particle
        self.N = self.N + 2 * self.NinP  # Add boundary to grid
        self.end = self.N - 1 * self.NinP  # End of domain

        # Allocate weighted densities
        self.weighted_densities = weighted_densities_1D(self.N, self.R)
        # Allocate differentials container
        self.differentials = differentials_1D(self.N, self.R)
        # Allocate weights
        self.weights = planar_weights(self.dr, self.R)

        # Calculate reduced pressure and excess chemical potential
        self.red_pressure = self.bulk_density * self.T * self.functional.compressibility(self.eta)
        self.excess_mu = self.functional.excess_chemical_potential(self.eta)

        # Left wall setup
        self.Vext = np.zeros(self.N)
        self.Vext[:self.NinP] = np.inf
        if wall.upper() == "SLIT":
            # Add right wall setup
            self.Vext[self.end:] = np.inf


if __name__ == "__main__":
    pass
