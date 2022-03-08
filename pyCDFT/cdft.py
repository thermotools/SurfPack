#!/usr/bin/env python3
import numpy as np
import fmt_functionals
from utility import weighted_densities_1D, differentials_1D, \
    packing_fraction_from_density, boundary_condition
from weight_functions import planar_weights
from constants import CONV_FFTW, CONV_SCIPY_FFT, CONV_NO_FFT, CONVOLUTIONS
import sys


class cdft1D:
    """
    Base classical DFT class for 1D problems
    """

    def __init__(self,
                 bulk_density,
                 wall="HW",
                 domain_length=40.0,
                 functional="Rosenfeld",
                 grid_dr=0.001,
                 temperature=1.0):
        """
        Object holding specifications for classical DFT problem.
        Reduced particle size assumed to be d=1.0, and all other sizes are relative to this scale.

        Args:
            bulk_density (float): Bulk fluid density ()
            wall (str): Wall type (HardWall, SlitHardWall)
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
        self.beta = 1.0 / temperature
        # Bulk density
        self.bulk_density = bulk_density
        self.eta = packing_fraction_from_density(bulk_density)
        # Length
        self.domain_length = domain_length
        # Grid spacing
        self.dr = grid_dr

        # Get functional
        self.functional = fmt_functionals.get_functional(functional)

        # FFT padding of grid
        if CONVOLUTIONS == CONV_FFTW:
            self.padding = 1
        else:
            self.padding = 0
        # Get grid info
        self.N = round(domain_length / grid_dr) + 1
        self.NinP = 2 * round(self.R / grid_dr)  # Number of grid points within particle
        self.padding *= self.NinP
        self.N = self.N + 2 * self.NinP + 2 * self.padding  # Add boundary and padding to grid
        self.end = self.N - self.NinP - self.padding  # End of domain

        # Allocate weighted densities
        self.weighted_densities = weighted_densities_1D(self.N, self.R)
        # Allocate differentials container
        self.differentials = differentials_1D(self.N, self.R)
        # Allocate weights
        self.weights = planar_weights(self.dr, self.R, self.N)

        # Calculate reduced pressure and excess chemical potential
        self.red_pressure = self.bulk_density * self.T * self.functional.compressibility(self.eta)
        self.excess_mu = self.functional.excess_chemical_potential(self.eta)

        # Set up wall
        self.wall_setup(wall)

        # Mask for updating density on inner domain
        self.NiWall = self.N - self.end
        self.domain_mask = np.full(self.N, False, dtype=bool)
        self.domain_mask[self.NiWall:self.end] = True

    def wall_setup(self, wall):
        self.left_boundary = boundary_condition["OPEN"]
        self.right_boundary = boundary_condition["OPEN"]
        # Wall setup
        self.Vext = np.zeros(self.N)
        hw = ("HW", "HARDWALL", "SHW")
        is_hard_wall = len([w for w in hw if w in wall.upper()]) > 0
        slit = ("SLIT", "SHW")
        is_slit = len([s for s in slit if s in wall.upper()]) > 0
        if is_hard_wall:
            self.Vext[:self.NinP] = np.inf
            self.left_boundary = boundary_condition["WALL"]
            self.wall = "HW"
            if is_slit:
                # Add right wall setup
                self.Vext[self.end:] = np.inf
                self.right_boundary = boundary_condition["WALL"]
                self.wall = "SHW"

    def grand_potential(self, density, update_convolutions=True):
        """
        Calculates the grand potential in the system.

        Args:
            density (array_like): Density profile
            update_convolutions (bool): Flag telling if convolutions should be calculated

        Returns:
            (float): Grand potential
            (array): Grand potential contribution for each grid point
        """

        # Make sure weighted densities are up-to-date
        if update_convolutions:
            self.weights.convolutions(self.weighted_densities, density)

        # Calculate chemical potential (excess + ideal)
        mu = self.excess_mu + self.T * np.log(self.bulk_density)

        # FMT hard-sphere part
        omega_a = self.T * self.functional.excess_free_energy(self.weighted_densities)

        # Ideal part
        omega_a[self.domain_mask] += self.T * density[self.domain_mask] * \
                                   (np.log(density[self.domain_mask]) - 1.0)

        # Extrinsic part
        omega_a[self.domain_mask] += density[self.domain_mask] \
                                   * (self.Vext[self.domain_mask] - mu)

        omega_a[:] *= self.dr

        # Integrate using trapezoidal method
        omega = np.sum(omega_a[self.domain_mask]) - 0.5*omega_a[self.NiWall] - 0.5*omega_a[self.end]

        return omega, omega_a[:]


if __name__ == "__main__":
    pass
