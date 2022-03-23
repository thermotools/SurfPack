#!/usr/bin/env python3
import numpy as np
import fmt_functionals
from utility import weighted_densities_1D, differentials_1D, \
    packing_fraction_from_density, boundary_condition, densities
from weight_functions import planar_weights_system_mc
from constants import CONV_FFTW, CONV_SCIPY_FFT, CONV_NO_FFT, CONVOLUTIONS
import sys


class cdft1D:
    """
    Base classical DFT class for 1D problems
    """

    def __init__(self,
                 bulk_densities,
                 particle_diameters=np.array([1.0]),
                 wall="HW",
                 domain_length=40.0,
                 functional="Rosenfeld",
                 grid_dr=0.001,
                 temperature=1.0,
                 quadrature="None"):
        """
        Object holding specifications for classical DFT problem.
        Reduced particle size assumed to be d=1.0, and all other sizes are relative to this scale.

        Args:
            bulk_densities (ndarray): Bulk fluid density ()
            particle_diameters (ndarray): Particle diameter
            wall (str): Wall type (HardWall, SlitHardWall)
            domain_length (float): Length of domain
            functional (str): Name of hard sphere functional: Rosenfeld, WhiteBear, WhiteBear Mark II, Default Rosenfeld
            grid_dr (float) : Grid spacing
            temperature (float): Reduced temperature
            quadrature (str): Quadrature to use during integration
        Returns:
            None
        """
        # Number of components
        self.nc = len(particle_diameters)
        # Particle radius
        self.R = 0.5*particle_diameters
        # Temperature
        self.T = temperature
        self.beta = 1.0 / temperature
        # Bulk density
        self.bulk_densities = bulk_densities
        self.eta = packing_fraction_from_density(
            bulk_densities, d=particle_diameters)
        # Bulk fractions
        self.bulk_fractions = bulk_densities/np.sum(bulk_densities)
        # Length
        self.domain_length = domain_length
        # Grid spacing
        self.dr = grid_dr

        # Get functional
        self.functional = fmt_functionals.get_functional(functional, self.R)

        # FFT padding of grid
        if CONVOLUTIONS in (CONV_FFTW, CONV_SCIPY_FFT):
            self.padding = 1
        else:
            self.padding = 0
        # Get grid info
        self.N = round(domain_length / grid_dr)  # Should be even
        self.NinP = []
        for i in range(self.nc):
            # Number of grid points within particle
            self.NinP.append(2 * round(self.R[i] / grid_dr))
        self.Nbc = np.max(self.NinP)
        self.padding *= self.Nbc
        # Add boundary and padding to grid
        self.N = self.N + 2 * self.Nbc + 2 * self.padding
        self.end = self.N - self.Nbc - self.padding  # End of domain

        # Allocate weighted densities, differentials container and weights
        self.weights_system = planar_weights_system_mc(functional=self.functional,
                                                       dr=self.dr,
                                                       R=self.R,
                                                       N=self.N,
                                                       quad=quadrature)

        # Calculate reduced pressure and excess chemical potential
        self.red_pressure = np.sum(self.bulk_densities) * self.T * \
            self.functional.bulk_compressibility(self.bulk_densities)
        self.excess_mu = self.functional.bulk_excess_chemical_potential(
            self.bulk_densities)
        print(self.red_pressure, self.excess_mu)

        # Mask for inner domain
        self.NiWall = self.N - self.end
        self.domain_mask = np.full(self.N, False, dtype=bool)
        self.domain_mask[self.NiWall:self.end] = True

        # Set up wall
        self.NiWallArrayLeft = [self.NiWall] * self.nc
        self.NiWallArrayRight = [self.NiWall] * self.nc
        self.Vext = densities(self.nc, self.N) # Use structure of densities class
        self.wall_setup(wall)


    def wall_setup(self, wall):
        """

        Args:
            wall (str): Wall type

        """
        self.left_boundary = boundary_condition["OPEN"]
        self.right_boundary = boundary_condition["OPEN"]
        # Wall setup
        hw = ("HW", "HARDWALL", "SHW")
        is_hard_wall = len([w for w in hw if w in wall.upper()]) > 0
        slit = ("SLIT", "SHW")
        is_slit = len([s for s in slit if s in wall.upper()]) > 0
        if is_hard_wall:
            self.left_boundary = boundary_condition["WALL"]
            self.wall = "HW"
            for i in range(self.nc):
                self.NiWallArrayLeft[i] += round(self.NinP[i]/2)
                self.Vext[i][:self.NiWallArrayLeft[i]] = 500.0
            if is_slit:
                # Add right wall setup
                self.right_boundary = boundary_condition["WALL"]
                self.wall = "SHW"
                for i in range(self.nc):
                    self.NiWallArrayRight[i] += round(self.NinP[i] / 2)
                    self.Vext[i][self.NiWallArrayRight[i]:] = 500.0

    def grand_potential(self, dens, update_convolutions=True):
        """
        Calculates the grand potential in the system.

        Args:
            dens (densities): Density profile
            update_convolutions(bool): Flag telling if convolutions should be calculated

        Returns:
            (float): Grand potential
            (array): Grand potential contribution for each grid point
        """

        # Make sure weighted densities are up-to-date
        if update_convolutions:
            self.weights_system.convolutions(dens)

        # Calculate chemical potential (excess + ideal)
        mu = self.excess_mu + self.T * np.log(self.bulk_densities)

        # FMT hard-sphere part
        omega_a = self.T * \
            self.functional.excess_free_energy(self.weights_system.weighted_densities)

        # Add ideal part and extrinsic part
        for i in range(self.nc):
            # Ideal part
            omega_a[self.domain_mask] += self.T * dens[i][self.domain_mask] * \
                (np.log(dens[i][self.domain_mask]) - 1.0)
            # Extrinsic part
            omega_a[self.domain_mask] += dens[i][self.domain_mask] \
                * (self.Vext[self.domain_mask] - mu[i])

        omega_a[:] *= self.dr

        # Integrate using trapezoidal method
        omega = np.sum(omega_a[self.domain_mask]) - 0.5 * \
            omega_a[self.NiWall] - 0.5*omega_a[self.end]

        return omega, omega_a[:]


if __name__ == "__main__":
    from utility import density_from_packing_fraction
    d = np.array([1.0, 3.0/5.0])
    bulk_density = density_from_packing_fraction(
        eta=np.array([0.3105, 0.0607]), d=d)
    cdft = cdft1D(bulk_densities=bulk_density,
                  particle_diameters=d,
                  domain_length=40.0,
                  functional="Rosenfeld",
                  grid_dr=0.001,
                  temperature=1.0,
                  quadrature="None")

    d = np.array([1.0])
    bulk_density = density_from_packing_fraction(eta=np.array([0.2]), d=d)
    cdft = cdft1D(bulk_densities=bulk_density,
                  particle_diameters=d,
                  domain_length=40.0,
                  functional="WHITEBEARMARKII",
                  grid_dr=0.001,
                  temperature=1.0,
                  quadrature="None")
