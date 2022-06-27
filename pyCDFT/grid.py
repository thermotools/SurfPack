#!/usr/bin/env python3
from enum import Enum
from dft_numerics import dft_solver
import sys
from constants import NA, KB, Geometry
from weight_functions_cosine_sine import planar_weights_system_mc, \
    planar_weights_system_mc_pc_saft
from utility import packing_fraction_from_density, \
    boundary_condition, densities, get_thermopack_model, \
    weighted_densities_pc_saft_1D, get_initial_densities_vle, \
    weighted_densities_1D
import fmt_functionals
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

class Densities():
    """
    Utility class. List of np.ndarrays.
    """

    def __init__(self, nc, N):
        """

        Args:
            nc (int): Number of components
            N (int): Number of grid points
        """
        self.nc = nc
        self.N = N
        self.densities = []
        for i in range(nc):
            self.densities.append(np.zeros(N))

    def __setitem__(self, ic, rho):
        self.densities[ic][:] = rho[:]

    def __getitem__(self, ic):
        return self.densities[ic]

    def assign_elements(self, other, mask=None):
        """
        Assign all arrays ot other instance of densities. Apply mask if given.
        Args:
            other (densities): Other densities
            mask (bool ndarray): Array values to be changes

        """
        for i in range(self.nc):
            if mask is None:
                self.densities[i][:] = other.densities[i][:]
            else:
                self.densities[i][mask] = other.densities[i][mask]

    def assign_components(self, rho):
        """
        Assign all arrays ot other instance of densities. Apply mask if given.
        Args:
            rho (ndarray): Other densities

        """
        for i in range(self.nc):
            self.densities[i][:] = rho[i]

    def set_mask(self, mask, value=0.0):
        """
        Assign all arrays with value. Apply mask.
        Args:
            mask (bool ndarray): Array values to be changes
            value (float): Set masked array to value

        """
        for i in range(self.nc):
            self.densities[i][mask[i]] = value

    def mult_mask(self, mask, value):
        """
        Multiply arrays with value. Apply mask.
        Args:
            mask (bool ndarray): Array values to be changes
            value (float): Multiply masked array with value

        """
        for i in range(self.nc):
            self.densities[i][mask] *= value

    def is_valid_reals(self):
        """

        Returns:
            bool: True if no elements are NaN of Inf
        """
        is_valid = True
        for i in range(self.nc):
            if np.any(np.isnan(self.densities[i])) or np.any(np.isinf(self.densities[i])):
                is_valid = False
                break
        return is_valid

    def diff_norms(self, other, order=np.inf):
        """

        Args:
            other (densities): Other densities
            order: Order of norm

        Returns:

        """
        nms = np.zeros(self.nc)
        for i in range(self.nc):
            nms[i] = np.linalg.norm(
                self.densities[i] - other.densities[i], ord=order)
        return nms

    def diff_norm_scaled(self, other, scale, mask, ord=np.inf):
        """

        Args:
            other (densities): Other densities
            scale (ndarray): Column scale
            ord: Order of norm

        Returns:

        """
        nd_copy = self.get_nd_copy()[:, mask]
        nd_copy_other = other.get_nd_copy()[:, mask]
        nd_copy -= nd_copy_other
        nd_copy = (nd_copy.T / scale).T
        norm = np.linalg.norm(nd_copy, ord=ord)
        return norm

    def get_nd_copy(self):
        """
        Make ndarray with component densities as columns
        Returns:
            ndarray: nc x N array.
        """
        nd_copy = np.zeros((self.nc, self.N))
        for i in range(self.nc):
            nd_copy[i, :] = self.densities[i][:]
        return nd_copy


class Grid(object):
    """

    """

    def __init__(self,
                 geometry,
                 n_comp,
                 domain_size=100.0,
                 n_grid=1024,
                 n_bc=0):
        """Class holding specifications for a gird

        Args:
            geometry (int): PLANAR/POLAR/SPHERICAL
            n_comp (int, optional): Number of components.
            domain_size (float, optional): Sisze of domain. Defaults to 100.0.
            n_grid (int, optional): Number of grid points. Defaults to 1024.
            n_bc (int, optional): Number of boundary points. Defaults to 0.

        Returns:
            None
        """
        self.geometry = geometry
        self.densities = densities(n_comp, n_grid)
        # Length
        self.domain_size = domain_size
        # Grid size
        self.n_grid_inner = n_grid
        self.n_grid = n_grid
        # Grid spacing
        self.dr = self.domain_length / self.N
        # Boundary
        self.n_bc = n_bc
        # FFT padding of grid
        self.padding = 0
        # Add boundary and padding to grid
        self.N = self.N + 2 * self.n_bc + 2 * self.padding
        self.end = self.N - self.n_bc - self.padding  # End of domain

        # Mask for inner domain
        self.NiWall = self.N - self.end
        self.domain_mask = np.full(self.N, False, dtype=bool)
        self.domain_mask[self.NiWall:self.end] = True
        self.weight_mask = np.full(self.N, False, dtype=bool)

        # Set grid positions or planar distance for plotting
        assert Geometry.POLAR != geometry
        self.z = np.linspace(0.5*self.dr, self.domain_size - 0.5*self.dr, self.n_grid_inner)
        # Store bulk for access to R and particle_diamater
        self.bulk = None

        # Mask for inner domain
        self.NiWall = self.N - self.end
        self.domain_mask = np.full(self.N, False, dtype=bool)
        self.domain_mask[self.NiWall:self.end] = True
        self.weight_mask = np.full(self.N, False, dtype=bool)

        # Set up wall
        self.NiWall_array_left = [self.NiWall] * self.nc
        self.NiWall_array_right = [self.N - self.NiWall] * self.nc
        self.left_boundary = None  # Handled in setup_wall
        self.right_boundary = None  # Handled in setup_wall

        # self.left_boundary_mask = []
        # self.right_boundary_mask = []
        # self.boundary_mask = []
        # for i in range(self.nc):
        #     self.left_boundary_mask.append(
        #         np.full(self.N, False, dtype=bool))
        #     self.right_boundary_mask.append(
        #         np.full(self.N, False, dtype=bool))
        #     self.boundary_mask.append(
        #         np.full(self.N, False, dtype=bool))
        #     self.left_boundary_mask[i][:self.NiWall_array_left[i]] = True
        #     self.right_boundary_mask[i][self.NiWall_array_right[i]:] = True
        #     self.boundary_mask[i] = np.logical_or(
        #         self.left_boundary_mask[i], self.right_boundary_mask[i])


    def integrate_number_of_molecules(self):
        """
        Calculates the overall number of molecules (reduced) of the system.

        Returns:
            (float): Reduced number of molecules (-)
        """

        n_tot = 0.0
        for ic in range(self.nc):
            n_tot += np.sum(self.density[ic])
        n_tot *= self.dr
        return n_tot

    def print_grid(self):
        """
        Debug function

        """
        print("N: ", self.N)
        print("NiWall: ", self.NiWall)
        print("NinP: ", self.NinP)
        print("Nbc: ", self.Nbc)
        print("end: ", self.end)
        print("domain_mask: ", self.domain_mask)
        print("weight_mask: ", self.weight_mask)
        for i in range(self.nc):
            print(f"NiWall_array_left {i}: ", self.NiWall_array_left[i])
            print(f"NiWall_array_right {i}: ", self.NiWall_array_right[i])
            print(f"left_boundary_mask {i}: ", self.left_boundary_mask[i])
            print(f"right_boundary_mask {i}: ", self.right_boundary_mask[i])
            print(f"boundary_mask {i}: ", self.boundary_mask[i])

    def set_tanh_profile(bulk, reduced_temperature, rel_pos_dividing_surface=0.5):
        """
        Calculate initial densities for gas-liquid interface calculation
        Args:
            rho_left (list of float): Left side density
            rho_right (list of float): Right side density
            reduced_temperature (float): T/Tc
            rel_pos_dividing_surface (float, optional): Relative location of initial dividing surface. Default value 0.5.
        """
        z_centered = np.zeros_like(self.z)
        z_centered[:] = self.z[:] - rel_dividing_surface*self.domain_size

        r_left = bulk.reduced_density_left
        r_right = bulk.reduced_density_right
        R = bulk.R
        self.bulk = bulk
        for i in range(np.shape(r_right)[0]):
        #         rho0.densities[i][:] = 0.5*(rho_g[i] + rho_l[i]) + 0.5 * \
            #             (rho_l[i] - rho_g[i])*np.tanh(0.3*z[:]/R[i])
            self.densities[i][:] = 0.5*(r_left[i] + r_right[i]) + 0.5 * \
                (r_right[i] - r_left[i])*np.tanh(z_centered[:]/(2*R[i])
                                                 * (2.4728 - 2.3625 * reduced_temperature))

    def set_constant_profile(bulk):
        """
        Calculate initial densities for gas-liquid interface calculation
        Args:
            rho (list of float): Densities
        """
        self.bulk = bulk
        r = bulk.reduced_density_left
        for i in range(np.shape(rho)[0]):
            self.densities[i][:] = r[i]

    def set_density_profile(densities):
        """
        Calculate initial densities for gas-liquid interface calculation
        Args:
            densities (Densities): Densities
        """
        self.densities = densities


class Bulk(object):
    """

    """

    def __init__(self,
                 functional,
                 left_state,
                 right_state):
        """Class holding specifications for a gird

        Args:
            geometry (int): PLANAR/POLAR/SPHERICAL
            n_comp (int, optional): Number of components.
            domain_size (float, optional): Sisze of domain. Defaults to 100.0.
            n_grid (int, optional): Number of grid points. Defaults to 1024.
            n_bc (int, optional): Number of boundary points. Defaults to 0.

        Returns:
            None
        """
        self.left_state = left_state
        self.right_state = right_state
        self.particle_diameters = left_state.eos.hard_sphere_diameters(left_state.temperature)
        self.R = np.zeros_like(particle_diameters) # Particle radius (Reduced)
        self.R[:] = 0.5*particle_diameters/particle_diameters[0]

        # Temperature
        self.temperature = left_state.temperature
        self.beta = 1.0 / temperature
        # Bulk density
        self.reduced_density_left = self.get_reduced_density(left_state.partial_density())
        self.reduced_density_right = self.get_reduced_density(right_state.partial_density())
        # Bulk fractions
        self.bulk_fractions = self.reduced_density_left/np.sum(self.reduced_density_left)

        # Calculate reduced pressure and excess chemical potential
        self.red_pressure = np.sum(self.reduced_density_right) * self.T * \
            functional.bulk_compressibility(self.reduced_density_right)
        # Extract normalized chemical potential (multiplied by beta) (mu/kbT)
        self.mu_res_scaled_beta = self.functional.bulk_excess_chemical_potential(
            self.reduced_density_right)
        self.mu_ig_scaled_beta = np.log(self.reduced_density_right)
        self.mu_scaled_beta = self.mu_ig_scaled_beta + self.mu_res_scaled_beta

    def get_reduced_density(self, partial_density):
        """
        Calculates the overall number of molecules (reduced) of the system.

        Returns:
            (float): Reduced number of molecules (-)
        """

        reduced_density = np.zeros_like(partial_density)
        reduced_density[:] = partial_density*NA*self.particle_diameters[0]**3
        return reduced_density

    def get_real_density(self, reduced_density):
        """
        Calculates the overall number of molecules (reduced) of the system.

        Returns:
            (float): Reduced number of molecules (-)
        """

        partial_density = np.zeros_like(reduced_density)
        partial_density[:] = reduced_density/(NA*self.particle_diameters[0]**3)
        return partial_density
