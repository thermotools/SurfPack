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
        self.real_mu = functional.thermo.chemical_potential(self.temperature, volume=1.0, n=left_state.partial_density())

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



class Grid(object):
    """

    """

    def __init__(self,
                 geometry,
                 domain_size=100.0,
                 n_grid=1024,
                 n_bc=0):
        """Class holding specifications for a gird

        Args:
            geometry (int): PLANAR/POLAR/SPHERICAL
            domain_size (float, optional): Sisze of domain. Defaults to 100.0.
            n_grid (int, optional): Number of grid points. Defaults to 1024.
            n_bc (int, optional): Number of boundary points. Defaults to 0.

        Returns:
            None
        """
        self.geometry = geometry
        # Length
        self.domain_size = domain_size
        # Grid size
        self.n_grid_inner = n_grid
        self.n_grid = n_grid

        # Set up grid
        if geometry == Geometry.PLANAR:
            self.planar_grid(domain_size=domain_size,
                            n_grid=n_grid)
        elif geometry == Geometry.POLAR:
            self.polar_grid(domain_size=domain_size,
                            n_grid=n_grid)
        elif geometry == Geometry.SPHERICAL:
            self.spherical_grid(domain_size=domain_size,
                                n_grid=n_grid)
        else:
            print("Wrong grid type specified!")
            sys.exit()

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
            (float): Reduced number of molecules (Unit depending on geometry)
        """
        n_tot = 0.0
        for ic in range(self.nc):
            n_tot += np.sum(self.density[ic]*self.integration_weights)
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

    def get_index_of_rel_pos(self, rel_pos_dividing_surface):
        """
        Calculate initial densities for gas-liquid interface calculation
        Args:
            rel_pos_dividing_surface (float, optional): Relative location of initial dividing surface.
        """
        idx = None
        if self.geometry == Geometry.PLANAR or \
           self.geometry == Geometry.SPERICAL:
            idx = int(rel_pos_dividing_surface*self.n_grid)
        else:
            pos = rel_pos_dividing_surface*self.domain_size
            for i in range(n_grid):
                if pos >= self.z_edge[i] and pos <= self.z_edge[i+1]:
                    idx = i
                    break
        return idx

    def polar_grid(self,
                   domain_size=15.0,
                   n_grid=1024):
        """Set up grid according tO Xi eta al. 2020
        An Efficient Algorithm for Molecular Density Functional Theory in Cylindrical Geometry:
        Application to Interfacial Statistical Associating Fluid Theory (iSAFT)
        DOI: 10.1021/acs.iecr.9b06895
        """

        # self.basic_init(geometry=Geometry.POLAR,
        #                 n_comp=n_comp,
        #                 domain_size=domain_size,
        #                 n_grid=n_grid)
        alpha = 0.002
        for _ in range(21):
            alpha = -np.log(1.0 - np.exp(-alpha)) / (n_grid - 1)
        self.alpha = alpha
        self.x0 = 0.5 * (np.exp(-alpha * n_grid) + np.exp(-alpha * (n_grid - 1)))
        # Setting the grid
        self.z = np.zeros(n_grid)
        for i in range(n_grid):
            self.z = domain_size*self.x0*np.exp(alpha*i)
        # Setting the edge grid
        self.z_edge = np.zeros(n_grid+1)
        for i in range(1:n_grid+1):
            self.z_edge = domain_size*np.exp(-alpha*(n_grid-i))
        # End correction factor
        k0 = np.exp(2*alpha)*(2*np.exp(alpha) + np.exp(2*alpha) - 1)/ \
            (1 + np.exp(alpha))**2/(np.exp(2*alpha) - 1)
        self.k = np.ones(n_grid)
        self.k[0] = k0
        k0v = np.exp(2*alpha)*(2*np.exp(alpha) + np.exp(2*alpha) - 5.0/3.0)/ \
            (1 + np.exp(alpha))**2/(np.exp(2*alpha) - 1)
        self.kv = np.ones(n_grid)
        self.kv[0] = k0v
        # Hankel paramaters
        #self.b = domain_size
        #fac = 1.0/(2*self.x0*(np.exp(alpha*(n_grid-1)) - np.exp(alpha*(n_grid-2))))
        #self.lam = int(0.5*fac/self.b)
        #self.gamma = self.lam*self.b
        # Defining integration weights
        self.integration_weights = np.zeros(n_grid)
        self.integration_weights[0] = k0 * np.exp(2 * alpha)
        self.integration_weights[1] = (np.exp(2 * alpha) - k0) * np.exp(2 * alpha)
        for i in range(2:n_grid):
            self.integration_weights[i] = np.exp(2 * alpha * i) * (np.exp(2 * alpha) - 1.0)
        self.integration_weights *= np.exp(-2 * alpha * n_grid) * np.pi * domain_size**2


    def planar_grid(self,
                    domain_size=50.0,
                    n_grid=1024):
        """
        """

        # self.basic_init(geometry=Geometry.POLAR,
        #                 n_comp=n_comp,
        #                 domain_size=domain_size,
        #                 n_grid=n_grid)

        # Grid spacing
        self.dr = self.domain_length / n_grid
        # Grid
        self.z = np.linspace(0.5*self.dr, domain_size - 0.5*self.dr, n_grid)
        # Setting the edge grid
        self.z_edge = np.linspace(0.0, self.domain_size, n_grid + 1)
        # Defining integration weights
        self.integration_weights = np.zeros(n_grid)
        self.integration_weights[:] = self.dr

    def spherical_grid(self,
                       domain_size=15.0,
                       n_grid=1024):
        """
        """

        # self.basic_init(geometry=Geometry.POLAR,
        #                 n_comp=n_comp,
        #                 domain_size=domain_size,
        #                 n_grid=n_grid)


        # Grid spacing
        self.dr = self.domain_length / n_grid
        # Grid
        self.z = np.linspace(0.5*self.dr, domain_size - 0.5*self.dr, n_grid)
        # Setting the edge grid
        self.z_edge = np.linspace(0.0, self.domain_size, n_grid + 1)
        # Defining integration weights
        self.integration_weights = np.zeros(n_grid)
        for k in range(n_grid):
            self.integration_weights[k] = 4.0*np.pi/3.0*dr**3*(3*k**2 + 3*k + 1)

    @staticmethod
    def Planar(domain_size=15.0,
               n_grid=1024):
        """Set up polar grid
        """

        return Grid(geometry=Geometry.PLANAR,
                    domain_size=domain_size,
                    n_grid=n_grid)

    @staticmethod
    def Polar(domain_size=15.0,
              n_grid=1024):
        """Set up polar grid
        """

        return Grid(geometry=Geometry.POLAR,
                    domain_size=domain_size,
                    n_grid=n_grid)

    @staticmethod
    def Sperical(domain_size=15.0,
                 n_grid=1024):
        """Set up spherical grid
        """

        return Grid(geometry=Geometry.SPHERICAL,
                    domain_size=domain_size,
                    n_grid=n_grid)


class Profile(object):
    """
    Profile and methods to initialize profiles
    """

    def __init__(self,
                 dens):
        """Class holding density profiles

        Args:
            geometry (int): PLANAR/POLAR/SPHERICAL
            n_comp (int, optional): Number of components.
            domain_size (float, optional): Sisze of domain. Defaults to 100.0.
            n_grid (int, optional): Number of grid points. Defaults to 1024.
            n_bc (int, optional): Number of boundary points. Defaults to 0.

        Returns:
            None
        """
        self.densities = dens

    @staticmethod
    def tanh_profile(grid, bulk, reduced_temperature, rel_pos_dividing_surface=0.5):
        """
        Calculate initial densities for gas-liquid interface calculation
        Args:
            rho_left (list of float): Left side density
            rho_right (list of float): Right side density
            reduced_temperature (float): T/Tc
            rel_pos_dividing_surface (float, optional): Relative location of initial dividing surface. Default value 0.5.
        """
        z_centered = np.zeros_like(grid.z)
        z_centered[:] = grid.z[:] - rel_dividing_surface*grid.domain_size

        r_left = bulk.reduced_density_left
        r_right = bulk.reduced_density_right
        R = bulk.R
        n_comp = np.shape(r_right)[0]
        dens = densities(n_comp, grid.n_grid)
        for i in range(n_comp):
        #         rho0.densities[i][:] = 0.5*(rho_g[i] + rho_l[i]) + 0.5 * \
            #             (rho_l[i] - rho_g[i])*np.tanh(0.3*z[:]/R[i])
            dens[i][:] = 0.5*(r_left[i] + r_right[i]) + 0.5 * \
                (r_right[i] - r_left[i])*np.tanh(z_centered[:]/(2*R[i])
                                                 * (2.4728 - 2.3625 * reduced_temperature))
        return Profile(dens)

    @staticmethod
    def constant_profile(grid, bulk, v_ext=None):
        """
        Calculate initial densities for gas-liquid interface calculation
        Args:
            grid (Grid): Grid
            bulk (Bulk): Bulk states
            v_ext (array, optional): Array of potentials evalauted in grid positions
        """
        r = bulk.reduced_density_left
        n_comp = np.shape(r)[0]
        dens = densities(n_comp, grid.n_grid)
        dens.assign_components(r)
        if v_ext:
            for i in range(n_comp):
                dens[i] *= np.minimum(np.exp(-self.Vext[i]), 1.0)
        return Profile(dens)

    @staticmethod
    def step_profile(grid, bulk, rel_pos_dividing_surface=0.5):
        """
        Calculate initial densities for gas-liquid interface calculation
        Args:
            grid (Grid): Grid
            bulk (Bulk): Bulk states
            rel_pos_dividing_surface (float, optional): Relative location of initial dividing surface. Default value 0.5.
        """
        r_left = bulk.reduced_density_left
        r_right = bulk.reduced_density_right
        idx = grid.get_index_of_rel_pos(rel_pos_dividing_surface)
        n_comp = np.shape(r_right)[0]
        dens = densities(n_comp, grid.n_grid)
        for i in range(n_comp):
            dens[i][0:idx+1] = r_left[i]
            dens[i][idx+1:] = r_right[i]
        return Profile(dens)

    @staticmethod
    def empty_profile(n_comp, n_grid):
        """
        Allocate memory for profile
        Args:
            n_comp (int): Number of components.
            n_grid (int): Number of grid points.
        """
        dens = densities(n_comp, n_grid)
        return Profile(dens)

    def copy_profile(self, prof):
        """
        Copy exsisting profile
        Args:
            prof (Profile): Exsisting profile
        """
        self.dens = densities(prof.nc, prof.N)
        self.dens.assign_elements(prof.densities)

