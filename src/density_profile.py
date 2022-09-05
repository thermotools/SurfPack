#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from constants import NA, KB, Geometry
import numpy as np

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

class Profile(object):
    """
    Profile and methods to initialize profiles
    """

    def __init__(self,
                 dens=None):
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
        z_centered[:] = grid.z[:] - rel_pos_dividing_surface*grid.domain_size
        r_left = bulk.reduced_density_left
        r_right = bulk.reduced_density_right
        R = bulk.R
        n_comp = np.shape(r_right)[0]
        dens = Densities(n_comp, grid.n_grid)
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
        dens = Densities(n_comp, grid.n_grid)
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
        dens = Densities(n_comp, grid.n_grid)
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
        dens = Densities(n_comp, n_grid)
        return Profile(dens)

    def copy_profile(self, prof):
        """
        Copy exsisting profile
        Args:
            prof (Profile): Exsisting profile
        """
        self.densities = Densities(prof.densities.nc, prof.densities.N)
        self.densities.assign_elements(prof.densities)

    @property
    def rho_mix(self):
        """
        Get mixture density
        Args:
            (np.ndarray): Mixture density
        """
        rho_mix = np.zeros(self.densities.N)
        for i in range(self.densities.nc):
            rho_mix[:] += self.densities[i][:]
        return rho_mix

if __name__ == "__main__":
    pass
