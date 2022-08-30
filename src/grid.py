#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from constants import NA, KB, Geometry
import numpy as np

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
        self.N = self.n_grid + 2 * self.n_bc + 2 * self.padding
        self.end = self.n_grid - self.n_bc - self.padding  # End of domain

        # Mask for inner domain
        self.NiWall = self.n_grid - self.end
        self.domain_mask = np.full(self.N, False, dtype=bool)
        self.domain_mask[self.NiWall:self.end] = True
        self.weight_mask = np.full(self.N, False, dtype=bool)

        # Mask for inner domain
        self.NiWall = self.n_grid - self.end
        self.domain_mask = np.full(self.N, False, dtype=bool)
        self.domain_mask[self.NiWall:self.end] = True
        self.weight_mask = np.full(self.N, False, dtype=bool)

        # Set up wall
        # self.NiWall_array_left = [self.NiWall] * self.nc
        # self.NiWall_array_right = [self.n_grid - self.NiWall] * self.nc
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
           self.geometry == Geometry.SPHERICAL:
            idx = int(rel_pos_dividing_surface*self.n_grid)
        else:
            pos = rel_pos_dividing_surface*self.domain_size
            for i in range(n_grid):
                if pos >= self.z_edge[i] and pos <= self.z_edge[i+1]:
                    idx = i
                    break
        return idx

    def get_left_weight(self, position):
        """
        Calculate left side weight in cell, given position
        Args:
            rel_pos_dividing_surface (float, optional): Relative location of initial dividing surface.
        """
        idx = self.get_index_of_rel_pos(position/self.domain_size)
        if self.geometry == Geometry.PLANAR:
            w_left = position - self.z_edge[idx]
        elif self.geometry == Geometry.SPHERICAL:
            w_left = 4.0*np.pi/3.0*(position**3 - self.z_edge[idx]**3)
        else:
            w_left = None
        return w_left

    def get_volume(self, position):
        """
        Calculate "volume", given position
        Args:
            position (float): Position (m)
        Return:
            float: Volume/Area/Length
        """
        if self.geometry == Geometry.PLANAR:
            vol =  position
        elif self.geometry == Geometry.SPHERICAL:
            vol = 4.0*np.pi/3.0*position**3
        else:
            vol = np.pi*position**2
        return vol

    @property
    def total_volume(self):
        """
        Calculate "volume", given position

        Return:
            float: System Volume/Area/Length
        """
        if self.geometry == Geometry.PLANAR:
            vol = self.domain_size
        elif self.geometry == Geometry.SPHERICAL:
            vol = 4.0*np.pi/3.0*self.domain_size**3
        else:
            vol = np.pi*self.domain_size**2
        return vol

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
        for i in range(1,n_grid+1):
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
        for i in range(2,n_grid):
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
        self.dr = self.domain_size / n_grid
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
        self.dr = self.domain_size / n_grid
        # Grid
        self.z = np.linspace(0.5*self.dr, domain_size - 0.5*self.dr, n_grid)
        # Setting the edge grid
        self.z_edge = np.linspace(0.0, self.domain_size, n_grid + 1)
        # Defining integration weights
        self.integration_weights = np.zeros(n_grid)
        for k in range(n_grid):
            self.integration_weights[k] = 4.0*np.pi/3.0*self.dr**3*(3*k**2 + 3*k + 1)

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
