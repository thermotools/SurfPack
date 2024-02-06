"""
Here we find one of two fundemental structures in the DFT-code: The Grid.

The Grid is "Dumb" in the sense that it does not perform calculations. It only holds information the Geometry (symmetry)
of the domain, as well as the real- and fourier-space gridpoints for the discretisation.

HOWEVER: The *whole point* of the Grid structure is that other code can be Geometry and domain-size agnostic.
        That means: If you find yourself writing
            if grid.geometry == Geometry.PLANAR:
                ...
            elif grid.geometry == Geometry.SPHERICAL:
                ...
        You are likely doing something wrong. Whatever you are trying to do can probably be handled by calling a method
        in the grid, that does whatever you want to do, and correctly handles the cases for different geometries.
        A nice example is computing the volume of a grid or subgrid. We do this using the Grid.volume() method, not
        by checking the geometry and treating the different cases every time we need the volume.
"""
from enum import IntEnum
import numpy as np
from scipy.optimize import root_scalar


class Geometry(IntEnum):
    PLANAR = 1
    POLAR = 2
    SPHERICAL = 3


class Grid:
    """
    Spacial discretisation, determined by a length (domain_size), number of gridpoints, and geometry.

    Computes the corresponding fourier-space grids for cosine- and sine transforms upon initialisation.
    """

    def __init__(self, n_grid, geometry, domain_size, domain_start=0):
        self.L = domain_size
        self.domain_start = domain_start
        self.domain_end = domain_start + domain_size
        self.N = n_grid
        self.geometry = geometry

        self.dz = domain_size / n_grid
        self.z = np.linspace(self.domain_start + self.dz/2, self.domain_end - self.dz/2, n_grid)

        if geometry == Geometry.PLANAR:
            self.k_cos = np.linspace(0.0, self.N - 1, self.N) / (2 * self.L)
            self.k_sin = np.linspace(1.0, self.N, self.N) / (2 * self.L)
        elif geometry == Geometry.SPHERICAL:
            self.k_cos = np.linspace(0, self.N - 1, self.N) / (2 * self.L)
            self.k_sin = np.linspace(1, n_grid, n_grid) / (2 * self.L)
        else:
            raise NotImplementedError('Grid is not implemented for Geometry : ' + str(geometry))

    @staticmethod
    def tanh_grid(n_grid, geometry, width_factors, eps=1e-3):
        def rootfunc(L, w):
            k = L / (2 * w)
            A = np.tanh(k)
            val = (2 / (np.exp(k) + np.exp(-k))) * A / w
            return val

        L_lst = np.empty_like(width_factors)
        for i, wi in enumerate(width_factors):
            Li = 10.
            while rootfunc(Li, wi) > eps:
                Li += 5.
            L_lst[i] = Li

        return Grid(n_grid, geometry, max(L_lst))

    def volume(self, z=None):
        """
        Compute the volume of the grid, if a position is supplied, use that position as the endpoint of the domain
        Useful to be able to treat volumes without thinking about geometry all the time

        Args:
            z (float, optional) : If supplied, compute the volume of the grid bounded by r = z.

        Returns:
            float : The volume
        """
        if self.geometry == Geometry.PLANAR:
            return self.L if z is None else z
        elif self.geometry == Geometry.SPHERICAL:
            return (4 / 3) * np.pi * ((self.L if z is None else z)**3 - self.domain_start**3)

        raise NotImplementedError(f'Grid volume not implemented for geometry {self.geometry}')

    def area(self, z):
        """
        Compute the area of a surface dividing the domain at position z.

        Args:
            z (float) : Position of the dividing surface

        Returns:
            float : The area of the dividing surface.
        """
        if self.geometry == Geometry.PLANAR:
            return 1
        elif self.geometry == Geometry.SPHERICAL:
            return 4 * np.pi * z**2
        raise NotImplementedError(f'Grid area not implemented for geometry {self.geometry}')

    def size(self, V):
        """
        Compute the size (radius or length) of a grid with a given volume (inverse of Grid.volume)

        Args:
            V (float) : The volume

        Returns:
            float : The radius or length of the grid with the supplied volume
        """
        if self.geometry == Geometry.PLANAR:
            return V
        elif self.geometry == Geometry.SPHERICAL:
            return (3 * V / (4 * np.pi))**(1 / 3)
        else:
            raise NotImplementedError(f'Grid volume not implemented for geometry {self.geometry}')

    def set_center(self, z0):
        """Utility
        Shift the start and end of the domain, while maintaining the domain width, such that `z0` is at the center of the domain.

        Args:
            z0 (float) : New center of domain
        """
        self.domain_start = z0 - self.L / 2
        self.domain_end = z0 + self.L / 2
        self.z = np.linspace(self.domain_start + self.dz / 2, self.domain_end - self.dz / 2, self.N)

    def set_domain_start(self, start):
        """Utility
        Set the start of the domain, while maintaining the domain width.

        Args:
            start (float) : New start of domain.
        """
        self.domain_start = start
        self.domain_end = start + self.L
        self.z = np.linspace(self.domain_start + self.dz / 2, self.domain_end - self.dz / 2, self.N)

    def reset_zero(self, z0):
        self.domain_start -= z0
        self.domain_end -= z0
        self.z -= z0

    def get_center(self):
        return (self.domain_end - self.domain_start) / 2

    def __getitem__(self, item):
        """Utility
        Grids can be sliced, not accessed by index. Slicing a Grid will return a new Grid with the same geometry, but with
        the number of points, and domain start/end/width adjusted as if you were slicing the Grid.z attribute. That is:
        ```
        all(grid.z[start : stop] == grid[start : stop].z) # True
        (grid.z[stop - 1] - grid.z[start]) == grid[start : stop].L # True
        len(grid.z[start : stop]) == grid[start : stop].N # True
        grid.dz == grid[start : stop].dz # True
        ```
        """
        if isinstance(item, slice):
            if item.start is None:
                start = 0
            elif item.start < 0:
                start = self.N + item.start
            else:
                start = item.start

            if item.stop is None:
                stop = self.N
            elif item.stop < 0:
                stop = self.N + item.stop
            else:
                stop = item.stop

            n_grid = stop - start
            domain_size = self.dz * n_grid

            return Grid(n_grid, self.geometry, domain_size)

        raise TypeError('Grids can be sliced, not accessed by index!')

    def __hash__(self):
        return hash((self.L, self.N, self.geometry))

    def __repr__(self):
        return f'Grid with L : {self.L}, N : {self.N}, geometry : {self.geometry}, domain_start : {self.domain_start}'

class PlanarGrid(Grid):

    def __init__(self, n_grid, domain_size, domain_start=0):
        super().__init__(n_grid, Geometry.PLANAR, domain_size, domain_start=domain_start)

class SphericalGrid(Grid):

    def __init__(self, n_grid, domain_size, domain_start=0):
        super().__init__(n_grid, Geometry.SPHERICAL, domain_size, domain_start=domain_start)

class GridSpec:
    def __init__(self, n_grid, geometry, domain_start=0, tanh_eps=1e-3):
        self.n_grid = n_grid
        self.geometry = geometry
        self.tanh_eps = tanh_eps

    def __repr__(self):
        return f'GridSpec with n_grid : {self.n_grid}, Geometry : {self.geometry}, tanh_eps : {self.tanh_eps}'

    @staticmethod
    def Planar(n_grid, domain_start=0, tanh_eps=1e-3):
        return GridSpec(n_grid, Geometry.PLANAR, domain_start=domain_start, tanh_eps=tanh_eps)

    @staticmethod
    def Spherical(n_grid, domain_start=0, tanh_eps=1e-3):
        return GridSpec(n_grid, Geometry.SPHERICAL, domain_start=domain_start, tanh_eps=tanh_eps)