import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann
from surfpack.grid import Grid, Geometry
from collections.abc import Iterable
import json
from surfpack.Convolver import convolve_ad

class Profile(np.ndarray):
    """
    Class for holding density profiles, Helmholtz energy densities etc.

    The idea is the following: We want to treat a profile (i.e. a vector of densities) as a numpy array for math purposes.
    However, a density profile is always tied to a discrete grid in space, which determines how convolutions are performed.

    So: Profile is a class that inherits from ndarray, such that Profile objects may be treated as ndarrays in calculations,
    but are also passed a Grid object upon initialisation, so that the spacial discretisation and geometry can be retrieved
    as needed, without needing to keep track of and pass around both the density, the grid and the geometry.

    Finally, a Profile can be a vector field (e.g. a vector weighted density). Therefore, we have the attribute
    is_vector_field, such that we can write generic code most places, and only check whether a profile is a vector field
    in the places where we need to handle that differently.

    Note: Some operations may implicitly lead to a Profile being converted to an ndarray, and losing the grid attribute.
        So be a bit careful


    """
    def __new__(cls, profile, grid=None, geometry=Geometry.PLANAR, domain_size=10, is_vector_field=False):
        """
        This is how you inherit from ndarray (using __new__ instead of __init__) see also: __array_finalize__

        Args:
            profile (Sized) : A list or array of densities
            grid (Grid) : The spacial grid associated with the densities in 'profile'
            geometry (Geometry) : Instead of passing an initialised Grid object, a Geometry and doman_size can be
                                passed instead, and the Grid object will be created.
            domain_size (int) : If no Grid is supplied, this is the number of gridpoints used
            is_vector_field (bool) : Whether or not this profile is a vector field.
        """
        obj = np.asarray(profile).view(cls)

        if grid is not None:
            obj.grid = grid
        else:
            obj.grid = Grid(len(profile), geometry, domain_size)

        obj.is_vector_field = is_vector_field

        return obj

    def __array_finalize__(self, obj):
        """
        This is how we inherit from ndarray
        """
        if obj is None: return
        self.grid = getattr(obj, 'grid', None)
        self.is_vector_field = getattr(obj, 'is_vector_field', None)

    def is_even(self):
        """Property
        Used in convolutions to determine what transforms to use
        """
        return not self.is_vector_field

    def is_odd(self):
        """Property
        Used in convolutions to determine what transforms to use
        """
        return not self.is_even()

    def __call__(self, z):
        """
        Get value of profile as a function of position, instead of by index. Uses linear interpolation between points.
        Args:
            z (float) : Position
        Returns:
            float : Value of profile
        """
        if isinstance(z, Iterable):
            return Profile([self(zi) for zi in z], Grid(len(z), self.grid.geometry, max(z) - min(z), domain_start=min(z)))

        if z <= self.grid.z[0]:
            return self[0]
        if z >= self.grid.z[-1]:
            return self[-1]

        left_idx = int((z - (self.grid.domain_start + (self.grid.dz / 2))) / self.grid.dz)
        p0 = self[left_idx]
        p1 = self[left_idx + 1]
        z0 = self.grid.z[left_idx]
        a = (p1 - p0) / self.grid.dz
        return a * (z - z0) + p0

    def on_grid(self, grid):
        """
        Create a copy this profile on a new grid. If the new grid spans a larger domain, the edge values of the Profile
        are used to fill out the grid. The Profile.__call__() method is used to get the value at each gridpoint on the
        new grid, such that when the new gridpoints do not align with the existing ones, linear interpolation between
        the two nearest gridpoints is used to compute the value at the new gridpoint.
        Can be useful to increase or decrease grid resolution, or extend or cut off the domain.

        Args:
            grid (Grid) : The new grid

        Returns:
            Profile : Copy of this profile interpolated onto the supplied grid
        """
        vals = [self(zi) for zi in grid.z]
        return Profile(vals, grid)

    @staticmethod
    def lst_on_grid(lst, grid):
        """
        See: Profile.on_grid
        This method does the same, but for a list[Profile]

        Args:
            lst (list[Profile]) : List of profiles to move to new grid
            grid (Grid) : The new grid

        Returns:
            list[Profile] : The new profiles
        """
        return [p.on_grid(grid) for p in lst]

    def integrate_cartesian(self, deg=1):
        """
        Integrates the profile using cartesian coordinates. To use with other geometries than Planar, the integral must
        first be transformed as appropriate (i.e. by multiplying with self.grid.z**2 for spherical geometry).

        Args:
            deg (int) : The degree of the interpolation polynomial to use.
        Returns:
            float : The integral of the Profile
        """
        if deg == 1:
            return float(np.trapz(self, dx=self.grid.dz))

        val = 0
        rest_points = self.grid.N % (deg + 1)
        for i in range(0, self.grid.N - deg, deg):
            a = self.grid.z[i]
            b = self.grid.z[i + deg]
            coeff = np.polyfit(self.grid.z[i: i + deg + 1], self[i: i + deg + 1], deg)
            for k in range(deg + 1):
                val += coeff[k] * (1 / (deg - k + 1)) * (b ** (deg - k + 1) - a ** (deg - k + 1))

        deg = rest_points
        a = self.grid.z[- (deg + 1)]
        b = self.grid.z[-1]

        coeff = np.polyfit(self.grid.z[- (deg + 1): self.grid.N], self[- (deg + 1): self.grid.N], deg)
        for k in range(deg + 1):
            val += coeff[k] * (1 / (deg - k + 1)) * (b ** (deg - k + 1) - a ** (deg - k + 1))

        return val


    def integrate(self, deg=1, error=False):
        if self.grid.geometry == Geometry.PLANAR:
            f = self
        elif self.grid.geometry == Geometry.SPHERICAL:
            f = 4 * np.pi * self * self.grid.z**2
        elif self.grid.geometry == Geometry.POLAR:
            raise NotImplementedError('Profile integration not implemented for Polar geometry!')
        else:
            raise ValueError(f'Geometry must be Planar, Spherical or Polar, but was {self.grid.geometry}.')

        I = f.integrate_cartesian(deg=deg)
        if error is True:
            err = integration_error(f, self.grid.dz, deg)
            return I, err
        return I

    def check_extends_to_bulk(self, left_side=True, right_side=True, bulk_width=0.05, change_tol=1e-3):
        z_start = self.grid.domain_start
        z_end = self.grid.domain_end
        dz = bulk_width * self.grid.L
        if left_side is True:
            if abs((self(z_start + dz) - self(z_start)) / (self[0])) > change_tol: return False
        if right_side is True:
            if abs((self(z_end) - self(z_end - dz)) / (self[-1])) > change_tol: return False
        return True


    def get_position_idx(self, z):
        if (z < self.grid.domain_start) or (z > self.grid.domain_end):
            raise ValueError(f'z ({z}) is outside the grid domain ({self.grid.domain_start}, {self.grid.domain_end})')
        return int((z - self.grid.domain_start) / self.grid.dz)

    def equimolar_interface_idx(self):
        return int(self.get_interface_position() / self.grid.dz)

    def equimolar_interface_position(self):
        bulk_1 = self[0]
        bulk_2 = self[-1]
        N = self.integrate()
        V = self.grid.volume()
        V1 = (N - V * bulk_2) / (bulk_1 - bulk_2)
        return self.grid.size(V1) # Handles differentiating between spherical and planar geometry

    def save(self, filename):
        """Utility
        Save the profile to <filename>, used in combination with load_dct(<filename>)
        """
        data = self.to_dict()
        json.dump(data, open(filename, 'w'), indent=4)

    def to_dict(self):
        """Utility
        Convert the Profile to a dict, used in combination with save-methods.
        """
        data = {'array': [self[i] for i in range(len(self))],
                'domain_size': self.grid.L,
                'geometry': self.grid.geometry,
                'is_vector_field': self.is_vector_field}
        return data

    @staticmethod
    def save_list(lst, filename):
        """Utility
        save a list[Profile] object to a json file, used in combination with load_file

        Args:
            lst (list[Profile]) : The profiles to save
            filename (str) : Write destination (does not check for overwrite)
        """
        data = {ci : lst[ci].to_dict() for ci in range(len(lst))}
        json.dump(data, open(filename, 'w'), indent=4)

    @staticmethod
    def load_dict(dct):
        """Construction
        Create a Profile from a dict created using to_dict
        """
        arr = dct['array']
        domain_size = dct['domain_size']
        geometry = dct['geometry']
        is_vector_field = dct['is_vector_field']
        return Profile(arr, geometry=geometry, domain_size=domain_size, is_vector_field=is_vector_field)

    @staticmethod
    def load_file(filename):
        """Construction
        Create a list[Profile] from a json file written using save_list
        """
        data = json.load(open(filename, 'r'))
        profiles = [None for _ in data.keys()]
        for ci, dct in data.items():
            profiles[int(ci)] = Profile.load_dict(dct)
        return profiles

    @staticmethod
    def step_function(grid, high_val, low_val=0):
        """
        Generates a profile that is a step function, like this

        high val -----
                     |
                     |
                     ------ low val

        Args:
            grid (Grid) : The Grid to use
            high_val (float) : High value
            low_val (float) : Low value
        """
        profile = np.ones(grid.N) * high_val
        profile[grid.N // 2 :] *= low_val / high_val
        return Profile(profile, grid)

    @staticmethod
    def tanh_profile(grid, left_val, right_val, width_factors=None, centre=None):
        """
        Generate a tanh density Profile, going from left_val to right_val

        Args:
            grid (Grid) : The grid used for all Profiles
            left_val (list[float]) : Value on the left side (minimum z in grid)
            right_val (list[float]) : Value on the right side (maximum z in grid)
            width_factors (list[float], optional) : Higher width_factor gives wider tanh profile
            centre (list[float], optional) : The positions to centre the tanh-profiles at
        Returns:
            list[Profile] : The tanh Profiles from left_val at min(grid.z) to right_val at max(grid.z)
        """
        if width_factors is None:
            width_factors = np.ones_like(left_val)
        if centre is None:
            centre = grid.L / 2

        profile_arr = [None for _ in range(len(left_val))]
        for i in range(len(profile_arr)):
            if np.tanh((max(grid.z) - centre) / width_factors[i]) - 1 > 1e-10:
                warnings.warn('Grid may be too narrow for tanh-Profile.', RuntimeWarning, stacklevel=2)

            profile_arr[i] = np.tanh((grid.z - centre) / width_factors[i]) / np.tanh(max(abs((grid.z - centre) / width_factors[i]))) # profile from -1 to 1
            profile_arr[i] += 1 # profile from 0 to 2
            profile_arr[i] /= 2 # profile from 0 to 1
            profile_arr[i] *= (right_val[i] - left_val[i]) # profile from 0 to (right_val - left_val)
            profile_arr[i] += left_val[i] # profile from left_val to right_val
            profile_arr[i] = Profile(profile_arr[i], grid)

        return profile_arr

    @staticmethod
    def from_potential(rho_bulk, T, grid, Vext, w3=None):
        """
        Generates the equilibrium profile for an ideal gas in the external potential Vext.
        Not tested for multicomponent i think

        Note: Potential is multiplied by 0.01, this is to prevent attractive potentials from leading to too high
        initial densities.

        Args:
            rho_bulk (list[float]) : bulk densities
            T (float) : Temperature
            grid (Grid) : The grid to be associated with the profile
            Vext (list[callable]) : The external potential
            w3 (list[callable], optional) : The weight functions for n_3 (FMT weighted density). Used to prevent too high inital densities.
        Returns:
            list[Profile] : The ideal gas density Profile for each species.
        """
        beta = 1 / (Boltzmann * T)

        if not isinstance(Vext, Iterable):
            Vext = [Vext for _ in rho_bulk]

        profile_arr = [Profile.zeros(grid) for _ in rho_bulk]
        for i, rb in enumerate(rho_bulk):
            profile_arr[i] = rb * np.exp(- beta * Vext[i](grid.z))

        if w3 is not None:
            n3 = np.zeros_like(profile_arr[0])
            for i in range(len(rho_bulk)):
                n3 += convolve_ad(w3[i], Profile(profile_arr[i], grid))
        else:
            for i in range(len(profile_arr)):
                profile_arr[i] = Profile(profile_arr[i], grid)
            return profile_arr

        ncomps = len(profile_arr)
        for ri in range(len(profile_arr[0])):
            if n3[ri] >= 0.99:
                nt = sum([p[ri] for p in profile_arr])
                if nt < 1e-4:
                    continue
                x = [p[ri] / nt for p in profile_arr]
                rho_t = 0.99 / sum([x[i] * w3[i].real_integral() for i in range(ncomps)])

                rho_ri = [rho_t * x[i] for i in range(ncomps)]
                for i in range(len(profile_arr)):
                    profile_arr[i][ri] = rho_ri[i]

        for i in range(len(profile_arr)):
            profile_arr[i] = Profile(profile_arr[i], grid)

        return profile_arr

    @staticmethod
    def bulk(bulk_value):
        """
        Implemented at a point in time when I wanted to compute bulk properties by using a profile with a constant
        value. Instead, the functionals are now implemented to handle bulk computations by being passed a single
        density, and the bulk=True option.
        """
        grid = Grid(10, Geometry.PLANAR, 10)
        profile = np.ones(10) * bulk_value
        return Profile(profile, grid)


    @staticmethod
    def zeros(grid, nprofiles=1):
        """
        Quickly allocate a profile or list of profiles (kind of like np.zeros)

        Args:
            grid (Grid) : The Grid to be associated with all profiles
            nprofiles (int) : The number of profiles to allocate

        Returns:
            Profile : If nprofiles=1
            list[Profile] : If nprofiles > 1.
        """
        if nprofiles == 1:
            return Profile(np.zeros(grid.N), grid)
        else:
            return [Profile(np.zeros(grid.N), grid) for _ in range(nprofiles)]


    @staticmethod
    def zeros_like(profile_arr):
        """
        Kind of like np.zeros_like

        Args:
            profile_arr (list[Profile]) : The template

        Returns:
            list[Profile] : The same number of profiles as in profile_arr, using the same grid.
        """
        arr = [Profile.zeros(profile_arr[0].grid) for _ in profile_arr]
        return arr

def integration_error(f, h, deg):
    """
    Standard equations for computing integration error on equidistant grids.

    Args:
        f (Iterable) : The evaluated function
        h (float) : The grid spacing
        deg (int) : The polynomial degree used for the quadrature rule (1 = Trapezoidal, 2 = Simpsons, etc.)
    Returns:
        float : The (absolute) maximum error, computed using numerical derivatives.
    """
    rest_points = len(f) % (deg + 1)
    if deg == 1:
        d2fdz2 = (np.roll(f, -1)[1: -1] - 2 * f[1: -1] + np.roll(f, + 1)[1: -1]) / (h ** 2)
        err = 0

        for i in range(0, len(d2fdz2) - 2):
            err += h ** 3 * max(abs(d2fdz2[i : i + 2])) / 12

        return err

    if deg == 2:
        fm1 = np.roll(f, -1)[2:-2]
        fm2 = np.roll(f, -2)[2:-2]
        fp1 = np.roll(f, +1)[2:-2]
        fp2 = np.roll(f, +2)[2:-2]
        d4fdz4 = (fm2 + fp2 - 4 * (fm1 + fp1) + 6 * f[2 : -2]) / h**4
        err = 0
        for i in range(0, len(d4fdz4) - 3):
            err += abs(h**5 * max(abs(d4fdz4[i : i + 3])) / 90)

        rest = integration_error(f[- (rest_points + 2):], h, rest_points)
        return err + rest

    warnings.warn(f'Error estimate not implemented for degree {deg} (> 2). Returning NaN.', RuntimeWarning, stacklevel=2)
    return np.nan