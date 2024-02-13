import abc
import copy
import os.path
import shutil
import warnings
import hashlib
from functools import wraps
import matplotlib.pyplot as plt
from surfpack.WeightFunction import get_FMT_weights
from surfpack.Convolver import convolve_ad
from surfpack.grid import Grid, GridSpec, Geometry
from surfpack.profile import Profile
from surfpack.external_potential import ExpandedSoft
from surfpack.solvers import picard_rhoT, anderson_rhoT, solve_sequential, picard_NT, anderson_NT, picard_muT, anderson_muT, solve_sequential_NT, solve_sequential_muT, solve_sequential_rhoT, SequentialSolver, GridRefiner
import numpy as np
from scipy.constants import Boltzmann, Avogadro, gas_constant
from collections.abc import Iterable
from enum import IntEnum

class PropertyFlag(IntEnum):
    total = 0
    residual = 1
    ideal = 2

def profilecaching(func):
    """Utility
    Decorator used to save computed density profiles. Used for most or all methods that
    evaluate density profiles at different conditions. By default, models will not save their
    output or do lookups, but if a save-directory and/or load-directory have been set on the
    model using `model.set_save_dir()`, `model.set_load_dir()` or `model.set_cache_dir()` these
    will automatically be used by methods decorated with this decorator.

    The file names are generated from a SHA-256 hash that guarantees a unique file name for every
    input and model, so different models can safely use the same save/load directory.

    The procedure automatically generates the file `000_REGISTER.txt` where the information
    used to generate each hash is stored.

    Note: For a file to successfully load, the input must be *exactly* the same, which means that
    this is mostly suited for scripts that are run several times with the same values, not so much
    for extracting parameters later. Some methods to parse `000_REGISTER.txt` in order to extract
    profiles may be useful, but for now you have to manually manage saving if you want to easily
    extract computed profiles later.
    """
    @wraps(func)
    def density_profile(*args, **kwargs):
        self = args[0]
        load_dir = self.get_load_dir()
        save_dir = self.get_save_dir()
        if (load_dir is not None) or (save_dir is not None):
            file_id = f"{self.get_caching_id()}\n" \
                      f"Func : {func.__name__}\n" \
                      f"Args : {', '.join([str(a) for a in args[1:]])}\n"
            if 'Vext' in kwargs.keys(): file_id += 'Vext = ' + str(kwargs['Vext'])
            string_bytes = file_id.encode('utf-8')
            sha256_hash = hashlib.sha256(string_bytes)
            filename = str(sha256_hash.hexdigest())
            loadpath = f'{load_dir}/{filename}.json'
            savepath = f'{save_dir}/{filename}.json'
        try:
            if load_dir is None: raise FileNotFoundError
            rho = Profile.load_file(loadpath)
        except FileNotFoundError:
            rho = func(*args, **kwargs)

        if save_dir is not None:
            Profile.save_list(rho, savepath)
            with open(f'{save_dir}/000_REGISTER.txt', 'a') as file:
                file.write(f'{filename}\n')
                for line in file_id.split('\n'):
                    file.write(f'\t{line}\n')
                file.write(('#' * 100) + '\n\n')
        return rho
    return density_profile

class Functional(metaclass=abc.ABCMeta):

    TWOPH = 0
    LIQPH = 1
    VAPPH = 2
    MINGIBBSPH = 3
    SINGLEPH = 4
    SOLIDPH = 5
    FAKEPH = 6

    def __init__(self, ncomps):
        """Internal
        Handles initialisation that is common for all functionals

        Args:
            ncomps (int) : Number of components
        """
        self.ncomps = ncomps
        self.weights = []
        if not hasattr(self, 'eos'):
            self.eos = None

        if not hasattr(self, 'pair_potentials'):
            self.pair_potentials = None

        self.computed_density_profiles = {}
        self.__save_dir__ = None
        self.__load_dir__ = None

    @abc.abstractmethod
    def get_caching_id(self):
        """Utility
        Returns a unique ID for an initialized model. Should include information about the model type, components,
        parameters and mixing parameters (if applicable). Used for caching profiles.

        Returns:
            str : A model identifier
        """
        pass

    def set_save_dir(self, save_dir):
        """Utility
        Sets this model to automatically save computed density profiles in the directory 'save_dir'.

        Args:
            save_dir (str) : Name of directory to save to

        Raises:
            FileExistsError : If save_dir is the name of an existing file that is not a directory.
        """
        self.__save_dir__ = save_dir
        if os.path.exists(self.__save_dir__) and not os.path.isdir(self.__save_dir__):
            raise FileExistsError(f"The file at {self.__save_dir__} exists, and is not a directory.")
        if not os.path.isdir(self.__save_dir__):
            os.makedirs(self.__save_dir__)

    def get_save_dir(self):
        """Utility
        Get the current directory used to save Profiles

        Returns:
            str : Path to the current save directory
        """
        return self.__save_dir__

    def set_load_dir(self, load_dir):
        """Utility
        Sets this model to automatically search for computed profiles in `load_dir`.

        Args:
            load_dir (str) : Name of directory to load files from.

        Raises:
            NotADirectoryError : If load_dir does not exist.
        """
        self.__load_dir__ = load_dir
        if not os.path.isdir(self.__load_dir__):
            raise NotADirectoryError(f"Load directory {self.__load_dir__} does not exist.")

    def get_load_dir(self):
        """Utility
        Get the current directory used to search and load Profiles

        Returns:
            str : Path to the current load directory
        """
        return self.__load_dir__

    def set_cache_dir(self, cache_dir):
        """Utility
        Forwards call to `self.set_load_dir` and `self.set_save_dir`.

        Args:
            cache_dir (str) : Name of directory save and load files in.

        Raises:
            FileExistsError : If cache_dir is the name of an existing file that is not a directory.
            NotADirectoryError : If cache_dir does not exist after successfull call to `self.set_save_dir`
        """
        self.set_save_dir(cache_dir)
        self.set_load_dir(cache_dir)

    def clear_cache_dir(self, clear_dir):
        """Utility
        Clear the directory `clear_dir`, after prompting for confirmation.

        Args:
            clear_dir (str) : The name of the directory.

        Raises:
            NotADirectoryError : If clear_dir does not exist or is not a directory.
        """
        if not os.path.isdir(clear_dir):
            raise NotADirectoryError(f"Directory {clear_dir} does not exist or is not a directory.")
        confirm = input(f'This action will permanently delete the contents of the directory {clear_dir}, are you sure? (Y/n)')
        if confirm != 'Y':
            return
        shutil.rmtree(clear_dir)
        if self.__save_dir__ == clear_dir:
            os.makedirs(self.__save_dir__)

    def validate_composition(self, z):
        """Internal
        Check that the composition `z` has length equal to number of components, and sums to one.

        Args:
            z (Iterable(float)) : The composition

        Raises:
            IndexError : If number of fractions does not match number of components.
            ValueError : If fractions do not sum to unity.
        """
        if len(z) != self.ncomps:
            raise IndexError(f'Number of mole fractions ({len(z)} did not match number of components ({self.ncomps}).')
        elif abs(sum(z) - 1) > 1e-12:
            raise ValueError(f'Mole fractions did not sum to unity but to {sum(z)}.')

    @abc.abstractmethod
    def __repr__(self):
        """Internal
        All Functionals must implement a unique `__repr__`, as these are used to generate the hashes that are used when saving
        profiles. The `__repr__` should contain a human-readable text with (at least) the name of the model and all model parameters.
        """
        pass

    @abc.abstractmethod
    def reduced_helmholtz_energy_density(self, *args, **kwargs):
        r"""Profile Property
        Returns the reduced, residual helmholtz energy density, $\phi$ (i.e. the integrand of eq. 3.4 in "introduction to DFT")

        $$\phi = \\frac{a^{res}}{k_B T} $$

        where $a^{res}$ is the residual helmholtz energy density (per particle), in [J Å$^{-3}$]
        """
        pass

    @abc.abstractmethod
    def get_characteristic_lengths(self):
        """Utility
        Used to generate initial guesses for density profiles. Should return lengths that give an indication of molecular
        sizes. For example diameters of hard-spheres, or the Barker-Henderson diameter.

        Returns:
             ndarray(float) : The characteristic length of the molecules.
        """
        pass

    @abc.abstractmethod
    def get_weights(self, T):
        """Weights
        Returns the weights for weighted densities in a 2D array, ordered as
        weight[<weight idx>][<component idx>]. See arrayshapes.md for a description of the ordering of different arrays.
        """
        pass

    @abc.abstractmethod
    def get_weighted_densities(self, *args, **kwargs):
        """Weighted density
        Compute the weighted densities, and optionally differentials

        Args:
            rho (ndarray[Profile]) : 2D array of component densities indexed as rho[<component index>][<position index>]
            bulk (bool) : If True, use simplified expressions for bulk - not requiring FFT
            dndrho (bool) : Flag to activate calculation of differential

        Returns:
            ndarray : array of weighted densities indexed as n[<weight index>][<position index>]
            ndarray : array of differentials
        """
        pass

    def reduce_temperature(self, T, c=0):
        """Utility
        Reduce the temperature in some meaningful manner, using LJ units when possible, doing nothing for hard spheres.

        Args:
            T (float) : The temperature
            c (float, optional) : ???
        Returns:
            float : The reduced temperature
        """
        pass

    def find_non_bulk_idx(self, profile, rho_b, tol=0.01, start_idx=-1):
        """Deprecated
        Possibly no longer working, used to find the the first index in a Profile which is no longer considered a point in
        the bulk phase, by checking the change in density and comparing to `tol`.

        Args:
            profile (Profile) : The density profile
            rho_b (float) : The bulk density
            tol (float) : Tolerance for relative change in density within bulk.
            start_idx (int) : Where to start searching. If start_idx < 0, search goes "right-to-left", otherwise "left-to-right".

        Returns:
             int : The index at which the bulk ends.
        """
        if abs((profile[start_idx] - rho_b) / rho_b) < tol:
            in_bulk = True
        else:
            in_bulk = False

        for i in range(len(profile)):
            if start_idx == -1:
                i = start_idx - i

            if (abs((profile[i] - rho_b) / rho_b) < tol) != in_bulk:
                return i

        warnings.warn(f'Did not find transition between bulk and non-bulk for\n'
                        f'bulk density : {rho_b}, tolerance : {tol}\n'
                        f'Profile edges are: {profile[0]}, {profile[-1]}', RuntimeWarning, stacklevel=2)

        return 0

    def adsorbed_thickness(self, profile):
        """Deprecated
        Find the thickness of an adsorbed film.

        Args:
            profile (Profile) : The density profile

        Returns:
            float : The thickness of the adsorbed layer.
        """
        idx = self.find_non_bulk_idx(profile, profile[-1])
        return profile.grid.z[idx]

    def adsorbed_mean_density(self, profile):
        """Deprecated
        Compute the mean density of an adsorbed film.

        Args:
            profile (Profile) : The density profile

        Returns:
            float : The mean density.
        """
        idx = self.find_non_bulk_idx(profile, profile[-1])
        return np.mean(profile[:idx])

    def interface_thickness(self, profile, positions=False):
        """Deprecated
        Find the thickness of an interface.

        Args:
            profile (Profile) : The density profile
            positions (bool) : Also return the position where the interface starts and stops.
        Returns:
            float : The thickness of the interface.
        """
        idx_l = self.find_non_bulk_idx(profile, profile[0], start_idx=0)
        idx_r = self.find_non_bulk_idx(profile, profile[-1])
        if positions is True:
            return profile.grid.z[idx_r] - profile.grid.z[idx_l], (profile.grid.z[idx_l], profile.grid.z[idx_r])
        return profile.grid.z[idx_r] - profile.grid.z[idx_l]

    def residual_chemical_potential(self, rho, T, bulk=True):
        """Bulk Property
        Compute the residual chemical potential [J]

        Args:
            rho (list[float]) : Density [particles / Å^3]
            T (float) : Temperature [K]
            bulk (bool) : Only True is implemented

        Returns:
            1d array (float) : The chemical potentials [J / particle]
        """
        if bulk is True:
            _, dphidn = self.reduced_helmholtz_energy_density(rho, T, dphidn=True, bulk=True, asarray=True)
            _, dndrho = self.get_weighted_densities(rho, T, dndrho=True, bulk=True)
            beta_mu = np.zeros(self.ncomps) # First compute mu / (k_B T)
            for i in range(self.ncomps):
                for a in range(len(dphidn)):
                    beta_mu[i] += dphidn[a] * dndrho[a][i]
        else:
            raise NotImplementedError('Chemical potential only implemented for bulk!')
        beta = 1 / (Boltzmann * T)
        return beta_mu / beta # Then multiply by (k_B T)

    def chemical_potential(self, rho, T, bulk=True, property_flag='IR'):
        """Bulk Property
        Compute the chemical potential [J]

        Args:
            rho (list[float]) : Density [particles / Å^3]
            T (float) : Temperature [K]
            bulk (bool) : Only True is implemented
            property_flag (str, optional) : 'I' for ideal, 'R' for residual, 'IR' for total.

        Returns:
            1d array (float) : The chemical potentials [J / particle]
        """
        if property_flag not in ('I', 'R', 'IR'):
            raise KeyError("Invalid property flag! Valid flags are 'I' (Ideal), 'R' (Residual), 'IR' (total)")
        mu_res = self.residual_chemical_potential(rho, T, bulk=bulk) if ('R' in property_flag) else 0
        if 'I' in property_flag:
            debroglie = np.array([self.eos.de_broglie_wavelength(i + 1, T) * 1e10 for i in range(self.ncomps)])
            mu_id = Boltzmann * T * np.log((debroglie ** 3) * rho)
        else:
            mu_id = 0

        return mu_id + mu_res

    def residual_helmholtz_energy_density(self, rho, T, bulk=False):
        """Profile Property
        Compute the residual Helmholtz energy density [J / Å^3]

        Args:
            rho (list[Profile] or 1d array [float]) : The density profile or bulk density for each component [particles / Å^3]
            T (float) : Temperature [K]
            bulk (bool) : Whether to compute for bulk phase
        Returns:
            Profile or float: The residual Helmholtz energy density [J / Å^3]
        """
        phi = self.reduced_helmholtz_energy_density(rho, T, bulk=bulk)
        return Boltzmann * T * phi

    def residual_entropy_density(self, rho, T):
        """Profile Property
        Compute the residual entropy density [J / Å^3 K]

        Args:
            rho (list[Profile]) : The density profile for each component [particles / Å^3]
            T (float) : Temperature [K]
        Returns:
            Profile : The residual entropy density [J / Å^3 K]
        """
        phi, dphidT = self.reduced_helmholtz_energy_density(rho, T, dphidT=True)
        return - Boltzmann * (phi + T * dphidT)

    def residual_internal_energy_density(self, rho, T):
        """Profile Property
        Compute the residual internal energy density [J / Å^3]

        Args:
            rho (list[Profile]) : The density profile for each component [particles / Å^3]
            T (float) : Temperature [K]
        Returns:
            Profile : The residual internal energy density [J / Å^3]
        """
        a = self.residual_helmholtz_energy_density(rho, T)
        s = self.residual_entropy_density(rho, T)
        return a + T * s

    def residual_enthalpy_density(self, rho, T):
        """Profile Property
        Compute the residual enthalpy density [J / Å^3]

        Args:
            rho (list[Profile]) : The density profile for each component [particles / Å^3]
            T (float) : Temperature [K]
        Returns:
            Profile : The residual enthalpy density [J / Å^3]
        """
        V = 1
        u_res = self.residual_internal_energy_density(rho, T)
        h_res = Profile.zeros(rho[0].grid)
        for ri in range(len(rho[0])):
            n = [rho[ci][ri] * 1e30 / Avogadro for ci in range(len(rho))]
            p_res, = self.eos.pressure_tv(T, V, n, property_flag='R') # In Pascal from thermopack
            h_res[ri] = u_res[ri] + p_res * 1e30
        return h_res

    def fugacity(self, rho, T, Vext=None):
        """Bulk Property
        Compute the fugacity at given density and temperature

        Args:
            rho (list[float]) : Particle density of each species
            T (flaot) : Temperature [K]
            Vext (ExternalPotential, optional) : External potential for each particle

        Returns:
            1d array (flaot) : The fugacity of each species.
        """
        Vext = self.sanitize_Vext(Vext)
        c = self.correlation(rho, T)
        beta = 1 / (Boltzmann * T)
        f = np.empty(self.ncomps)
        grid = rho[0].grid
        for i in range(self.ncomps):
            f[i] = rho[i].integrate() / Profile(np.exp(c[i] - beta * Vext[i](grid.z)), grid).integrate()
        return f / beta

    def correlation(self, rho, T):
        """Profile Property
        Compute the one-body correlation function (sec. 4.2.3 in "introduction to DFT")

        Args:
            rho (list[Profile]) : Density profiles [particles / Å^3]
            T (float) : Temperature [K]

        Returns:
            list[Profile] : One body correlation function of each species as a function of position,
                            indexed as c[<component idx>][<grid idx>]
        """
        _, dphidn = self.reduced_helmholtz_energy_density(rho, T, dphidn=True)

        c = np.zeros((self.ncomps, rho[0].grid.N))
        weights = self.get_weights(T)
        for wi, comp_weights in enumerate(weights): # Iterate over weights
            for ci, w in enumerate(comp_weights): # Iterate over components
                if w == 0:
                    continue
                c[ci] -= convolve_ad(w, dphidn[wi]) * (-1 if w.is_odd() else 1)

        return c

    def grand_potential(self, rho, T, Vext=None, bulk=False, property_flag='IR'):
        """Profile Property
        Compute the Grand Potential, as defined in sec. 2.7 of R. Roth - Introduction to Density Functional Theory of
        Classical Systems: Theory and Applications.

        Args:
            rho (list[Profile] or Iterable[float]) : The density profile for each component,
                                                    or the bulk density of each component [particles / Å^3]
            T (float) : Temperature [K]
            Vext (list[callable] or callable) : External potential for each component, it is recomended to use the
                                                callables inherriting ExtPotential in external_potential.py. Defaults to None.
            bulk (bool) : Is this a bulk computation? Defaults to False. Note: For bulk computations, the grand
                            potential density [J / Å^3] is returned.
            property_flag (str) : Return Residual ('R'), Ideal ('I') or total ('IR') grand potential
        Returns:
            float : The Grand Potential [J]. NOTE: For bulk computations, the grand potential
                    density (which is equal to the bulk pressure) [J / Å^3] is returned.
        """
        if (bulk is True) and (Vext is not None):
            raise ValueError('Vext must be None for bulk!')
        Vext = self.sanitize_Vext(Vext)

        A_res = self.residual_helmholtz_energy_density(rho, T, bulk=bulk)

        if bulk is True:
            Vext = np.zeros(self.ncomps)
            mu = self.residual_chemical_potential(rho, T) + Boltzmann * T * np.log(rho)
        else:
            Vext = [Vext[i](rho[i].grid.z) for i in range(self.ncomps)]
            rho_b = [r[0] for r in rho]
            mu = self.residual_chemical_potential(rho_b, T) + Boltzmann * T * np.log(rho_b)

        omega_id = 0 if bulk else Profile.zeros(rho[0].grid)
        omega_res = A_res
        for i in range(self.ncomps):
            omega_res += rho[i] * (Vext[i] - mu[i] + Boltzmann * T * np.log(rho[i]))
            omega_id -= rho[i] * Boltzmann * T

        if bulk is False:
            omega_id = omega_id.integrate()
            omega_res = omega_res.integrate()

        if property_flag == 'IR':
            return omega_id + omega_res
        elif property_flag == 'R':
            return omega_res
        elif property_flag == 'I':
            return omega_id
        raise KeyError(f"Invalid property flag {property_flag}, valid flags are 'IR' (total), 'I' (ideal), and 'R' (residual)")

    def grand_potential_density(self, rho, T, Vext=None, property_flag='IR'):
        """Profile Property
        Compute the Grand Potential density.

        Args:
            rho (list[Profile] or Iterable[float]) : The density profile for each component,
                                                    or the bulk density of each component [particles / Å^3]
            T (float) : Temperature [K]
            Vext (list[callable] or callable) : External potential for each component, it is recomended to use the
                                                callables inherriting ExtPotential in external_potential.py. Defaults to None.
            property_flag (str) : Return Residual ('R'), Ideal ('I') or total ('IR') grand potential density
        Returns
            Profile : The Grand Potential density [J / Å^3].
        """

        Vext = self.sanitize_Vext(Vext)
        A_res = self.residual_helmholtz_energy_density(rho, T)
        Vext = [Vext[i](rho[i].grid.z) for i in range(self.ncomps)]
        rho_b = [r[0] for r in rho]
        mu = self.residual_chemical_potential(rho_b, T) + Boltzmann * T * np.log(rho_b)

        omega_id = Profile.zeros(rho[0].grid)
        omega_res = A_res
        for i in range(self.ncomps):
            omega_res += rho[i] * (Vext[i] - mu[i] + Boltzmann * T * np.log(rho[i]))
            omega_id -= rho[i] * Boltzmann * T

        if property_flag == 'IR':
            return omega_id + omega_res
        elif property_flag == 'R':
            return omega_res
        elif property_flag == 'I':
            return omega_id
        raise KeyError(f"Invalid property flag {property_flag}, valid flags are 'IR' (total), 'I' (ideal), and 'R' (residual)")


    def dividing_surface_position(self, rho, T=None, dividing_surface='e'):
        """Utility
        Compute the position of a dividing surface on a given density profile.

        Args:
            rho (list[Profile]) : The density profile for each species
            T (float) : The temperature
            dividing_surface (str, optional) : Can be '(e)quimolar' (default) or '(t)ension'.

        Returns:
            float : Position of the dividing surface.
        """
        if dividing_surface in ('e', 'equimolar'):
            return self.equimolar_surface_position(rho)
        elif dividing_surface in ('t', 'tension'):
            return self.surface_of_tension_position(rho, T)

        raise KeyError(f"Invalid dividing surface identifier '{dividing_surface}'!\n"
                       f"Valid identifiers are ('e' / 'equimolar') or ('t' / 'tension').")

    @staticmethod
    def equimolar_surface_position(rho):
        """Utility
        Calculate the position of the equimolar surface for a given density profile

        Args:
            rho (list[Profile]) : The density profile for each component [1 / Å^3]

        Returns:
            float : The position of the equimolar surface [Å]
        """
        rho_b1 = sum(r[0] for r in rho)
        rho_b2 = sum(r[-1] for r in rho)
        N = sum([r.integrate() for r in rho])
        V = rho[0].grid.volume()
        V1 = (N - V * rho_b2) / (rho_b1 - rho_b2)
        return rho[0].grid.size(V1) # Handles differetiating between spherical and planar geometry

    def surface_of_tension_position(self, rho, T):
        """Utility
        Calculate the position of the surface of tension on a given density profile

        Args:
            rho (list[Profile]) : The density profile for each component [1 / Å^3]
            T (float) : Temperature [K]

        Returns:
            float : The position of the surface of tension [Å]
        """
        if rho[0].grid.geometry == Geometry.PLANAR:
            return self.equimolar_surface_position(rho)
        rho_b1 = [r[0] for r in rho]
        rho_b2 = [r[-1] for r in rho]
        omega = self.grand_potential(rho, T)
        p_1 = - self.grand_potential(rho_b1, T, bulk=True)
        p_2 = - self.grand_potential(rho_b2, T, bulk=True)
        V = rho[0].grid.volume()
        delta_omega = omega + p_2 * V
        delta_p = p_1 - p_2
        return (3 * delta_omega / (2 * np.pi * delta_p))**(1 / 3)

    def tolmann_length(self, rho, T):
        """Profile Property
        Compute the Tolmann length, given a density profile.

        Args:
            rho (list[Profile]) : The density profile of each species.
            T (float) : Temperature [K]

        Returns:
            float : The Tolmann lenth
        """
        R_eq = self.equimolar_surface_position(rho)
        R_s = self.surface_of_tension_position(rho, T)
        return R_eq - R_s

    def surface_tension(self, rho, T, Vext=None, dividing_surface='equimolar'):
        """Profile Property
        Compute the surface tension for a given density profile
        Args:
            rho (list[Profile]) : Density profile for each component [particles / Å^3]
            T (float) : Temperature [K]
            Vext (list[callable], optional) : External potential
            dividing_surface (str, optional) : Which deviding surface to use
                                                'e' or 'equimolar' for Equimolar surface
                                                't' or 'tension' for Surface of tension
        Returns:
            float : Surface tension [J / Å^2]
        """
        for rho_i in rho:
            if not rho_i.check_extends_to_bulk():
                warnings.warn(f'Density profile domain at T = {T:.2f} K is likely too narrow (does not extend into bulk phases). The computed surface tension will '
                              'likely be incorrect. Attempt to recompute with a wider domain (see Profile.on_grid and Profile.lst_on_grid for info '
                              'on how to efficiently translate your profiles to a wider grid).', RuntimeWarning, stacklevel=2)
                break

        Vext = self.sanitize_Vext(Vext)

        rho_b1 = [r[0] for r in rho]
        rho_b2 = [r[-1] for r in rho]

        omega_b1 = self.grand_potential(rho_b1, T, bulk=True)
        omega_b2 = self.grand_potential(rho_b2, T, bulk=True)
        omega = self.grand_potential(rho, T, Vext=Vext)

        if dividing_surface.lower() in ('e', 'equimolar'):
            R_surf = self.equimolar_surface_position(rho)
            z_interface_idx = rho[0].get_position_idx(R_surf)
            V1 = rho[0].grid[:z_interface_idx].volume()
            V2 = rho[0].grid[z_interface_idx:].volume()

            omega_b = omega_b1 * V1 + omega_b2 * V2
            return omega - omega_b

        elif dividing_surface.lower() in ('t', 'tension'):
            if rho[0].grid.geometry == Geometry.PLANAR:
                V = rho[0].grid.volume()
                R_surf = self.surface_of_tension_position(rho, T)
                V1 = rho[0].grid.volume(R_surf)
                V2 = V - V1
                return omega - (omega_b1 * V1 + omega_b2 * V2)
            p_1 = - omega_b1
            p_2 = - omega_b2
            V = rho[0].grid.volume()
            delta_omega = omega + p_1 * V
            delta_p = p_2 - p_1
            gamma = (3 * delta_omega * delta_p**2 / (16 * np.pi))**(1 / 3)
            return gamma


        raise KeyError(f"Invalid dividing surface identifier '{dividing_surface}'!\n"
                       f"Valid identifiers are ('e' / 'equimolar') or ('t' / 'tension').")

    def surface_tension_isotherm(self, T, n_points=30, dividing_surface='t', solver=None, rho0=None, calc_lve=False, verbose=False, cache_dir=''):
        """rhoT Property
        Compute the surface tension as a function of molar composition along an isotherm

        Args:
            T (float) : Temperature [K]
            n_points (int) : Number of (evenly distriubted) points to compute. If an array is supplied, those points are used instead.
            dividing_surface (str, optional) : 't' or 'tension' for surface of tension, 'e' or 'equimolar' for equimolar surface
            solver (SequentialSolver, optional) : Custom solver object to use
            rho0 (list[Profile], optional) : Initial guess for denisty profile at x = [0, 1]
            calc_lve (bool, optional) : If true, return a tuple (x, y, p) with pressure (p), liquid (x) and vapour (y) composition. If false, return only liquid composition.
            verbose (int, optional) : Print progress information, higher number gives more output, default 0

        Returns
            tuple(gamma, x) or tuple(gamma, lve) : Surface tension and composition (of first component)
        """
        if isinstance(n_points, Iterable):
            x_lst = n_points
        else:
            x_lst = np.linspace(1e-3, 1 - 1e-3, n_points)

        if rho0 is None:
            d, _ = self.eos.hard_sphere_diameters(T)
            d *= 1e10
            w = [abs((di / mi) / (2.4728 - 2.3625 * self.reduce_temperature(T, i))) for i, (di, mi) in enumerate(zip(d, self.ms))]
            grid = Grid.tanh_grid(200, Geometry.PLANAR, w, eps=5e-4)
        else:
            grid = rho0[0].grid
        rho = rho0

        gamma_lst = np.empty_like(x_lst)
        for i, x in enumerate(x_lst):
            rho = self.density_profile_tz(T, [x, 1 - x], grid=grid, rho_0=rho, solver=solver, verbose=verbose - 1)
            gamma_lst[i] = self.surface_tension(rho, T, dividing_surface=dividing_surface)
            if verbose > 0:
                print(f'Finished x = {x} ({round(100 * (i + 1) / len(x_lst), 2)} %)')

        if calc_lve is True:
            p_lst = np.empty_like(x_lst)
            y_lst = np.empty_like(x_lst)
            for i, x in enumerate(x_lst):
                p_lst[i], y = self.eos.bubble_pressure(T, [x, 1 - x])
                y_lst[i] = y[0]
            lve = (x_lst, y_lst, p_lst)
            return gamma_lst, lve

        return gamma_lst, x_lst

    def surface_tension_singlecomp(self, n_points=30, t_min=0.5, t_max=0.99, grid=None, solver=None, rho0=None, verbose=0):
        """Pure Property
        Compute the surface tension of a pure component for a series of temperatures.

        Args:
            n_points (int) : Number of points to compute
            t_min (float) : Start temperature, if 0 < t_min < 1, start temperature will be t_min * Tc, where Tc is the critical temperature.
            t_max (float) : Stop temperature, if 0 < t_max < 1, stop temperature will be t_max * Tc, where Tc is the critical temperature.
            grid (Grid) : Grid to use for initial calculation.
            solver (Solver) : Solver to use for all calculations
            rho0 (list[Profile]) : Initial guess for first density profile.
            verbose (int) : Larger number gives more output during progress.

        Returns:
            tuple(gamma, T) : Where gamma and T are matching 1d arrays of the surface tension and temperature.
        """
        tc, _, _ = self.eos.critical([1])
        t_min = tc * t_min if (0 < t_min < 1) else t_min
        t_max = tc * t_max if (0 < t_max < 1) else t_max

        T_lst = np.linspace(t_min, t_max, n_points)
        if rho0 is None:
            grid = GridSpec(200, Geometry.PLANAR) if (grid is None) else grid
        else:
            grid = rho0[0].grid

        gamma_lst = np.empty_like(T_lst)
        rho = rho0
        for i, T in enumerate(T_lst):
            rho = self.density_profile_singlecomp(T, grid, rho, solver=solver, verbose=verbose - 1)
            gamma_lst[i] = self.surface_tension(rho, T, dividing_surface='e')

            if verbose > 0:
                print(f'Finished T = {T} / {tc} ({100 * (i + 1) / len(T_lst):.2f} % of points)')

        return gamma_lst, T_lst

    def adsorbtion(self, rho, T=None, dividing_surface='e'):
        """Profile Property
        Compute the adsorbtion of each on the interface in a given density profile.

        Args:
            rho (list[Profile]) : The density profile for each component [1 / Å^3]
            T (float) : Temperature [K], only required if `dividing_surface` is 't' or 'tension'
            dividing_surface (str) : The dividing surface to use:
                                    'e' or 'equimolar' for equimolar surface
                                    't' or 'tension' for surface of tension

        Returns:
            1D array : The adsobtion of each component [1 / Å^2]
        """
        R = self.dividing_surface_position(rho, T, dividing_surface)
        print(R)
        A = rho[0].grid.area(R)
        V = rho[0].grid.volume()
        V_inner = rho[0].grid.volume(R)
        V_outer = V - V_inner
        N = np.array([r.integrate() for r in rho])
        N_ref = V_inner * np.array([r[0] for r in rho]) + V_outer * np.array([r[-1] for r in rho])
        return (N - N_ref) / A

    def adsorbtion_isotherm(self, T, n_points=30, dividing_surface='t', x_min=1e-3, x_max=(1 - 1e-3), solver=None, rho0=None, calc_lve=False, verbose=False):
        """rhoT Property
        Compute the adsorbtion as a function of molar composition along an isotherm

        Args:
            T (float) : Temperature [K]
            n_points (int) : Number of (evenly distriubted) points to compute. If an array is supplied, those points are used instead.
            dividing_surface (str, optional) : 't' or 'tension' for surface of tension, 'e' or 'equimolar' for equimolar surface
            x_min (float, optional) : Minimum liquid mole fraction of the first component.
            x_max (float, optional) : Maximum liquid mole fraction of the first component.
            solver (SequentialSolver, optional) : Custom solver object to use
            rho0 (list[Profile], optional) : Initial guess for denisty profile at x = [0, 1]
            calc_lve (bool, optional) : If true, return a tuple (x, y, p) with pressure (p), liquid (x) and vapour (y) composition. If false, return only liquid composition.
            verbose (int, optional) : Print progress information, higher number gives more output, default 0

        Returns
            tuple(gamma, x) or tuple(gamma, lve) : Adsorbtion and composition (of first component)
        """
        if isinstance(n_points, Iterable):
            x_lst = n_points
        else:
            x_lst = np.linspace(x_min, x_max, n_points)
            x_lst = np.linspace(-2, 2, n_points)
            x_lst = np.tanh(x_lst / 2)
            bufx = 1e-3
            minx = min(x_lst)
            maxx = - minx
            dx = (maxx - minx)
            x_lst = (x_lst - minx) / dx
            x_lst = bufx + x_lst * (1 - bufx)

        if rho0 is None:
            d, _ = self.eos.hard_sphere_diameters(T)
            d *= 1e10
            w = [abs((di / mi) / (2.4728 - 2.3625 * self.reduce_temperature(T, i))) for i, (di, mi) in
                 enumerate(zip(d, self.ms))]
            grid = Grid.tanh_grid(200, Geometry.PLANAR, w, eps=5e-4)
        else:
            grid = rho0[0].grid
        rho = rho0

        ads_array = np.empty((self.ncomps, n_points))
        for i, x in enumerate(x_lst):
            rho = self.density_profile_tz(T, [x, 1 - x], grid=grid, rho_0=rho, solver=solver, verbose=verbose - 1)

            ads_array[:, i] = self.adsorbtion(rho, T, dividing_surface=dividing_surface)
            if verbose > 0:
                print(f'Finished x = {x} ({round(100 * (i + 1) / len(x_lst), 2)} %)')

        if calc_lve is True:
            p_lst = np.empty_like(x_lst)
            y_lst = np.empty_like(x_lst)
            for i, x in enumerate(x_lst):
                p_lst[i], y = self.eos.bubble_pressure(T, [x, 1 - x])
                y_lst[i] = y[0]
            lve = (x_lst, y_lst, p_lst)
            return ads_array, lve

        return ads_array, x_lst

    def N_adsorbed(self, rho, T=None, dividing_surface='e'):
        """Profile Property
        Compute the adsorbtion of each on the interface in a given density profile

        Args:
            rho (list[Profile]) : The density profile for each component [1 / Å^3]
            T (float) : Temperature [K], only required if `dividing_surface` is 't' or 'tension'
            dividing_surface (str) : The dividing surface to use:
                                    'e' or 'equimolar' for equimolar surface
                                    't' or 'tension' for surface of tension

        Returns:
            1D array : The adsobtion of each component [1 / Å^2]
        """
        R = self.dividing_surface_position(rho, T, dividing_surface)
        A = rho[0].grid.area(R)
        adsorbtion = self.adsorbtion(rho, T, dividing_surface)
        return adsorbtion * A

    def radial_distribution_functions(self, rho_b, T, comp_idx=0, grid=None):
        """rhoT Property
        Compute the radial distribution functions $g_{i,j}$ for $i =$ `comp_idx` using the "Percus trick". To help convergence:
        First converge the profile for a planar geometry, exposed to an ExtendedSoft potential with a core radius $5R$, where
        $R$ is the maximum `characteristic_length` of the mixture. Then, shift that profile to the left, and use it as
        an initial guess for the spherical case.
        If that doesn't work, the profile can be shifted in several steps (by gradually reducing the core radius of the
        ExtendedSoft potential). The latter possibility is not implemented, but is just a matter of putting the "shift
        and recompute" part of this method in a for-loop, and adding some appropriate kwargs.

        Args:
            rho_b (list[float]) : The bulk densities [particles / Å^3]
            T (float) : Temperature [K]
            comp_idx (int) : The first component in the pair, defaults to the first component
            grid (Grid) : The spatial discretisation (should have Spherical geometry for results to make sense)
        Returns:
            list[Profile] : The radial distribution functions around a particle of type `comp_idx`
        """
        R = self.get_characteristic_lengths()
        if grid is None:
            grid = Grid(2500, Geometry.SPHERICAL, 25 * R)

        tolerances = [1e-4, 1e-8]
        solvers = [picard_rhoT, anderson_rhoT]
        solver_kwargs = [{'mixing_alpha': 0.05, 'max_iter': 100},
                         {'beta_mix' : 0.05, 'max_iter' : 300}]

        # First, converge for a planar geometry
        grid_p = Grid(grid.N, Geometry.PLANAR, grid.L)
        Vext = [ExpandedSoft(R, lambda r: self.pair_potential(comp_idx, i, r)) for i in range(self.ncomps)]
        rho = Profile.from_potential(rho_b, T, grid_p, Vext=Vext)
        sol = solve_sequential(self, rho_b, T, rho, solvers, tolerances, solver_kwargs, Vext=Vext)

        if sol.converged is False:
            warnings.warn('Initial computation for planar profile did not converge!', RuntimeWarning, stacklevel=2)

        # Shift profile by R, convert geometry to spherical, and reconverge profile
        shift_idx = - int(R / grid.dz)
        for i in range(self.ncomps):
            rho[i] = np.roll(np.asarray(sol.profile[i]), shift_idx)
            rho[i][shift_idx:] = rho_b[i]
        rho = [Profile(rho[i], grid) for i in range(self.ncomps)]

        sol = solve_sequential(self, rho_b, T, rho, solvers, tolerances, solver_kwargs, Vext=Vext)
        if sol.converged is False:
            warnings.warn('Density profile did not converge after maximum number of iterations', RuntimeWarning, stacklevel=2)

        # Divide by bulk densities to get RDF.
        rdf = [Profile(sol.profile[i] / rho_b[i], grid) for i in range(self.ncomps)]
        return rdf

    @profilecaching
    def density_profile_wall(self, rho_b, T, grid, Vext=None, rho_0=None, verbose=False):
        """Density Profile
        Calculate equilibrium density profile for a given external potential
        Note: Uses lazy evaluation for (rho_b, T, x, grid, Vext) to return a copy of previous result if the same
                calculation is done several times.

        Args:
            rho_b (list[float]) : The bulk densities [particles / Å^3]
            T (float) : Temperature [K]
            grid (Grid) : Spatial discretization
            Vext (ExtPotential, optional) : External potential as a function of position (default : Vext(r) = 0)
                                                    Note: Must be hashable, to use with lazy evaluation
                                                    Recomended: Use the callable classes inherriting ExtPotential
            rho_0 (list[Profile], optional) : Initial guess for density profiles.
            verbose (bool) : Print progression information during run
        Returns:
            list[Profile] : The equilibrium density profiles
        """
        Vext = self.sanitize_Vext(Vext)

        if (tuple(rho_b), T, grid, tuple(Vext)) in self.computed_density_profiles.keys():
            profile = copy.deepcopy(self.computed_density_profiles[(tuple(rho_b), T, grid, tuple(Vext))])
            return profile

        if rho_0 is None:
            wts = self.get_weights(T)
            rho_eq = Profile.from_potential(rho_b, T, grid, Vext, w3=wts[3])
        else:
            rho_eq = rho_0

        solvers = [picard_rhoT, picard_rhoT, anderson_rhoT]
        tolerances = [1e-3, 1e-5, 1e-9]
        solver_kwargs = [{'mixing_alpha': 0.01, 'max_iter': 500},
                         {'mixing_alpha': 0.05, 'max_iter': 500},
                         {'beta_mix': 0.05, 'max_iter': 50}]
        sol = solve_sequential_rhoT(self, rho_eq, rho_b, T, solvers=solvers, tolerances=tolerances,
                               solver_kwargs=solver_kwargs, Vext=Vext, verbose=verbose)

        profile = sol.profile
        if (tuple(rho_b), T, grid, tuple(Vext)) not in self.computed_density_profiles.keys():
            self.computed_density_profiles[(tuple(rho_b), T, grid, tuple(Vext))] = copy.deepcopy(profile)
        return profile

    @profilecaching
    def density_profile_wall_tp(self, T, p, z, grid, Vext, rho0=None, verbose=0):
        """Density Profile
        Calculate equilibrium density profile for a given external potential

        Args:
            T (float) : Temperature [K]
            p (float) : Pressure [Pa]
            z (list[float]) : Bulk composition
            grid (Grid) : Spatial discretization
            Vext (ExtPotential, optional) : External potential as a function of position (default : Vext(r) = 0)
                                                    Note: Must be hashable, to use with lazy evaluation
                                                    Recomended: Use the callable classes inherriting ExtPotential
            rho0 (list[Profile], optional) : Initial guess for density profiles.
            verbose (bool) : Print progression information during run
        Returns:
            list[Profile] : The equilibrium density profiles
        """
        flsh = self.eos.two_phase_tpflash(T, p, z)
        if flsh.phase == self.eos.TWOPH:
            raise ValueError('Supplied state is a two phase state')

        Vm, = self.eos.specific_volume(T, p, z, self.eos.MINGIBBSPH)
        rho = np.array(z) * (1 / Vm) * Avogadro / 1e30
        return self.density_profile_wall(rho, T, grid, Vext, rho_0=rho0, verbose=verbose)

    @profilecaching
    def density_profile_tp(self, T, p, z, grid, rho_0=None, solver=None, verbose=False):
        """Density Profile
        Compute the equilibrium density profile across a gas-liquid interface.
        For multicomponent systems, twophase_tpflash is used to compute the composition of the two phases.
        For single component systems, p is ignored, and pressure is computed from dew_pressure.

        Args:
             T (float) : Temperature [K]
             p (float) : Pressure [Pa]
             z (Iterable) : Total Molar composition
             grid (Grid) : Spatial discretisation
             rho_0 (list[Profile], optional) : Initial guess for density profile
             solver (SequentialSolver or GridRefiner) : Solver to use, uses a default SequentialSolver if none is supplied
             verbose (bool, optional) : Whether to print progress info
        Returns:
            list[Profile] : The density profile for each component across the interface.
        """

        self.validate_composition(z)
        if self.ncomps == 1:
            p, _ = self.eos.bubble_pressure(T, z)
            vl = self.eos.specific_volume(T, p, z, self.eos.LIQPH)[0] * 1e30 / Avogadro
            vg = self.eos.specific_volume(T, p, z, self.eos.VAPPH)[0] * 1e30 / Avogadro
            x = y = np.array([1])
        else:
            flsh = self.eos.two_phase_tpflash(T, p, z)
            if flsh.betaV < 1e-10:
                warnings.warn(f'Mixture at (T, p) = ({T}, {p}) is a pure liquid phase.', RuntimeWarning, stacklevel=2)
                vl = self.eos.specific_volume(T, p, flsh.x, self.eos.LIQPH)[0] * 1e30 / Avogadro  # Converting to Å^3
                vg = vl
                flsh.y = flsh.x
            elif flsh.betaL < 1e-10:
                warnings.warn(f'Mixture at (T, p) = ({T}, {p}) is a pure vapour phase.', RuntimeWarning, stacklevel=2)
                vg = self.eos.specific_volume(T, p, flsh.y, self.eos.VAPPH)[0] * 1e30 / Avogadro  # Converting to Å^3
                vl = vg
                flsh.x = flsh.y
            else:
                vl = self.eos.specific_volume(T, p, flsh.x, self.eos.LIQPH)[0] * 1e30 / Avogadro # Converting to Å^3
                vg = self.eos.specific_volume(T, p, flsh.y, self.eos.VAPPH)[0] * 1e30 / Avogadro # Converting to Å^3

            x, y, beta_V = flsh.x, flsh.y, flsh.betaV
        rho_l = x / vl
        rho_g = y / vg
        return self.density_profile_twophase(rho_g, rho_l, T, grid, beta_V=0.5, rho_0=rho_0, verbose=verbose, solver=solver)

    @profilecaching
    def density_profile_tz(self, T, z, grid, z_phase=1, rho_0=None, solver=None, verbose=0):
        """Density Profile
        Compute the density profile separating two phases at temperature T, with liquid (or optionally vapour) composition x

        Args:
            T (float) : Temperature [K]
            z (ndarray[float]) : Molar composition of liquid phase (unless x_phase=2)
            grid (Grid or GridSpec) : The spatial discretization
            z_phase (int) : ThermoPack Phase flag, indicating which phase has composition `x`. `x_phase=1` for liquid (default)
                            `x_phase=2` for vapour.
            rho_0 (list[Profile], optional) : Initial guess
            solver (SequentialSolver or GridRefiner, optional) : Solver to use
            verbose (int) : Print debugging information (higher number gives more output), default 0

        Returns:
            list[Profile] : The converged density profiles.
        """
        self.validate_composition(z)
        if z_phase == self.eos.LIQPH:
            x = z
            p, y = self.eos.bubble_pressure(T, z)
        elif z_phase == self.eos.VAPPH:
            y = z
            p, x = self.eos.dew_pressure(T, z)
        else:
            raise KeyError(f'Invalid phase flag {z_phase}. Valid options are 1 (LIQPH) or 2 (VAPPH).')
        x, y = np.array(x), np.array(y)

        vl, = self.eos.specific_volume(T, p, x, self.eos.LIQPH)
        vg, = self.eos.specific_volume(T, p, y, self.eos.VAPPH)
        rho_l = (x / vl) * Avogadro / 1e30
        rho_g = (y / vg) * Avogadro / 1e30
        return self.density_profile_twophase(rho_g, rho_l, T, grid, rho_0=rho_0, solver=solver, verbose=verbose)

    @profilecaching
    def density_profile_singlecomp(self, T, grid, rho_0=None, solver=None, verbose=False):
        """Density Profile
        Compute the equilibrium density profile across a gas-liquid interface.
        For multicomponent systems, twophase_tpflash is used to compute the composition of the two phases.
        For single component systems, p is ignored, and pressure is computed from dew_pressure.

        Args:
             T (float) : Temperature [K]
             grid (Grid) : Spatial discretisation
             rho_0 (list[Profile], optional) : Initial guess for density profile
             solver (SequentialSolver or GridRefiner) : Solver to use, uses a default SequentialSolver if none is supplied
             verbose (bool, optional) : Whether to print progress info
        Returns:
            list[Profile] : The density profile for each component across the interface.
        """

        if self.ncomps != 1: raise AttributeError(f"density_profile_singlecomp invoked for model with {self.ncomps} components.")
        p, _ = self.eos.bubble_pressure(T, [1])
        vl = self.eos.specific_volume(T, p, [1], self.eos.LIQPH)[0] * 1e30 / Avogadro
        vg = self.eos.specific_volume(T, p, [1], self.eos.VAPPH)[0] * 1e30 / Avogadro

        rho_l = np.array([1 / vl])
        rho_g = np.array([1 / vg])
        return self.density_profile_twophase(rho_g, rho_l, T, grid, beta_V=0.5, rho_0=rho_0, verbose=verbose, solver=solver)

    @profilecaching
    def density_profile_twophase(self, rho_g, rho_l, T, grid, beta_V=0.5, rho_0=None, solver=None, verbose=0):
        """Density Profile
        Compute the density profile separating two phases with denisties rho_g and rho_l

        Args:
            rho_g (list[float]) : Density of each component in phase 1 [particles / Å^3]
            rho_l (list[float]) : Density of each component in phase 2 [particles / Å^3]
            beta_V (float) : Liquid fraction
            T (float) : Temperature [K]
            grid (Grid or GridSpec) : The spacial discretisation. Using a GridSpec is preferred, as the method then generates a
                                        grid that is likely to be a suitable width.
            rho_0 (list[Profile], optional) : Initial guess, optional
            solver (SequentialSolver or GridRefiner) : Solver to use, uses a default SequentialSolver if none is supplied
            verbose (bool, options) : Print progress info while running solver

        Returns:
            list[Profile] : The density profile for of each component
        """

        rho_g = np.array(rho_g)
        rho_l = np.array(rho_l)

        if rho_0 is None:
            d, _ = self.eos.hard_sphere_diameters(T)
            d *= 1e10
            w = [abs((di / mi) / (2.4728 - 2.3625 * self.reduce_temperature(T, i))) for i, (di, mi) in enumerate(zip(d, self.ms))]

            if isinstance(grid, GridSpec):
                grid = Grid.tanh_grid(grid.n_grid, grid.geometry, w, grid.tanh_eps)
                print(f'Generated Grid with width {grid.L}')

            rho_vle = Profile.tanh_profile(grid, rho_l, rho_g, width_factors=w)
        else:
            rho_vle = rho_0
            grid = rho_vle[0].grid

        N = grid.volume() * ( beta_V * rho_g + (1 - beta_V) * rho_l)

        if solver is None:
            solver = SequentialSolver('muT')
            solver.add_picard(3e-3, mixing_alpha=0.01, max_iter=2000)
            solver.add_picard(1e-3, mixing_alpha=0.015, max_iter=1000)
            solver.add_picard(3e-4, mixing_alpha=0.02, max_iter=1000)
            solver.add_picard(3e-5, mixing_alpha=0.04, max_iter=1000)
            solver.add_picard(1e-5, mixing_alpha=0.05, max_iter=1000)
            solver.add_anderson(1e-7, beta_mix=0.02, max_iter=500)
            solver.add_anderson(1e-8, beta_mix=0.03, max_iter=500)
            solver.add_anderson(1e-9, beta_mix=0.05, max_iter=500)
            solver.add_anderson(1e-10, beta_mix=0.10, max_iter=500)

        if solver.spec == 'NT':
            solver.set_constraints((N, T))
        elif solver.spec == 'muT':
            N = sum(N)
            mu = self.chemical_potential(rho_l, T)
            solver.set_constraints((mu, N, T))
        else:
            raise KeyError(f"Solver spec must be 'NT' or 'muT' but was {solver.spec}.")


        sol = solver(self, rho_vle, verbose=verbose - 1)
        # After converging the profile: Check that it extends sufficiently into the
        # Bulk phase on either side of the interface for reliable further calculations.
        # If not, expand the grid. This method calls itself recursively, with progressively wider
        # grids until the profiles extend into the bulk phases.
        rho_vle = sol.profile
        for ri in rho_vle:
            if not ri.check_extends_to_bulk():
                center = ri.grid.get_center()
                grid = Grid(ri.grid.N, ri.grid.geometry, ri.grid.L * 1.2, ri.grid.domain_start)
                grid.set_center(center)
                rho_vle = Profile.lst_on_grid(rho_vle, grid)
                for i in range(self.ncomps):
                    rho_vle[i].grid.set_domain_start(0)
                if verbose > 0:
                    print("Grid appears to be too narrow (profile does not extend into bulk)")
                    print(f'Expanding domain width from {ri.grid.L / 1.2:.2f} to {ri.grid.L:.2f}')
                return self.density_profile_twophase(rho_g, rho_l, T, rho_vle[0].grid, beta_V=beta_V,
                                                         rho_0=rho_vle, solver=solver, verbose=verbose)
        return sol.profile

    def __density_profile_twophase(self, rho_left, rho_right, T, grid, rho_0=None, constraints=None):
        """Deprecated
        """
        valid_constrains = ['rho', 'rho_frac']
        if constraints is None:
            return self.density_profile_twophase(rho_left, rho_right, T, grid, rho_0=rho_0)

    def density_profile_NT(self, rho1, rho2, T, grid, rho, rho_is_frac, rho_0=None):
        """Deprecated
        """
        rho1 = np.array(rho1)
        rho2 = np.array(rho2)

        V = grid.volume()
        if rho_is_frac:
            if grid.geometry == Geometry.SPHERICAL:
                r_inner = (1 - rho) * grid.L
                V_inner = (4 / 3) * np.pi * r_inner ** 3
                V_outer = V - V_inner
            else:
                V_inner = rho * V
                V_outer = V - V_inner

            N = rho1 * V_inner + rho2 * V_outer
            initial_profile_centre = sum(rho) * grid.L
        else:
            N = np.array(rho) * V
            V1 = (N - V * rho2) / (rho1 - rho2)
            initial_profile_centre = grid.size(V1)

        if rho_0 is None:
            d, _ = self.eos.hard_sphere_diameters(T)
            d *= 1e10
            w = [di / (2.4728 - 2.3625 * self.reduce_temperature(T, i)) for i, di in enumerate(d)]

            rho_vle = Profile.tanh_profile(grid, rho1, rho2, width_factors=w, centre=initial_profile_centre)
        else:
            rho_vle = rho_0

        solvers = [picard_NT, picard_NT, picard_NT, picard_NT, picard_NT, anderson_NT, anderson_NT, anderson_NT]
        tolerances = [1e-3, 5e-4, 1e-4, 5e-5, 1e-6, 1e-7, 1e-8, 1e-9]
        solver_kwargs = [{'mixing_alpha' : 0.01, 'max_iter': 500}
                        ,{'mixing_alpha' : 0.05, 'max_iter': 500}
                        ,{'mixing_alpha' : 0.075, 'max_iter': 500}
                        ,{'mixing_alpha' : 0.10, 'max_iter': 500}
                        ,{'mixing_alpha': 0.125, 'max_iter': 500}
                        ,{'beta_mix': 0.02, 'max_iter': 200}
                        ,{'beta_mix': 0.03, 'max_iter': 200}
                        ,{'beta_mix': 0.05, 'max_iter': 200}]

        sol = solve_sequential_NT(self, rho_vle, N, T, solvers=solvers, tolerances=tolerances,
                               solver_kwargs=solver_kwargs, verbose=True)
        return sol.profile

    def density_profile_muT(self, rho1, rho2, T, grid, rho, rho_is_frac, rho_0=None):
        """Deprecated
        """
        rho1 = np.array(rho1)
        rho2 = np.array(rho2)

        V = grid.volume()
        if rho_is_frac:
            if grid.geometry == Geometry.SPHERICAL:
                r_inner = (1 - rho) * grid.L
                V_inner = (4 / 3) * np.pi * r_inner ** 3
                V_outer = V - V_inner
            else:
                V_inner = rho * V
                V_outer = V - V_inner

            N = rho1 * V_inner + rho2 * V_outer
            initial_profile_centre = rho * grid.L
        else:
            N = np.array(rho) * V
            V1 = (N - V * sum(rho2)) / (sum(rho1 - rho2))
            initial_profile_centre = grid.size(V1)

        if rho_0 is None:
            d, _ = self.eos.hard_sphere_diameters(T)
            d *= 1e10
            w = [di / abs(2.4728 - 2.3625 * self.reduce_temperature(T, i)) for i, di in enumerate(d)]

            rho_vle = Profile.tanh_profile(grid, rho1, rho2, width_factors=w, centre=initial_profile_centre)
        else:
            rho_vle = rho_0

        solvers = [picard_muT, picard_muT, picard_muT, picard_muT, picard_muT, anderson_muT, anderson_muT, anderson_muT]
        tolerances = [1e-3, 5e-4, 1e-4, 5e-5, 1e-6, 1e-7, 1e-8, 1e-9]
        solver_kwargs = [{'mixing_alpha' : 0.01, 'max_iter': 500}
                        ,{'mixing_alpha' : 0.05, 'max_iter': 500}
                        ,{'mixing_alpha' : 0.075, 'max_iter': 500}
                        ,{'mixing_alpha' : 0.10, 'max_iter': 500}
                        ,{'mixing_alpha': 0.125, 'max_iter': 500}
                        ,{'beta_mix': 0.02, 'max_iter': 200}
                        ,{'beta_mix': 0.03, 'max_iter': 200}
                        ,{'beta_mix': 0.05, 'max_iter': 200}]

        sol = solve_sequential_muT(self, rho_vle, N, T, solvers=solvers, tolerances=tolerances,
                               solver_kwargs=solver_kwargs, verbose=True)

        return sol.profile

    def drubble_profile_rT(self, rho_i, rho_o, T, r, grid, rho_0=None):
        """Density Profile
        Compute the density profile across the interface of a droplet or bubble (drubble).

        Args:
            rho_i (list[float]) : The particle densities inside the drubble.
            rho_o (list[float]) : The particle densities outside the drubble.
            T (float) : Temperature [K]
            r (float) : Drubble radius [Å]
            grid (Grid) : The grid to use (must have Geometry.SPHERICAL)
            rho_0 (list[Profile], optional) : Initial guess

        Returns:
            list[Profile] : The density profiles across the interface.
        """
        if grid.geometry != Geometry.SPHERICAL:
            raise ValueError(f'Drops / Bubbles must have spherical geometry, but grid has geometry {grid.geometry}!')
        elif r > grid.L:
            raise ValueError(f'The grid must be larger than the drubble size, but was r_max = {grid.L}, (r = {r}).')

        V_inner = (4 / 3) * np.pi * r**3
        N_inner = rho_i * V_inner
        N_outer = rho_o * (grid.volume() - V_inner)
        N = N_inner + N_outer
        rho_tot = N / grid.volume()
        return self.density_profile_NT(rho_i, rho_o, T, grid, rho_tot, rho_is_frac=False, rho_0=rho_0)

    def sanitize_Vext(self, Vext):
        """Internal
        Ensure that Vext is a tuple with the proper ammount of elements.
        """
        if Vext is None:
            return tuple(lambda z: 0 for _ in range(self.ncomps))
        elif not isinstance(Vext, Iterable):
            return tuple(Vext for _ in range(self.ncomps))
        return Vext

    def split_component_weighted_densities(self, n):
        """Internal
        Unsure if this is still in use, but I believe it takes in fmt-weighted densities as a 1d array and returns
        the same densities as a 2d array.

        Args:
            n (list[Profile]) : FMT-weighted densities

        Returns:
            list[list[Profile]] : FMT-weighted densities, indexed as n[<component_idx>][<weight_idx>]
        :return:
        """
        return [n[ci : ci + 6] for ci in range(self.ncomps)]