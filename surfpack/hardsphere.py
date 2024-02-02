"""
Note: The hard sphere functionals used for the hard sphere contribution in saft
models are defined in SAFT_functional.py. They inherit the classes defined here,
but override the get_R method to return temperature dependent radii.
"""
import copy

import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable
import abc
from surfpack.Functional import Functional
from surfpack.WeightFunction import get_FMT_weights, get_FMT_weight_derivatives
from surfpack.profile import Profile
from surfpack.Convolver import convolve_ad
from surfpack.external_potential import HardWall
from scipy.constants import Avogadro

class FMT_Functional(Functional):
    """
    Parent class for hard-sphere FMT functionals (Rosenfeld, White Bear and White Bear Mark II in "introduction to DFT")
    """

    def __init__(self, R, ms=None):
        try:
            ncomps = len(R)
            self._R = np.array(R)
        except TypeError:
            ncomps = 1
            self._R = np.array([R])

        if ms is None:
            self.ms = np.ones_like(self._R)
        elif isinstance(ms, Iterable):
            self.ms = np.array(ms)
        else:
            self.ms = np.array([ms])

        super().__init__(ncomps)
        self.pair_potentials = [[HardWall(0.5 * (2 * self._R[i] + 2 * self._R[j]), is_pore=False) for i in range(self.ncomps)] for j in range(self.ncomps)]

    @abc.abstractmethod
    def reduced_helmholtz_energy_density(self, rho, T, dphidn=False, bulk=False, asarray=False, dphidT=False): pass

    def get_characteristic_lengths(self):
        return 2 * self._R

    def pressure_tv(self, rho, T):
        """
        Unsure if this still works, should be Eq. 3.7 in "introduction to DFT"

        Args:
            rho (list[float]) : Density [particles / Å^3]
            T (float) : Temperature [K]

        Returns:
            float : Pressure (unsure about unit)
        """
        n = self.get_weighted_densities(rho, T, bulk=True)
        phi, dphidn = self.reduced_helmholtz_energy_density(rho, T, dphidn=True, bulk=True)
        p_id = n[0]
        p_res = - phi
        for i in range(1, len(n) - 2):
            p_res += dphidn[i] * n[i]

        p = p_id + p_res

        return p * 1e10

    def get_R(self, T, dRdT=False):
        """
        Use this to get R, because inheriting classes may override it to set a temperature dependent radius

        NOTE: RADIUS - NOT DIAMETER
        """
        if dRdT is False:
            return self._R

        return self._R, np.zeros_like(self._R)

    def get_weights(self, T, dwdT=False, R=None, dRdT=None):
        """
        Same as above: Inheriting classes may have temperature dependent radii, so use this!

        Args:
            T (float) : Temperature [K]
            dwdT (bool) : Return derivatives (instead of weights)
            R (1d array, optional) : Hard sphere radii [Å]
            dRdT (1d array, optional) : Hard sphere radii derivatives wrt. temperature [Å / K]
        Returns
            list[float] : weights *or* derivatives
        """

        if dwdT is False:
            if R is None:
                R = self.get_R(T)
            return get_FMT_weights(R)

        if (R is None) or (dRdT is None):
            R, dRdT = self.get_R(T, dRdT=True)
        dwdT = get_FMT_weight_derivatives(R) # this is dwdR, multiplying with dRdT in the following loop.
        for wi in range(len(dwdT)):
            for ci in range(self.ncomps):
                dwdT[wi][ci] *= dRdT[ci]
        return dwdT

    def get_component_weighted_densities(self, rho, T, bulk=False, dndrho=False, dndT=False):
        """
        Compute the weighted densities for FMT (the same for all FMT)

        Args:
            rho (list[Profile]) : Density profiles [particles / Å^3] indexed as rho_arr[<comp idx>][<grid idx>]
            T (float) : Temperature [K], used to get radii and weights (in case of temp. dependent radii)
            bulk (bool) : If true, take a list[float] for rho_arr, and return a list[float]
            dndrho (bool) : If true, also return derivative (only for bulk)
            dndT (bool) : If true, also return derivative
        Returns:
            list[Profile] : weighted densities [particles / Å^3], indexed as wts[<comp idx>][<grid idx>]
            Optional 2d array (float) : dndrho, derivatives of weighted densities, indexed as dndrho[<wt idx>][<comp idx>]
        """
        weights = self.get_weights(T)

        if bulk is False:
            weighted_densities = [Profile.zeros(rho[0].grid, len(weights)) for _ in range(self.ncomps)]  # Returns a list of len(weights) profiles, with the grid taken from rho_arr[0]
            for wi, weight in enumerate(weights):  # Iterate over weights
                for ci, (rho_i, comp_weight) in enumerate(zip(rho, weight)):  # Iterate over components
                    weighted_densities[ci][wi] = Profile(convolve_ad(comp_weight, rho_i) + 1e-12, rho_i.grid)  # Todo: Prevent division by zero in a better way
                    weighted_densities[ci][wi].is_vector_field = weights[wi][0].is_vector_valued
        else:
            if not isinstance(rho, Iterable):
                rho = np.array([rho])

            weighted_densities = np.empty((self.ncomps, 6))
            for ci in range(self.ncomps):
                for wi in range(6):
                    weighted_densities[ci][wi] = rho[ci] * weights[wi][ci].real_integral()

        if dndrho is True:
            if bulk is True:
                """
                dndrho is indexed as dndrho[<wt_idx>][<comp_idx>], such that for the weighted densities n_i, and the actual densities rho_j, we have
                d n_i / d rho_j = dndrho[i][j].
                Further, the weighted densities are indexed as n_ij = rho_i * w_j = n[i * 6 + j], such that we have
                n = [rho_1 * w_1, rho_1 * w_2, ..., rho_1 * w_6, rho_2 * w_1, ..., rho_2 * w_6, ... rho_n * w_n]
                """
                dndrho = np.zeros((6 * self.ncomps, self.ncomps))
                for ci in range(self.ncomps):
                    for wi in range(6):
                        dndrho[ci * 6 + wi][ci] = weights[wi][ci].real_integral()

            else:
                raise NotImplementedError('dndrho only implemented for bulk (is actually equal to weight functions)')

            return weighted_densities, dndrho

        if dndT is True:
            dwdT = self.get_weights(T, dwdT=True)
            dndT = [Profile.zeros(rho[0].grid, len(weights)) for _ in range(self.ncomps)]
            for wi, weight in enumerate(dwdT):  # Iterate over weights
                for ci, (rho, comp_weight) in enumerate(zip(rho, weight)):  # Iterate over components
                    dndT[ci][wi] = convolve_ad(comp_weight, rho) + 1e-12  # Todo: Prevent division by zero in a better way
                    dndT[ci][wi].is_vector_field = weights[wi][0].is_vector_valued
            return weighted_densities, dndT

        return weighted_densities

    @staticmethod
    def get_weighted_density_conversion_factors():
        return np.array([1e30, 1e20, 1e10, 1, 1e20, 1e10]) / Avogadro

    def weighted_densities_to_SI(self, n_alpha, comp_wts=True):
        n_alpha = copy.deepcopy(n_alpha)
        wt_conv_f = self.get_weighted_density_conversion_factors()
        if comp_wts is True:
            for ci in range(self.ncomps):
                for wi in range(6):
                    n_alpha[ci][wi] *= wt_conv_f[wi]
            return n_alpha

        for ci in range(self.ncomps):
            for wi in range(6):
                n_alpha[wi] *= wt_conv_f[wi]
        return n_alpha

    def get_weighted_densities(self, rho, T, bulk=False, dndrho=False, dndT=False, n_comp=None):
        """
        Compute the weighted densities for FMT (the same for all FMT)

        Args:
            rho (list[Profile]) : Density profiles [particles / Å^3] indexed as rho_arr[<comp idx>][<grid idx>]
            T (float) : Temperature [K], used to get radii and weights (in case of temp. dependent radii)
            bulk (bool) : If true, take a list[float] for rho_arr, and return a list[float]
            dndrho (bool) : If true, also return derivative (only for bulk)
            dndT (bool) : If true, also return derivative
            n_comp (list[Profile]) : Component weighted densities

        Returns:
            list[Profile] : weighted densities [particles / Å^3], indexed as wts[<comp idx>][<grid idx>]
            Optional 2d array (float) : dndrho, derivatives of weighted densities, indexed as dndrho[<wt idx>][<comp idx>]
        """
        if (n_comp is not None) and (dndrho is False) and (dndT is False) and (bulk is False):
            weighted_densities = np.sum(n_comp, axis=0)
            weighted_densities = [Profile(n, rho[0].grid) for n in weighted_densities]
            return weighted_densities

        weights = self.get_weights(T)
        if bulk is False:
            weighted_densities = Profile.zeros(rho[0].grid, len(weights)) # Returns a list of len(weights) profiles, with the grid taken from rho_arr[0]
            for wi, weight in enumerate(weights):  # Iterate over weights
                for rho_i, comp_weight in zip(rho, weight): # Iterate over components
                    weighted_densities[wi] += convolve_ad(comp_weight, rho_i) + 1e-12 # Todo: Prevent division by zero in a better way
                weighted_densities[wi].is_vector_field = weights[wi][0].is_vector_valued
        else:
            if not isinstance(rho, Iterable):
                rho = np.array([rho])

            weighted_densities = [0 for _ in range(6)]
            for wi in range(6):
                for ci in range(self.ncomps):
                    weighted_densities[wi] += rho[ci] * weights[wi][ci].real_integral()

        if dndrho is True:
            if bulk is True:
                dndrho = np.empty((6, self.ncomps))
                for ni in range(6):
                    for ci in range(self.ncomps):
                        dndrho[ni][ci] = weights[ni][ci].real_integral()

            else:
                raise NotImplementedError('dndrho only implemented for bulk (is actually equal to weight functions)')

            return weighted_densities, dndrho

        if dndT is True:
            dwdT = self.get_weights(T, dwdT=True)
            dndT = Profile.zeros(rho[0].grid, len(weights)) # Returns a list of len(weights) profiles, with the grid taken from rho_arr[0]
            for wi, weight in enumerate(dwdT):  # Iterate over weights
                for rho, comp_weight in zip(rho, weight): # Iterate over components
                    dndT[wi] += convolve_ad(comp_weight, rho) + 1e-12 # Todo: Prevent division by zero in a better way
                dndT[wi].is_vector_field = weights[wi][0].is_vector_valued
            return weighted_densities, dndT

        return weighted_densities

    def pressure(self, rho, T):
        """
        See: pressure_tv
        """
        n = self.get_weighted_densities(rho, T, bulk=True)
        phi, dphidn = self.reduced_helmholtz_energy_density(rho, T, dphidn=True, bulk=True)
        p_id = n[0]
        p_res = - phi
        for i in range(1, len(n) - 2):
            p_res += dphidn[i] * n[i]

        return p_id + p_res

class Rosenfeld(FMT_Functional):

    def __init__(self, R, ms=None):
        super().__init__(R, ms)

    def reduced_helmholtz_energy_density(self, rho, T, dphidn=False, bulk=False, asarray=False, dphidT=False):
        """
        Compute the reduced helmholtz energy density (i.e. f / k_b T) See method in Functional for explanation

        Args:
            rho (list[Profile] or list[float]) : Density profiles or bulk densities
            T (float) : Temperature [K] needed in case inheriting methods have temp. dependent diameters
            dphidn (bool) : Return derivatives
            bulk (bool) : Passed on to get_weighted_densities, because for bulk no convolution is required
            asarray (bool) : Return the helmholtz density profile as a numpy array rather than the (default) list[Profile]
            dphidT (bool) : Return temperature derivative
        Returns:
             list[Profile] : Reduced Helmholtz energy densities [Å^{-3}]
             Optional list[Profile] : Derivatives wrt. weighted densities
        """

        n0, n1, n2, n3, nv1, nv2 = self.get_weighted_densities(rho, T, bulk=bulk)

        phi1 = - n0 * np.log(1 - n3)
        phi2 = (n1 * n2 - nv1 * nv2) / (1 - n3)
        phi3 = (n2 ** 3 - 3 * n2 * nv2 * nv2) / (24 * np.pi * (1 - n3) ** 2)

        if dphidn is False:
            if asarray is False:
                return phi1 + phi2 + phi3
            return np.array(phi1 + phi2 + phi3)

        dphidn0 = - np.log(1 - n3)
        dphidn1 = n2 / (1 - n3)
        dphidn2 = n1 / (1 - n3) + (n2 ** 2 - nv2 ** 2) / (8 * np.pi * (1 - n3) ** 2)
        dphidn3 = n0 / (1 - n3) + (n1 * n2 - nv1 * nv2) / (1 - n3) ** 2 \
                  + (n2 ** 3 - 3 * n2 * nv2 * nv2) / (12 * np.pi * (1 - n3) ** 3)
        dphidnv1 = - nv2 / (1 - n3)
        dphidnv2 = - nv1 / (1 - n3) - n2 * nv2 / (4 * np.pi * (1 - n3) ** 2)

        if asarray is False:
            return (phi1 + phi2 + phi3), [dphidn0, dphidn1, dphidn2, dphidn3, dphidnv1, dphidnv2]
        return np.array(phi1 + phi2 + phi3), np.array([dphidn0, dphidn1, dphidn2, dphidn3, dphidnv1, dphidnv2])

class WhiteBear(FMT_Functional):

    def __init__(self, R, ms=None):
        super().__init__(R, ms)

    def reduced_helmholtz_energy_density(self, rho, T, dphidn=False, bulk=False, asarray=False, dphidT=False, dphidrho=False, n_fmt=None):
        """
        Compute the reduced helmholtz energy density (i.e. f / k_b T) See method in Functional for explanation

        Args:
            rho (list[Profile] or list[float]) : Density profiles or bulk densities
            T (float) : Temperature [K] needed in case inheriting methods have temp. dependent diameters
            dphidn (bool) : Return derivatives
            bulk (bool) : Passed on to get_weighted_densities, because for bulk no convolution is required
            asarray (bool) : Return the helmholtz density profile as a numpy array rather than the (default) list[Profile]
            dphidT (bool) : Compute derivative
            dphidrho (bool) : Compute derivative
            n_fmt (bool, optional) : FMT weighted densities, computed internally if not supplied, can be pre-computed for better performance

        Returns:
             list[Profile] : Reduced Helmholtz energy densities [Å^{-3}]
             Optional list[Profile] : Derivatives wrt. weighted densities
        """
        if dphidT is True:
            n_fmt, dndT = self.get_weighted_densities(rho, T, bulk=bulk, dndT=True)
        elif n_fmt is None:
            n_fmt = self.get_weighted_densities(rho, T, bulk=bulk)

        if dphidrho is True:
            phi, dphidn = self.reduced_helmholtz_energy_density(rho, T, dphidn=True, n_fmt=n_fmt)
            wts = self.get_weights(T)
            if bulk is True:
                dphidrho = np.zeros(self.ncomps)
                print('R : ', self.get_R(T), ', ms : ', self.ms)
                for wi, comp_weights in enumerate(wts):
                    print(f'dphidn : {dphidn[wi]}', end=' Weights : ')
                    for ci, w in enumerate(comp_weights):
                        dphidrho[ci] += dphidn[wi] * w.real_integral()
                        print(w.real_integral(), end=' ')
                    print()
            else:
                dphidrho = Profile.zeros_like(rho)
                for wi, comp_weights in enumerate(wts):
                    for ci, w in enumerate(comp_weights):
                        dphidrho[ci] += convolve_ad(w, dphidn[wi]) * (-1 if w.is_odd() else 1)
            return phi, dphidrho

        n0, n1, n2, n3, nv1, nv2 = n_fmt

        phi1 = - n0 * np.log(1 - n3)
        phi2 = (n1 * n2 - nv1 * nv2) / (1 - n3)
        phi3 = (n2**3 - 3 * n2 * nv2 * nv2) * (n3 + (1 - n3)**2 * np.log(1 - n3)) / (36 * np.pi * n3 ** 2 * (1 - n3)**2)

        phi = phi1 + phi2 + phi3

        if (dphidn is False) and (dphidT is False):
            if asarray is False:
                return phi
            return np.array(phi)

        dphidn0 = - np.log(1 - n3)
        dphidn1 = n2 / (1 - n3)

        dphidn2 = n1 / (1 - n3) + (n2 ** 2 - nv2 ** 2) * (n3 + (1 - n3) ** 2 * np.log(1 - n3)) / (12 * np.pi * n3 ** 2 * (1 - n3) ** 2)
        dphidnv2 = - nv1 / (1 - n3) - n2 * nv2 * (n3 + (1 - n3) ** 2 * np.log(1 - n3)) \
                   / (6 * np.pi * n3 ** 2 * (1 - n3) ** 2)

        dphidn3 = n0 / (1 - n3) + (n1 * n2 - nv1 * nv2) / (1 - n3) ** 2 \
                  - (n2 ** 3 - 3 * n2 * nv2 ** 2) * ((((n3 * (n3**2 - 5 * n3 + 2)) / (36 * np.pi * n3**3 * (1 - n3)**3))
                                                      + np.log(1 - n3) / (18 * np.pi * n3**3)))
        dphidnv1 = - nv2 / (1 - n3)

        dphidn = [dphidn0, dphidn1, dphidn2, dphidn3, dphidnv1, dphidnv2]

        if dphidT is True:
            dphidT = 0
            for dphidn_a, dn_adT in zip(dphidn, dndT):
                dphidT += dphidn_a * dn_adT
            return phi, dphidT

        if asarray is False:
            return phi, dphidn
        return phi, np.array(dphidn)

class WhiteBearMarkII(FMT_Functional):

    def __init__(self, R, ms=None):
        super().__init__(R, ms)

    def reduced_helmholtz_energy_density(self, rho, T, dphidn=False, bulk=False, asarray=False):
        """
        Compute the reduced helmholtz energy density (i.e. f / k_b T) See method in Functional for explanation

        Args:
            rho (list[Profile] or list[float]) : Density profiles or bulk densities
            T (float) : Temperature [K] needed in case inheriting methods have temp. dependent diameters
            dphidn (bool) : Return derivatives
            bulk (bool) : Passed on to get_weighted_densities, because for bulk no convolution is required
            asarray (bool) : Return the helmholtz density profile as a numpy array rather than the (default) list[Profile]

        Returns:
             list[Profile] : Reduced Helmholtz energy densities [Å^{-3}]
             Optional list[Profile] : Derivatives wrt. weighted densities
        """
        n0, n1, n2, n3, nv1, nv2 = self.get_weighted_densities(rho, T, bulk=bulk)

        phi2 = (2 * n3 - n3**2 + 2 * (1 - n3) * np.log(1 - n3)) / n3
        phi3 = (2 * n3 - 3 * n3 ** 2 + 2 * n3 ** 3 + 2 * (1 - n3)**2 * np.log(1 - n3)) / (n3 ** 2)
        phi = - n0 * np.log(1 - n3) + (n1 * n2 - nv1 * nv2) * (1 + (1 / 3) * phi2) / (1 - n3) \
                + (n2**3 - 3 * n2 * nv2 * nv2) * (1 - (1 / 3) * phi3) / (24 * np.pi * (1 - n3) ** 2)

        if dphidn is False:
            if asarray is True:
                return np.array(phi)
            return phi

        d_phi2_dn3 = -1 - (2 / n3) - 2 * np.log(1 - n3) / n3**2
        d_phi3_dn3 = 2 + (2 / n3) - (4 / n3**2) - 4 * (1 - n3) * np.log(1 - n3) / n3**3

        # Derivatives of Helmholtz energy density
        dphidn0 = - np.log(1 - n3)
        dphidn1 = n2 * (1 + (1 / 3) * phi2) / (1 - n3)
        dphidn2 = n1 * (1 + (1 / 3) * phi2) / (1 - n3) + (n2**2 - nv2 * nv2) * (1 - (1 / 3) * phi3) / (8 * np.pi * (1 - n3)**2)

        dphidn3 = n0 / (1 - n3) + (n1 * n2 - nv1 * nv2) * ((d_phi2_dn3 / 3) / (1 - n3) + (1 + phi2 / 3) / (1 - n3)**2) \
                  + ((n2**3 - 3 * n2 * nv2 * nv2) / (24 * np.pi * (1 - n3)**2)) * (- (d_phi3_dn3 / 3) + (2 * 1 - phi3 / 3) / (1 - n3))

        dphidnv1 = -  nv2 * (1 + (1 / 3) * phi2) / (1 - n3)
        dphidnv2 = - nv1 * (1 + (1 / 3) * phi2) / (1 - n3) \
                   - n2 * nv2 * (1 - (1 / 3) * phi3) / (4 * np.pi * (1 - n3) ** 2)

        dphidn = [dphidn0, dphidn1, dphidn2, dphidn3, dphidnv1, dphidnv2]

        if asarray is True:
            return np.array(phi), np.array(dphidn)
        return phi, dphidn