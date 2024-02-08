import numpy as np
from surfpack.hardsphere import FMT_Functional, WhiteBear, WhiteBearMarkII, Rosenfeld
from surfpack.WeightFunction import get_FMT_weights, get_FMT_weight_derivatives
from surfpack.profile import Profile
from collections.abc import Iterable
import copy

class SAFT_WhiteBear(WhiteBear):
    """
    This is the class used for the hard sphere contribution to the saft models, it overrides the method get_R from
    the class FMT_functional (defined in HardSphere_functional.py) in order to get temperature dependen radii.
    """
    def __init__(self, comps, eos):
        self.eos = eos
        assert self.eos.test_fmt_compatibility()[0], f'eos : {eos} is not FMT compatible'
        self.comps = comps
        ncomps = len(comps.split(','))
        R = np.zeros(ncomps)
        ms = np.array([self.eos.get_pure_fluid_param(i + 1)[0] for i in range(ncomps)])
        super().__init__(R, ms)

        self.computed_weights = {}  # lazy evaluation in get_weights(T)
        self.computed_weight_differentials = {}  # lazy evaluation in get_weights(T)

    def __repr__(self):
        ostr = f'SAFT-Whitebear model for {self.comps}\n' \
               f'with segment numbers : {self.ms}\n' \
               f'Underlying eos params : {[self.eos.get_pure_fluid_param(i + 1) for i in range(self.ncomps)]}\n'
        return ostr

    def get_caching_id(self):
        ostr = f'SAFT-WhiteBear model : {self.comps} \n' \
               f'ms : {self.ms}, \n' \
               f'params : {[self.eos.get_pure_fluid_param(i + 1) for i in range(self.ncomps)]}\n' \
                + super().get_caching_id()
        return ostr

    def get_R(self, T, dRdT=False):
        """
        The length unit used everywhere is Å, so must convert from SI when calling thermopack
        """
        d, d_T = self.eos.hard_sphere_diameters(T)
        d /= 1e-10 # Convert unit to [Å]
        R = d / 2
        if dRdT is False:
            return R

        d_T /= 1e-10 # Converting to [Å]
        return R, d_T / 2

    def get_characteristic_lengths(self):
        d, _ = self.eos.hard_sphere_diameters(300)
        return d * 1e10

    def get_weights(self, T, dwdT=False, R=None, dRdT=None):
        """
        self.computed_weights and self.computed_weight_differentials are dictionaries used for lazy evaluation.
        """
        if T not in self.computed_weights.keys():
            if R is None:
                R = self.get_R(T)
            self.computed_weights[T] = get_FMT_weights(R, self.ms)

        if dwdT is False:
            return copy.deepcopy(self.computed_weights[T])

        if (R is None) or (dRdT is None):
            R, dRdT = self.get_R(T, dRdT=True)

        if T not in self.computed_weight_differentials.keys():
            dwdT = get_FMT_weight_derivatives(R)
            for wi in range(len(dwdT)):
                for ci in range(self.ncomps):
                    dwdT[wi][ci] *= dRdT[ci]
            self.computed_weight_differentials[T] = dwdT
        return copy.deepcopy(self.computed_weight_differentials[T])

    def pair_potential(self, i, j, r):
        raise NotImplementedError

    def packing_fraction(self, rho, T):
        R = self.get_R(T)
        d = 2 * R
        return (1. / 6.) * np.pi * self.ms * d**3 * rho

class SAFT_HardSphere(FMT_Functional):
    """
    This is the class used for the hard sphere contribution to the saft models, it overrides the method get_R from
    the class FMT_functional (defined in HardSphere_functional.py) in order to get temperature dependen radii.
    """
    def __init__(self, comps, eos, fmt_model):
        self.eos = eos(comps)
        assert self.eos.test_fmt_compatibility()[0], f'eos : {eos} is not FMT compatible'
        self.comps = comps
        self.fmt_model = fmt_model
        ncomps = len(comps.split(','))
        R = np.zeros(ncomps) # get_R is overridden, so this list is never accessed.
        ms = np.array([self.eos.get_pure_fluid_param(i + 1)[0] for i in range(ncomps)])
        super().__init__(R, ms)

        self.computed_weights = {}  # lazy evaluation in get_weights(T), contains (T => array[weights]) map
        self.computed_weight_differentials = {}  # lazy evaluation in get_weights(T)

    def get_R(self, T, dRdT=False):
        """
        The length unit used everywhere is Å, so must convert from SI when calling thermopack
        """
        d, d_T = self.eos.hard_sphere_diameters(T)
        d /= 1e-10 # Convert unit to [Å]
        R = d / 2
        if dRdT is False:
            return R

        d_T /= 1e-10 # Converting to [Å]
        return R, d_T / 2

    def get_characteristic_lengths(self):
        d, _ = self.eos.hard_sphere_diameters(300)
        return d * 1e10

    def get_weights(self, T, dwdT=False):
        """
        self.computed_weights and self.computed_weight_differentials are dictionaries used for lazy evaluation.
        """
        if T not in self.computed_weights.keys():
            R = self.get_R(T)
            self.computed_weights[T] = get_FMT_weights(R)

        if dwdT is False:
            return self.computed_weights[T]

        R, dRdT = self.get_R(T, dRdT=True)
        if T not in self.computed_weight_differentials.keys():
            dwdT = get_FMT_weight_derivatives(R)
            for wi in range(len(dwdT)):
                for ci in range(self.ncomps):
                    dwdT[wi][ci] *= dRdT[ci]
            self.computed_weight_differentials[T] = dwdT
        return self.computed_weight_differentials[T]

    def reduced_helmholtz_energy_density(self, rho, T, dphidn=False, phi_nn=False, dphidT=False, bulk=False, asarray=False):
        if dphidT is True:
            n, dndT = self.get_weighted_densities(rho, T, bulk=bulk, dndT=True)
            phi, dphidn = self.eos.fmt_energy_density(n, phi_n=True, fmt_model=self.fmt_model)
            dphidT = 0
            for dphidn_a, dn_adT in zip(dphidn, dndT):
                dphidT += dphidn_a * dn_adT
            if bulk is False:
                return Profile(phi, rho[0].grid), Profile(dphidT, rho[0].grid)
            return phi, dphidT

        n = self.get_weighted_densities(rho, T, bulk=bulk)
        if dphidn is True:
            phi, dphidn =  self.eos.fmt_energy_density(n, phi_n=True, fmt_model=self.fmt_model)
            if bulk is False:
                return Profile(phi, rho[0].grid), Profile(dphidn, rho[0].grid)
            return phi, dphidn
        elif phi_nn is True:
            phi, phi_nn = self.eos.fmt_energy_density(n, phi_nn=True, fmt_model=self.fmt_model)
            if bulk is False:
                return Profile(phi, rho[0].grid), Profile(phi_nn, rho[0].grid)
            return phi, phi_nn

        phi = self.eos.fmt_energy_density(n)
        if bulk is False:
            phi = Profile(phi, rho[0].grid)
        return phi

    def pair_potential(self, i, j, r, T):
        """
        Args:
            i (int) : Component 1
            j (int) : Component 2
            r (float or Iterable) : Inter-particle separation [Å]
            T (float) : Temperature [K]
        Returns:
            float or ndarray : The pair potential (inf if r < sigma else 0)
        """
        R = self.get_R(T)
        sigma = R[i] + R[j]
        if not isinstance(r, Iterable):
            return np.inf if r < sigma else 0

        V = np.zeros_like(r)
        V[r < sigma] = np.inf
        return V

# class SAFT_Rosenfeld(SAFT_HardSphere):
#
#     def __init__(self, comps, eos):
#         super().__init__(comps, eos, 'RF')
#
# class SAFT_WhiteBear(SAFT_HardSphere):
#
#     def __init__(self, comps, eos):
#         super().__init__(comps, eos, 'WB')
#
# class SAFT_WhiteBearII(SAFT_HardSphere):
#
#     def __init__(self, comps, eos):
#         super().__init__(comps, eos, 'WBII')