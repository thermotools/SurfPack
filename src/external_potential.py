#!/usr/bin/env python3
from enum import Enum
from dft_numerics import dft_solver
import sys
from constants import NA, KB, Dft_enum
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
import scipy.special.gamma as gamma

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


class External_potential(Dft_enum):
    ENUM_HARDWALL = 11
    ENUM_LJ93 = 22
    ENUM_STEELE = 33
    ENUM_LJ93_SIMPLE = 44
    ENUM_LJ93_CUSTOM = 55
    ENUM_FF_POTENTIAL = 66

class Eps_mixing(Dft_enum):
    GEOMETRIC = 111
    MOD_GEOMETRIC = 222

class PotentialParam(object):
    """
    """
    def __init__(self, potential_type, eps_mixing=GEOMETRIC):
        self.potential_type = potential_type
        self.eps_mixing = eps_mixing

class Hardwall(PotentialParam):
    """
    """
    def __init__(self, sigma_ss):
        PotentialParam.__init__(self, HARDWALL)
        self.sigma = sigma_ss

class Steele(PotentialParam):
    """
    """
    def __init__(self, rho_s, epsilon, sigma, alpha=0.61, delta=3.35, lambda_r=12.0, lambda_a=6.0, prefactor=1.0, eps_mixing=GEOMETRIC):
        PotentialParam.__init__(self, ENUM_STEELE, eps_mixing)
        self.sigma = sigma
        self.epsilon = epsilon
        self.rho_s = rho_s
        self.alpha = alpha
        self.delta = delta
        self.lambda_r = lambda_r
        self.lambda_a = lambda_a

class LJ93(PotentialParam):
    """
    """
    def __init__(self, rho_s, epsilon, sigma, eps_mixing=GEOMETRIC):
        PotentialParam.__init__(self, ENUM_LJ93, eps_mixing)
        self.sigma = sigma
        self.epsilon = epsilon
        self.rho_s = rho_s

class LJ93Simple(PotentialParam):
    """
    """

    def __init__(self, epsilon, sigma, eps_mixing=GEOMETRIC):
        Potential.__init__(self, ENUM_LJ93_SIMPLE, eps_mixing)
        self.sigma = sigma
        self.epsilon = epsilon

class LJ93Custom(PotentialParam):
    """
    """

    def __init__(self, epsilon, sigma):
        Potential.__init__(self, ENUM_LJ93_CUSTOM)
        self.sigma = sigma
        self.epsilon = epsilon

class FluidFluid(PotentialParam):
    """
    """

    def __init__(self, comp):
        Potential.__init__(self, ENUM_FF_POTENTIAL)
        self.comp = comp

class Potential(object):
    """
    """
    def __init__(self, potential_type, geometry, prefactor=1.0):
        self.potential_type = potential_type
        self.prefactor = prefactor
        self.geometry = geometry

    @abstractmethod
    def pot(self, r, pore_size, temperature):
        pass


class HardwallPotential(Potential):
    """
    """

    def __init__(self, geometry, sigma_ss, perfactor=1.0):
        Potential.__init__(self, ENUM_HARDWALL, geometry, perfactor)
        self.sigma_ss = sigma_ss

    def pot(self, r, pore_size, temperature):
        if r > self.sigma_ss:
            p = 0
        else:
            p = 500.0
        return p

class SteelePotential(Potential):
    """
    Jiménez-Serratos et. al. doi:10.1080/00268976.2019.1669836
    Siderius and Gelb. doi: 10.1063/1.3626804
    """

    def __init__(self, geometry, rho_s, epsilon, sigma, alpha=0.61, delta=3.35, lambda_r=12.0, lambda_a=6.0, prefactor=1.0):
        Potential.__init__(self, ENUM_STEELE, geometry, prefactor)
        self.sigma = sigma
        self.epsilon = epsilon
        self.rho_s = rho_s
        self.alpha = alpha
        self.delta = delta
        self.lambda_r = lambda_r
        self.lambda_a = lambda_a
        self.C = (lambda_r/(lambda_r-lambda_a))*(lambda_r/lambda_a)**(lambda_a/(lambda_r-lambda_a))
        if geometry != PLANAR and (abs(lambda_r-12.0) > 1.0e-12 or abs(lambda_a-6.0) > 1.0e-12):
            raise ValueError("Steel Mie potential, unsupported geometry")

    def pot(self, r, pore_size, temperature):
        prefac = self.prefactor*2*np.pi*self.rho_s*self.epsilon*self.sigma**2*self.delta
        if self.Geometry == PLANAR:
            p = prefac*(
                (self.sigma/r)**(self.lambda_r-2)/(self.lambda_r-2)
                -(self.sigma/r)**(self.lambda_a-2)/(self.lambda_a-2)
                -self.sigma**(self.lambda_a-2)/((self.lambda_a-2)*(self.lambda_a-3)*self.delta*(r + self.alpha*self.delta)**(self.lambda_a-3)))
        elif self.Geometry == POLAR:
            p = prefac * (psi(6, r / pore_size, self.sigma / pore_size)
                          - psi(3, r / pore_size, self.sigma / pore_size)
                          - self.sigma / self.delta
                          * phi(3, r / (pore_size + self.delta * self.alpha),
                                self.sigma / (pore_size + self.delta * self.alpha)))
        elif self.Geometry == SPHERICAL:
            p = prefac * (2.0 / 5.0 * sum_n(10, r, self.sigma, pore_size)
                          - sum_n(4, r, self.sigma, pore_size)
                          - self.sigma / (3.0 * self.delta)
                                * ((self.sigma / (pore_size + 0.61 * self.delta - r))**3
                          + (self.sigma / (pore_size + self.alpha * self.delta + r))**3)
                          + 1.5 * sum_n(3, r, self.sigma, pore_size + self.alpha * self.delta))
        else:
            raise ValueError("Steel potential, unsupported geometry")

        return p

    def phi(n, r_r, sigma_r):
        m3n2 = 3.0 - 2.0 * n
        n2m3 = 2.0 * n - 3.0
        p = (1.0 - r_r**2)**m3n2 * 4.0 * np.sqrt(np.pi) / n2m3 \
            * sigma_r**n2m3 * gamma(n - 0.5) / gamma(n) \
            * taylor_2f1_phi(r_r, n)
        return p

    def psi(n, r_r, sigma_r):
        p = (1.0 - r_r**2)**(2.0 - 2.0 * n) \
            * 4.0*np.sqrt(np.pi) * gamma(n - 0.5) / gamma(n) \
            * sigma_r**(2.0 * n - 2.0) * taylor_2f1_psi(r_r, n)
        return p

    def sum_n(n, r, sigma, pore_size):
        s = np.zeros_like(r)
        for i in range(n):
            s += sigma**n / (pore_size**i * (pore_size - r)**(n - i)) \
                + sigma**n / (pore_size**i * (pore_size + r)**(n - i))
        return s

    def taylor_2f1_phi(x, n):
        if n == 3:
            2f1 = 1.0 + 3.0 / 4.0 * x**2 + 3.0 / 64.0 * x**4 \
                - 1.0 / 256.0 * x**6 - 15.0 / 16384.0 * x**8
        elif n == 6:
            2f1 = 1.0 + 63.0 / 4.0 * x**2 + 2205.0 / 64.0 * x**4 \
                + 3675.0 / 256.0 * x**6 + 11025.0 / 16384.0 * x**8
        else:
            raise ValueError("taylor_2f1_phi, unsupported n")
        return 2f1

    def taylor_2f1_psi(x, n):
        if n == 3:
            2f1 = 1.0 + 9.0 / 4.0 * x**2 + 9.0 / 64.0 * x**4 \
                + 1.0 / 256.0 * x**6 + 9.0 / 16384.0 * x**8
        elif n == 6:
            2f1 = 1.0 + 81.0 / 4.0 * x**2 + 3969.0 / 64.0 * x**4 \
                + 11025.0 / 256.0 * x**6 + 99125.0 / 16384.0 * x**8
        else:
            raise ValueError("taylor_2f1_psi, unsupported n")
        return 2f1

class LJ93Potential(Potential):
    """
    Siderius and Gelb. doi: 10.1063/1.362680
    Ravikovitch and Neimark. doi: 10.1021/la0107594
    """

    def __init__(self, geometry, rho_s, epsilon, sigma, prefactor=1.0):
        Potential.__init__(self, ENUM_LJ93, geometry, prefactor)
        self.sigma = sigma
        self.epsilon = epsilon
        self.rho_s = rho_s

    def pot(self, r, pore_size, temperature):
        prefac = self.prefactor * 2.0 * np.pi * self.epsilon * self.sigma**3 * self.rho_s
        if self.Geometry == PLANAR:
            p = prefac / 45.0 * (2.0 * (self.sigma / r)**9 - 15.0 * (self.sigma / r)**3)
        elif self.Geometry == POLAR:
            p = prefac * (phi(6, r / pore_size, self.sigma / pore_size)
                          - phi(3, r / pore_size, self.sigma / pore_size))
        elif self.Geometry == SPHERICAL:
            p = prefac * 0.5 * (
                * self.sigma**12 / 90.
                * ((r - 9.0 * pore_size) / (r - pore_size)**9
                   - (r + 9.0 * pore_size) / (r + pore_size)**9)
                - self.sigma**6 / 3.
                * ((r - 3.0 * pore_size) / (r - pore_size)**3
                   - (r + 3.0 * pore_size) / (r + pore_size)**3)) / r
        else:
            raise ValueError("LJ93 potential, unsupported geometry")

        return p

class LJ93SimplePotential(Potential):
    """
    """

    def __init__(self, geometry, epsilon, sigma):
        Potential.__init__(self, ENUM_LJ93_SIMPLE, geometry)
        self.sigma = sigma
        self.epsilon = epsilon

    def pot(self, r, pore_size):
        if self.Geometry == PLANAR:
            p = self.epsilon * ((self.sigma / r)**9 - (self.sigma / r)**3)
        else:
            raise ValueError("LJ93Simple potential, unsupported geometry")

        return p

class FluidFluidPotential(Potential):
    """
    """

    def __init__(self, thermo, comp1, comp2):
        Potential.__init__(self, ENUM_FF_POTENTIAL)
        self.thermo = thermo
        self.comp1 = comp1
        self.comp2 = comp2

    def pot(self, r, pore_size, temperature):
        p = self.thermo.potential(self.comp1, self.comp2, r, temperature)
        return p

def mix_epsilon_solid_fluid(eps_mixing, eps_ss, eps_ff, sigma_ss=1.0, sigma_ff=1.0):
    eps_sf = np.sqrt(eps_ss*eps_ff)
    if eps_mixing == MOD_GEOMETRIC:
        eps_sf *= np.sqrt(sigma_ss**3*sigma_ff**3)/(sigma_ss + sigma_ff)**3*8
    return eps_sf

def mix_lambda_solid_fluid(lambda_ss, lambda_ff):
    lambda_sf = 3 + np.sqrt((lambda_ss-3)*(lambda_ff-3))
    return lambda_sf

class ExternalPotentials(object):

    def __init__(self, geometry, potential_param, functional):
        self.vext = []
        for i in functional.nc:
            prefactor = functional.thermo.m[i]
            if not (potential_param.potential_type == ENUM_LJ93_CUSTOM or
                    potential_param.potential_type == ENUM_FF_POTENTIAL):
                sigma_sf = (potential_param.sigma + functional.thermo.sigma[i])*0.5
                if potential_param.potential_type != ENUM_HARDWALL:
                    epsilon_sf = mix_epsilon_solid_fluid(potential_param.eps_mixing,
                                                         potential_param.epsilon,
                                                         functional.thermo.eps_div_kb[i],
                                                         sigma_ss=potential_param.sigma,
                                                         sigma_ff=functional.thermo.sigma[i])
            if potential_param.potential_type == ENUM_HARDWALL:
                potential = HardwallPotential(geometry, sigma_sf)
            elif potential_param.potential_type == ENUM_STEELE:
                lambda_r = mix_lambda_solid_fluid(potential_param.lambda_r, functional.thermo.lambda_r[i])
                lambda_a = mix_lambda_solid_fluid(potential_param.lambda_a, functional.thermo.lambda_a[i])
                potential = SteelePotential(geometry,
                                            potential_param.rho_s,
                                            epsilon_sf,
                                            sigma_sf,
                                            alpha=potential_param.alpha,
                                            delta=potential_param.delta,
                                            lambda_r=lambda_r,
                                            lambda_a=lambda_a,
                                            prefactor=prefactor*potential_param.prefactor)
            elif potential_param.potential_type == ENUM_LJ93:
                potential = LJ93Potential(geometry, potential_param.rho_s, epsilon_sf, sigma_sf,
                                          prefactor=prefactor*potential_param.prefactor)
            elif potential_param.potential_type == ENUM_LJ93_SIMPLE:
                potential = LJ93SimplePotential(geometry, epsilon_sf, sigma_sf)
            elif potential_param.potential_type == ENUM_LJ93_CUSTOM:
                potential = LJ93SimplePotential(geometry, potential_param.epsilon[i], potential_param.sigma[i])
            elif potential_param.potential_type == ENUM_FF_POTENTIAL:
                potential = FliuidFluidPotential(functional.thermo,
                                                 potential_param.comp, i+1)
            else:
                raise ValueError("Unsupported potential")
            v_ext.append(potential)

class Pore(Interface):
    """

    """

    def __init__(self,
                 geometry,
                 functional,
                 external_potential_param,
                 domain_size=100.0,
                 n_grid=1024):
        """
        Object holding specifications for classical DFT problem.
        Reduced particle size assumed to be d=1.0, and all other sizes are relative to this scale.

        Args:
            bulk_densities (ndarray): Bulk fluid density ()
            particle_diameters (ndarray): Particle diameter
            wall (str): Wall type (HardWall, SlitHardWall, None)
            domain_length (float): Length of domain
            functional (str): Name of hard sphere functional: Rosenfeld, WhiteBear, WhiteBear Mark II, Default Rosenfeld
            grid (int) : Grid size
            temperature (float): Reduced temperature
            quadrature (str): Quadrature to use during integration
        Returns:
            None
        """

        # Init of base class
        Interface.__init__(self,
                           geometry=geometry,
                           functional=functional,
                           domain_size=domain_size,
                           n_grid=n_grid)
        self.ext_potentials = ExternalPotential(geometry=geometry,
                                                potential_param=external_potential_param,
                                                functional=functional)
        self.v_ext = []
        # for i in functional.nc:
        #     r = self.grid
        #     T = functional
        #     v = self.ext_potentials.pot(r, domain_size, temperature)
        #     self.v_ext.append(v)

    def residual(self, x):
        return self.grid.convolve(x).residual


class PairCorrelation(Pore):

    def __init__(self,
                 functional,
                 state,
                 comp=0,
                 domain_size=15.0,
                 n_grid=1024):
        """
        Object holding specifications for classical DFT problem.
        Reduced particle size assumed to be d=1.0, and all other sizes are relative to this scale.

        Args:
            bulk_densities (ndarray): Bulk fluid density ()
            particle_diameters (ndarray): Particle diameter
            wall (str): Wall type (HardWall, SlitHardWall, None)
            domain_length (float): Length of domain
            functional (str): Name of hard sphere functional: Rosenfeld, WhiteBear, WhiteBear Mark II, Default Rosenfeld
            grid (int) : Grid size
            temperature (float): Reduced temperature
            quadrature (str): Quadrature to use during integration
        Returns:
            None
        """

        fluid_fluid_pot_param = FluidFluid(comp)
        # Init of base class
        Pore.__init__(self,
                      geometry=Geometry.SPHERICAL,
                      functional=functional,
                      external_potential_param=fluid_fluid_pot_param,
                      domain_size=domain_size,
                      n_grid=n_grid)


if __name__ == "__main__":
    pass
