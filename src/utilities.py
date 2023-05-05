import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
from thermopack.thermo import thermo
from constants import Geometry
from scipy.linalg import svd

def extrapolate_mu_in_inverse_radius(eos: thermo,
                                     sigma_0: float,
                                     temperature: float,
                                     rho_l_eq: float,
                                     rho_g_eq: float,
                                     radius: float,
                                     geometry: Geometry,
                                     phase: int):
    """
    Corrections for curved surfaces:
    Assuming constant temperature extrapolate mu in 1/R to first order,
    and calculate bulk phase properties. A constant compoision path is taken.
    rho_l and rho_g represent the bulk equilibrium for wich sigma_0 is calculated
    For details see Aasen et al. 2018, doi: 10.1063/1.5026747

        Args;
            eos (thermo): Thermopack object
            sigma_0 (float): Planar surface tension (N/m)
            temperature (float): Temperature (K)
            rho_l_eq (float): Liquid densities (mol/m3)
            rho_g_eq (float): Gas densities (mol/m3)
            radius (float): Droplet/bubble radius (m)
            geometry (Geometry): Spherical (SPHERICAL) or cylindrical (POLAR) geometry
            phase (int): Phase flag

        Returns:
            (float): Chemichal potential (J/mol)
            (float): Liquid densities (mol/m3)
            (float): Gas densities (mol/m3)
    """

    unity = np.ones(eos.nc)
    V = 1.0
    mu_0, mu_rho_l = eos.chemical_potential_tv(temperature, V, rho_l_eq, dmudn=True)
    mu_0, mu_rho_g = eos.chemical_potential_tv(temperature, V, rho_g_eq, dmudn=True)

    rho = np.zeros(eos.nc)
    mu_rho = np.zeros(eos.nc)
    if phase == eos.LIQPH:
        rho[:] = rho_l_eq
        mu_rho[:] = mu_rho_l
    elif phase == eos.VAPPH:
        rho[:] = rho_g_eq
        mu_rho[:] = mu_rho_g
    else:
        raise ValueError("Wrong phase identifyer in extrapolate_mu_in_inverse_radius")

    # Equation S6
    rho_tot = sum(rho)
    M = np.outer(rho, unity)
    for i in range(eos.nc):
      M[i,i] += rho_tot
    # Find null-space defining mu_1
    w = svd(M)[0][:,-1]
    mu_1 = np.zeros(eos.nc)
    mu_1[:] = np.matmul(mu_rho,w)
    if geometry == Geometry.SPHERICAL:
      g = 2
    elif geometry == Geometry.POLAR:
      g = 1
    else:
      raise ValueError("Wrong geometry in extrapolate_mu_in_inverse_radius")

    # Equation 14
    fac = g*sigma_0 / np.sum(mu_1*(rho_l_eq-rho_g_eq))
    mu_1 = mu_1*fac

    # Equation 8 truncated after first order correction
    mu = np.zeros(eos.nc)
    mu[:] = mu_0 + mu_1/radius

    # Solve for "bulk" states
    # Equation 9 truncated after first order correction
    rho_l_1 = np.zeros(eos.nc)
    rho_l_1[:] = rho_l_eq + np.matmul(1.0/mu_rho_l,mu_1)/radius
    if np.amin(rho_l_1) < 0.0:
        rho_l_1[:] = rho_l_eq
    rho_l = eos.solve_mu_t(temperature,mu,rho_l_1,phase=eos.LIQPH)

    # Equation 9 truncated after first order correction
    rho_g_1 = np.zeros(eos.nc)
    rho_g_1[:] = rho_g_eq + np.matmul(1.0/mu_rho_g,mu_1)/radius
    if np.amin(rho_g_1) < 0.0:
        rho_g_1[:] = rho_g_eq
    rho_g = eos.solve_mu_t(temperature,mu,rho_g_1,phase=eos.LIQPH)

    return mu, rho_l, rho_g
