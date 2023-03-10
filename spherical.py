# Run script for the cDFT code that interfaces with Thermopack

import numpy as np
import sys
from thermopack.pcsaft import pcsaft
from thermopack.thermopack_state import Equilibrium
from src.interface import SphericalInterface
from src.interface import PlanarInterface
from src.constants import LenghtUnit

# Set up thermopack and equilibrium state
thermopack = pcsaft()
thermopack.init("C1")
T = 140.0
vle = Equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))

# Define interface with initial tanh density profile
#sigma0 = PlanarInterface.from_tanh_profile(vle, thermopack.critical_temperature(1), domain_size=100.0, n_grid=1024).solve().surface_tension_real_units()
sigma0 = 0.00815622572569111

# Define interface with initial tanh density profile
spi = SphericalInterface.from_tanh_profile(vle,
                                           thermopack.critical_temperature(1),
                                           radius=25.0,
                                           domain_radius=50.0,
                                           n_grid=1024,
                                           calculate_bubble=False,
                                           sigma0=sigma0)

# Solve for equilibrium profile
spi.solve(log_iter=True)

# Plot profile
spi.plot_property_profiles(plot_reduced_property=True,
                           plot_equimolar_surface=True,
                           plot_bulk=True,
                           include_legend=True,
                           grid_unit=LenghtUnit.REDUCED)

# Surface tension
print(f"Planar surface tension: {1.0e3*sigma0} mN/m")
print(f"Surface tension: {1.0e3*spi.surface_tension_real_units()} mN/m")
gamma_s, r_s, delta = spi.surface_of_tension()
print(f"Surface of tension: {1.0e3*gamma_s} mN/m")
print(f"Tolman length: {1.0e10*delta} Å")
