# Run script for the cDFT code that interfaces with Thermopack

import numpy as np
import sys
from pyctp.pcsaft import pcsaft
from pyctp.thermopack_state import equilibrium
from pyCDFT.interface import SphericalInterface
from pyCDFT.interface import PlanarInterface

# Set up thermopack and equilibrium state
thermopack = pcsaft()
thermopack.init("C1")
T = 140.0
vle = equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))

# Define interface with initial tanh density profile
sigma0 = 0.00815622572569111 #PlanarInterface.from_tanh_profile(vle, thermopack.critical_temperature(1), domain_size=100.0, n_grid=1024).solve().surface_tension_real_units()

# Define interface with initial tanh density profile
interf = SphericalInterface.from_tanh_profile(vle,
                                              thermopack.critical_temperature(1),
                                              radius=25.0,
                                              domain_radius=100.0,
                                              n_grid=1024,
                                              calculate_bubble=False,
                                              sigma0=sigma0)

# Solve for equilibrium profile
interf.solve(log_iter=True)

# Plot profile
interf.plot_equilibrium_density_profiles(plot_actual_densities=False,
                                         plot_equimolar_surface=True)

# Surface tension
print("Surface tension: ", interf.surface_tension_real_units())
