# Run script for the cDFT code that interfaces with Thermopack
import numpy as np
import sys
from thermopack.pcsaft import pcsaft
from thermopack.thermopack_state import Equilibrium
from src.interface import PlanarInterface
from src.constants import LenghtUnit, Properties

# Set up thermopack and equilibrium state
thermopack = pcsaft("C1")
T = 140.0
vle = Equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))

# Define interface with initial tanh density profile
interf = PlanarInterface.from_tanh_profile(vle, thermopack.critical_temperature(
    1), domain_size=100.0, n_grid=1024, invert_states=True)

# Solve for equilibrium profile
interf.solve(log_iter=True)

# Plot profile
interf.plot_property_profiles(prop=Properties.RHO,
                              plot_reduced_property=False,
                              plot_equimolar_surface=True)

# Surface tension
print("Surface tension: ", interf.surface_tension_real_units())
