# Run script for the classical DFT code that interfaces with Thermopack

import numpy as np
import sys
from pyctp.pets import pets
from pyctp.thermopack_state import phase_diagram, equilibrium, state
from src.constants import LenghtUnit, NA, KB, Properties, Specification
from src.interface import PlanarInterface, SphericalInterface
import matplotlib.pyplot as plt
from pets_functional import surface_tension_LJTS
from src.dft_numerics import dft_solver
from src.surface_tension_diagram import SphericalDiagram

#Set up thermopack and equilibrium curve
thermopack = pets()
thermopack.init()
T_star = 0.741
T = T_star*thermopack.eps_div_kb[0]
thermopack.set_tmin(0.3*thermopack.eps_div_kb)
vle = equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))

spdia = SphericalDiagram(vle,
                         initial_radius=40.0,
                         n_grid=1024,
                         calculate_bubble=True)

sys.exit()
# interf = PlanarInterface.from_tanh_profile(vle,
#                                            thermopack.critical_temperature(1),
#                                            domain_size=200.0,
#                                            n_grid=1024,
#                                            invert_states=False)
# interf.solve(log_iter=True)
# sigma0 = interf.surface_tension_real_units()
sigma0 = 0.008148477000941151
print(sigma0)
# n = 25 # Number of meta-stable points
# vz, rho = PeTS.map_meta_isotherm(temperature=T,
#                                  z=z,
#                                  phase=PeTS.LIQPH,
#                                  n=n)

# Define interface with initial tanh density profile
spi = SphericalInterface.from_tanh_profile(vle,
                                           thermopack.critical_temperature(1),
                                           radius=40.0,
                                           domain_radius=80.0,
                                           n_grid=1024,
                                           calculate_bubble=True,
                                           sigma0=sigma0,
                                           specification = Specification.NUMBER_OF_MOLES) #CHEMICHAL_POTENTIAL, NUMBER_OF_MOLES

spi.plot_property_profiles()

ls = state.new_nvt(spi.bulk.left_state.eos, spi.bulk.left_state.T, spi.bulk.left_state.V, spi.bulk.left_state.x)
rs = state.new_nvt(spi.bulk.right_state.eos, spi.bulk.right_state.T, spi.bulk.right_state.V, spi.bulk.right_state.x)
vle_modified = equilibrium(spi.bulk.left_state, spi.bulk.right_state)
vle_modified = equilibrium(ls, rs)
#solver=dft_solver().picard(tolerance=1.0e-5,max_iter=200,beta=0.05,ng_frequency=100).anderson(mmax=50, beta=0.05, tolerance=1.0e-10,
#                                                                                                max_iter=200)
solver=dft_solver().anderson(mmax=50, beta=0.05, tolerance=1.0e-10, max_iter=200)
# Solve for equilibrium profile
spi.solve(solver=solver, log_iter=True)

spi.plot_property_profiles()

sys.exit()

vle_modified = equilibrium(spi.bulk.left_state, spi.bulk.right_state)
spi_profile = SphericalInterface.from_profile(vle_modified,
                                              spi.profile,
                                              domain_radius=100.0,
                                              n_grid=1024,
                                              specification=Specification.NUMBER_OF_MOLES)

#NUMBER_OF_MOLES, CHEMICHAL_POTENTIAL

spi_profile.plot_property_profiles()

# Solve for equilibrium profile
spi_profile.solve(solver=solver, log_iter=True)

spi_profile.plot_property_profiles()
