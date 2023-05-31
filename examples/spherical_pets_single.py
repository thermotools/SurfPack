# Run script for the classical DFT code that interfaces with Thermopack

import numpy as np
import sys
from thermopack.pets import pets
from thermopack.thermopack_state import PhaseDiagram, Equilibrium, State
from src.constants import LenghtUnit, NA, KB, Properties, Specification
from src.interface import PlanarInterface, SphericalInterface
import matplotlib.pyplot as plt
from pets_functional import surface_tension_LJTS
from src.dft_numerics import dft_solver
from src.surface_tension_diagram import SphericalDiagram
from density_profile import Profile

bubble = False

#Set up thermopack and equilibrium curve
thermopack = pets()
thermopack.init()
T_star = 0.741
#T_star = 0.625

T = T_star*thermopack.eps_div_kb[0]
thermopack.set_tmin(0.3*thermopack.eps_div_kb)
vle = Equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))


interf = PlanarInterface.from_tanh_profile(vle,
                                           thermopack.critical_temperature(1),
                                           domain_size=200.0,
                                           n_grid=1024,
                                           invert_states=False)
interf.solve(log_iter=True)
sigma0 = interf.surface_tension(reduced_unit=False)

n_grid = 4096
radius = 230.0
rd = 320.0
radius = 1000.0
rd = 1100.0
radius = 100.0
rd = 200.0



# Define interface with initial tanh density profile
spi = SphericalInterface.from_tanh_profile(vle,
                                           thermopack.critical_temperature(1),
                                           radius=radius,
                                           domain_radius=rd,
                                           n_grid=n_grid,
                                           calculate_bubble=True,
                                           sigma0=sigma0,
                                           specification = Specification.CHEMICHAL_POTENTIAL) #CHEMICHAL_POTENTIAL, NUMBER_OF_MOLES


print(spi.bulk.left_state.chemical_potential, vle.vapour.chemical_potential)

#spi.plot_property_profiles()

#solver=dft_solver().anderson(mmax=50, beta=0.05, tolerance=1.0e-10, max_iter=200)
# Solve for equilibrium profile
solver=dft_solver().picard(tolerance=1.0e-8,max_iter=500,beta=0.05,ng_frequency=None).\
    anderson(mmax=50, beta=0.05, tolerance=1.0e-12,max_iter=500)

spi.solve(solver=solver, log_iter=True)

# Resolve using chemical potential specification
vle_modified = Equilibrium(spi.bulk.left_state, spi.bulk.right_state)
spi_profile = SphericalInterface.from_profile(vle_modified,
                                              spi.profile,
                                              domain_radius=rd,
                                              n_grid=n_grid,
                                              invert_states=False,
                                              specification=Specification.CHEMICHAL_POTENTIAL)

solver2=dft_solver().anderson(mmax=50, beta=0.05, tolerance=1.0e-12,max_iter=500)

spi_profile.solve(solver=solver2, log_iter=True)
print(sigma0, spi.surface_tension(reduced_unit=False), spi_profile.surface_tension(reduced_unit=False))

print(spi_profile.r_equimolar)

print(spi_profile.bulk.left_state.chemical_potential, vle.vapour.chemical_potential)

spi.plot_property_profiles()

sys.exit()

mu = np.array([3046.0])

rho_left = spi_profile.bulk.left_state.partial_density()
sl = State.new_mut(thermopack, mu, T, rho0=rho_left)
rho_right = spi_profile.bulk.right_state.partial_density()
sr = State.new_mut(thermopack, mu, T, rho0=rho_right)
meta = Equilibrium(sl, sr)
profile = Profile()
profile.copy_profile(spi.profile)
spi2 = SphericalInterface.from_profile(meta,
                                       profile,
                                       domain_radius=rd,
                                       n_grid=n_grid,
                                       invert_states=False,
                                       specification=Specification.CHEMICHAL_POTENTIAL)
spi2.solve(solver=solver, log_iter=True)

print(spi2.r_equimolar)



vle_modified = Equilibrium(spi.bulk.left_state, spi.bulk.right_state)
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
