# Run script for the classical DFT code that interfaces with Thermopack

import numpy as np
import sys
from thermopack.pets import pets
from thermopack.thermopack_state import PhaseDiagram, Equilibrium
from src.constants import LenghtUnit, NA, KB, Properties
from src.interface import PlanarInterface
from src.surface_tension_diagram import SurfaceTensionDiagram
import matplotlib.pyplot as plt
from pets_functional import surface_tension_LJTS
from dft_numerics import dft_solver


# Set up thermopack and equilibrium curve
thermopack = pets()
T_star = 0.625
T = T_star*thermopack.eps_div_kb[0]
thermopack.set_tmin(0.3*thermopack.eps_div_kb)

n = 20
curve = PhaseDiagram.pure_saturation_curve(thermopack, T, n=n)
diagram = SurfaceTensionDiagram(curve)
data = surface_tension_LJTS()

plt.plot(data["T"], data["gamma"], marker="o", label=r"MD", linestyle="None")
plt.plot(curve.temperatures /
         thermopack.eps_div_kb[0], diagram.surface_tension_reduced)
plt.ylabel(r"$\gamma^*$")
plt.xlabel("$T^*$")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.savefig("pets_surface_tension.pdf")
plt.show()

# Get surface tension values for article temperatures
Tv = [0.625, 0.7, 0.741, 0.8, 0.9, 1.0]
gamma = []
solver=dft_solver().anderson(mmax=50, beta=0.05, tolerance=1.0e-12,max_iter=500)
for T_star in Tv:
    T = T_star*thermopack.eps_div_kb[0]
    vle = Equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))

    # Define interface with initial tanh density profile
    interf = PlanarInterface.from_tanh_profile(vle, thermopack.critical_temperature(1), domain_size=200.0, n_grid=1024, invert_states=True)

    interf.grid.get_domain_weights(position=0.55*interf.grid.domain_size)
    # Solve for equilibrium profile
    interf.solve(solver=solver, log_iter=True)

    # Plot profile
    interf.plot_property_profiles(plot_equimolar_surface=True, continue_after_plotting=True)


    gamma.append(interf.surface_tension(reduced_unit=True))

    print("surface tension:",interf.get_surface_of_tension(reduced_unit=True))
    sys.exit()

print(gamma)
