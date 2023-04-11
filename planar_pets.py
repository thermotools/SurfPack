# Run script for the classical DFT code that interfaces with Thermopack

import numpy as np
import sys
from thermopack.pets import pets
from thermopack.thermopack_state import PhaseDiagram
from src.constants import LenghtUnit, NA, KB, Properties
from src.surface_tension_diagram import SurfaceTensionDiagram
import matplotlib.pyplot as plt
from pets_functional import surface_tension_LJTS

# Set up thermopack and equilibrium curve
thermopack = pets()
thermopack.init()
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
