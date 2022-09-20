# Run script for the cDFT code that interfaces with Thermopack

import numpy as np
import sys
from pyctp.ljs_wca import ljs_uv, ljs_wca
from pyctp.ljs_bh import ljs_bh
from pyctp.thermopack_state import phase_diagram
from src.interface import PlanarInterface
from src.surface_tension_diagram import SurfaceTensionDiagram
from src.constants import LenghtUnit, NA
import matplotlib.pyplot as plt

data = np.loadtxt("surf_tens_ljs.dat", skiprows=1)

n = 20
# Set up thermopack and equilibrium state
thermopack = ljs_uv()
thermopack.init("Ar")
T = 0.5*thermopack.eps_div_kb[0]
thermopack.set_tmin(0.3*thermopack.eps_div_kb)
curve = phase_diagram.pure_saturation_curve(thermopack, T, n=n)
diagram = SurfaceTensionDiagram(curve)
print(diagram.surface_tension_reduced)

# Set up Barker-Henderson model
thermopack_bh = ljs_bh()
thermopack_bh.init("Ar")
thermopack.set_tmin(0.3*thermopack.eps_div_kb)
curve_bh = phase_diagram.pure_saturation_curve(thermopack_bh, T, n=n)
diagram_bh = SurfaceTensionDiagram(curve_bh)
print(diagram_bh.surface_tension_reduced)

# Set up WCA model
thermopack_wca = ljs_wca()
thermopack_wca.init("Ar")
thermopack.set_tmin(0.3*thermopack.eps_div_kb)
curve_wca = phase_diagram.pure_saturation_curve(thermopack_wca, T, n=n)
diagram_wca = SurfaceTensionDiagram(curve_wca)
print(diagram_wca.surface_tension_reduced)

fac = 0.7
plt.plot(data[:,0], data[:,1], marker="o", label=r"MD", linestyle="None")
plt.plot(curve.temperatures/thermopack.eps_div_kb[0], diagram.surface_tension_reduced, label=r"UV")
plt.plot(curve_bh.temperatures/thermopack.eps_div_kb[0], diagram_bh.surface_tension_reduced, label=r"BH")
plt.plot(curve_wca.temperatures/thermopack.eps_div_kb[0], diagram_bh.surface_tension_reduced, label=r"WCA")
plt.plot(curve.temperatures/thermopack.eps_div_kb[0], diagram.surface_tension_reduced*fac, label=r"70% UV")
plt.ylabel(r"$\gamma^*$")
plt.xlabel("$T^*$")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.savefig("ljs_surface_tension.pdf")
plt.show()
