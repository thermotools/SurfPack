# Run script for the classical DFT code that interfaces with Thermopack
import numpy as np
import sys
from pyctp.ljs_wca import ljs_uv, ljs_wca
from pyctp.ljs_bh import ljs_bh
from pyctp.thermopack_state import PhaseDiagram, Equilibrium
from src.interface import PlanarInterface
from src.surface_tension_diagram import SurfaceTensionDiagram
from src.constants import LenghtUnit, NA
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def error_function_surf_tens(psi, gamma_star_exp, vle_curve):
    psi_disp = psi[0]
    psi_soft_rep = psi_disp
    kwargs={"psi_disp": psi_disp,
            "psi_soft_rep": psi_soft_rep}
    diagram = SurfaceTensionDiagram(vle_curve, functional_kwargs=kwargs)
    gamma_star = diagram.surface_tension_reduced
    error = np.zeros_like(gamma_star_exp)
    error[:] = gamma_star - gamma_star_exp
    return error


def regress_psi():
    data = np.loadtxt("surf_tens_ljs.dat", skiprows=1)
    thermopack = ljs_uv()
    thermopack.init("Ar")
    thermopack.set_tmin(0.3*thermopack.eps_div_kb)
    z = np.ones(1)
    vle_list = []
    for i in range(np.shape(data)[0]):
        T_star = data[i,0]
        T = T_star*thermopack.eps_div_kb[0]
        vle_list.append(Equilibrium.bubble_pressure(thermopack, T, z))
    vle_curve = PhaseDiagram(vle_list)
    gamma_star_exp = data[:,1]

    psi = np.ones(1)
    #error = error_function_surf_tens(psi, gamma_star_exp, vle_curve)
    #print(error)
    res_lsq = least_squares(error_function_surf_tens,
                            psi,
                            bounds=(0.9, 1.21),
                            verbose=1,
                            args=(gamma_star_exp, vle_curve))
    print(res_lsq)
    psi = res_lsq["x"]
    err = res_lsq["fun"]
    # err = np.array([-0.08912383, -0.12636836,  0.02354231, -0.0746587 , -0.00938688,
    #             -0.00563867, -0.0080268 , -0.00177602,  0.01528261,  0.00965159,
    #             -0.00013098,  0.00120795, -0.00214915, -0.01252554, -0.00538378,
    #             -0.00587265,  0.00030623, -0.00031588, -0.00063681, -0.00037987,
    #             -0.00283436])
    mae = np.sum(np.abs(err))/np.shape(err)[0]
    print(f"mae={mae:.3f}")
    sys.exit()

psi_disp = 1.0002
psi_soft_rep = psi_disp
sr_functional_kwargs={"psi_disp": psi_disp,
                      "psi_soft_rep": psi_soft_rep}
functional_kwargs={"psi_disp": psi_disp}

data = np.loadtxt("surf_tens_ljs.dat", skiprows=1)

n = 20
# Set up thermopack and equilibrium state
thermopack = ljs_uv()
thermopack.init("Ar")
T = 0.5*thermopack.eps_div_kb[0]
thermopack.set_tmin(0.3*thermopack.eps_div_kb)
curve = PhaseDiagram.pure_saturation_curve(thermopack, T, n=n)
diagram = SurfaceTensionDiagram(curve, functional_kwargs=sr_functional_kwargs)
print(diagram.surface_tension_reduced)

# Set up Barker-Henderson model
thermopack_bh = ljs_bh()
thermopack_bh.init("Ar")
thermopack.set_tmin(0.3*thermopack.eps_div_kb)
curve_bh = PhaseDiagram.pure_saturation_curve(thermopack_bh, T, n=n)
diagram_bh = SurfaceTensionDiagram(curve_bh, functional_kwargs=functional_kwargs)
print(diagram_bh.surface_tension_reduced)

# Set up WCA model
thermopack_wca = ljs_wca()
thermopack_wca.init("Ar")
thermopack.set_tmin(0.3*thermopack.eps_div_kb)
curve_wca = PhaseDiagram.pure_saturation_curve(thermopack_wca, T, n=n)
diagram_wca = SurfaceTensionDiagram(curve_wca, functional_kwargs=sr_functional_kwargs)
print(diagram_wca.surface_tension_reduced)

plt.plot(data[:,0], data[:,1], marker="o", label=r"MD", linestyle="None")
plt.plot(curve.temperatures/thermopack.eps_div_kb[0], diagram.surface_tension_reduced, label=r"UV")
plt.plot(curve_bh.temperatures/thermopack.eps_div_kb[0], diagram_bh.surface_tension_reduced, label=r"BH")
plt.plot(curve_wca.temperatures/thermopack.eps_div_kb[0], diagram_bh.surface_tension_reduced, label=r"WCA")
plt.ylabel(r"$\gamma^*$")
plt.xlabel("$T^*$")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.savefig("ljs_surface_tension.pdf")
plt.show()
