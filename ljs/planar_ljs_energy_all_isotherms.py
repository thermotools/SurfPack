# Run script for the classical DFT code that interfaces with Thermopack

import numpy as np
import sys
from pyctp.ljs_wca import ljs_uv, ljs_wca
from pyctp.ljs_bh import ljs_bh
from pyctp.thermopack_state import equilibrium
from src.interface import PlanarInterface
from src.constants import LenghtUnit, NA, KB, Properties
from src.dft_numerics import dft_solver
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

psi_disp = 1.4
psi_soft_rep = psi_disp
functional_kwargs={"psi_disp": psi_disp,
                   "psi_soft_rep": psi_soft_rep}
#functional_kwargs={"psi_disp": psi_disp}

# Dict of all temperatures:
temperatures = {}
temperatures["0.56"] = {"T_star": 0.5612, "filename": "T056.csv"}
temperatures["0.6"] = {"T_star": 0.6, "filename": "T06.csv"}
temperatures["0.64"] = {"T_star": 0.6415, "filename": "T065.csv"}
temperatures["0.69"] = {"T_star": 0.69, "filename": "T07.csv"}
# Select temperature:
#temperature = temperatures["0.7"]

colors = ["b", "g", "k", "orange"]

# Set up thermopack and equilibrium state
thermopack = ljs_uv()
thermopack.init("Ar")
thermopack.set_tmin(0.5*thermopack.eps_div_kb)
solver=dft_solver()
DATA_X = 0
DATA_T = 1
DATA_RHO = 2
DATA_P_N = 3
DATA_P_T = 4
DATA_H = 6
DATA_U_POT = 7

def offset_error(delta_z, rho, z, rho_of_z):
    dz = delta_z[0]
    error = np.zeros_like(rho)
    error[:] = rho - rho_of_z(z + delta_z)
    return error

for it, temp in enumerate(temperatures):
    temperature = temperatures[temp]
    T =  temperature["T_star"]*thermopack.eps_div_kb[0]
    vle = equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))

    # Define interface with initial tanh density profile
    interf = PlanarInterface.from_tanh_profile(vle,
                                           thermopack.critical_temperature(1),
                                           domain_size=200.0,
                                           n_grid=1024,
                                           invert_states=False,
                                           functional_kwargs=functional_kwargs)
    sigma = interf.functional.thermo.sigma[0]
    eps = interf.functional.thermo.eps_div_kb[0]*KB
    len_fac = interf.functional.grid_reducing_lenght/interf.functional.thermo.sigma[0]

    # Solve for equilibrium profile
    interf.solve(solver=solver, log_iter=True)
    # Load experimental data
    data = np.loadtxt(temperature["filename"], skiprows=1, delimiter=";")

    z = np.zeros_like(interf.grid.z)
    z[:] = interf.grid.z*len_fac
    rho_star = np.zeros_like(interf.grid.z)
    rho_star[:] = interf.profile.densities[0][:]*(sigma/interf.functional.grid_reducing_lenght)**3
    rho_of_z = interp1d(z, rho_star, kind="cubic", fill_value=(rho_star[0], rho_star[-1]), bounds_error=False)
    z_test = np.linspace(z[0],z[-1],10000)

    delta_z = np.array([-15.0])
    data_start = 67

    sol = least_squares(offset_error, delta_z, bounds=(- 50, 50), verbose=0, args=(data[data_start:,DATA_RHO], data[data_start:,DATA_X], rho_of_z))
    dz = sol.x

    # plt.plot(z - dz, rho_star,label=r"DFT 1.4")
    # plt.plot(data[:,DATA_X], data[:,DATA_RHO],label=r"MD")
    # plt.ylabel(r"$\rho^*$")
    # plt.xlabel("$z^*$")
    # leg = plt.legend(loc="best", numpoints=1, frameon=False)
    # plt.savefig("ljs_density_T_is_0.7.pdf")
    # plt.show()
    energy_scaling = sigma**3/eps

    h_E = interf.get_excess_enthalpy_density()
    dh = 1.0
    #plt.plot(z - dz, h_E,label=r"DFT-1.4")
    plt.plot(z - dz, h_E + dh*rho_star,label=r"$T^*=$"+temp, linestyle="--", color=colors[it])
    plt.plot(data[:,DATA_X], data[:,DATA_RHO]*(data[:,DATA_H] - data[:,DATA_T]), color=colors[it])
plt.ylabel(r"$h_{\rm{E}}^*$")
plt.xlabel("$z^*$")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.savefig("ljs_enthalpy.pdf")
plt.show()
