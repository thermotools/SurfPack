# Run script for the classical DFT code that interfaces with Thermopack

import numpy as np
import sys
from thermopack.ljs_wca import ljs_uv, ljs_wca
from thermopack.ljs_bh import ljs_bh
from thermopack.thermopack_state import Equilibrium
from src.interface import PlanarInterface
from src.constants import LenghtUnit, NA, KB, Properties
from src.dft_numerics import dft_solver
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

psi_disp = 1.4
psi_soft_rep = psi_disp
functional_kwargs = {"psi_disp": psi_disp,
                     "psi_soft_rep": psi_soft_rep}
#functional_kwargs={"psi_disp": psi_disp}

# Dict of all temperatures:
temperatures = {}
temperatures["0.56"] = {"T_star": 0.5612, "filename": "T056.csv"}
temperatures["0.6"] = {"T_star": 0.6, "filename": "T06.csv"}
temperatures["0.65"] = {"T_star": 0.6415, "filename": "T065.csv"}
temperatures["0.7"] = {"T_star": 0.69, "filename": "T07.csv"}
# Select temperature:
temperature = temperatures["0.7"]

# Set up thermopack and equilibrium state
thermopack = ljs_uv()
thermopack.init("Ar")
T_star = temperature["T_star"]
T = T_star*thermopack.eps_div_kb[0]
thermopack.set_tmin(0.5*thermopack.eps_div_kb)
vle = Equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))

# Define interface with initial tanh density profile
interf = PlanarInterface.from_tanh_profile(vle,
                                           thermopack.critical_temperature(1),
                                           domain_size=200.0,
                                           n_grid=1024,
                                           invert_states=False,
                                           functional_kwargs=functional_kwargs)

solver = dft_solver()
# .picard(tolerance=1.0e-10,max_iter=600,beta=0.05,ng_frequency=None).\
#    anderson(tolerance=1.0e-10,max_iter=200,beta=0.05)

# Solve for equilibrium profile
interf.solve(solver=solver, log_iter=True)

# Plot profile
# interf.plot_property_profiles(plot_reduced_property=True,
#                               plot_equimolar_surface=True,
#                               plot_bulk=True,
#                               include_legend=True,
#                               grid_unit=LenghtUnit.REDUCED)

sigma = interf.functional.thermo.sigma[0]
eps = interf.functional.thermo.eps_div_kb[0]*KB
len_fac = interf.functional.grid_reducing_lenght / \
    interf.functional.thermo.sigma[0]

# Load experimental data
data = np.loadtxt(temperature["filename"], skiprows=1, delimiter=";")
DATA_X = 0
DATA_T = 1
DATA_RHO = 2
DATA_P_N = 3
DATA_P_T = 4
DATA_H = 6
DATA_U_POT = 7

z = np.zeros_like(interf.grid.z)
z[:] = interf.grid.z*len_fac
rho_star = np.zeros_like(interf.grid.z)
rho_star[:] = interf.profile.densities[0][:] * \
    (sigma/interf.functional.grid_reducing_lenght)**3
rho_of_z = interp1d(z, rho_star, kind="cubic", fill_value=(
    rho_star[0], rho_star[-1]), bounds_error=False)
z_test = np.linspace(z[0], z[-1], 10000)


def offset_error(delta_z, rho, z, rho_of_z):
    dz = delta_z[0]
    error = np.zeros_like(rho)
    error[:] = rho - rho_of_z(z + delta_z)
    return error


delta_z = np.array([-15.0])
data_start = 67

sol = least_squares(offset_error, delta_z, bounds=(- 50, 50), verbose=0,
                    args=(data[data_start:, DATA_RHO], data[data_start:, DATA_X], rho_of_z))
dz = sol.x

plt.plot(z - dz, rho_star, label=r"DFT 1.4")
plt.plot(data[:, DATA_X], data[:, DATA_RHO], label=r"MD")
plt.ylabel(r"$\rho^*$")
plt.xlabel("$z^*$")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.savefig("ljs_density_T_is_0.7.pdf")
plt.show()

plt.plot(data[:, DATA_X], data[:, DATA_T], label=r"MD")
plt.ylabel(r"$T^*$")
plt.xlabel("$z^*$")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.savefig("ljs_temperature_T_is_0.7.pdf")
plt.show()

s_scaling = NA*sigma**3/interf.functional.thermo.Rgas
energy_scaling = sigma**3/eps

s_E = interf.get_excess_entropy_density()
a_E = interf.get_excess_free_energy_density()
p_T = interf.parallel_pressure()
sum_rho_mu_E = interf.get_excess_chemical_potential_density_sum()
h_E = interf.get_excess_enthalpy_density()
#T_star*s_E + sum_rho_mu_E
u_E = interf.get_excess_energy_density()
#a_E + T_star*s_E

plt.plot(z - dz, u_E, label=r"DFT-1.4")
plt.plot(data[:, DATA_X], data[:, DATA_RHO]*data[:, DATA_U_POT], label=r"MD")

plt.ylabel(r"$u_{\rm{E}}^*$")
plt.xlabel("$z^*$")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.savefig("ljs_energy_T_is_0.7 .pdf")
plt.show()

plt.plot(z - dz, p_T, label=r"DFT-1.4")
plt.plot(data[:, DATA_X], data[:, DATA_P_T], label=r"MD")
plt.ylabel(r"$p_\parallel^*$")
plt.xlabel("$z^*$")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.savefig("ljs_parallel_pressure_T_is_0.7 .pdf")
plt.show()

dh = 1.0
#plt.plot(z - dz, h_E,label=r"DFT-1.4")
plt.plot(z - dz, h_E + dh*rho_star, label=r"DFT-1.4")
plt.plot(data[:, DATA_X], data[:, DATA_RHO] *
         (data[:, DATA_H] - data[:, DATA_T]), label=r"MD")
plt.ylabel(r"$h_{\rm{E}}^*$")
plt.xlabel("$z^*$")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.savefig("ljs_enthalpy_T_is_0.7 .pdf")
plt.show()

plt.plot(z - dz, s_E, label=r"DFT-1.4")
plt.ylabel(r"$s_{\rm{E}}^*$")
plt.xlabel("$z^*$")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.savefig("ljs_entropt_T_is_0.7 .pdf")
plt.show()
