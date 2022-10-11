# Run script for the cDFT code that interfaces with Thermopack

import numpy as np
import sys
from pyctp.pcsaft import pcsaft
from pyctp.thermopack_state import equilibrium
from src.interface import PlanarInterface
from src.constants import LenghtUnit, NA
import matplotlib.pyplot as plt

domain_size = 200.0
n_grid = 32
# Set up thermopack and equilibrium state
thermopack = pcsaft()
thermopack.init("C3")
m = thermopack.m[0]
T = 230.0 #231.036 # NBP
vle = equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))

# Define interface with initial tanh density profile
interf = PlanarInterface.from_tanh_profile(vle, thermopack.critical_temperature(1), domain_size=domain_size, n_grid=n_grid)

eps_T = 1.0e-3
T_p = T + eps_T
T_m = T - eps_T

#interf.functional.test_eos_differentials(vle.liquid.v, vle.liquid.x)
#interf.test_functional_differential("rho")
#interf.test_functional_differential("w_rho_hc")
#interf.test_functional_differential("w_lambda_hc")
#interf.test_functional_differential("w_disp")
#interf.test_functional_in_bulk()

# Solve for equilibrium profile
#interf.solve(log_iter=True)
interf.single_convolution()

# Test dFdT
dFdT = interf.functional.temperature_differential(interf.convolver.weighted_densities)
#F = interf.functional.excess_free_energy(interf.convolver.weighted_densities)
interf_pt = PlanarInterface.from_profile(vle, interf.profile, domain_size=domain_size, n_grid=n_grid)
interf_pt.single_convolution()
interf_pt.functional.T = T_p
F_pt = interf_pt.functional.excess_free_energy(interf_pt.convolver.weighted_densities)
interf_mt = PlanarInterface.from_profile(vle, interf.profile, domain_size=domain_size, n_grid=n_grid)
interf_mt.single_convolution()
interf_mt.functional.T = T_m
F_mt = interf_mt.functional.excess_free_energy(interf_mt.convolver.weighted_densities)


dFdT_num = (F_pt-F_mt)/(2*eps_T)
plt.plot(interf.grid.z, dFdT_num,label="Num. dFdT")
plt.plot(interf.grid.z, dFdT,label="Anal. dFdT")
#plt.plot(interf.grid.z, dFdT_num-dFdT,label="Diff. dFdT")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.show()
plt.clf()
#sys.exit()


# Plot profile
# interf.plot_equilibrium_density_profiles(plot_actual_densities=True,
#                                          plot_equimolar_surface=True,
#                                          unit=LenghtUnit.ANGSTROM)

# Surface tension
# print("Surface tension: ", interf.surface_tension_real_units())


# Perturbate in temperature using density profile from solution
vle_p = equilibrium.bubble_pressure(thermopack, T_p, z=np.ones(1))
interf_p = PlanarInterface.from_profile(vle_p, interf.profile, domain_size=domain_size, n_grid=n_grid)

interf_p.single_convolution()
F_p = interf_p.get_excess_free_energy_density()
n_0_p = interf_p.convolver.weighted_densities.n0
n_1_p = interf_p.convolver.weighted_densities.n1
n_2_p = interf_p.convolver.weighted_densities.n2
n_3_p = interf_p.convolver.weighted_densities.n3
n_1v_p = interf_p.convolver.weighted_densities.n1v
n_2v_p = interf_p.convolver.weighted_densities.n2v
n_disp_p = interf_p.convolver.weighted_densities.n["w_disp"]
n_rho_hc_p = interf_p.convolver.weighted_densities.n["w_rho_hc"]
n_lambda_hc_p = interf_p.convolver.weighted_densities.n["w_lambda_hc"]

vle_m = equilibrium.bubble_pressure(thermopack, T_m, z=np.ones(1))
interf_m = PlanarInterface.from_profile(vle_m, interf.profile, domain_size=domain_size, n_grid=n_grid)
interf_m.single_convolution()
F_m = interf_m.get_excess_free_energy_density()
n_0_m = interf_m.convolver.weighted_densities.n0
n_1_m = interf_m.convolver.weighted_densities.n1
n_2_m = interf_m.convolver.weighted_densities.n2
n_3_m = interf_m.convolver.weighted_densities.n3
n_1v_m = interf_m.convolver.weighted_densities.n1v
n_2v_m = interf_m.convolver.weighted_densities.n2v
n_disp_m = interf_m.convolver.weighted_densities.n["w_disp"]
n_rho_hc_m = interf_m.convolver.weighted_densities.n["w_rho_hc"]
n_lambda_hc_m = interf_m.convolver.weighted_densities.n["w_lambda_hc"]

vol_fac = (interf.functional.thermo.sigma[0]/interf.functional.grid_reducing_lenght)**3
s_num = -interf.functional.thermo.eps_div_kb[0]*(F_p-F_m)/(2*eps_T)
s = interf.get_excess_entropy_density()
plt.plot(interf.grid.z, s_num,label="Numerical")
plt.plot(interf.grid.z, s,label="Analytical")
#plt.plot(interf.grid.z, s-s_num,label="Analytical")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.show()
plt.clf()

print("s err: ", np.abs((s - s_num)/s_num))


interf.convolver.convolve_density_profile_T(interf.profile.densities)

dndT = m*interf.convolver.weighted_densities_T.n["w0"]
dndT_num = (n_0_p-n_0_m)/(2*eps_T)
print("n0: ",np.abs((dndT - dndT_num)/dndT_num))
plt.plot(interf.grid.z, dndT_num,label="Num. n0")
plt.plot(interf.grid.z, dndT,label="Anal. n0")

# dndT = m*interf.convolver.weighted_densities_T.n["w1"]
# dndT_num = (n_1_p-n_1_m)/(2*eps_T)
# plt.plot(interf.grid.z, dndT_num,label="Num. n1")
# plt.plot(interf.grid.z, dndT,label="Anal. n1")

# dndT = m*interf.convolver.weighted_densities_T.n["w2"]
# dndT_num = (n_2_p-n_2_m)/(2*eps_T)
# plt.plot(interf.grid.z, dndT_num,label="Num. n2")
# plt.plot(interf.grid.z, dndT,label="Anal. n2")

# dndT = m*interf.convolver.weighted_densities_T.n["w3"]
# dndT_num = (n_3_p-n_3_m)/(2*eps_T)
# plt.plot(interf.grid.z, dndT_num,label="Num. n3")
# plt.plot(interf.grid.z, dndT,label="Anal. n3")

# dndT = m*interf.convolver.weighted_densities_T.n["wv2"]
# dndT_num = (n_2v_p-n_2v_m)/(2*eps_T)
# plt.plot(interf.grid.z, dndT_num,label="Num. nv2")
# plt.plot(interf.grid.z, dndT,label="Anal. nv2")

# dndT = m*interf.convolver.weighted_densities_T.n["wv1"]
# dndT_num = (n_1v_p-n_1v_m)/(2*eps_T)
# plt.plot(interf.grid.z, dndT_num,label="Num. nv1")
# plt.plot(interf.grid.z, dndT,label="Anal. nv1")

# comp = 0
# dndT = interf.convolver.weighted_densities_T.n["w_disp"]
# dndT_num = (n_disp_p[comp,:]-n_disp_m[comp,:])/(2*eps_T)
# plt.plot(interf.grid.z, dndT_num,label="Num. n_disp")
# plt.plot(interf.grid.z, dndT[comp,:],label="Anal. n_disp")

# comp = 0
# dndT = interf.convolver.weighted_densities_T.n["w_rho_hc"]
# dndT_num = (n_rho_hc_p[comp,:]-n_rho_hc_m[comp,:])/(2*eps_T)
# plt.plot(interf.grid.z, dndT_num,label="Num. rho_hc")
# plt.plot(interf.grid.z, dndT[comp,:],label="Anal. rho_hc")

# comp = 0
# dndT = interf.convolver.weighted_densities_T.n["w_lambda_hc"]
# dndT_num = (n_lambda_hc_p[comp,:]-n_lambda_hc_m[comp,:])/(2*eps_T)
# plt.plot(interf.grid.z, dndT_num,label="Num. lambda_hc")
# plt.plot(interf.grid.z, dndT[comp,:],label="Anal. lambda_hc")

leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.show()
sys.exit()

s_scaling = 1.0e-6
s_E = interf.get_excess_entropy_density_real_units()
plt.plot(interf.grid.z, s_E*s_scaling,label=r"$s^{\rm{E}}$ functional")
plt.plot([interf.grid.z[0]], s_scaling*np.array([vle.liquid.specific_excess_entropy()/vle.liquid.specific_volume()]),
         label=r"$s^{\rm{E}}$ bulk liquid", linestyle="None", marker="o")
plt.plot([interf.grid.z[-1]], s_scaling*np.array([vle.vapor.specific_excess_entropy()/vle.vapor.specific_volume()]),
         label=r"$s^{\rm{E}}$ bulk vapour", linestyle="None", marker="o")
plt.ylabel(r"$s^{\rm{E}}$ (MJ/m$^3$/K)")
plt.xlabel("$z$ (Å)")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
#plt.savefig("propane.pdf")
plt.show()

s_scaling = 1.0
rho = interf.profile.rho_mix/(NA*interf.functional.grid_reducing_lenght**3)
plt.plot(interf.grid.z, s_E/rho,label=r"$s^{\rm{E}}$ functional")
plt.plot([interf.grid.z[0]], s_scaling*np.array([vle.liquid.specific_excess_entropy()]),
         label=r"$s^{\rm{E}}$ bulk liquid", linestyle="None", marker="o")
plt.plot([interf.grid.z[-1]], s_scaling*np.array([vle.vapor.specific_excess_entropy()]),
         label=r"$s^{\rm{E}}$ bulk vapour", linestyle="None", marker="o")
plt.ylabel(r"$s^{\rm{E}}$ (J/mol/K)")
plt.xlabel("$z$ (Å)")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.show()
