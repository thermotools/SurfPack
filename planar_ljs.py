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

psi_disp = 1.0
psi_soft_rep = psi_disp
functional_kwargs={"psi_disp": psi_disp,
                   "psi_soft_rep": psi_soft_rep}

# Set up thermopack and equilibrium state
thermopack = ljs_uv()
thermopack.init("Ar")
T_star = 0.75
T = T_star*thermopack.eps_div_kb[0]
thermopack.set_tmin(0.5*thermopack.eps_div_kb)
vle = equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))

# Define interface with initial tanh density profile
interf = PlanarInterface.from_tanh_profile(vle,
                                           thermopack.critical_temperature(1),
                                           domain_size=200.0,
                                           n_grid=1024,
                                           invert_states=True,
                                           functional_kwargs=functional_kwargs)

solver=dft_solver()
#.picard(tolerance=1.0e-10,max_iter=600,beta=0.05,ng_frequency=None).\
#    anderson(tolerance=1.0e-10,max_iter=200,beta=0.05)

# Solve for equilibrium profile
interf.solve(solver=solver, log_iter=True)

# Plot profile
interf.plot_property_profiles(plot_reduced_property=True,
                              plot_equimolar_surface=True,
                              plot_bulk=True,
                              include_legend=True,
                              grid_unit=LenghtUnit.REDUCED)

# Surface tension
print("Surface tension: ", interf.surface_tension())

# Perturbate in temperature using density profile from solution
# eps_T = 1.0e-5
# T_p = T + eps_T
# vle_p = equilibrium.bubble_pressure(thermopack, T_p, z=np.ones(1))
# interf_p = PlanarInterface.from_profile(vle_p, interf.profile, domain_size=100.0, n_grid=1024, invert_states=True)

# interf_p.single_convolution()
# F_p = interf_p.get_excess_helmholtz_energy_density()
# n_0_p = interf_p.convolver.weighted_densities.n0
# n_1_p = interf_p.convolver.weighted_densities.n1
# n_2_p = interf_p.convolver.weighted_densities.n2
# n_3_p = interf_p.convolver.weighted_densities.n3
# n_1v_p = interf_p.convolver.weighted_densities.n1v
# n_2v_p = interf_p.convolver.weighted_densities.n2v
# n_disp_p = interf_p.convolver.weighted_densities.n["w_disp"]
# n_soft_rep_p = interf_p.convolver.weighted_densities.n["w_soft_rep"]

# T_m = T - eps_T
# vle_m = equilibrium.bubble_pressure(thermopack, T_m, z=np.ones(1))
# interf_m = PlanarInterface.from_profile(vle_m, interf.profile, domain_size=100.0, n_grid=1024, invert_states=True)
# interf_m.single_convolution()
# F_m = interf_m.get_excess_helmholtz_energy_density()
# n_0_m = interf_m.convolver.weighted_densities.n0
# n_1_m = interf_m.convolver.weighted_densities.n1
# n_2_m = interf_m.convolver.weighted_densities.n2
# n_3_m = interf_m.convolver.weighted_densities.n3
# n_1v_m = interf_m.convolver.weighted_densities.n1v
# n_2v_m = interf_m.convolver.weighted_densities.n2v
# n_disp_m = interf_m.convolver.weighted_densities.n["w_disp"]
# n_soft_rep_m = interf_m.convolver.weighted_densities.n["w_soft_rep"]

# vol_fac = (interf.functional.thermo.sigma[0]/interf.functional.grid_reducing_lenght)**3
# s_num = -interf.functional.thermo.eps_div_kb[0]*vol_fac*(F_p-F_m)/(2*eps_T)
# s = interf.get_excess_entropy_density()
# plt.plot(interf.grid.z, s_num,label="Numerical")
# plt.plot(interf.grid.z, s,label="Analytical")
# leg = plt.legend(loc="best", numpoints=1, frameon=False)
# plt.show()
# plt.clf()

# interf.convolver.convolve_density_profile_T(interf.profile.densities)

# dndT = interf.convolver.weighted_densities_T.n["w0"]
# dndT_num = (n_0_p-n_0_m)/(2*eps_T)
# plt.plot(interf.grid.z, dndT_num,label="Num. n0")
# plt.plot(interf.grid.z, dndT,label="Anal. n0")

# dndT = interf.convolver.weighted_densities_T.n["w1"]
# dndT_num = (n_1_p-n_1_m)/(2*eps_T)
# plt.plot(interf.grid.z, dndT_num,label="Num. n1")
# plt.plot(interf.grid.z, dndT,label="Anal. n1")

# dndT = interf.convolver.weighted_densities_T.n["w2"]
# dndT_num = (n_2_p-n_2_m)/(2*eps_T)
# plt.plot(interf.grid.z, dndT_num,label="Num. n2")
# plt.plot(interf.grid.z, dndT,label="Anal. n2")

# dndT = interf.convolver.weighted_densities_T.n["w3"]
# dndT_num = (n_3_p-n_3_m)/(2*eps_T)
# plt.plot(interf.grid.z, dndT_num,label="Num. n3")
# plt.plot(interf.grid.z, dndT,label="Anal. n3")

# dndT = interf.convolver.weighted_densities_T.n["wv2"]
# dndT_num = (n_2v_p-n_2v_m)/(2*eps_T)
# plt.plot(interf.grid.z, dndT_num,label="Num. nv2")
# plt.plot(interf.grid.z, dndT,label="Anal. nv2")

# dndT = interf.convolver.weighted_densities_T.n["wv1"]
# dndT_num = (n_1v_p-n_1v_m)/(2*eps_T)
# plt.plot(interf.grid.z, dndT_num,label="Num. nv1")
# plt.plot(interf.grid.z, dndT,label="Anal. nv1")

# comp = 0
# dndT = interf.convolver.weighted_densities_T.n["w_disp"]
# dndT_num = (n_disp_p[comp,:]-n_disp_m[comp,:])/(2*eps_T)
# plt.plot(interf.grid.z, dndT_num,label="Num. n_disp")
# plt.plot(interf.grid.z, dndT[comp,:],label="Anal. n_disp")

# dndT = interf.convolver.weighted_densities_T.n["w_soft_rep"]
# dndT_num = (n_soft_rep_p[comp,:]-n_soft_rep_m[comp,:])/(2*eps_T)
# plt.plot(interf.grid.z, dndT_num,label="Num. n_soft_rep")
# plt.plot(interf.grid.z, dndT[comp,:],label="Anal. n_soft_rep")

# leg = plt.legend(loc="best", numpoints=1, frameon=False)
# plt.show()


sigma = interf.functional.thermo.sigma[0]
eps = interf.functional.thermo.eps_div_kb[0]*KB
len_fac = interf.functional.grid_reducing_lenght/interf.functional.thermo.sigma[0]

s_scaling = NA*sigma**3/interf.functional.thermo.Rgas
s_E = interf.get_excess_entropy_density()
plt.plot(interf.grid.z*len_fac, s_E,label=r"Functional")
plt.plot([interf.grid.z[0]*len_fac], s_scaling*np.array([vle.liquid.specific_excess_entropy()/vle.liquid.specific_volume()]),
         label=r"Bulk liquid", linestyle="None", marker="o")
plt.plot([interf.grid.z[-1]*len_fac], s_scaling*np.array([vle.vapor.specific_excess_entropy()/vle.vapor.specific_volume()]),
         label=r"Bulk vapour", linestyle="None", marker="o")
plt.ylabel(r"$s_{\rm{E}}^*$")
plt.xlabel("$z/\sigma$")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.savefig("ljs_entropy_T_is_0.75.pdf")
plt.show()

# Plot profile
interf.plot_property_profiles(prop=Properties.ENTROPY,
                              plot_reduced_property=True,
                              plot_equimolar_surface=True,
                              plot_bulk=True,
                              include_legend=True,
                              grid_unit=LenghtUnit.REDUCED)


a_E = interf.get_excess_free_energy_density()
p = interf.parallel_pressure()
sum_rho_mu_E = interf.get_excess_chemical_potential_density_sum()
h_E = interf.get_excess_enthalpy_density()
#T_star*s_E + sum_rho_mu_E
u_E = interf.get_excess_energy_density()
#a_E + T_star*s_E

plt.plot(interf.grid.z*len_fac, p,label=r"Functional")
p_scaling = sigma**3/eps
plt.plot([interf.grid.z[0]*len_fac], p_scaling*np.array([vle.liquid.pressure()]),
         label=r"Bulk liquid", linestyle="None", marker="o")
plt.plot([interf.grid.z[-1]*len_fac], p_scaling*np.array([vle.vapor.pressure()]),
         label=r"Bulk vapour", linestyle="None", marker="o")
plt.ylabel(r"$p^*$")
plt.xlabel("$z/\sigma$")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.savefig("ljs_pressure_T_is_0.75.pdf")
plt.show()

#
interf.plot_property_profiles(prop=Properties.PARALLEL_PRESSURE,
                              plot_reduced_property=True,
                              plot_equimolar_surface=True,
                              plot_bulk=True,
                              include_legend=True,
                              grid_unit=LenghtUnit.REDUCED)

energy_scaling = sigma**3/eps
plt.plot(interf.grid.z*len_fac, a_E,label=r"Functional")
plt.plot([interf.grid.z[0]*len_fac], energy_scaling*np.array([vle.liquid.specific_excess_free_energy()/vle.liquid.specific_volume()]),
         label=r"Bulk liquid", linestyle="None", marker="o")
plt.plot([interf.grid.z[-1]*len_fac], energy_scaling*np.array([vle.vapor.specific_excess_free_energy()/vle.vapor.specific_volume()]),
         label=r"Bulk vapour", linestyle="None", marker="o")
plt.ylabel(r"$a_{\rm{E}}^*$")
plt.xlabel("$z/\sigma$")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.savefig("ljs_free_energy_T_is_0.75.pdf")
plt.show()

#
interf.plot_property_profiles(prop=Properties.FREE_ENERGY,
                              plot_reduced_property=True,
                              plot_equimolar_surface=True,
                              plot_bulk=True,
                              include_legend=True,
                              grid_unit=LenghtUnit.REDUCED)

plt.plot(interf.grid.z*len_fac, h_E,label=r"Functional")
plt.plot([interf.grid.z[0]*len_fac], energy_scaling*np.array([vle.liquid.specific_excess_enthalpy()/vle.liquid.specific_volume()]),
         label=r"Bulk liquid", linestyle="None", marker="o")
plt.plot([interf.grid.z[-1]*len_fac], energy_scaling*np.array([vle.vapor.specific_excess_enthalpy()/vle.vapor.specific_volume()]),
         label=r"Bulk vapour", linestyle="None", marker="o")
plt.ylabel(r"$h_{\rm{E}}^*$")
plt.xlabel("$z/\sigma$")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.savefig("ljs_enthalpy_T_is_0.75.pdf")
plt.show()

interf.plot_property_profiles(prop=Properties.ENTHALPY,
                              plot_reduced_property=True,
                              plot_equimolar_surface=True,
                              plot_bulk=True,
                              include_legend=True,
                              grid_unit=LenghtUnit.REDUCED)

plt.plot(interf.grid.z*len_fac, u_E,label=r"Functional")
plt.plot([interf.grid.z[0]*len_fac], energy_scaling*np.array([vle.liquid.specific_excess_energy()/vle.liquid.specific_volume()]),
         label=r"Bulk liquid", linestyle="None", marker="o")
plt.plot([interf.grid.z[-1]*len_fac], energy_scaling*np.array([vle.vapor.specific_excess_energy()/vle.vapor.specific_volume()]),
         label=r"Bulk vapour", linestyle="None", marker="o")
plt.ylabel(r"$u_{\rm{E}}^*$")
plt.xlabel("$z/\sigma$")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.savefig("ljs_energy_T_is_0.75.pdf")
plt.show()

interf.plot_property_profiles(prop=Properties.ENERGY,
                              plot_reduced_property=True,
                              plot_equimolar_surface=True,
                              plot_bulk=True,
                              include_legend=True,
                              grid_unit=LenghtUnit.REDUCED)

plt.plot(interf.grid.z*len_fac, sum_rho_mu_E,label=r"Functional")
plt.plot([interf.grid.z[0]*len_fac], energy_scaling*vle.liquid.excess_chemical_potential()/vle.liquid.specific_volume(),
         label=r"Bulk liquid", linestyle="None", marker="o")
plt.plot([interf.grid.z[-1]*len_fac], energy_scaling*vle.vapor.excess_chemical_potential()/vle.vapor.specific_volume(),
         label=r"Bulk vapour", linestyle="None", marker="o")
plt.ylabel(r"$\mu_{\rm{E}}^*$")
plt.xlabel("$z/\sigma$")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.savefig("ljs_chempot_T_is_0.75.pdf")
plt.show()

interf.plot_property_profiles(prop=Properties.CHEMPOT_SUM,
                              plot_reduced_property=True,
                              plot_equimolar_surface=True,
                              plot_bulk=True,
                              include_legend=True,
                              grid_unit=LenghtUnit.REDUCED)

# # Set up Barker-Henderson model
# bh_functional_kwargs={"psi_disp": psi_disp}
# thermopack_bh = ljs_bh()
# thermopack_bh.init("Ar")
# thermopack.set_tmin(0.5*thermopack.eps_div_kb)
# vle_bh = equilibrium.bubble_pressure(thermopack_bh, T, z=np.ones(1))

# # Define interface with initial tanh density profile
# interf_bh = PlanarInterface.from_tanh_profile(vle_bh,
#                                               thermopack_bh.critical_temperature(1),
#                                               domain_size=100.0,
#                                               n_grid=1024,
#                                               invert_states=True,
#                                               functional_kwargs=bh_functional_kwargs)

# # Solve for equilibrium profile
# interf_bh.solve(log_iter=True)

# # Plot profile
# interf_bh.plot_equilibrium_density_profiles(plot_reduced_densities=True,
#                                             plot_equimolar_surface=True,
#                                             grid_unit=LenghtUnit.REDUCED)

# # Set up WCA model
# thermopack_wca = ljs_wca()
# thermopack_wca.init("Ar")
# thermopack.set_tmin(0.5*thermopack.eps_div_kb)
# vle_wca = equilibrium.bubble_pressure(thermopack_wca, T, z=np.ones(1))

# # Define interface with initial tanh density profile
# interf_wca = PlanarInterface.from_tanh_profile(vle_wca,
#                                               thermopack_wca.critical_temperature(1),
#                                               domain_size=100.0,
#                                               n_grid=1024,
#                                                invert_states=True,
#                                                functional_kwargs=functional_kwargs)

# # Solve for equilibrium profile
# interf_wca.solve(log_iter=True)

# # Plot profile
# interf_wca.plot_equilibrium_density_profiles(plot_reduced_densities=True,
#                                             plot_equimolar_surface=True,
#                                             grid_unit=LenghtUnit.REDUCED)

# s_E_bh = interf_bh.get_excess_entropy_density()
# s_E_wca = interf_wca.get_excess_entropy_density()
# plt.plot(interf.grid.z*len_fac, s_E,label=r"UV")
# plt.plot(interf.grid.z*len_fac, s_E_bh,label=r"BH")
# plt.plot(interf.grid.z*len_fac, s_E_wca,label=r"WCA")
# plt.ylabel(r"$s_{\rm{E}}^*$")
# plt.xlabel("$z/\sigma$")
# leg = plt.legend(loc="best", numpoints=1, frameon=False)
# plt.savefig("ljs_entropy.pdf")
# plt.show()
