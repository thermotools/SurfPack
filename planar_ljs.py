# Run script for the cDFT code that interfaces with Thermopack

import numpy as np
import sys
from pyctp.ljs_wca import ljs_uv
from pyctp.ljs_bh import ljs_bh
from pyctp.thermopack_state import equilibrium
from src.interface import PlanarInterface
from src.constants import LenghtUnit, NA
import matplotlib.pyplot as plt

# Set up thermopack and equilibrium state
thermopack = ljs_uv()
thermopack.init("Ar")
T = 0.75*thermopack.eps_div_kb[0]
thermopack.set_tmin(0.5*thermopack.eps_div_kb)
vle = equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))

# Define interface with initial tanh density profile
interf = PlanarInterface.from_tanh_profile(vle,
                                           thermopack.critical_temperature(1),
                                           domain_size=100.0,
                                           n_grid=1024,
                                           invert_states=True)

# Solve for equilibrium profile
interf.solve(log_iter=True)

# Plot profile
interf.plot_equilibrium_density_profiles(plot_reduced_densities=True,
                                         plot_equimolar_surface=True,
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

len_fac = interf.functional.grid_reducing_lenght/interf.functional.thermo.sigma[0]
s_scaling = NA*interf.functional.thermo.sigma[0]**3/interf.functional.thermo.Rgas
s_E = interf.get_excess_entropy_density()
plt.plot(interf.grid.z*len_fac, s_E,label=r"$s_{\rm{E}}^*$ functional")
plt.plot([interf.grid.z[0]*len_fac], s_scaling*np.array([vle.liquid.specific_excess_entropy()/vle.liquid.specific_volume()]),
         label=r"$s_{\rm{E}}^*$ bulk liquid", linestyle="None", marker="o")
plt.plot([interf.grid.z[-1]*len_fac], s_scaling*np.array([vle.vapor.specific_excess_entropy()/vle.vapor.specific_volume()]),
         label=r"$s_{\rm{E}}^*$ bulk vapour", linestyle="None", marker="o")
plt.ylabel(r"$s_{\rm{E}}^*$")
plt.xlabel("$z/\sigma$")
leg = plt.legend(loc="best", numpoints=1, frameon=False)
plt.savefig("ljs_T_is_0.75.pdf")
plt.show()
