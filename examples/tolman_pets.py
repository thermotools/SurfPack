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
from curvature_expansion import CurvatureExpansionInterface

thermopack = pets()
T_star = [0.625, 0.7, 0.741, 0.8, 0.9, 1.0]
thermopack.set_tmin(0.1*thermopack.eps_div_kb[0])

solver=dft_solver().anderson(mmax=50, beta=0.05, tolerance=1.0e-12,max_iter=500)


T = 1.0*thermopack.eps_div_kb[0]
vle = Equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))
cei = CurvatureExpansionInterface(vle,n_grid=4096)
cei.solve(log_iter=True)
sys.exit()

#CurvatureExpansionInterface()




tolman = []
for Ts in T_star:
    T = Ts*thermopack.eps_div_kb[0]
    vle = Equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))
    interf = PlanarInterface.from_tanh_profile(vle, thermopack.critical_temperature(1), domain_size=200.0, n_grid=4096, invert_states=True)
    # Solve for equilibrium profile
    interf.solve(solver=solver, log_iter=True)

    tolman.append(interf.get_surface_of_tension(reduced_unit=True))
    print(tolman[-1])

print(tolman)
sys.exit()
# interf = PlanarInterface.from_tanh_profile(vle, thermopack.critical_temperature(1), domain_size=l, n_grid=512, invert_states=True)

# IV2=20
# I2=0
# I3=10
# interf.single_convolution()
# print("R",interf.convolver.comp_wfs[0]["w2"].R)
# print("w2",interf.convolver.comp_wfs[0]["w2"].w_conv_steady)
# print("w3",interf.convolver.comp_wfs[0]["w3"].w_conv_steady)
# print("wv2",interf.convolver.comp_wfs[0]["wv2"].prefactor_evaluated)
# prefac = {}
# prefac["w2"] = interf.convolver.comp_wfs[0]["w2"].w_conv_steady
# prefac["w3"] = interf.convolver.comp_wfs[0]["w3"].w_conv_steady
# prefac["wv2"] = interf.convolver.comp_wfs[0]["wv2"].prefactor_evaluated
# interf.convolver.comp_wfs[0]["w2"].generate_planar_rigidity_fourier_weights()
# interf.convolver.comp_wfs[0]["w3"].generate_planar_rigidity_fourier_weights()
# interf.convolver.comp_wfs[0]["wv2"].generate_planar_rigidity_fourier_weights()
# interf.convolver.comp_wfs[0]["w_disp"].generate_planar_rigidity_fourier_weights()
#interf.convolver.comp_wfs[0]["w2"].w_conv_steady

# plt.figure()
# plt.plot(interf.convolver.comp_wfs[0]["w2"].k_grid/(2*np.pi))
# plt.plot(data_k)
# plt.title("k")
# plt.show()


# alias = "w3"
# I0 = I3

# for i in range(0,5):
#     if i == 0:
#         fw = interf.convolver.comp_wfs[0][alias].fw_complex
#     elif i == 1:
#         fw = interf.convolver.comp_wfs[0][alias].fzw_minus
#     elif i == 2:
#         fw = interf.convolver.comp_wfs[0][alias].fzw_pluss
#     elif i == 3:
#         fw = interf.convolver.comp_wfs[0][alias].fw_tilde_pluss
#     elif i == 4:
#         fw = interf.convolver.comp_wfs[0][alias].fw_tilde_minus
#     plt.figure()
#     plt.plot(fw.imag, label="tp")
#     plt.plot(data_fmt[:,I0+5+i],label="itt")
#     plt.title(alias + " imag " + str(i))
#     #/prefac[alias]
#     plt.figure()
#     plt.plot(fw.real,label="tp")
#     plt.plot(data_fmt[:,I0+i],label="itt")
#     plt.title(alias + " real " + str(i))
# plt.show()
# sys.exit()

print(vle,thermopack.critical_temperature(1))
# Define interface with initial tanh density profile
interf = PlanarInterface.from_tanh_profile(vle, thermopack.critical_temperature(1), domain_size=200.0, n_grid=4096, invert_states=True)

# Solve for equilibrium profile
interf.solve(solver=solver, log_iter=True)

# Plot profile
#interf.plot_property_profiles(plot_equimolar_surface=True, continue_after_plotting=True)

print(interf.get_surface_of_tension(reduced_unit=True))
