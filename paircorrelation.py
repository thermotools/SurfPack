# Run script for the cDFT code that interfaces with Thermopack

import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pyCDFT.cdft import cdft1D, cdft_thermopack
from pyCDFT.utility import boundary_condition, density_from_packing_fraction, \
    get_data_container, plot_data_container, \
    quadratic_polynomial, densities
from pyCDFT.constants import DEBUG, LCOLORS, Specification, Geometry, \
    Lenght_unit
import pyCDFT.ng_extrapolation
from pyCDFT.fmt_functionals import bulk_weighted_densities
from pyCDFT.geometry_solvers import picard_geometry_solver

# Initialize Thermopack

temperature = 27.5
cdft_tp = cdft_thermopack(model="SAFT-VRQ Mie",
                          comp_names="H2",
                          comp=np.array([1.0]),
                          temperature=temperature,
                          pressure=8.34275808728558e5,
                          bubble_point_pressure=False,
                          domain_length=15.0/3.13444844,
                          grid=1024,
                          geometry=Geometry.SPHERICAL,
                          no_bc=True,
                          kwthermoargs={"feynman_hibbs_order": 1,
                                        "parameter_reference": "AASEN2019-FH1"})
cdft_tp.thermo.print_saft_parameters(1)
print(cdft_tp.thermo.hard_sphere_diameters(temperature))
#sys.exit()

#3.27761130924748e-5 m³/mol


print(cdft_tp.eps, cdft_tp.T, cdft_tp.beta)
r_red = np.linspace(cdft_tp.dr/2, cdft_tp.domain_length - cdft_tp.dr/2, cdft_tp.N)
cdft_tp.Vext[0][:] = cdft_tp.thermo.potential(1,1,
                                              r_red*cdft_tp.functional.d_hs[0],
                                              temperature)/cdft_tp.thermo.eps_div_kb[0]

#plt.plot(cdft_tp.Vext[0][:])
#plt.show()

# Initialize the solver
solver = picard_geometry_solver(cDFT=cdft_tp, alpha_min=0.1, alpha_max=0.5,
                                alpha_initial=0.025, n_alpha_initial=500,
                                ng_extrapolations=None, line_search="NONE",
                                density_init="VEXT",
                                specification=Specification.CHEMICHAL_POTENTIAL)
#sys.exit()
# Make the calculations
# solver.minimise(print_frequency=250,
#                 plot="ERROR",
#                 tolerance=1.0e-10)
#solver.picard_iteration_if(beta=0.05, tolerance=1.0e-10,
#                           log_iter=True, max_iter=600)

solver.anderson_mixing(mmax=50, beta=0.05, tolerance=1.0e-10,
                       log_iter=True, use_scipy=False, max_iter=200)
# Plot the profiles
data_dict = {}
data_dict["filename"] = "hydrogen_rdf.dat"
data_dict["y"] = [1]
data_dict["x"] = 0
data_dict["labels"] = ["feos"]
data_dict["colors"] = ["b"]
solver.plot_equilibrium_density_profiles(data_dict=data_dict,
                                         unit=Lenght_unit.ANGSTROM,
                                         xlim=[2.0, 12.0])

# Print the surface tension
print("gamma", cdft_tp.surface_tension_real_units(solver.densities))
