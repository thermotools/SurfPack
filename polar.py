# Run script for the cDFT code that interfaces with Thermopack

import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pyCDFT.cdft import cdft1D, cdft_thermopack
from pyCDFT.utility import boundary_condition, density_from_packing_fraction, \
    get_data_container, plot_data_container, \
    quadratic_polynomial, densities
from pyCDFT.constants import DEBUG, LCOLORS, Geometry, Specification
import pyCDFT.ng_extrapolation
from pyCDFT.fmt_functionals import bulk_weighted_densities
from pyCDFT.geometry_solvers import picard_geometry_solver
from pyCDFT.weight_functions_polar import polar
# Initialize Thermopack
# cdft_tp = cdft_thermopack(model="PC-SAFT",
#                           comp_names="C1",
#                           comp=np.array([1.0]),
#                           temperature=140.0,
#                           pressure=0.0,
#                           bubble_point_pressure=True,
#                           domain_length=20.0,
#                           grid=24,
#                           geometry=Geometry.SPHERICAL,
#                           no_bc=True)

pol = polar(R=0.5,
            domain_size=3.0,
            n_grid=4096)
pol.test_weigthed_densities()
sys.exit()

#  # wf_i = [[31.235829714960325,
          
# pol = polar(R=1.5780162409077065,
#             domain_size=15.0,
#             n_grid=64)

# #pol.test_weigthed_densities()
# pol.test_pd()
# #pol.tests()
# sys.exit()


cdft_tp = cdft_thermopack(model="SAFT-VRQ Mie",
                          comp_names="H2",
                          comp=np.array([1.0]),
                          temperature=25.0,
                          pressure=0.0,
                          bubble_point_pressure=True,
                          domain_length=20.0,
                          grid=4048,
                          geometry=Geometry.POLAR,
                          no_bc=True,
                          kwthermoargs={"feynman_hibbs_order": 1,
                                        "parameter_reference": "AASEN2019-FH1"})
cdft_tp.thermo.print_saft_parameters(1)
print(cdft_tp.thermo.hard_sphere_diameters(25.0))

# Initialize the solver
solver = picard_geometry_solver(cDFT=cdft_tp, alpha_min=0.1, alpha_max=0.5,
                                alpha_initial=0.025, n_alpha_initial=500,
                                ng_extrapolations=None, line_search="NONE",
                                density_init="VLE",
                                specification=Specification.NUMBER_OF_MOLES)

# Make the calculations
# solver.minimise(print_frequency=250,
#                 plot="ERROR",
#                 tolerance=1.0e-10)
solver.picard_iteration_if(beta=0.05, tolerance=1.0e-10,
                           log_iter=True, max_iter=500)

solver.anderson_mixing(mmax=50, beta=0.05, tolerance=1.0e-10,
                       log_iter=True, use_scipy=False, max_iter=200)
# Plot the profiles
solver.plot_equilibrium_density_profiles(plot_equimolar_surface=True)

# Print the surface tension
print("gamma", cdft_tp.surface_tension_real_units(solver.densities))
