# Run script for the cDFT code that interfaces with Thermopack

import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pyCDFT.cdft import cdft1D, cdft_thermopack
from pyCDFT.utility import boundary_condition, density_from_packing_fraction, \
    get_data_container, plot_data_container, \
    quadratic_polynomial, densities
from pyCDFT.constants import DEBUG, LCOLORS
import pyCDFT.ng_extrapolation
from pyCDFT.fmt_functionals import bulk_weighted_densities
from pyCDFT.geometry_solvers import picard_geometry_solver

# Initialize Thermopack
cdft_tp = cdft_thermopack(model="PC-SAFT",
                          comp_names="C1",
                          comp=np.array([1.0]),
                          temperature=100.0,
                          pressure=0.0,
                          bubble_point_pressure=True,
                          domain_length=50,
                          grid=1024)

# Initialize the solver
solver = picard_geometry_solver(cDFT=cdft_tp, alpha_min=0.1, alpha_max=0.5,
                                alpha_initial=0.025, n_alpha_initial=250,
                                ng_extrapolations=10, line_search="ERROR",
                                density_init="VLE")

# Make the calculations
# solver.minimise(print_frequency=250,
#                    plot="ERROR",
#                    tolerance=1.0e-10)
solver.anderson_mixing(mmax=50, beta=0.05, tolerance=1.0e-10,
                       log_iter=True, use_scipy=False)
# Plot the profiles
solver.plot_equilibrium_density_profiles(
    xlim=[25.75, 35.0], ylim=[0.97, 1.005])

# Print the surface tension
print("gamma", cdft_tp.surface_tension_real_units(solver.densities))
