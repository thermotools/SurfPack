"""Simple set of (unit)tests for thermopack_dft."""
import numpy as np
from pyctp.saftvrqmie import saftvrqmie
from pyctp.thermopack_state import equilibrium
from src.interface import PlanarInterface
from src.constants import LenghtUnit, Properties
import sys
import matplotlib.pyplot as plt
from fmt_functionals import bulk_weighted_densities
from src.dft_numerics import dft_solver
import time

def test_saftvrqmie_surface_tension():
    """Test saftvrqmie functional"""

    # Set up thermopack and equilibrium state
    thermopack = saftvrqmie()
    thermopack.init("H2,Ne",additive_hard_sphere_reference=True)
    T = 24.59
    thermopack.set_tmin(5.0)
    z=np.array([0.0144,1.0-0.0144])
    vle = equilibrium.bubble_pressure(thermopack, T, z)
    Tc, _, _ = thermopack.critical(z)
    n_grid = 512
    domain_size=200.0

    # Define interface with initial tanh density profile
    interf = PlanarInterface.from_tanh_profile(vle,
                                               Tc,
                                               domain_size=domain_size,
                                               n_grid=n_grid)
    # Solve for equilibrium profile
    interf.solve(log_iter=True)
    # Plot profile
    # interf.plot_property_profiles(plot_reduced_property=True,
    #                           plot_equimolar_surface=False,
    #                           plot_bulk=True,
    #                           include_legend=True,
    #                           grid_unit=LenghtUnit.REDUCED)

    # Test differentials
    # interf.single_convolution()
    # for wf in interf.convolver.weighted_densities.wfs:
    #     interf.test_functional_differential(wf,ic=0)
    #     interf.test_functional_differential(wf,ic=1)

    # Test temperature differentials
    # eps_T = 1.0e-5
    # T_p = T + eps_T
    # T_m = T - eps_T
    # interf.single_convolution()
    # # Test dFdT
    # dFdT = interf.functional.temperature_differential(interf.convolver.weighted_densities)
    # vle_pt = equilibrium.bubble_pressure(thermopack, T_p, z=np.array([0.4,0.6]))
    # interf_pt = PlanarInterface.from_profile(vle_pt, interf.profile, domain_size=domain_size, n_grid=n_grid)
    # F_pt = interf_pt.functional.excess_free_energy(interf.convolver.weighted_densities)
    # vle_mt = equilibrium.bubble_pressure(thermopack, T_m, z=np.array([0.4,0.6]))
    # interf_mt = PlanarInterface.from_profile(vle_mt, interf.profile, domain_size=domain_size, n_grid=n_grid)
    # F_mt = interf_mt.functional.excess_free_energy(interf.convolver.weighted_densities)

    # dFdT_num = (F_pt-F_mt)/(2*eps_T)
    # plt.plot(interf.grid.z, dFdT_num,label="Num. dFdT")
    # plt.plot(interf.grid.z, dFdT,label="Anal. dFdT")
    # #plt.plot(interf.grid.z, dFdT_num-dFdT,label="Diff. dFdT")
    # leg = plt.legend(loc="best", numpoints=1, frameon=False)
    # plt.show()
    # plt.clf()

    # Test bulk properties
    interf.test_functional_in_bulk()
    # Calculate surface tension
    gamma = interf.surface_tension_real_units()*1.0e3
    print(f"SAFT-VRQ Mie surface tension {gamma} mN/m")
    print(f"SAFT-VRQ Mie (feos) surface tension 4.24514776308888 mN/m")

if __name__ == "__main__":
    st = time.process_time()
    test_saftvrqmie_surface_tension()
    et = time.process_time()
    res = et - st
    print('CPU Execution time:', res, 'seconds')
