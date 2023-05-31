"""Simple set of (unit)tests for thermopack_dft."""
import numpy as np
from thermopack.pcsaft import pcsaft
from thermopack.thermopack_state import Equilibrium
from src.interface import PlanarInterface
from src.constants import LenghtUnit, NA, KB, Properties
import sys
from src.dft_numerics import dft_solver

def test_pcsaft_mixture_surface_tension():
    """Test PC-SAFT functional"""

    # Set up thermopack and equilibrium state
    thermopack = pcsaft()
    thermopack.init("C1,N2")
    T = 111.667
    thermopack.set_tmin(0.5*thermopack.eps_div_kb[0])
    vle = Equilibrium.bubble_pressure(thermopack, T, z=np.array([0.5,0.5]))
    print(vle)
    print(vle.pressure)
    Tc, _, _ = thermopack.critical(np.array([0.5,0.5]))
    #print(Tc)
    # Define interface with initial tanh density profile
    interf = PlanarInterface.from_tanh_profile(vle,
                                               Tc,
                                               domain_size=200.0,
                                               n_grid=1024)



    # interf.test_functional_in_bulk()
    # sys.exit()
    # interf.single_convolution()
    # interf.test_grand_potential_bulk()
    # sys.exit()
    # Plot profile
    interf.plot_property_profiles(plot_reduced_property=True,
                              plot_equimolar_surface=False,
                              plot_bulk=True,
                              include_legend=True)

    solver=dft_solver().picard(tolerance=1.0e-10,max_iter=1,beta=0.05,ng_frequency=None)
    #.\
    #    anderson(tolerance=1.0e-10,max_iter=200,beta=0.05)
    solver=dft_solver()
    # Solve for equilibrium profile
    interf.solve(solver=solver,log_iter=True)
    # Test differentials
    # interf.single_convolution()
    #alias = "w_rho_hc"
    #alias = "w_lambda_hc"
    #alias = "wv2"
    #alias = "w_disp"
    #interf.test_functional_differential(alias,ic=0)
    #interf.test_functional_differential(alias,ic=1)

    interf.test_grand_potential_bulk()

    # Plot profile
    interf.plot_property_profiles(plot_reduced_property=True,
                              plot_equimolar_surface=False,
                              plot_bulk=True,
                              include_legend=True)

    #interf.test_functional_in_bulk()

    # Calculate surface tension
    gamma = interf.surface_tension_real_units()*1.0e3
    print(f"PC-SAFT surface tension {gamma} mN/m")
    print(f"PC-SAFT (feos) surface tension 6.634872247359591 mN/m")

if __name__ == "__main__":
    test_pcsaft_mixture_surface_tension()
