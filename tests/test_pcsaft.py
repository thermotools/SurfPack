"""Simple set of (unit)tests for thermopack_dft."""
import numpy as np
from pyctp.pcsaft import pcsaft
from pyctp.thermopack_state import equilibrium
from src.interface import PlanarInterface
#from src.constants import LenghtUnit, NA, KB, Properties
from pytest import approx
import pytest

@pytest.mark.parametrize('inpt', [{"Ts": 0.6, "gamma": 4.444379370511722},
                                  {"Ts": 0.8, "gamma": 3.6704466235468596}])
def test_pcsaft_dispersion_surface_tension(inpt):
    """Test ljs functional"""

    # Set up thermopack and equilibrium state
    thermopack = pcsaft()
    thermopack.init("C1")
    T_star = inpt["Ts"]
    T = T_star*thermopack.eps_div_kb[0]
    thermopack.set_tmin(0.5*thermopack.eps_div_kb)
    vle = equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))
    # Define interface with initial tanh density profile
    interf = PlanarInterface.from_tanh_profile(vle,
                                               thermopack.critical_temperature(1),
                                               domain_size=200.0,
                                               n_grid=512)

    # Solve for equilibrium profile
    interf.solve()

    # Calculate surface tension
    gamma = interf.surface_tension_real_units()
    print(f"PC-SAFT surface tension {gamma}")

    # Test result
    assert(gamma == approx(inpt["gamma"], rel=1.0e-8))

def test_pcsaft_dispersion_entropy():
    """Test PC-SAFT entropy"""

    domain_size=200.0
    n_grid=32

    # Set up thermopack and equilibrium state
    thermopack = pcsaft()
    thermopack.init("C1")
    T = 130.0
    thermopack.set_tmin(0.5*thermopack.eps_div_kb)
    vle = equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))
    # Define interface with initial tanh density profile
    interf = PlanarInterface.from_tanh_profile(vle,
                                               thermopack.critical_temperature(1),
                                               domain_size=domain_size,
                                               n_grid=n_grid)

    # Perform single convolution
    interf.single_convolution()

    # Perturbate in temperature using density profile from solution
    eps_T = 1.0e-5
    T_p = T + eps_T
    vle_p = equilibrium.bubble_pressure(thermopack, T_p, z=np.ones(1))
    interf_p = PlanarInterface.from_profile(vle_p,
                                            interf.profile,
                                            domain_size=domain_size,
                                            n_grid=n_grid)

    interf_p.single_convolution()
    F_p = interf_p.get_excess_free_energy_density()
    T_m = T - eps_T
    vle_m = equilibrium.bubble_pressure(thermopack, T_m, z=np.ones(1))
    interf_m = PlanarInterface.from_profile(vle_m,
                                            interf.profile,
                                            domain_size=domain_size,
                                            n_grid=n_grid)
    interf_m.single_convolution()
    F_m = interf_m.get_excess_free_energy_density()

    s_num = -interf.functional.thermo.eps_div_kb[0]*(F_p-F_m)/(2*eps_T)
    s = interf.get_excess_entropy_density()

    print((s - s_num)/s_num)
    # Test result
    assert(s == approx(s_num, rel=1.0e-7))
