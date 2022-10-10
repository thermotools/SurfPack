"""Simple set of (unit)tests for thermopack_dft."""
import numpy as np
from pyctp.ljs_wca import ljs_uv, ljs_wca
from pyctp.ljs_bh import ljs_bh
from pyctp.thermopack_state import equilibrium
from src.interface import PlanarInterface
#from src.constants import LenghtUnit, NA, KB, Properties
from pytest import approx
import pytest


@pytest.mark.parametrize('inpt', [{"functional_kwargs":{}, "gamma": 0.22309092128554645},
                                  {"functional_kwargs":{"psi_disp": 1.4, "psi_soft_rep": 1.4},
                                   "gamma": 0.34711467901407866}])
def test_ljs_surface_tension(inpt):
    """Test ljs functional"""

    # Set up thermopack and equilibrium state
    thermopack = ljs_uv()
    thermopack.init("Ar")
    T_star = 0.7
    T = T_star*thermopack.eps_div_kb[0]
    thermopack.set_tmin(0.5*thermopack.eps_div_kb)
    vle = equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))
    # Define interface with initial tanh density profile
    interf = PlanarInterface.from_tanh_profile(vle,
                                               thermopack.critical_temperature(1),
                                               domain_size=200.0,
                                               n_grid=1024,
                                               invert_states=True,
                                               functional_kwargs=inpt["functional_kwargs"])

    # Solve for equilibrium profile
    interf.solve()

    # Calculate surface tension
    gamma = interf.surface_tension()
    print(f"LJs surface tension {gamma}")

    # Test result
    assert(gamma == approx(inpt["gamma"], rel=1.0e-8))

def test_ljs_entropy():
    """Test ljs entropy"""

    domain_size=200.0
    n_grid=32

    # Set up thermopack and equilibrium state
    thermopack = ljs_uv()
    thermopack.init("Ar")
    T_star = 0.7
    T = T_star*thermopack.eps_div_kb[0]
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
    assert(s == approx(s_num, rel=1.0e-6))

def test_ljs_proporties():
    """Test ljs entropy"""

    domain_size=100.0
    n_grid=128

    # Set up thermopack and equilibrium state
    thermopack = ljs_bh()
    thermopack.init("Ar")
    T_star = 0.7
    T = T_star*thermopack.eps_div_kb[0]
    thermopack.set_tmin(0.5*thermopack.eps_div_kb)
    vle = equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))
    # Define interface with initial tanh density profile
    interf = PlanarInterface.from_tanh_profile(vle,
                                               thermopack.critical_temperature(1),
                                               domain_size=domain_size,
                                               n_grid=n_grid)

    # Perform single convolution
    interf.solve()

    # Get properties
    a_E = interf.get_excess_free_energy_density()
    p = interf.parallel_pressure()
    h_E = interf.get_excess_enthalpy_density()
    u_E = interf.get_excess_energy_density()
    s_E = interf.get_excess_entropy_density()

    idx = 64
    print(f"a_E={a_E[idx]}, h_E={h_E[idx]}, u_E={u_E[idx]}, s_E={s_E[idx]}, p={p[idx]}")
    # Test result
    assert((a_E[idx], h_E[idx], u_E[idx], s_E[idx], p[idx]) ==
           approx((-0.6300822402716728, -1.732418370953793, -1.1989924891427794, -0.8127289269587238, -0.17852334676992881), rel=1.0e-8))
