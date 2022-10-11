"""Simple set of (unit)tests for thermopack_dft."""
import numpy as np
from pyctp.pets import pets
from pyctp.thermopack_state import equilibrium
from src.interface import PlanarInterface
#from src.constants import LenghtUnit, NA, KB, Properties
from pytest import approx
import pytest


@pytest.mark.parametrize('inpt', [{"Ts": 0.625, "gamma": 0.7158132731155862},
                                  {"Ts": 0.8, "gamma": 0.40430364035092536}])
def test_ljs_surface_tension(inpt):
    """Test PeTS functional"""

    # Set up thermopack and equilibrium state
    thermopack = pets()
    thermopack.init("Ar")
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
    gamma = interf.surface_tension()
    print(f"PeTS surface tension {gamma}")

    # Test result
    assert(gamma == approx(inpt["gamma"], rel=1.0e-8))