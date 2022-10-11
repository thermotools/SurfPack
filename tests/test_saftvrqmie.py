"""Simple set of (unit)tests for thermopack_dft."""
import numpy as np
from pyctp.saftvrqmie import saftvrqmie
from pyctp.thermopack_state import equilibrium
from src.interface import PlanarInterface
#from src.constants import LenghtUnit, NA, KB, Properties
from pytest import approx
import pytest


@pytest.mark.parametrize('inpt', [{"T": 20.369, "gamma": 0.0018429563519096466},
                                  {"T": 25.0, "gamma": 0.0010842767252945544},
                                  {"T": 30.0, "gamma": 0.00032485807783920787}])
def test_hydrogen_surface_tension(inpt):
    """Test SAFT-VRQ Mie functional for hydrogen"""

    # Set up thermopack and equilibrium state
    thermopack = saftvrqmie()
    thermopack.init("H2")
    T = inpt["T"]
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
    print(f"Hydrogen surface tension {gamma} N/m")

    # Test result
    assert(gamma == approx(inpt["gamma"], rel=1.0e-8))
