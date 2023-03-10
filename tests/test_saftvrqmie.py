"""Simple set of (unit)tests for thermopack_dft."""
import numpy as np
from thermopack.saftvrqmie import saftvrqmie
from thermopack.thermopack_state import Equilibrium
from src.interface import PlanarInterface
#from src.constants import LenghtUnit, NA, KB, Properties
from pytest import approx
import pytest


@pytest.mark.parametrize('inpt', [{"T": 20.369, "gamma": 0.0018429563521940614},
                                  {"T": 25.0, "gamma": 0.0010842767231806062},
                                  {"T": 30.0, "gamma": 0.00032485786159997345}])
def test_hydrogen_surface_tension(inpt):
    """Test SAFT-VRQ Mie functional for hydrogen"""

    # Set up thermopack and equilibrium state
    thermopack = saftvrqmie()
    thermopack.init("H2")
    T = inpt["T"]
    thermopack.set_tmin(0.5*thermopack.eps_div_kb)
    vle = Equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))
    # Define interface with initial tanh density profile
    interf = PlanarInterface.from_tanh_profile(vle,
                                               thermopack.critical_temperature(
                                                   1),
                                               domain_size=200.0,
                                               n_grid=512)

    # Solve for equilibrium profile
    interf.solve()

    # Calculate surface tension
    gamma = interf.surface_tension_real_units()
    print(f"Hydrogen surface tension {gamma} N/m")

    # Test result
    assert(gamma == approx(inpt["gamma"], rel=1.0e-5))


def test_saftvrqmie_mixture_surface_tension():
    """Test SAFT-VRQ Mie functional for hydrogen-neno"""

    # Set up thermopack and equilibrium state
    thermopack = saftvrqmie()
    thermopack.init("H2,Ne", additive_hard_sphere_reference=True)
    T = 24.59
    thermopack.set_tmin(5.0)
    z = np.array([0.0144, 1.0-0.0144])
    vle = Equilibrium.bubble_pressure(thermopack, T, z)
    Tc, _, _ = thermopack.critical(z)
    n_grid = 512
    domain_size = 200.0

    # Define interface with initial tanh density profile
    interf = PlanarInterface.from_tanh_profile(vle,
                                               Tc,
                                               domain_size=domain_size,
                                               n_grid=n_grid)
    # Solve for equilibrium profile
    interf.solve()

    # Calculate surface tension
    gamma = interf.surface_tension_real_units()*1.0e3
    print(f"SAFT-VRQ Mie surface tension {gamma} mN/m")
    print(f"SAFT-VRQ Mie (feos) surface tension 4.24514776308888 mN/m")

    # Test result
    assert(gamma == approx(4.24514776308888, rel=1.0e-5))
