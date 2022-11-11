"""Simple set of (unit)tests for thermopack_dft."""
import numpy as np
from pyctp.pcsaft import pcsaft
from pyctp.thermopack_state import equilibrium
from src.interface import PlanarInterface, SphericalInterface
#from src.constants import LenghtUnit, NA, KB, Properties
from pytest import approx
import pytest

@pytest.mark.parametrize('inpt', [{"T": 140.0, "gamma": 8.077618584439763, "tolman": 0.6114414737046836} ])
def test_pcsaft_dispersion_surface_tension(inpt):
     # Set up thermopack and equilibrium state
    thermopack = pcsaft()
    thermopack.init("C1")
    T = inpt["T"]
    vle = equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))

    sigma0 = PlanarInterface.from_tanh_profile(vle,
                                               thermopack.critical_temperature(1),
                                               domain_size=100.0,
                                               n_grid=1024).solve().surface_tension_real_units()

    # Define interface with initial tanh density profile
    spi = SphericalInterface.from_tanh_profile(vle,
                                               thermopack.critical_temperature(1),
                                               radius=25.0,
                                               domain_radius=50.0,
                                               n_grid=1024,
                                               calculate_bubble=False,
                                               sigma0=sigma0)

    # Solve for equilibrium profile
    spi.solve()

    # Surface tension
    gamma_s, r_s, tolman = spi.surface_of_tension()
    gamma_s *= 1.0e3
    tolman *= 1.0e10
    print(f"Surface of tension: {gamma_s} mN/m")
    print(f"Tolman length: {tolman} Å")

    # Test result
    assert(gamma_s == approx(inpt["gamma"], rel=1.0e-6))
    assert(tolman == approx(inpt["tolman"], rel=1.0e-6))



