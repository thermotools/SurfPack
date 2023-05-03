"""Simple set of (unit)tests for thermopack_dft."""
import numpy as np
from thermopack.pcsaft import pcsaft
from thermopack.thermopack_state import Equilibrium
from src.interface import PlanarInterface, SphericalInterface
#from src.constants import LenghtUnit, NA, KB, Properties
from pytest import approx
import pytest

@pytest.mark.parametrize('inpt', [{"T": 140.0, "gamma": 8.08457702550937, "r_s": 25.213970735985264} ])
def test_pcsaft_dispersion_surface_tension(inpt):
     # Set up thermopack and equilibrium state
    thermopack = pcsaft()
    thermopack.init("C1")
    T = inpt["T"]
    vle = Equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))

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
    gamma_s, r_s = spi.surface_of_tension()
    gamma_s *= 1.0e3
    r_s *= 1.0e10
    print(f"Surface tension: {gamma_s} mN/m")
    print(f"R_s: {r_s} Å")

    # Test result
    assert(gamma_s == approx(inpt["gamma"], rel=1.0e-6))
    assert(r_s == approx(inpt["r_s"], rel=1.0e-6))



