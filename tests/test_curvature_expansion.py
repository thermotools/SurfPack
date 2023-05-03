"""Simple set of (unit)tests for thermopack_dft."""
import numpy as np
from thermopack.pcsaft import pcsaft
from thermopack.thermopack_state import Equilibrium
from src.curvature_expansion import CurvatureExpansionInterface
from pytest import approx
import pytest


@pytest.mark.parametrize('inpt', [{"T": 140.0, "tolman": 0.0949322276640371, "k": -0.6431094215959555, "k_bar": 0.38224182944537377}])
def test_curvature_expoansion(inpt):
    """Test curvature expansion calculation"""

    thermopack = pcsaft()
    thermopack.init("C1")
    T = inpt["T"]
    thermopack.set_tmin(0.1*thermopack.eps_div_kb[0])
    vle = Equilibrium.bubble_pressure(thermopack, T, z=np.ones(1))
    cei = CurvatureExpansionInterface(vle)
    cei.solve(log_iter=False)
    tolman, k, k_bar = cei.get_curvature_corrections(reduced_unit=True)
    print(f"Methane tolman length {tolman} and Helfrich coefficients: {k} {k_bar}")

    # Test result
    assert((tolman, k, k_bar) ==
           approx((inpt["tolman"], inpt["k"], inpt["k_bar"]), rel=1.0e-6))
