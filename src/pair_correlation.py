#!/usr/bin/env python3
from grid import Grid
from profile import Profile
from bulk import Bulk
import numpy as np
import fmt_functionals
from pore import Pore
from thermopack import pcsaft
from convolver import Convolver
from constants import NA, KB, Geometry, Specification, LenghtUnit
from dft_numerics import dft_solver
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


class PairCorrelation(Pore):

    def __init__(self,
                 state,
                 comp=0,
                 domain_size=15.0,
                 n_grid=1024):
        """
        Object holding specifications for classical DFT problem.
        Reduced particle size assumed to be d=1.0, and all other sizes are relative to this scale.

        Args:
            bulk_densities (ndarray): Bulk fluid density ()
            particle_diameters (ndarray): Particle diameter
            wall (str): Wall type (HardWall, SlitHardWall, None)
            domain_length (float): Length of domain
            functional (str): Name of hard sphere functional: Rosenfeld, WhiteBear, WhiteBear Mark II, Default Rosenfeld
            grid (int) : Grid size
            temperature (float): Reduced temperature
            quadrature (str): Quadrature to use during integration
        Returns:
            None
        """

        v_ext = []
        for i in functional.nc:
            v_ext.append(functional.potential(i, comp, state.T))
        # Init of base class
        Pore.__init__(self,
                      geometry=Geometry.SPHERICAL,
                      thermopack=state.eos,
                      temperature=state.T,
                      external_potential=v_ext,
                      domain_size=domain_size,
                      n_grid=n_grid)

        self.constant_profile(state)


if __name__ == "__main__":
    pass
