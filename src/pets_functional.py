#!/usr/bin/env python3
import numpy as np
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from constants import NA, RGAS, LenghtUnit
from pcsaft_functional import saft_dispersion
from pyctp.pets import pets

class PeTS_functional(saft_dispersion):
    """
    Functional as published by Heier et al. 2018:
    Equation of state for the Lennard-Jones truncated and shifted fluid
    with a cut-off radius of 2.5 sigma based on perturbation theory and
    its applications to interfacial thermodynamics, Molecular Physics,
    116:15-16, 2083-2094, DOI: 10.1080/00268976.2018.1447153
    """

    def __init__(self, N, eos: pets, T_red, psi_disp=1.21, grid_unit=LenghtUnit.ANGSTROM):
        """
        Set up PeTS functional
        Args:
            N (int): Size of grid
            eos (pets): Thermopack object
            T_red (float): Reduced temperature
            psi_disp (float): Width for weighted dispersion density
            grid_unit (LenghtUnit): Unit used for grid
        """
        saft_dispersion.__init__(self,
                                 N,
                                 eos,
                                 T_red,
                                 psi_disp=psi_disp,
                                 grid_unit=grid_unit)
        self.name += "(Perturbed truncated and shifted Lennard-Jones"
        self.short_name = "PeTS"


def surface_tension_LJTS():
    """
    J. Vrabec et al. (2006)
    Comprehensive study of the vapour–liquid coexistence of the truncated
    and shifted Lennard–Jones fluid including planar and spherical
    interface properties, Molecular Physics, 104:9, 1509-1527
    doi: 10.1080/00268970600556774
    """
    data = {}
    data["T"] = np.array([0.625, 0.650, 0.675, 0.700, 0.725, 0.750,
                          0.775, 0.800, 0.825, 0.850, 0.875, 0.900,
                          0.925, 0.950, 0.975, 1.000, 1.025, 1.050])
    data["gamma"] = np.array([0.7279, 0.6759, 0.6339, 0.5818, 0.5499, 0.4938,
                              0.4558, 0.4037, 0.3697, 0.3227, 0.2756, 0.2396,
                              0.1916, 0.1566, 0.1225, 0.0815, 0.0574, 0.0304])
    return data


if __name__ == "__main__":
    # Model testing
    pass
