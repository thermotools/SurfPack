#!/usr/bin/env python3
import numpy as np
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from constants import NA, RGAS, LenghtUnit
from pcsaft_functional import saft_dispersion
from pyctp.saftvrmie import saftvrmie
from pyctp.saftvrqmie import saftvrqmie

class saftvrqmie_functional(saft_dispersion):
    """

    """

    def __init__(self, N, svrqm: saftvrqmie, T_red, psi_disp=1.3862, grid_unit=LenghtUnit.ANGSTROM):
        """

        Args:
            N (int): Size of grid
            svrqm (saftvrqmie): Thermopack object
            T_red (float): Reduced temperature
            psi_disp (float): Width for weighted dispersion density
            grid_unit (LenghtUnit): Unit used for grid
        """
        saft_dispersion.__init__(self,
                                 N,
                                 svrqm,
                                 T_red,
                                 psi_disp=psi_disp,
                                 grid_unit=grid_unit)
        self.name = "SAFTVRQ-MIE"
        self.short_name = "SVRQM"

class saftvrmie_functional(saft_dispersion):
    """

    """

    def __init__(self, N, svrm: saftvrmie, T_red, psi_disp=1.3862, grid_unit=LenghtUnit.ANGSTROM):
        """

        Args:
            N (int): Size of grid
            svrm (saftvrmie): Thermopack object
            T_red (float): Reduced temperature
            psi_disp (float): Width for weighted dispersion density
            grid_unit (LenghtUnit): Unit used for grid
        """
        saft_dispersion.__init__(self,
                                 N,
                                 svrm,
                                 T_red,
                                 psi_disp=psi_disp,
                                 grid_unit=grid_unit)
        self.name = "SAFTVR-MIE"
        self.short_name = "SVRM"


if __name__ == "__main__":
    # Model testing
    pass
