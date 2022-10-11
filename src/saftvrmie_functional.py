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
        # Set up non-additive correction
        self.na_enabled = False
        self.delta_ij = None
        self.delta_T_ij = None
        self.d_ij = None
        self.d_T_ij = None
        if svrqm.nc > 1:
            _, self.na_enabled = svrqm.test_fmt_compatibility()
        if self.na_enabled:
            self.d_ij = np.zeors((svrqm.nc, svrqm.nc))
            self.delta_ij = np.zeors((svrqm.nc, svrqm.nc))
            self.d_T_ij = np.zeors((svrqm.nc, svrqm.nc))
            self.delta_T_ij = np.zeors((svrqm.nc, svrqm.nc))
            for i in range(svrqm.nc):
                self.d_ij[i,i] = self.d_hs[i]
                self.d_T_ij[i,i] = self.d_T_hs[i]
                self.delta_ij[i,i] = self.d_ij[i,i]
                self.delta_T_ij[i,i] = self.d_T_ij[i,i]
                for j in range(i+1,svrqm.nc):
                    self.d_ij[i,j] = 0.5*(self.d_hs[i]+self.d_hs[j])
                    self.d_ij[j,i] = self.d_ij[i,j]
                    self.d_T_ij[i,j] = 0.5*(self.d_T_hs[i]+self.d_T_hs[j])
                    self.d_T_ij[j,i] = self.d_T_ij[i,j]
                    self.delta_ij[i,j], self.delta_T_ij[i,j] = svrqm.hard_sphere_diameters_ij(i+1, j+1, self.T)
                    self.delta_ij[j,i] = self.delta_ij[i,j]
                    self.delta_T_ij[j,i] = self.delta_T_ij[i,j]

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
